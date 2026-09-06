"""Groq API client for durable-notebook's eval-only gates (Gate 1, Gate 2).

Reads GROQ_API_KEY from environments/durable_notebook/.env via
python-dotenv, or from a real environment variable if already set. This
module never reads or prints the key itself -- load_dotenv() populates
os.environ directly, and only the variable's *presence*, never its value,
is checked here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from verifiers.clients import OpenAIChatCompletionsClient

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class _StripNullReasoningContentTransport(httpx.AsyncHTTPTransport):
    """verifiers always serializes AssistantMessage.reasoning_content into
    the outgoing request, even when it's None (it's an always-present field
    on the type, not something we can omit by not setting it). Groq's
    request schema rejects the key's mere presence on a non-reasoning
    model's assistant turns ("property 'reasoning_content' is unsupported"),
    which surfaced as every multi-turn rollout's 2nd+ request failing with a
    400 the moment history included a prior assistant turn. Confirmed via a
    live run before this fix existed. Strip the key here, at the transport
    layer, rather than patching verifiers' client code directly."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json") and request.content:
            try:
                body = json.loads(request.content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            if isinstance(body, dict) and isinstance(body.get("messages"), list):
                changed = False
                for msg in body["messages"]:
                    if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
                        continue
                    # reasoning_content: Groq rejects this key on a replayed
                    # assistant input message outright, value or not -- not
                    # just when null. First diagnosed against a non-reasoning
                    # Llama model where verifiers always sends it as null, so
                    # the initial fix only stripped null values. That missed
                    # qwen3.6-27b, a reasoning model whose own responses
                    # populate it with real text: 12 of 15 Gate 1 episodes
                    # died to the identical error before this was corrected
                    # to an unconditional strip.
                    if "reasoning_content" in msg:
                        del msg["reasoning_content"]
                        changed = True
                    # tool_calls: Groq's schema wants this key omitted, not
                    # present-as-null, when there were no tool calls -- but
                    # DOES want the real list present when there were.
                    if msg.get("tool_calls", "not-present") is None:
                        del msg["tool_calls"]
                        changed = True
                if changed:
                    new_content = json.dumps(body).encode("utf-8")
                    # The body shrank, but request.headers still carries the
                    # old Content-Length -- must drop it so httpx recomputes
                    # from the new body instead of declaring a stale, now
                    # too-large length (confirmed live: this exact bug
                    # produced "Too much data for declared Content-Length").
                    headers = httpx.Headers(request.headers)
                    del headers["content-length"]
                    request = httpx.Request(
                        method=request.method,
                        url=request.url,
                        headers=headers,
                        content=new_content,
                        extensions=request.extensions,
                    )
        return await super().handle_async_request(request)


def _patch_openai_sdk_for_groq_service_tier() -> None:
    """Groq's chat completion responses set service_tier to Groq-specific
    values (e.g. "on_demand") that aren't in the openai SDK's hardcoded
    Literal of OpenAI's own tier names, so strict pydantic validation
    rejects every response outright before verifiers ever sees it. Widen
    the field to `str` -- confirmed via a local round-trip test that this
    is enough for parsing to succeed; nothing here depends on the tier
    value itself.
    """
    if ChatCompletion.model_fields["service_tier"].annotation is not Optional[str]:
        ChatCompletion.model_fields["service_tier"].annotation = Optional[str]
        ChatCompletion.model_rebuild(force=True)


def get_groq_vf_client(max_retries: int = 16) -> OpenAIChatCompletionsClient:
    """A verifiers Client wrapping a raw AsyncOpenAI client pointed at Groq,
    suitable to pass directly as `client=` to env.evaluate_sync()/generate_sync().

    `max_retries` relies on the openai SDK's own retry logic, which reads
    the real `Retry-After` value Groq returns (confirmed against SDK source:
    openai._base_client._calculate_retry_timeout) and honors waits up to
    its own MAX_RETRY_AFTER_DELAY (120s). That's enough to ride out the
    per-minute TPM limit we hit live (resets were all well under 90s), and
    it correctly does NOT retry a daily-cap 429 (those came back asking to
    wait 8+ minutes) rather than hanging a run -- verifiers' own
    max_retries on generate()/evaluate() doesn't cover RateLimitError at
    all (only vf.InfraError/InvalidModelResponseError by default), so this
    SDK-level retry is what actually helps, not that one.

    Default raised 5 -> 16 on 2026-08-18. Measured, not guessed: a
    12-episode Gate 1 batch at max_concurrent=2 lost 3 of its first 6
    rollouts, and every one of them died to a TPM 429 (per-minute, limit
    8000) carrying a Retry-After of 1-10 seconds -- not to the daily cap.
    Five retries is too few when a multi-turn rollout issues a dozen
    requests and any of them can collide: the waits are seconds, while an
    abort forfeits the whole episode's already-spent tokens and returns no
    data. A daily-cap 429 asks for 8+ minutes, still above the SDK's
    120s MAX_RETRY_AFTER_DELAY, so this does NOT make a TPD-exhausted run
    hang -- it still fails fast on the one limit that should stop a run.
    """
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GROQ_API_KEY not set. Create {ENV_PATH} with a line "
            "GROQ_API_KEY=gsk_... (see docs/specs for setup notes)."
        )
    _patch_openai_sdk_for_groq_service_tier()
    http_client = httpx.AsyncClient(transport=_StripNullReasoningContentTransport())
    raw_client = AsyncOpenAI(
        api_key=api_key, base_url=GROQ_BASE_URL, http_client=http_client,
        max_retries=max_retries,
    )
    return OpenAIChatCompletionsClient(raw_client)
