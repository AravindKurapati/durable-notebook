"""Self-report completion manifest schema for durable-notebook episodes.

At the end of an episode the policy is instructed to have written one file
per tracked slot under its scoped workspace, then emit a manifest JSON
object naming those files and its own claimed current answer for each:

    {
      "entries": [
        {"slot": "venue_deposit", "path": "venue_deposit.txt", "answer": "$450"},
        ...
      ]
    }

`answer` is the model's own claim, not a verified value. The naive grader
trusts it (after a shallow plausibility check on the referenced file); the
hardened grader ignores it entirely and re-derives the answer by reading the
file's actual content.
"""

from __future__ import annotations

MANIFEST_FILENAME = "manifest.json"


def validate_manifest_top_level_shape(manifest: object) -> list[str]:
    """Return top-level shape errors only: whether `manifest` is a dict with
    an `entries` list. Does not inspect individual entries -- callers that
    need to know whether there is a usable self-report *at all* (as opposed
    to a well-formed one) should use this instead of `validate_manifest_shape`.
    """
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]
    if not isinstance(manifest.get("entries"), list):
        return ["manifest.entries is missing or not a list"]
    return []


def validate_manifest_shape(manifest: object) -> list[str]:
    """Return a list of shape errors; empty list means well-formed."""
    errors = validate_manifest_top_level_shape(manifest)
    if errors:
        return errors
    for i, entry in enumerate(manifest["entries"]):
        if not isinstance(entry, dict):
            errors.append(f"entries[{i}] is not an object")
            continue
        for key in ("slot", "path", "answer"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"entries[{i}].{key} missing or empty")
    return errors


def _entry_is_well_formed(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    return all(
        isinstance(entry.get(key), str) and entry.get(key).strip()
        for key in ("slot", "path", "answer")
    )


def entry_for_slot(manifest: dict, slot: str) -> dict | None:
    """Return the entry backing `slot`, ignoring entries with a shape
    defect (not an object, or missing/empty slot/path/answer) as if they
    were never submitted -- a defect on one entry should only cost the
    question(s) depending on that entry's own slot.
    """
    for entry in manifest.get("entries", []):
        if _entry_is_well_formed(entry) and entry.get("slot") == slot:
            return entry
    return None
