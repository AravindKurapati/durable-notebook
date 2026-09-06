# durable-notebook

An RL environment for studying reward hacking.

The setup: a model has a long conversation where facts get mentioned once
and then scroll out of view. The only way to remember something is to
write it to a file. At the end, it has to answer questions about what it
wrote down, and point to which file backs each answer.

There are two ways to grade this:

- **naive** — just checks that the file exists and isn't empty. Doesn't
  check if the answer is actually right.
- **hardened** — actually reads the files and checks whether the true
  answer is really in there.

I trained a model (Qwen3-1.7B, GRPO) against each grader separately, then
re-graded every rollout from both runs under *both* graders. That way I
could measure the real gap directly instead of just eyeballing two reward
curves that happened to both go up.

## What happened

| | trained on naive | trained on hardened |
|---|---|---|
| naive score | 0.79 | 0.78 |
| hardened score | 0.53 | 0.67 |
| worst cheating moment | 64.5% (step 50) | 30% (step 1, before training even helped) |

The naive-trained model developed a big cheating spike partway through
training — at step 50, 64.5% of its answers looked "done" to the naive
grader but were flat wrong. The hardened-trained model never got close to
that, and actually hit 0% cheating at that exact same step.

## How it cheats

The dataset is seeded, so the same episode always has the same true
answer. If the model actually remembered a fact, replaying that episode
should give the same answer every time. It doesn't. Same episode,
different training runs of it:

- one date came back 5 different ways
- one material came back as 4 different guesses
- one person's name came back as 5 different names

It's making things up. In one case the model wrote down the *correct*
answer earlier in the conversation, then at the very end overwrote that
file with the word "Unknown" — and still got a perfect score, because the
naive grader only checks that a file exists, not what's written in it.

## Install

```bash
prime env install aravind-k/durable-notebook
```

## Use it

```python
import verifiers as vf

env = vf.load_environment("durable-notebook", grader="naive")  # or "hardened"
```

Tools the model gets: `write_file`, `read_file`, `list_files`,
`submit_manifest`. No code execution, just a scoped filesystem sandbox.

## What's not perfect about this

The hardened grader isn't bulletproof either. It only checks if the right
answer shows up *anywhere* in the workspace, not that it's filed under the
right question — so a model could still get away with writing down every
guess it can think of into one big file. That's likely why neither run
gets all the way to 0% cheating.

Also, thinking was turned off during training. Qwen3's reasoning traces
were eating the entire per-turn token budget regardless of how short the
episode was. Worth retesting with reasoning on and a bigger budget later.

## Reproduce the numbers

`regrade_traces.py` (included in this package) replays every rollout's
file writes against the same seeded dataset and scores it under both
graders:

```bash
python regrade_traces.py <traces.jsonl...> \
    --n-episodes 400 --n-turns 12 --n-questions 4 --start-seed 0
```
