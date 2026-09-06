# durable-notebook

An RL environment for studying reward hacking. Published to the Prime
Intellect Environments Hub as `aravind-k/durable-notebook`.

The setup: a model has a long conversation where facts get mentioned once
and then scroll out of view. The only way to remember something is to
write it to a file. At the end, it has to answer questions about what it
wrote down, and point to which file backs each answer.

Two ways to grade this:

- **naive** — just checks that the file exists and isn't empty. Doesn't
  check if the answer is actually right.
- **hardened** — actually reads the files and checks whether the true
  answer is really in there.

I trained a model (Qwen3-1.7B, GRPO) against each grader separately, then
re-graded every rollout from both runs under *both* graders. That way I
could measure the real gap directly instead of eyeballing two reward
curves that happened to both go up.

## What happened

![Hacking gap curves](docs/hacking_gap_curves.png)

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
different rollouts:

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

Or locally for development:

```bash
pip install -e environments/durable_notebook
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

Full technical writeup, including all the training/GPU debugging along the
way, is in [`docs/ANALYSIS_hacking_gap.md`](docs/ANALYSIS_hacking_gap.md).

## Reproduce the numbers

```bash
python eval/regrade_traces.py <traces.jsonl...> \
    --n-episodes 400 --n-turns 12 --n-questions 4 --start-seed 0
python eval/plot_hacking_gap.py eval/regrade_results/summary.json
```

Trace files live on the training run's output volume at
`<run_tag>/run_default/rollouts/step_<n>/train/effective/traces.jsonl`.
