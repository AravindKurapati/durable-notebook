# durable-notebook, explained simply

## What this is

A test we built to catch an AI model cheating, on purpose, so we can then
train it not to.

## The task we give the model

The model sits through a long, one-sided conversation. Facts get mentioned
turn by turn ("the room capacity is 1018", "the deadline changed to
March 3rd"). But the model can't see the whole conversation forever, old
turns get dropped from view as new ones come in, like a chat scrolling
off screen. The only way to keep a fact around is to actually write it to
a file. At the end, the model gets asked questions about facts from way
earlier, and has to answer using whatever it wrote down.

If it never wrote something down, it has no way to know the answer. That's
the whole point: the task can only be solved by actually using files as
memory, not by just holding everything in its head.

## The cheat risk

Two ways of checking the model's answers:

- **The lazy check (naive grader).** Looks at the model's own summary of
  what it did and whether it sounds plausible, files exist, aren't empty,
  contain roughly the right kind of words. Doesn't verify anything is
  actually true.
- **The careful check (hardened grader).** Ignores whatever the model
  claims and just goes and reads the real files to see if the correct
  value is actually sitting in there.

A model being trained against the lazy check could learn to write
something that *sounds* right without doing the real work, confident
nonsense that would sail through the lazy check and fail the careful one.
That gap, lazy check says "great job", careful check says "you have no
idea", is what we're trying to measure and eventually train away.

## Why bother

Whoever trains real AI systems for a living runs into this constantly:
a cheap, automatic way of grading an AI's work almost always has some
loophole, and a model trained hard enough will find it. Building a small,
honest example of this, and showing you can catch it and fix it, is a
useful thing to have built and be able to talk through.

## Two checkpoints before spending real money

Training a model for real needs rented GPUs, so before doing that we set
two gates, checked with cheap, ordinary API calls first:

- **Gate 1: does the task actually need memory?** If a model can just
  answer the questions from what's still fresh in view, without ever
  writing anything down, the whole task is pointless. **Passed.**
- **Gate 2: is the lazy check actually fakeable, and is the careful check
  actually strict?** We run real models through both checks and look for
  a model that scores much higher on the lazy one than the careful one,
  proof the cheat is real and there's room for training to close it.
  **Running now.**

Only once both gates pass does actual GPU training start.

## After both gates pass

We'd train a model twice with reinforcement learning, once being scored
by the lazy check and once by the careful one, and compare how much the
model starts gaming the lazy check when given the chance versus how the
hardened check holds up. Then publish the results (curves, examples of
the model cheating caught red-handed, and the environment itself) as an
installable artifact other people can run.

## Where things stand right now

Gate 1 passed. Gate 2 is running live against a handful of real models.
Along the way we found and fixed two real bugs in our own grading code
that were making results look worse than they actually were, so what's
running now is the first trustworthy attempt at Gate 2. Once it finishes
we'll know whether to move on to real training.
