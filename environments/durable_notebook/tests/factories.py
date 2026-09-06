"""Small builders for hand-crafted Episodes, used by grader unit tests that
need precise control over facts/questions rather than a procedurally
generated episode."""

from durable_notebook.generator import Episode, Fact, Question


def make_fact(id, slot, label, value, turn_index=0, is_update=False, supersedes=None, scenario="test"):
    return Fact(
        id=id, scenario=scenario, slot=slot, label=label, value=value,
        turn_index=turn_index, is_update=is_update, supersedes=supersedes,
    )


def make_question(id, kind, text, answer, supporting_fact_ids):
    return Question(id=id, kind=kind, text=text, answer=answer,
                     supporting_fact_ids=tuple(supporting_fact_ids))


def make_episode(facts, questions, episode_id="test-ep", scenario="test"):
    return Episode(
        episode_id=episode_id, seed=0, scenario=scenario, split="train",
        turns=(), questions=tuple(questions), facts=tuple(facts),
    )
