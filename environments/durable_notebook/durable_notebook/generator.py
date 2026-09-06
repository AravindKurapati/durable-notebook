"""Deterministic, procedural generator for durable-notebook episodes.

An episode is a sequence of turns in a fictional, generic scenario (event
planning, a software project, a renovation, ...). Some turns state a fact
about the scenario ("the venue deposit is $450"); some later turns update an
earlier fact ("the venue deposit changed to $500"); other turns are
distractor filler unrelated to any tracked fact. At the end, a fixed set of
questions probe recall of the facts, including whether an updated value was
correctly retained rather than the stale original.

Everything here is pure and seed-deterministic: the same seed always
produces the same episode, so a fixed train/held-out split can be generated
once and never regenerated per rollout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NAME_POOL = [
    "Priya Chen", "Marcus Webb", "Dana Okafor", "Liam Torres", "Yuki Tanaka",
    "Sofia Rossi", "Amara Diallo", "Noah Becker", "Elena Petrova", "Kwame Mensah",
    "Ingrid Larsen", "Tariq Hassan", "Mei Lin", "Owen Fitzgerald", "Ana Reyes",
]

FILLER_TURNS = [
    "By the way, do you think it'll rain later this week?",
    "Quick unrelated question: what's a good way to organize a bookshelf?",
    "Thanks, that's helpful context.",
    "Can you remind me what day of the week it is?",
    "I saw an interesting article about coral reefs today.",
    "Let's take a short break before continuing.",
    "What's your favorite way to make coffee?",
    "Just checking in, are you still with me?",
    "That reminds me of a trip I took a while back.",
    "No action needed here, just thinking out loud.",
    "Do you have a recommendation for a good podcast?",
    "This is unrelated, but I like your explanation style.",
]

SCENARIOS: dict[str, dict] = {
    "event_planning": {
        "fact_template": "For the event, the {label} is {value}.",
        "update_template": "Update: the {label} for the event has changed to {value}.",
        "slots": {
            "venue_deposit": {"label": "venue deposit", "type": "money"},
            "guest_count": {"label": "guest count", "type": "int", "range": (20, 300)},
            "catering_budget": {"label": "catering budget", "type": "money"},
            "deposit_due_date": {"label": "deposit due date", "type": "date"},
            "theme_color": {"label": "theme color", "type": "word",
                             "choices": ["navy", "emerald", "blush", "charcoal", "gold"]},
            "rsvp_deadline": {"label": "RSVP deadline", "type": "date"},
        },
    },
    "software_project": {
        "fact_template": "For the project, the {label} is {value}.",
        "update_template": "Update: the {label} for the project has changed to {value}.",
        "slots": {
            "sprint_length_days": {"label": "sprint length in days", "type": "int", "range": (5, 21)},
            "staging_db_engine": {"label": "staging database engine", "type": "word",
                                   "choices": ["postgres", "mysql", "sqlite", "mariadb", "cockroachdb"]},
            "api_rate_limit": {"label": "API rate limit", "type": "int", "range": (10, 5000)},
            "on_call_engineer": {"label": "on-call engineer", "type": "name"},
            "release_date": {"label": "release date", "type": "date"},
            "infra_budget": {"label": "infrastructure budget", "type": "money"},
        },
    },
    "apartment_renovation": {
        "fact_template": "For the renovation, the {label} is {value}.",
        "update_template": "Update: the {label} for the renovation has changed to {value}.",
        "slots": {
            "contractor_name": {"label": "contractor name", "type": "name"},
            "permit_number": {"label": "permit number", "type": "int", "range": (10000, 99999)},
            "budget_total": {"label": "total budget", "type": "money"},
            "paint_color": {"label": "paint color", "type": "word",
                             "choices": ["sage", "cream", "slate", "terracotta", "ivory"]},
            "flooring_material": {"label": "flooring material", "type": "word",
                                   "choices": ["oak", "bamboo", "tile", "polished concrete", "laminate"]},
            "completion_date": {"label": "completion date", "type": "date"},
        },
    },
    "conference_logistics": {
        "fact_template": "For the conference, the {label} is {value}.",
        "update_template": "Update: the {label} for the conference has changed to {value}.",
        "slots": {
            "keynote_speaker": {"label": "keynote speaker", "type": "name"},
            "room_capacity": {"label": "room capacity", "type": "int", "range": (50, 2000)},
            "av_vendor": {"label": "AV vendor", "type": "name"},
            "catering_headcount": {"label": "catering headcount", "type": "int", "range": (50, 2000)},
            "badge_color": {"label": "badge color", "type": "word",
                             "choices": ["red", "blue", "green", "yellow", "purple"]},
            "sponsor_budget": {"label": "sponsor budget", "type": "money"},
        },
    },
    "research_study": {
        "fact_template": "For the study, the {label} is {value}.",
        "update_template": "Update: the {label} for the study has changed to {value}.",
        "slots": {
            "irb_protocol_number": {"label": "IRB protocol number", "type": "int", "range": (1000, 9999)},
            "sample_size": {"label": "target sample size", "type": "int", "range": (20, 5000)},
            "enrollment_deadline": {"label": "enrollment deadline", "type": "date"},
            "data_monitoring_lead": {"label": "data monitoring lead", "type": "name"},
            "funding_amount": {"label": "funding amount", "type": "money"},
            "primary_site": {"label": "primary site city", "type": "word",
                              "choices": ["Boston", "Austin", "Chicago", "Denver", "Seattle"]},
        },
    },
}

MONEY_CHOICES = [250, 300, 450, 500, 620, 750, 900, 1200, 1500, 2000, 2800, 3500]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


@dataclass(frozen=True)
class Fact:
    id: str
    scenario: str
    slot: str
    label: str
    value: str
    turn_index: int
    is_update: bool
    supersedes: str | None


@dataclass(frozen=True)
class Turn:
    index: int
    kind: str  # "fact" | "filler"
    text: str
    fact_id: str | None


@dataclass(frozen=True)
class Question:
    id: str
    kind: str  # "single_hop" | "multi_hop" | "temporal_update"
    text: str
    answer: str
    supporting_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    seed: int
    scenario: str
    split: str  # "train" | "held_out"
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...]
    facts: tuple[Fact, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "scenario": self.scenario,
            "split": self.split,
            "turns": [t.__dict__ for t in self.turns],
            "questions": [
                {**q.__dict__, "supporting_fact_ids": list(q.supporting_fact_ids)}
                for q in self.questions
            ],
            "facts": [f.__dict__ for f in self.facts],
        }


def episode_from_dict(data: dict) -> Episode:
    """Inverse of Episode.to_dict(). Used to reconstruct an Episode from a
    dataset row's "info" column inside the running environment."""
    return Episode(
        episode_id=data["episode_id"],
        seed=data["seed"],
        scenario=data["scenario"],
        split=data["split"],
        turns=tuple(Turn(**t) for t in data["turns"]),
        questions=tuple(
            Question(**{**q, "supporting_fact_ids": tuple(q["supporting_fact_ids"])})
            for q in data["questions"]
        ),
        facts=tuple(Fact(**f) for f in data["facts"]),
    )


def slots_for_question(episode: Episode, question: Question) -> list[str]:
    """The distinct slot(s) a question depends on, in supporting_fact_ids
    order. Shared by naive.py, hardened.py, and judge.py -- was duplicated
    identically across the first two before being pulled out here."""
    fact_by_id = {f.id: f for f in episode.facts}
    slots: list[str] = []
    for fid in question.supporting_fact_ids:
        slot = fact_by_id[fid].slot
        if slot not in slots:
            slots.append(slot)
    return slots


def slot_label(episode: Episode, slot: str) -> str:
    for f in episode.facts:
        if f.slot == slot:
            return f.label
    raise KeyError(slot)


def true_value_for_slot(episode: Episode, slot: str) -> str:
    """The slot's current (i.e. latest, post-update) ground-truth value.
    Was duplicated in hardened.py as `_true_value_for_slot`."""
    matches = [f for f in episode.facts if f.slot == slot]
    matches.sort(key=lambda f: f.turn_index)
    return matches[-1].value


def _generate_value(rng: random.Random, scenario: dict, slot: str, exclude: str | None = None) -> str:
    spec = scenario["slots"][slot]
    kind = spec["type"]
    for _ in range(20):
        if kind == "money":
            value = f"${rng.choice(MONEY_CHOICES)}"
        elif kind == "date":
            month = rng.choice(MONTHS)
            day = rng.randint(1, 28)
            value = f"{month} {day}"
        elif kind == "int":
            lo, hi = spec.get("range", (5, 200))
            value = str(rng.randint(lo, hi))
        elif kind == "name":
            value = rng.choice(NAME_POOL)
        elif kind == "word":
            value = rng.choice(spec["choices"])
        else:
            raise ValueError(f"unknown slot type: {kind!r}")
        if value != exclude:
            return value
    return value  # exhausted retries (tiny value pool) -- accept the collision


def _split_for_seed(seed: int) -> str:
    return "held_out" if seed % 5 == 0 else "train"


def _generate_questions(
    rng: random.Random,
    facts: list[Fact],
    slot_latest: dict[str, Fact],
    n_questions: int,
) -> list[Question]:
    updated_slots = {f.slot for f in facts if f.is_update}
    single_slots = [s for s in slot_latest if s not in updated_slots]

    pool: list[Question] = []

    # Every question names the internal slot key(s) it's asking about,
    # e.g. "(key: keynote_speaker)". A live run showed why this is load
    # bearing, not decorative: without it, a model that answered every
    # fact correctly still scored 0, because it had no way to know what
    # string to put in a manifest entry's "slot" field and invented its
    # own numbering instead. The key also appears on the fact/update turn
    # itself (see generate_episode) so it survives compaction the same way
    # the value would need to.
    for slot in sorted(updated_slots):
        f = slot_latest[slot]
        pool.append(Question(
            id=f"q-temporal-{slot}",
            kind="temporal_update",
            text=f"What is the current {f.label}? (key: {slot})",
            answer=f.value,
            supporting_fact_ids=(f.id,),
        ))

    for slot in sorted(single_slots):
        f = slot_latest[slot]
        pool.append(Question(
            id=f"q-single-{slot}",
            kind="single_hop",
            text=f"What is the {f.label}? (key: {slot})",
            answer=f.value,
            supporting_fact_ids=(f.id,),
        ))

    all_slots = sorted(slot_latest.keys())
    rng.shuffle(all_slots)
    for i in range(0, len(all_slots) - 1, 2):
        s1, s2 = all_slots[i], all_slots[i + 1]
        f1, f2 = slot_latest[s1], slot_latest[s2]
        pool.append(Question(
            id=f"q-multi-{s1}-{s2}",
            kind="multi_hop",
            text=f"What are the {f1.label} (key: {s1}) and the {f2.label} (key: {s2})?",
            answer=f"{f1.label}: {f1.value}; {f2.label}: {f2.value}",
            supporting_fact_ids=(f1.id, f2.id),
        ))

    rng.shuffle(pool)
    return pool[:n_questions]


def generate_episode(
    seed: int,
    n_turns: int = 15,
    n_questions: int = 6,
    update_rate: float = 0.3,
) -> Episode:
    """Deterministically build one episode from `seed`. Same seed -> identical episode.

    Every slot gets exactly one initial fact turn; `update_rate` is the
    independent per-slot probability of one additional, always
    explicitly-labeled ("Update: ...") revisit. A slot is never silently
    overwritten by a second unlabeled "fact" turn -- every value change is
    textually signaled, so "single_hop" (never revisited) and
    "temporal_update" (revisited) are honest, non-overlapping categories.
    """
    rng = random.Random(seed)
    scenario_name = rng.choice(sorted(SCENARIOS))
    scenario = SCENARIOS[scenario_name]
    slots = sorted(scenario["slots"])
    rng.shuffle(slots)

    # Decide up front which slots get updated, so an "update" event can only
    # ever be scheduled after that same slot's "initial" event.
    slot_events: dict[str, list[str]] = {}
    for slot in slots:
        events = ["initial"]
        if rng.random() < update_rate:
            events.append("update")
        slot_events[slot] = events

    n_fact_turns = sum(len(events) for events in slot_events.values())
    if n_fact_turns > n_turns:
        raise ValueError(
            f"n_turns={n_turns} too small to fit {n_fact_turns} fact events "
            f"for scenario {scenario_name!r} ({len(slots)} slots)"
        )
    n_filler_turns = n_turns - n_fact_turns

    # Order fact events across slots while preserving each slot's own
    # initial-before-update order (pop from the front of each slot's queue).
    pending_slots = [s for s in slots if slot_events[s]]
    ordered_events: list[tuple[str, str]] = []  # (slot, event_kind)
    while pending_slots:
        slot = pending_slots[rng.randrange(len(pending_slots))]
        ordered_events.append((slot, slot_events[slot].pop(0)))
        if not slot_events[slot]:
            pending_slots.remove(slot)

    turn_kinds = ["fact"] * n_fact_turns + ["filler"] * n_filler_turns
    rng.shuffle(turn_kinds)
    fact_turn_indices = [i for i, k in enumerate(turn_kinds) if k == "fact"]

    slot_latest: dict[str, Fact] = {}
    facts: list[Fact] = []
    fact_by_turn: dict[int, Fact] = {}
    fact_counter = 0

    for turn_index, (slot, event_kind) in zip(fact_turn_indices, ordered_events):
        is_update = event_kind == "update"
        prior = slot_latest.get(slot)
        value = _generate_value(rng, scenario, slot, exclude=prior.value if prior else None)
        fact = Fact(
            id=f"f{fact_counter}",
            scenario=scenario_name,
            slot=slot,
            label=scenario["slots"][slot]["label"],
            value=value,
            turn_index=turn_index,
            is_update=is_update,
            supersedes=prior.id if is_update else None,
        )
        fact_counter += 1
        facts.append(fact)
        slot_latest[slot] = fact
        fact_by_turn[turn_index] = fact

    turns: list[Turn] = []
    for turn_index, kind in enumerate(turn_kinds):
        if kind == "fact":
            f = fact_by_turn[turn_index]
            template = scenario["update_template"] if f.is_update else scenario["fact_template"]
            # (key: ...) is the identifier the model must echo back in a
            # manifest entry's "slot" field -- it's otherwise never shown
            # anywhere, see the note in _generate_questions.
            text = template.format(label=f.label, value=f.value) + f" (key: {f.slot})"
            turns.append(Turn(index=turn_index, kind="fact", text=text, fact_id=f.id))
        else:
            filler = FILLER_TURNS[rng.randrange(len(FILLER_TURNS))]
            turns.append(Turn(index=turn_index, kind="filler", text=filler, fact_id=None))

    questions = _generate_questions(rng, facts, slot_latest, n_questions)

    return Episode(
        episode_id=f"{scenario_name}-{seed}",
        seed=seed,
        scenario=scenario_name,
        split=_split_for_seed(seed),
        turns=tuple(turns),
        questions=tuple(questions),
        facts=tuple(facts),
    )


def generate_dataset(
    n_episodes: int,
    start_seed: int = 0,
    **episode_kwargs,
) -> tuple[list[Episode], list[Episode]]:
    """Generate `n_episodes` episodes from consecutive seeds, split deterministically."""
    train: list[Episode] = []
    held_out: list[Episode] = []
    for seed in range(start_seed, start_seed + n_episodes):
        episode = generate_episode(seed, **episode_kwargs)
        (held_out if episode.split == "held_out" else train).append(episode)
    return train, held_out
