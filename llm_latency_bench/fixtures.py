"""Deterministic benchmark text: a fictional asset register (system message) and user prompts about it.
Fixed seeds make every run send the same text. Sizes assume 4 characters per token."""

import random
from functools import cache

CHARS_PER_TOKEN = 4
_FIXTURE_SEED = 20260930

_DISTRICTS = ("Harbor", "Northgate", "Millbrook", "Eastfield", "Riverside", "Kingsway", "Oakridge", "Saltmarsh")
_ASSET_KINDS = ("pump station", "reservoir", "treatment works", "booster station", "water tower", "intake")
_FINDINGS = (
    "minor corrosion on the inlet valve",
    "a worn bearing on pump two",
    "sediment build-up in the wet well",
    "a faulty level sensor that reads 8% high",
    "no defects",
    "a cracked access hatch seal",
    "vibration above the alert threshold on the main motor",
    "a backup generator that failed its monthly load test",
)
_QUESTIONS = (
    "Which asset in the register has the oldest commissioning year, and what was its last inspection finding?",
    "List the three assets with the highest rated capacity and give their districts.",
    "Which district has the most assets in the register, and how many does it have?",
    "Which assets had a defect that affects pumping reliability? Name up to four.",
    "What is the combined rated capacity of all reservoirs in the register?",
    "Which asset was inspected least recently, and when was that?",
    "Name two assets that share a district and were commissioned in the same decade.",
    "Which asset has the lowest rated capacity, and what kind of asset is it?",
    "Summarise the most common inspection finding in the register in one sentence.",
    "If one crew can visit two assets per day, which four assets should they visit first and why?",
)


def _asset_record(rng: random.Random, index: int) -> str:
    district = rng.choice(_DISTRICTS)
    kind = rng.choice(_ASSET_KINDS)
    year = rng.randint(1952, 2021)
    capacity = rng.randint(4, 180) * 50
    inspected = f"20{rng.randint(22, 26)}-{rng.randint(1, 12):02d}"
    finding = rng.choice(_FINDINGS)
    return (
        f"Asset A-{index:04d}: {district} {kind}, commissioned {year}, "
        f"rated capacity {capacity} cubic metres per hour. "
        f"Last inspected {inspected}; the inspection found {finding}."
    )


def _field_note(rng: random.Random, question_index: int, index: int) -> str:
    district = rng.choice(_DISTRICTS)
    pressure = rng.randint(28, 71)
    finding = rng.choice(_FINDINGS)
    return (
        f"Note {question_index + 1}.{index + 1}: crew visit in {district}, mains pressure {pressure} metres head, "
        f"crew observed {finding}."
    )


def _fill(target_chars: int, pieces: list[str]) -> str:
    """Join pieces, one per line, until the target length is reached. Always keeps at least one piece."""
    chosen: list[str] = []
    length = -1  # The first piece has no newline before it.
    for piece in pieces:
        if chosen and length >= target_chars:
            break
        chosen.append(piece)
        length += len(piece) + 1
    return "\n".join(chosen)


@cache
def reference_context(target_tokens: int) -> str:
    """The register. Always a prefix of the same generated sequence, so larger sizes only add records."""
    rng = random.Random(_FIXTURE_SEED)
    target_chars = target_tokens * CHARS_PER_TOKEN
    records = [_asset_record(rng, index) for index in range(1, target_chars // 80 + 2)]
    return _fill(target_chars, records)


_ANALYST_BRIEF = (
    "Context for this question: I am preparing the weekly operations briefing for the network control room. "
    "The audience is duty managers who need to decide where to send maintenance crews in the next few days. "
    "They read the briefing on a phone between calls, so they need the answer first and the reasoning second. "
    "Please rely only on the asset register you were given, and say plainly when the register does not contain "
    "enough information to answer. Do not invent asset identifiers, dates, or capacities. When you mention an "
    "asset, use its identifier exactly as it appears in the register, for example A-0001. If two assets tie, "
    "mention both. Prefer concrete numbers over adjectives, and keep units as written in the register."
)


@cache
def user_prompt(target_tokens: int, question_index: int, answer_words: int) -> str:
    """One user message: a question, plus a brief and pasted notes when the size allows. Same input, same text."""
    question = _QUESTIONS[question_index]
    closing = f"Answer in at most {answer_words} words."
    target_chars = target_tokens * CHARS_PER_TOKEN
    fixed = f"{question} {closing}"
    if target_chars < len(fixed) + len(_ANALYST_BRIEF):
        return fixed

    brief = _ANALYST_BRIEF
    remaining = target_chars - len(fixed) - len(brief)
    if remaining < 200:
        return f"{brief}\n\n{fixed}"

    rng = random.Random(_FIXTURE_SEED + 1 + question_index)
    notes = [_field_note(rng, question_index, index) for index in range(remaining // 60 + 2)]
    pasted = _fill(remaining, notes)
    return f"{brief}\n\nHere are my crew's field notes from this week:\n{pasted}\n\n{fixed}"


MAX_TURNS = len(_QUESTIONS)
_FINAL_QUESTION = 0


def conversation_prompts(target_tokens: int, turn_count: int, answer_words: int) -> list[str]:
    """The user prompts of one scenario, in turn order.

    The final turn always sends the same prompt, so the final turns of 1-, 5- and 10-turn scenarios differ only in
    the history before them. The earlier turns ask the other questions in a fixed order, so the first turns of
    scenarios with more than one turn are identical too.
    """
    if not 1 <= turn_count <= MAX_TURNS:
        raise ValueError(f"turn count must be between 1 and {MAX_TURNS}, got {turn_count}")
    order = [*range(1, turn_count), _FINAL_QUESTION]
    return [user_prompt(target_tokens, question_index, answer_words) for question_index in order]


def system_prompt(session_nonce: str, context: str) -> str:
    # The nonce comes first, so scenarios cannot share a prompt-cache prefix.
    # Turns inside one conversation can still hit the cache, as in real use.
    return (
        f"Session {session_nonce}.\n"
        "You are an assistant for a water utility's operations team. Answer questions using the asset register below.\n"
        f"<asset_register>\n{context}\n</asset_register>"
    )
