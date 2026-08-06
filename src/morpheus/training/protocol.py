"""The pre-sleep / WBTB conditioning protocol.

This is the evidence-backed core of Morpheus, and it is worth being explicit
about that: in the published trials the cue itself is inert without this. The
sound does nothing on its own — it works because it has been bound to a state of
critical self-awareness beforehand. A system that cued perfectly and trained
badly would be a worse version of a system that only trained.

So the trainer is not a wrapper around a wav file. It is the intervention, and
the cue is its delivery mechanism.

Adherence is recorded per step because it is the most plausible alternative
explanation for any positive result. If lucidity rises and training engagement
rose alongside it, the cue timing has explained nothing (design.md §14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StepKind(str, Enum):
    CUE = "cue"           # the cue is played during this step
    PROMPT = "prompt"     # user reflects, no input required
    ACTION = "action"     # user does something physical
    INPUT = "input"       # user types a response


@dataclass
class TrainingStep:
    key: str
    kind: StepKind
    title: str
    body: str
    seconds: int
    plays_cue: bool = False
    capture: Optional[str] = None  # field name when kind is INPUT


# The ordering follows the published TLR protocol. Steps 2-5 build the critical
# attitude, step 6 binds it to the cue, step 7 rehearses the target behaviour.
# Reordering these is a protocol change, not a UI tweak.
PROTOCOL: list[TrainingStep] = [
    TrainingStep(
        key="settle",
        kind=StepKind.PROMPT,
        title="Settle",
        body=(
            "Sit or lie comfortably. Let your breathing slow. You are not trying to "
            "fall asleep yet — you are getting attentive."
        ),
        seconds=45,
    ),
    TrainingStep(
        key="cue_intro",
        kind=StepKind.CUE,
        title="Listen to the cue",
        body=(
            "This is the sound. Give it your full attention. You are going to teach "
            "yourself that hearing it means: check whether this is a dream."
        ),
        seconds=20,
        plays_cue=True,
    ),
    TrainingStep(
        key="how_did_i_get_here",
        kind=StepKind.PROMPT,
        title="How did you get here?",
        body=(
            "Trace the last few minutes backwards, concretely. Where were you before "
            "this? And before that? In a dream this chain breaks or turns vague — "
            "that break is the thing you are training yourself to notice."
        ),
        seconds=60,
    ),
    TrainingStep(
        key="memory_review",
        kind=StepKind.PROMPT,
        title="Review the day",
        body=(
            "Recall three specific things that happened today, in order. Continuous, "
            "detailed autobiographical memory is the clearest difference between "
            "waking and dreaming. Dreams cannot usually produce it on demand."
        ),
        seconds=60,
    ),
    TrainingStep(
        key="scan_impossible",
        kind=StepKind.PROMPT,
        title="Look for what does not fit",
        body=(
            "Look around slowly. Is anything odd, out of place, impossible? Ask it "
            "genuinely, not as a formality. The habit only transfers into dreams if "
            "you actually mean it while awake."
        ),
        seconds=45,
    ),
    TrainingStep(
        key="reality_check",
        kind=StepKind.ACTION,
        title="Reality check",
        body=(
            "Pinch your nose closed and try to breathe in through it. Really try. In "
            "a dream the breath often passes through. Alternatively, read a line of "
            "text, look away, and read it again — dream text tends not to hold still."
        ),
        seconds=30,
    ),
    TrainingStep(
        key="cue_binding",
        kind=StepKind.CUE,
        title="Bind the cue",
        body=(
            "The sound plays again. As you hear it, imagine you are asleep and "
            "dreaming, and this sound reaches you inside the dream. Picture yourself "
            "recognising it and thinking clearly: this is a dream. Hold that."
        ),
        seconds=45,
        plays_cue=True,
    ),
    TrainingStep(
        key="rehearse_lucidity",
        kind=StepKind.PROMPT,
        title="Rehearse becoming lucid",
        body=(
            "In first person, vividly: you are in a dream, you hear the sound, you "
            "realise you are dreaming, and you stay calm and stay asleep. Rehearse "
            "the staying-calm part — excitement is what usually ends a lucid dream."
        ),
        seconds=60,
    ),
    TrainingStep(
        key="dream_signs",
        kind=StepKind.INPUT,
        title="Your dream signs",
        body=(
            "Name a recurring feature of your dreams — a place, person, situation, or "
            "feeling. Picture noticing it and becoming lucid. Leave blank if you do "
            "not have one yet; your journal will surface them over time."
        ),
        seconds=45,
        capture="dream_signs",
    ),
    TrainingStep(
        key="mild_intention",
        kind=StepKind.INPUT,
        title="Set the intention",
        body=(
            "Write the intention you will carry into sleep. The standard form is: "
            "'Next time I am dreaming, I will remember that I am dreaming.' Repeat it "
            "silently as you fall asleep, meaning it each time rather than chanting."
        ),
        seconds=60,
        capture="intention",
    ),
]

WBTB_SKIP = {"settle", "memory_review"}


@dataclass
class TrainingResult:
    kind: str
    completed: bool
    duration_s: float
    steps: dict = field(default_factory=dict)
    engagement_rating: Optional[int] = None
    captured: dict = field(default_factory=dict)
    notes: Optional[str] = None


def protocol_for(kind: str) -> list[TrainingStep]:
    """Steps for an evening or WBTB session.

    The WBTB version is shorter because it runs at 04:00 after a deliberate
    awakening, when the user is half asleep and the priority is getting back to
    sleep quickly with the intention held. Cutting the settle and day-review
    steps preserves the cue binding and the intention, which are the parts that
    carry the effect.
    """
    if kind == "wbtb":
        return [s for s in PROTOCOL if s.key not in WBTB_SKIP]
    return list(PROTOCOL)


def total_seconds(kind: str = "evening") -> int:
    return sum(s.seconds for s in protocol_for(kind))
