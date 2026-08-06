"""The waking calibration protocol (design.md §13.1).

The centre of this is one segment: **deliberate left-right eye movement with the
eyes closed**, compared against **eyes closed and still**. That pair is the
positive control, and it is the cheapest possible test of H1.

The logic is worth stating plainly. If a camera cannot detect large, deliberate,
supervised eye movements from a fully cooperative awake subject holding still in
good conditions, it will certainly not detect small involuntary ones from a
sleeping subject at an unknown angle in the dark. The waking case is a strict
upper bound on the sleeping case. Failing it ends the eye-movement branch in
fifteen minutes rather than four months, which is the entire point of running it
first.

The remaining segments exist to characterise the confounds (blinks, expressions,
head turns) and to measure the visibility cliff across the postures actually
slept in. For a side sleeper the posture segments may be the most informative
part of the whole exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentSetup(str, Enum):
    """Where the camera has to be for a segment to mean anything.

    The protocol asks two questions that need two different rigs, and running
    them together produces numbers that describe neither. This was found the
    hard way: a first full run reported 98-99% eye availability for all four
    sleep postures, which is not credible for a side sleeper and was an artefact
    of performing "lie on your left side" in front of a laptop on a desk.

    DESK segments ask whether the signal exists at all. They want you close,
    frontal and well lit, because that is the upper bound the sleeping case is
    measured against.

    BED segments ask whether the eye region is visible where you actually sleep.
    They mean nothing except from the camera's real overnight mount, at its real
    distance, in its real lighting.
    """

    DESK = "desk"
    BED = "bed"


class SegmentRole(str, Enum):
    """What a segment contributes to the profile."""

    BASELINE = "baseline"          # the noise floor everything is measured against
    POSITIVE_CONTROL = "positive"  # the signal that must be detectable
    CONFOUND = "confound"          # things that must NOT read as eye movement
    POSTURE = "posture"            # visibility across sleep positions
    ROBUSTNESS = "robustness"      # occlusion, lighting, frame exits


@dataclass(frozen=True)
class Segment:
    key: str
    role: SegmentRole
    setup: SegmentSetup
    title: str
    instruction: str
    seconds: int
    # Segments the analysis compares this one against. Effect sizes are always
    # relative — absolute flow magnitudes mean nothing across cameras.
    contrast_with: tuple[str, ...] = ()


PROTOCOL: tuple[Segment, ...] = (
    Segment(
        key="eyes_closed_still",
        role=SegmentRole.BASELINE,
        setup=SegmentSetup.DESK,
        title="Eyes closed, completely still",
        instruction=(
            "Close your eyes and hold as still as you can. Breathe normally. Try not "
            "to move your eyes behind the lids — this is the noise floor everything "
            "else is measured against, so it is the most important segment here."
        ),
        seconds=30,
    ),
    Segment(
        key="slow_saccades",
        role=SegmentRole.POSITIVE_CONTROL,
        setup=SegmentSetup.DESK,
        title="Slow left-right eye movement, eyes closed",
        instruction=(
            "Keep your eyes closed and your head perfectly still. Move your eyes "
            "slowly left, then right, about once per second. Make them large, "
            "deliberate movements — as far as is comfortable in each direction."
        ),
        seconds=30,
        contrast_with=("eyes_closed_still",),
    ),
    Segment(
        key="fast_saccades",
        role=SegmentRole.POSITIVE_CONTROL,
        setup=SegmentSetup.DESK,
        title="Fast left-right eye movement, eyes closed",
        instruction=(
            "Same again, but quickly — several movements per second. Head still. "
            "This is the closest available proxy for the bursts this system is "
            "eventually looking for."
        ),
        seconds=30,
        contrast_with=("eyes_closed_still",),
    ),
    Segment(
        key="blinks",
        role=SegmentRole.CONFOUND,
        setup=SegmentSetup.DESK,
        title="Blinking",
        instruction=(
            "Eyes open, looking at the camera. Blink normally, roughly once every "
            "two seconds. Blinks are large lid movements that must not be mistaken "
            "for eye movement."
        ),
        seconds=20,
        contrast_with=("eyes_closed_still",),
    ),
    Segment(
        key="facial_movement",
        role=SegmentRole.CONFOUND,
        setup=SegmentSetup.DESK,
        title="Small facial movements",
        instruction=(
            "Eyes closed. Frown, smile slightly, twitch your nose, purse your lips. "
            "Small expressions, no head movement."
        ),
        seconds=20,
        contrast_with=("eyes_closed_still",),
    ),
    Segment(
        key="head_turn",
        role=SegmentRole.CONFOUND,
        setup=SegmentSetup.DESK,
        title="Slow head turn",
        instruction=(
            "Eyes closed and still behind the lids. Slowly turn your head left, then "
            "right, then back to centre. This tests whether head movement leaks into "
            "the eye signal — the single most likely way for this system to fool "
            "itself."
        ),
        seconds=25,
        contrast_with=("eyes_closed_still", "slow_saccades"),
    ),
    Segment(
        key="posture_supine",
        role=SegmentRole.POSTURE,
        setup=SegmentSetup.BED,
        title="Lie on your back",
        instruction=(
            "Lie down as you would to sleep, on your back, eyes closed. Get "
            "comfortable and stay there."
        ),
        seconds=25,
    ),
    Segment(
        key="posture_left",
        role=SegmentRole.POSTURE,
        setup=SegmentSetup.BED,
        title="Lie on your left side",
        instruction="Roll onto your left side as you would to sleep. Eyes closed.",
        seconds=25,
    ),
    Segment(
        key="posture_right",
        role=SegmentRole.POSTURE,
        setup=SegmentSetup.BED,
        title="Lie on your right side",
        instruction="Roll onto your right side as you would to sleep. Eyes closed.",
        seconds=25,
    ),
    Segment(
        key="posture_prone",
        role=SegmentRole.POSTURE,
        setup=SegmentSetup.BED,
        title="Lie face down",
        instruction=(
            "Lie on your front, however you actually sleep. If your face ends up in "
            "the pillow, leave it there — that is the real condition, and a bad "
            "number here is a finding rather than a mistake."
        ),
        seconds=25,
    ),
    Segment(
        key="occlusion",
        role=SegmentRole.ROBUSTNESS,
        setup=SegmentSetup.BED,
        title="Partial occlusion",
        instruction=(
            "Eyes closed. Pull the duvet up over part of your face, or rest an arm "
            "across it, or let hair fall over your eyes."
        ),
        seconds=20,
    ),
    Segment(
        key="leave_frame",
        role=SegmentRole.ROBUSTNESS,
        setup=SegmentSetup.BED,
        title="Leave and return",
        instruction=(
            "Sit up, move out of the camera's view for a few seconds, then return "
            "and lie back down."
        ),
        seconds=25,
    ),
)

SEGMENTS_BY_KEY = {segment.key: segment for segment in PROTOCOL}

# The two halves normally run weeks apart: the signal test can be done today on
# a laptop, while the posture test has to wait for the IR camera to be mounted.
STAGES: dict[str, tuple[str, ...]] = {
    "signal": tuple(s.key for s in PROTOCOL if s.setup is SegmentSetup.DESK),
    "posture": tuple(s.key for s in PROTOCOL if s.setup is SegmentSetup.BED),
    "all": tuple(s.key for s in PROTOCOL),
}

STAGE_GUIDANCE: dict[str, str] = {
    "signal": (
        "Sit at the camera, roughly arm's length, face on, in normal light. This "
        "is the H1 positive control and it wants the most favourable conditions "
        "you can give it, because the sleeping case only gets harder."
    ),
    "posture": (
        "Put the camera exactly where it will sit overnight and do not move it "
        "again. Lie in bed as you actually sleep. These numbers describe that "
        "mount and no other, so a laptop on a desk will tell you nothing."
    ),
    "all": (
        "Runs both stages. Only correct if the camera is already in its final "
        "overnight position AND you can reach it to sit frontally, which is "
        "unusual. Prefer running the stages separately."
    ),
}


def segments_for(stage: str) -> tuple[Segment, ...]:
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; available: {sorted(STAGES)}")
    return tuple(SEGMENTS_BY_KEY[k] for k in STAGES[stage])


def stage_seconds(stage: str) -> int:
    return sum(s.seconds for s in segments_for(stage))


def total_seconds() -> int:
    return sum(segment.seconds for segment in PROTOCOL)


def positive_controls() -> tuple[Segment, ...]:
    return tuple(s for s in PROTOCOL if s.role is SegmentRole.POSITIVE_CONTROL)


def posture_segments() -> tuple[Segment, ...]:
    return tuple(s for s in PROTOCOL if s.role is SegmentRole.POSTURE)
