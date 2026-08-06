"""Mining recurring motifs from dream narratives.

Personal dream signs are the recurring features of someone's dreams — a place,
a person, a situation that keeps reappearing. The conditioning protocol has a
step for visualising them, and a generic prompt there is much weaker than one
naming the things that actually recur in your dreams. Whether this improves
lucidity is speculative (design.md §4.3, H5); that it improves the prompt over
"think of something" is not.

The method is deliberately crude: frequency counting over stopword-filtered
unigrams and bigrams, ranked by how many separate nights a term appears in
rather than raw count. Document frequency is the right ranking here because one
vivid dream that mentions "staircase" nine times is not a recurring motif; nine
dreams that each mention it once are.

No NLP dependency. A proper noun-phrase extractor would be better, and would
cost a model download and a new dependency to marginally improve a prompt the
user reads and filters themselves. Not worth it. The output is a shortlist for a
human to pick from, not an answer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

_WORD = re.compile(r"[a-z][a-z'-]{2,}")

# Ordinary English plus dream-journal boilerplate. The second group matters:
# "dream", "remember" and "woke" appear in nearly every entry and would
# otherwise dominate every ranking while carrying no information.
STOPWORDS: frozenset[str] = frozenset(
    """
    the and was were had have has been being are for but not you your yours
    with that this those these they them their there here then than when where
    which who whom whose what why how all any both each few more most other
    some such only own same too very can will just don't should now got get
    getting into onto out off over under again further once about against
    between during before after above below from down out its it's himself
    herself myself itself themselves ourselves yourself would could must might
    shall may because while though although since until upon among across
    behind beside toward towards around through
    dream dreams dreamt dreaming remember remembered recall vague woke wake
    waking asleep sleep slept night morning last felt feel feeling something
    someone somewhere anything everything nothing kind sort like really quite
    somehow suddenly seemed seem back went going come came going know knew
    think thought thing things time times bit lot around way saw see seeing
    look looked looking said say saying told tell
    kept keep keeps keeping made make makes making took take takes taken
    put puts putting gave give gives given
    lucid lucidity lucidly nonlucid
    """.split()
)
# "lucid" earns its place in that list for a specific reason: the lucidity tag
# lives in the narrative text, so without filtering it ranks as one of the most
# recurring "motifs" in any journal that tags lucid nights — which is an
# artefact of the annotation, not a feature of the dreams.


@dataclass(frozen=True)
class DreamSign:
    term: str
    nights: int          # how many separate entries mention it
    occurrences: int     # total mentions
    examples: tuple[str, ...] = ()

    @property
    def prevalence(self) -> float:
        return self.nights


_SENTENCE = re.compile(r"[.!?;:\n]+")


def _sentences(text: str) -> list[list[str]]:
    """Tokens grouped by sentence.

    Bigrams must not span a sentence boundary. "...the old house. The staircase
    went sideways" would otherwise yield "house staircase" as a recurring
    phrase, which is an artefact of adjacency across a full stop rather than a
    motif that ever appeared together.
    """
    return [
        [w for w in _WORD.findall(part.lower()) if w not in STOPWORDS]
        for part in _SENTENCE.split(text)
    ]


def _tokens(text: str) -> list[str]:
    return [w for sentence in _sentences(text) for w in sentence]


def _bigrams_by_sentence(sentences: Sequence[Sequence[str]]) -> list[str]:
    return [
        f"{a} {b}"
        for sentence in sentences
        for a, b in zip(sentence, sentence[1:])
    ]


def extract(
    narratives: Iterable[str],
    *,
    min_nights: int = 3,
    top_n: int = 20,
    include_bigrams: bool = True,
) -> list[DreamSign]:
    """Rank recurring terms by the number of nights they appear in."""
    narratives = [n for n in narratives if n and n.strip()]
    if not narratives:
        return []

    night_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()

    for text in narratives:
        sentences = _sentences(text)
        tokens = [w for sentence in sentences for w in sentence]
        terms = tokens + (_bigrams_by_sentence(sentences) if include_bigrams else [])
        total_counts.update(terms)
        night_counts.update(set(terms))

    signs: list[DreamSign] = []
    for term, nights in night_counts.items():
        if nights < min_nights:
            continue
        signs.append(DreamSign(term=term, nights=nights, occurrences=total_counts[term]))

    # A bigram and its component words are redundant: if "old house" recurs,
    # "house" almost certainly does too, and listing both wastes a slot in a
    # shortlist a human is going to read.
    bigram_parts = {w for s in signs if " " in s.term for w in s.term.split()}
    deduped = [
        s for s in signs
        if " " in s.term or s.term not in bigram_parts
        or night_counts[s.term] > max(
            (night_counts[b] for b in night_counts if " " in b and s.term in b.split()),
            default=0,
        ) * 1.5
    ]

    deduped.sort(key=lambda s: (-s.nights, -s.occurrences, s.term))
    return deduped[:top_n]


def format_signs(signs: Sequence[DreamSign], total_nights: int) -> str:
    if not signs:
        return (
            "No motif appeared on enough separate nights to call it recurring.\n"
            "With more entries this will fill in; until then the training step\n"
            "will keep asking you to name one yourself."
        )

    lines = [
        f"Recurring motifs across {total_nights} entries",
        "=" * 60,
        f"  {'term':<26} {'nights':>7} {'mentions':>9}",
        "  " + "-" * 44,
    ]
    for sign in signs:
        share = sign.nights / total_nights * 100 if total_nights else 0
        lines.append(f"  {sign.term:<26} {sign.nights:>7} {sign.occurrences:>9}   ({share:.0f}%)")
    lines += [
        "",
        "These are frequency counts, not interpretation. Pick the ones you",
        "recognise as genuinely characteristic of your dreams and ignore the",
        "rest — the list is a prompt for your judgement, not a result.",
    ]
    return "\n".join(lines)
