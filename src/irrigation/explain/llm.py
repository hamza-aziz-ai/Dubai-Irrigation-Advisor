"""Ollama-backed explanation engine, with the grounding guarantee enforced.

CLAUDE.md invariant 3 says the language model never computes a number. Every
LLM project claims something like that; almost none of them check it. A prompt
saying "do not calculate" is a request, and a request is not a guarantee - the
failure mode is not the model refusing, it is the model quietly producing
`14.2 mm` where the true figure was 13.8, in a fluent sentence that reads
exactly like the correct one.

So the rule is enforced after generation instead of merely asked for before
it. Every number the model emits is extracted and matched against the facts it
was given. An unmatched number means the model computed, inferred, or invented
something, and the output is discarded and the deterministic explainer used
instead. The check is cheap, total, and cannot be talked out of.

That inverts the usual dependency. The LLM is not load-bearing: it improves
phrasing when it is available and behaving, and the system loses nothing but
fluency when it is not. An outage, a bad sample, or a model swap degrades the
wording and never the reasoning.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .advisor import Explanation, OfflineExplainer

#: Ollama model. `gpt-oss:120b-cloud` runs on Ollama's servers rather than
#: locally, so it needs network and a signed-in Ollama install. Point this at
#: a local tag (`gpt-oss:20b`, `llama3.1:8b`) to keep everything on-machine.
DEFAULT_MODEL = os.environ.get("IRRIGATION_OLLAMA_MODEL", "gpt-oss:120b-cloud")

#: Opt-in. The offline explainer stays the default so that tests, the demo and
#: a fresh clone are deterministic and need nothing installed.
ENGINE_ENV_VAR = "IRRIGATION_EXPLAIN_ENGINE"

#: Matches integers and decimals, including negatives and thousands
#: separators. Deliberately greedy about what counts as a number: anything it
#: fails to catch is a number that escapes verification.
_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")

#: Numbers that carry no quantitative claim and would otherwise force a
#: needless fallback - a model writing "FAO-56" or "24 hours" is not
#: calculating anything about this field.
_ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 24.0, 56.0, 100.0}

#: Unicode the models actually emit, mapped to the ASCII the rest of this
#: repository uses. Applied BEFORE verification, not after, and that ordering
#: is the whole point: `ET₀` and `mm day⁻¹` carry digits in subscript and
#: superscript form, which `_NUMBER_PATTERN` cannot see. A model could state a
#: fabricated `¹⁵` and walk straight past the grounding check. Folding those
#: code points down to ASCII digits first closes the hole.
_UNICODE_REPLACEMENTS = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
    "≈": "~", "×": "x", "°": " deg ",
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
    "⁸": "8", "⁹": "9",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9",
}

#: A negative exponent bound directly to a unit symbol, as in `mm day-1` or
#: `MJ m-2` once superscripts have been folded down. These are units, not
#: quantities, and matching them as numbers would fail every explanation that
#: writes ET0 in SI form.
_UNIT_EXPONENT_PATTERN = re.compile(r"(?<=[A-Za-z])-\d+\b")


def normalise(text: str) -> str:
    """Fold model output to ASCII and collapse whitespace.

    Beyond closing the verification hole above, this keeps generated prose
    consistent with the rest of the project - and printable. A Windows console
    on cp1252 raises `UnicodeEncodeError` on a non-breaking hyphen, which
    turns a cosmetic difference into a crash in the demo script.
    """
    for source, target in _UNICODE_REPLACEMENTS.items():
        text = text.replace(source, target)
    # Anything still outside ASCII is replaced rather than dropped: deleting a
    # character could fuse two numbers into one that verifies against neither.
    text = "".join(char if ord(char) < 128 else " " for char in text)
    return " ".join(text.split())


class GroundingError(Exception):
    """Raised when generated text contains a number that was not supplied."""


@dataclass(frozen=True)
class NumericFact:
    """One pre-computed quantity the model is permitted to state.

    `label` is what the model is told the number means; it appears verbatim in
    the prompt. `value` is what the verifier matches against.
    """

    label: str
    value: float
    unit: str

    def render(self) -> str:
        return f"- {self.label}: {self.value:.1f} {self.unit}".rstrip()


# --------------------------------------------------------------------------
# Fact extraction
# --------------------------------------------------------------------------
def facts_from_context(context: dict[str, Any]) -> list[NumericFact]:
    """Every number the model may legitimately use, and no others.

    Built from the decision object rather than from free text, so there is no
    path by which an un-verified quantity reaches the prompt.
    """
    decision = context["decision"]
    taw = decision.taw_mm
    depletion_pct = (decision.predicted_depletion_mm / taw * 100.0) if taw else 0.0

    facts = [
        NumericFact("root-zone depletion", decision.predicted_depletion_mm, "mm"),
        NumericFact("total available water", taw, "mm"),
        NumericFact("readily available water", decision.raw_mm, "mm"),
        NumericFact("depletion as percent of total available", depletion_pct, "%"),
        NumericFact("tomorrow's reference ET0", context["et0_forecast_mm"], "mm/day"),
    ]
    if decision.action == "irrigate":
        facts.append(NumericFact("recommended application depth", decision.depth_mm, "mm"))
    return facts


def _numbers_in(text: str) -> list[float]:
    text = _UNIT_EXPONENT_PATTERN.sub("", text)
    values = []
    for match in _NUMBER_PATTERN.findall(text):
        cleaned = match.replace(",", "").rstrip(".")
        if cleaned in ("", "-"):
            continue
        values.append(float(cleaned))
    return values


def _permitted_values(
    facts: list[NumericFact], citations: list[dict[str, str]]
) -> set[float]:
    """Fact values plus any number appearing in the retrieved passages.

    Citation text is pasted into the prompt and the model is encouraged to
    reference it, so quoting "70 mm per metre of depth" from FAO-56 Table 19
    is correct behaviour, not fabrication. Those numbers are already grounded -
    in a cited source rather than in a computation.
    """
    allowed = set(_ALWAYS_ALLOWED)
    allowed.update(fact.value for fact in facts)
    for citation in citations:
        allowed.update(_numbers_in(citation.get("text", "")))
        allowed.update(_numbers_in(citation.get("source", "")))
    return allowed


def _is_grounded(value: float, permitted: set[float]) -> bool:
    """True if `value` is a permitted quantity, allowing sane rounding.

    A model writing 13.8 for 13.84, or 14 for 13.84, is presenting a supplied
    number at a sensible precision - that is desirable, not a violation. A
    model writing 15.2 is not.
    """
    for allowed in permitted:
        if abs(value - allowed) < 1e-9:
            return True
        if any(round(allowed, dp) == value for dp in (0, 1, 2)):
            return True
    return False


def ungrounded_numbers(
    text: str, facts: list[NumericFact], citations: list[dict[str, str]]
) -> list[float]:
    """Numbers in `text` that were never supplied. Empty means verified."""
    permitted = _permitted_values(facts, citations)
    return [v for v in _numbers_in(text) if not _is_grounded(v, permitted)]


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an irrigation agronomist writing a short note to a grounds "
    "manager in Dubai.\n\n"
    "Absolute rule: you may only state numbers that appear in the FACTS or "
    "CITATIONS given to you. Never calculate, convert, sum, average, "
    "estimate, or infer a number - not even one that seems obvious. If a "
    "quantity you want is not listed, describe it in words instead.\n\n"
    "Write two or three sentences of plain prose. Explain what the soil "
    "condition is, what the recommendation is, and why - referring to the "
    "agronomic reasoning in the citations. No bullet points, no headings, no "
    "restating these instructions."
)


def build_prompt(context: dict[str, Any], facts: list[NumericFact]) -> str:
    """Assemble the user message: facts, citations, and the decision taken."""
    decision = context["decision"]
    action = (
        f"irrigate with {decision.depth_mm:.1f} mm today"
        if decision.action == "irrigate"
        else "hold - apply no water today"
    )
    citations = "\n".join(
        f"- [{c['source']}] {c['text']}" for c in context["citations"]
    )
    return (
        f"SITE: {context['crop_name']} on {context['soil_name']}\n"
        f"DECISION ALREADY TAKEN: {action}\n\n"
        f"FACTS (the only numbers you may use):\n"
        + "\n".join(fact.render() for fact in facts)
        + f"\n\nCITATIONS (agronomic justification):\n{citations}\n\n"
        "Write the note."
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class OllamaExplainer:
    """Explanation engine backed by a local or cloud Ollama model.

    Falls back to `OfflineExplainer` on any of: Ollama not installed, model
    not pulled, request failure, empty output, or a failed grounding check.
    The fallback is silent in behaviour but not in reporting - the returned
    `Explanation.engine` names what actually produced the text, so a dashboard
    or a log can show that the model was bypassed and why.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._fallback = OfflineExplainer()
        self.last_failure: str | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def available(self) -> bool:
        """Whether the model can be reached right now. Never raises."""
        try:
            import ollama

            names = {m.model for m in ollama.list().models}
            return self.model in names
        except Exception:
            return False

    def _generate(self, context: dict[str, Any], facts: list[NumericFact]) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context, facts)},
            ],
            options={"temperature": self.temperature},
        )
        # gpt-oss models expose chain-of-thought in a separate `thinking`
        # field. Only `content` is used: reasoning text is unverified by
        # construction and must never reach the grounds manager.
        return normalise(response.message.content or "")

    def explain(self, context: dict[str, Any]) -> Explanation:
        self.last_failure = None
        facts = facts_from_context(context)

        try:
            body = self._generate(context, facts)
        except Exception as error:                       # noqa: BLE001
            self.last_failure = f"generation failed: {type(error).__name__}: {error}"
            return self._fall_back(context)

        if not body:
            self.last_failure = "model returned empty output"
            return self._fall_back(context)

        ungrounded = ungrounded_numbers(body, facts, context["citations"])
        if ungrounded:
            self.last_failure = (
                "ungrounded numbers in output: "
                + ", ".join(f"{v:g}" for v in ungrounded)
            )
            return self._fall_back(context)

        decision = context["decision"]
        headline = (
            f"IRRIGATE {decision.depth_mm:.1f} mm today "
            f"({context['crop_name']}, {context['soil_name']})"
            if decision.action == "irrigate"
            else f"HOLD - no irrigation today ({context['crop_name']})"
        )
        # The headline is templated even on the LLM path. It carries the
        # actionable number, and it is the one string that must be correct
        # even if everything else is discarded.
        return Explanation(
            headline=headline,
            body="  " + body,
            citations=context["citations"],
            engine=self.name,
        )

    def _fall_back(self, context: dict[str, Any]) -> Explanation:
        explanation = self._fallback.explain(context)
        return Explanation(
            headline=explanation.headline,
            body=explanation.body,
            citations=explanation.citations,
            engine=f"offline (fallback: {self.last_failure})",
        )


def build_llm_explainer(model: str = DEFAULT_MODEL) -> OllamaExplainer:
    """Construct the Ollama engine without checking availability.

    Availability is deliberately not asserted here: the engine degrades on its
    own, so a caller that builds one and uses it later never has to handle
    "the model went away in between".
    """
    return OllamaExplainer(model=model)


def explainer_from_environment():
    """Engine selected by `IRRIGATION_EXPLAIN_ENGINE`; offline unless asked.

    Opt-in rather than auto-detected on purpose. Silently using an LLM because
    one happened to be installed would make the demo non-reproducible on one
    machine and not another, which is the failure this project spends most of
    its effort avoiding elsewhere.
    """
    if os.environ.get(ENGINE_ENV_VAR, "").lower() == "ollama":
        return build_llm_explainer()
    return OfflineExplainer()
