"""
Unified LLM Judge module.

Two judge types:
- text_judge: evaluates text-based inputs against multiple criteria
- audio_judge: evaluates audio + reference text against multiple criteria (for TTS)

Both accept a list of evaluation criteria and return per-criterion results.
"""

import base64
from typing import Literal, Optional

import instructor
from pydantic import BaseModel, Field, create_model

from calibrate.langfuse import AsyncOpenAI, observe, langfuse, langfuse_enabled


# ── Default models ──────────────────────────────────────────────────────────
DEFAULT_TEXT_JUDGE_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_AUDIO_JUDGE_MODEL = "gpt-audio-2025-08-28"

# Simulation uses a stronger model by default for grading multi-turn conversations
DEFAULT_SIMULATION_JUDGE_MODEL = "gpt-5.2-2025-12-11"

# ── Default criteria per test type ──────────────────────────────────────────

DEFAULT_LLM_TEST_CRITERIA = [
    {
        "name": "criteria",
        "description": "",  # filled in from the test case's criteria string
    }
]

DEFAULT_STT_CRITERIA = [
    {
        "name": "llm_judge",
        "description": (
            "Check whether the values represented by both the strings match. "
            "E.g. if one string says 1,2,3 but the other string says 'one, two, three' "
            "or 'one, 2, three', they should be considered the same as their underlying "
            "value is the same. However, if the actual values itself are different, "
            "e.g. for the name of a person or address or the value of any other key "
            "detail - that difference should be noted. "
            "Ignore differences like a word being split up into more than 1 word by spaces. "
            "Look at whether the values mean the same in both the strings. "
            "If all the values for the strings match, mark it as True. Else, False."
        ),
    }
]

DEFAULT_TTS_CRITERIA = [
    {
        "name": "llm_judge",
        "description": (
            "Evaluate if the text is easily understandable from the audio. "
            "Check whether the spoken words match the reference text and "
            "the audio is clear enough to convey the intended message."
        ),
    }
]

# ── System prompts (internal, not user-configurable) ────────────────────────

_TEXT_JUDGE_SYSTEM_PROMPT = (
    "You are a highly accurate evaluator.\n\n"
    "You will be given some context and a set of evaluation criteria.\n\n"
    "You need to evaluate the context against each criterion and determine "
    "whether it passes or fails. Always give your reasoning in English "
    "irrespective of the language of the content."
)

_SIMULATION_JUDGE_SYSTEM_PROMPT = (
    "You are a highly accurate grader.\n\n"
    "You will be given a conversation between a user and an agent along "
    "with evaluation criteria to use for evaluating the agent's behaviour."
    "{agent_instructions_section}\n\n"
    "You need to evaluate if the agent's behaviour adheres to the evaluation "
    "criteria. Always give your reasoning in english irrespective of the "
    "language of the conversation."
)

_AUDIO_JUDGE_SYSTEM_PROMPT = (
    "You are a highly accurate evaluator evaluating the audio output of a "
    "Text-to-Speech model.\n\n"
    "You will be given the audio and the text that should have been spoken "
    "in the audio, along with evaluation criteria.\n\n"
    "You need to evaluate the audio against each criterion."
)


# ── Criterion types ─────────────────────────────────────────────────────────

BINARY = "binary"
RATING = "rating"


def is_rating(criterion: dict) -> bool:
    """Return True if the criterion is a rating-type criterion."""
    return criterion.get("type") == RATING


def criterion_result_value(criterion: dict, result: dict) -> float:
    """Extract the numeric value from a per-criterion judge result.

    - binary criterion result ``{match: bool, reasoning: str}`` → 0.0 or 1.0
    - rating criterion result ``{score: int, reasoning: str}`` → the score as float

    Used by downstream aggregation (CSV columns, metrics.json means).
    """
    if is_rating(criterion):
        return float(result["score"])
    return float(int(bool(result["match"])))


def _rating_range(criterion: dict) -> list[int]:
    """Return the list of allowed score values for a rating criterion."""
    lo = int(criterion["scale_min"])
    hi = int(criterion["scale_max"])
    if hi < lo:
        raise ValueError(
            f"Rating criterion '{criterion.get('name', '?')}' has scale_max ({hi}) "
            f"less than scale_min ({lo})."
        )
    return list(range(lo, hi + 1))


# ── Shared Pydantic helpers ─────────────────────────────────────────────────

class CriterionResult(BaseModel):
    """Binary result for a single evaluation criterion."""
    reasoning: str = Field(
        ...,
        description="Step-by-step analysis of whether the criterion is met; be concise.",
    )
    match: bool = Field(
        ...,
        description="True if the criterion is met. False if it is not.",
    )


def _build_rating_result_model(criterion: dict) -> type[BaseModel]:
    """Dynamically build a Pydantic model for a rating criterion with a Literal-constrained score."""
    values = _rating_range(criterion)
    # Literal[tuple(...)] expands to Literal[1, 2, 3, ...] — safe across Python 3.11+
    ScoreType = Literal[tuple(values)]  # type: ignore[valid-type]

    scale_min = criterion["scale_min"]
    scale_max = criterion["scale_max"]
    model_name = f"RatingResult_{criterion.get('name', 'criterion')}"

    return create_model(
        model_name,
        reasoning=(
            str,
            Field(
                ...,
                description=(
                    "Step-by-step analysis of how the context measures against "
                    "the criterion; be concise."
                ),
            ),
        ),
        score=(
            ScoreType,
            Field(
                ...,
                description=(
                    f"Integer rating score from {scale_min} (lowest) to "
                    f"{scale_max} (highest) as defined in the criterion description."
                ),
            ),
        ),
    )


def _result_model_for_criterion(criterion: dict) -> type[BaseModel]:
    """Return the Pydantic result model for a single criterion, based on its type."""
    if is_rating(criterion):
        return _build_rating_result_model(criterion)
    return CriterionResult


def build_criteria_output_model(criteria: list[dict]) -> type[BaseModel]:
    """Create a dynamic Pydantic model with one result field per criterion.

    Each criterion contributes a field whose shape depends on its ``type``:
    - binary (default): ``{reasoning: str, match: bool}`` via ``CriterionResult``
    - rating: ``{reasoning: str, score: Literal[scale_min..scale_max]}``
    """
    field_definitions: dict[str, tuple[type, ...]] = {}
    for c in criteria:
        field_definitions[c["name"]] = (_result_model_for_criterion(c), ...)
    return create_model("JudgeOutput", **field_definitions)


def format_criteria_prompt(criteria: list[dict]) -> str:
    """Format evaluation criteria list into prompt text.

    Rating criteria include a ``(rating N-M)`` hint so the judge LLM knows
    the score range it must return.
    """
    lines = []
    for c in criteria:
        if is_rating(c):
            header = f"**{c['name']}** (rating {c['scale_min']}-{c['scale_max']})"
        else:
            header = f"**{c['name']}**"
        lines.append(f"{header}: {c['description']}")
    return "\n\n".join(lines)


def normalize_criteria(criteria) -> list[dict]:
    """Normalize criteria to list[dict] format.

    - string → [{"name": "criteria", "description": "<string>"}]
    - list[dict] → returned as-is
    """
    if isinstance(criteria, str):
        return [{"name": "criteria", "description": criteria}]
    return criteria


# ── Text judge ──────────────────────────────────────────────────────────────

@observe(name="text_judge", capture_input=False)
async def text_judge(
    criteria: list[dict],
    user_prompt: str,
    model: str = DEFAULT_TEXT_JUDGE_MODEL,
    system_prompt: str = None,
) -> dict:
    """Multi-criteria text judge.

    Args:
        criteria: List of {"name": str, "description": str} dicts.
        user_prompt: The full user prompt containing the context to evaluate.
        model: LLM model to use.
        system_prompt: Override for the system prompt (used internally for simulation judge).

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
    """
    client = AsyncOpenAI()

    Output = build_criteria_output_model(criteria)
    _system_prompt = system_prompt or _TEXT_JUDGE_SYSTEM_PROMPT

    criteria_prompt = format_criteria_prompt(criteria)
    full_user_prompt = f"{user_prompt}\n\n`Evaluation criteria`:\n\n{criteria_prompt}"

    response = await client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": full_user_prompt},
        ],
        text_format=Output,
        temperature=0,
        max_output_tokens=8192,
        store=True,
    )

    result = response.output_parsed.model_dump()

    if langfuse_enabled and langfuse:
        langfuse.update_current_span(
            metadata={
                "input": full_user_prompt,
                "output": result,
                "system_prompt": _system_prompt,
                "output_schema": Output.model_json_schema(),
            }
        )

    return result


# ── Audio judge ─────────────────────────────────────────────────────────────

@observe(name="audio_judge", capture_input=False, capture_output=False)
async def audio_judge(
    criteria: list[dict],
    audio_path: str,
    reference_text: str,
    model: str = DEFAULT_AUDIO_JUDGE_MODEL,
) -> dict:
    """Multi-criteria audio judge for TTS evaluation.

    Args:
        criteria: List of {"name": str, "description": str} dicts.
        audio_path: Path to the WAV audio file to evaluate.
        reference_text: The text that should have been spoken.
        model: LLM model to use (must be audio-capable).

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
    """
    client = instructor.apatch(AsyncOpenAI())

    Output = build_criteria_output_model(criteria)
    criteria_prompt = format_criteria_prompt(criteria)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _AUDIO_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Reference text: {reference_text}\n\n"
                            f"`Evaluation criteria`:\n\n{criteria_prompt}\n\n"
                            f"Audio:"
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(
                                open(audio_path, "rb").read()
                            ).decode("utf-8"),
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        response_model=Output,
        modalities=["text"],
        temperature=0,
        max_completion_tokens=8192,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=True,
    )

    result = response.model_dump()

    if langfuse_enabled and langfuse:
        from calibrate.langfuse import create_langfuse_audio_media
        audio_media = create_langfuse_audio_media(audio_path)
        langfuse.update_current_trace(
            input={"audio": audio_media, "reference_text": reference_text},
            output=result,
            metadata={
                "input": f"Reference text: {reference_text}",
                "output": result,
                "criteria": criteria,
            },
        )

    return result


# ── Simulation judge (thin wrapper over text_judge) ─────────────────────────

@observe(name="simulation_judge")
async def simulation_judge(
    conversation: list[dict],
    evaluation_criteria: list[dict],
    agent_system_prompt: str = "",
    model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
) -> dict:
    """Evaluate a full conversation transcript against multiple criteria.

    This is a specialization of text_judge for simulation evaluation. It
    formats the conversation transcript and builds a prompt that includes
    the agent's system prompt for context.

    Args:
        conversation: List of message dicts (role/content, may include tool_calls).
        evaluation_criteria: List of {"name": str, "description": str} dicts.
        agent_system_prompt: The agent's system prompt (for grading context).
        model: LLM model to use.

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
    """
    # Format conversation including tool calls
    lines = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if content:
            lines.append(f"{role}: {content}")

        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "unknown")
                func_args = func.get("arguments", "{}")
                lines.append(f"[Tool Call] {func_name}({func_args})")

    conversation_as_prompt = "\n".join(lines)

    agent_instructions_section = ""
    if agent_system_prompt:
        agent_instructions_section = (
            f"\n\nThe agent was given the following instructions:\n\n"
            f"<agent_instructions>\n\n{agent_system_prompt}\n\n</agent_instructions>\n\n"
            f"Use these instructions to understand what the agent was supposed to do "
            f"and evaluate if the agent followed its instructions correctly."
        )

    system_prompt = _SIMULATION_JUDGE_SYSTEM_PROMPT.format(
        agent_instructions_section=agent_instructions_section
    )

    user_prompt = f"`Chat history`:\n\n{conversation_as_prompt}"

    return await text_judge(
        criteria=evaluation_criteria,
        user_prompt=user_prompt,
        model=model,
        system_prompt=system_prompt,
    )
