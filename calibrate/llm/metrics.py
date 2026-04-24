"""
LLM evaluation metrics.

Thin wrappers around calibrate.judges for backward compatibility.
"""

from calibrate.judges import (
    text_judge,
    simulation_judge,
    normalize_criteria,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_SIMULATION_JUDGE_MODEL,
)
from calibrate.langfuse import observe

# Re-export defaults for existing imports
DEFAULT_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL
DEFAULT_SIMULATION_JUDGE_MODEL = DEFAULT_SIMULATION_JUDGE_MODEL


@observe(
    name="llm_test_llm_judge",
    capture_input=False,
)
async def test_response_llm_judge(
    conversation: list[dict],
    response: str,
    criteria,
    model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    """Evaluate an LLM response against one or more criteria.

    Args:
        conversation: Chat history (list of role/content dicts).
        response: The LLM's response text to evaluate.
        criteria: Either a string (single criterion, backward compat) or a list of
            {"name": str, "description": str} dicts for multi-criteria evaluation.
        model: Judge model to use.

    Returns:
        When criteria is a string: {"reasoning": str, "match": bool}
        When criteria is a list: {criterion_name: {"reasoning": str, "match": bool}, ...}
    """
    criteria_list = normalize_criteria(criteria)

    conversation_as_prompt = "\n".join(
        [f'{msg["role"]}: {msg["content"]}' for msg in conversation if "content" in msg]
    )

    user_prompt = (
        f"`Chat history`:\n\n{conversation_as_prompt}\n\n"
        f"`Response to evaluation`:\n\n{response}"
    )

    result = await text_judge(
        criteria=criteria_list,
        user_prompt=user_prompt,
        model=model,
    )

    # Backward compat: if original criteria was a string, return flat {reasoning, match}
    if isinstance(criteria, str):
        return result[criteria_list[0]["name"]]

    return result


async def evaluate_simuation(
    conversation: list[dict],
    evaluation_criteria: list[dict],
    agent_system_prompt: str = "",
    model: str = DEFAULT_SIMULATION_JUDGE_MODEL,
    **kwargs,
) -> dict:
    """Evaluate a simulation transcript against multiple criteria.

    Delegates to calibrate.judges.simulation_judge.

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
    """
    return await simulation_judge(
        conversation=conversation,
        evaluation_criteria=evaluation_criteria,
        agent_system_prompt=agent_system_prompt,
        model=model,
    )
