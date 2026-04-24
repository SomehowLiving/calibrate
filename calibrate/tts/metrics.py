"""
TTS evaluation metrics.
"""

from typing import List, Optional

import numpy as np
from tqdm.asyncio import tqdm_asyncio
import backoff

from calibrate.judges import (
    audio_judge,
    is_rating,
    criterion_result_value,
    DEFAULT_AUDIO_JUDGE_MODEL,
    DEFAULT_TTS_CRITERIA,
)
from calibrate.langfuse import observe

# Re-export for existing imports
DEFAULT_TTS_JUDGE_MODEL = DEFAULT_AUDIO_JUDGE_MODEL


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="tts_llm_judge",
    capture_input=False,
    capture_output=False,
)
async def tts_llm_judge(
    audio_path: str,
    reference_text: str,
    model: str = DEFAULT_TTS_JUDGE_MODEL,
    criteria: Optional[List[dict]] = None,
) -> dict:
    """Evaluate a TTS audio output against one or more criteria.

    Args:
        audio_path: Path to the synthesized WAV audio file.
        reference_text: The text that should have been spoken.
        model: Judge model to use (must be audio-capable).
        criteria: List of {"name", "description"} dicts. Defaults to DEFAULT_TTS_CRITERIA.

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
    """
    criteria_list = criteria if criteria else DEFAULT_TTS_CRITERIA

    return await audio_judge(
        criteria=criteria_list,
        audio_path=audio_path,
        reference_text=reference_text,
        model=model,
    )


async def get_tts_llm_judge_score(
    audio_paths: List[str],
    reference_texts: List[str],
    model: str = DEFAULT_TTS_JUDGE_MODEL,
    criteria: Optional[List[dict]] = None,
) -> dict:
    """Run TTS judge across all rows and aggregate per-criterion scores.

    Returns:
        {
            "criteria_names": ["llm_judge", ...],
            "scores": {"llm_judge": float, ...},
            "score": float,  # overall mean (backward compat)
            "per_row": [
                {"llm_judge": {"reasoning": ..., "match": ...}, ...},
                ...
            ]
        }
    """
    criteria_list = criteria if criteria else DEFAULT_TTS_CRITERIA

    coroutines = []
    for audio_path, reference_text in zip(audio_paths, reference_texts):
        coroutines.append(
            tts_llm_judge(audio_path, reference_text, model=model, criteria=criteria_list)
        )

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running TTS LLM Judge",
    )

    criteria_names = [c["name"] for c in criteria_list]

    # Aggregate per-criterion scores — binary: mean 0/1, rating: mean score
    scores: dict = {}
    for c in criteria_list:
        name = c["name"]
        per_row_values = [criterion_result_value(c, row[name]) for row in results]
        if is_rating(c):
            scores[name] = {
                "type": "rating",
                "mean": float(np.mean(per_row_values)),
                "scale_min": int(c["scale_min"]),
                "scale_max": int(c["scale_max"]),
            }
        else:
            scores[name] = {
                "type": "binary",
                "mean": float(np.mean(per_row_values)),
            }

    overall_score = float(np.mean([s["mean"] for s in scores.values()]))

    return {
        "criteria_names": criteria_names,
        "scores": scores,
        "score": overall_score,
        "per_row": results,
    }
