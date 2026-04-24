"""
STT evaluation metrics.
"""

from typing import List, Optional

import difflib
import numpy as np
from evaluate import load
from tqdm.asyncio import tqdm_asyncio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import backoff

from calibrate.judges import (
    text_judge,
    is_rating,
    criterion_result_value,
    DEFAULT_TEXT_JUDGE_MODEL,
    DEFAULT_STT_CRITERIA,
)
from calibrate.langfuse import observe, langfuse, langfuse_enabled

normalizer = BasicTextNormalizer()

# Re-export for existing imports
DEFAULT_STT_JUDGE_MODEL = DEFAULT_TEXT_JUDGE_MODEL


def get_wer_score(references: List[str], predictions: List[str]) -> float:
    wer_metric = load("wer")

    references = [normalizer(str(ref)) for ref in references]
    predictions = [normalizer(str(pred)) if isinstance(pred, str) else "" for pred in predictions]

    per_row_wer = [
        wer_metric.compute(predictions=[p], references=[r])
        for p, r in zip(predictions, references)
    ]

    return {"score": np.mean(per_row_wer), "per_row": per_row_wer}


def get_string_similarity(references: List[str], predictions: List[str]) -> float:
    similarities = []

    for reference, prediction in zip(references, predictions):
        seq = difflib.SequenceMatcher(
            None,
            normalizer(str(reference)),
            normalizer(str(prediction)) if isinstance(prediction, str) else "",
        )
        similarities.append(seq.ratio())

    return {
        "score": np.mean(similarities),
        "per_row": similarities,
    }


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
@observe(
    name="stt_llm_judge",
    capture_input=False,
)
async def stt_llm_judge(
    reference: str,
    prediction: str,
    model: str = DEFAULT_STT_JUDGE_MODEL,
    criteria: Optional[List[dict]] = None,
) -> dict:
    """Evaluate an STT transcription against one or more criteria.

    Args:
        reference: The source/ground-truth text.
        prediction: The STT transcription output.
        model: Judge model to use.
        criteria: List of {"name", "description"} dicts. Defaults to DEFAULT_STT_CRITERIA.

    Returns:
        Dict keyed by criterion name, each value {"reasoning": str, "match": bool}.
        With default single criterion, returns {"llm_judge": {"reasoning": ..., "match": ...}}.
    """
    criteria_list = criteria if criteria else DEFAULT_STT_CRITERIA

    user_prompt = f"Source: {reference}\nTranscription: {prediction}"

    result = await text_judge(
        criteria=criteria_list,
        user_prompt=user_prompt,
        model=model,
    )

    if langfuse_enabled and langfuse:
        langfuse.update_current_trace(
            input={"reference": reference, "prediction": prediction},
            metadata={
                "reference": reference,
                "prediction": prediction,
                "output": result,
            },
        )

    return result


async def get_llm_judge_score(
    references: List[str],
    predictions: List[str],
    model: str = DEFAULT_STT_JUDGE_MODEL,
    criteria: Optional[List[dict]] = None,
) -> dict:
    """Run STT judge across all rows and aggregate per-criterion scores.

    Returns:
        {
            "criteria_names": ["llm_judge", ...],
            "scores": {"llm_judge": float, ...},  # mean match rate per criterion
            "per_row": [
                {"llm_judge": {"reasoning": ..., "match": ...}, ...},
                ...
            ]
        }
    """
    criteria_list = criteria if criteria else DEFAULT_STT_CRITERIA

    coroutines = []
    for reference, prediction in zip(references, predictions):
        coroutines.append(
            stt_llm_judge(str(reference), str(prediction), model=model, criteria=criteria_list)
        )

    results = await tqdm_asyncio.gather(
        *coroutines,
        desc="Running STT LLM Judge",
    )

    criteria_names = [c["name"] for c in criteria_list]

    # Aggregate per-criterion scores — mean of 0/1 for binary, mean of scores for rating.
    # Returns per-criterion aggregate dicts so downstream code can distinguish types.
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
                "mean": float(np.mean(per_row_values)),  # pass-rate fraction 0.0–1.0
            }

    # Backward compat: top-level "score" = mean across criteria means (same as before
    # for binary-only configs). Works for mixed configs too, with caveat that rating
    # means are on a different scale.
    overall_score = float(np.mean([s["mean"] for s in scores.values()]))

    return {
        "criteria_names": criteria_names,
        "scores": scores,
        "score": overall_score,
        "per_row": results,
    }
