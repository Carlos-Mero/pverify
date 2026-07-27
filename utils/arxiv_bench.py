from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from utils.common import ASYNC_LOOP, LLMClient, extract_xml_content, strip_think_simple


ARXIV_MATH_GRADING_BENCH = "LukeBailey181Pub/ArxivMathGradingBench"
DEFAULT_ARXIV_DATA_DIR = Path("NP_dataset/arxiv_math_grading_bench")
MANIFEST_NAME = "manifest.jsonl"


def read_manifest(data_dir: str | Path = DEFAULT_ARXIV_DATA_DIR) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ArxivMathGradingBench manifest not found at {manifest_path}. "
            "Run `python scripts/prepare_arxiv_math_grading_bench.py` first."
        )
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {manifest_path} at line {line_number}"
                ) from exc
    return rows


def resolve_data_path(data_dir: str | Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_dir) / path


def load_prepared_rows(
    data_dir: str | Path = DEFAULT_ARXIV_DATA_DIR,
    expected_count: int | None = 35,
) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    rows = read_manifest(data_dir)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} prepared ArxivMathGradingBench papers in "
            f"{data_dir / MANIFEST_NAME}, found {len(rows)}. Re-run the preparation "
            "script to complete missing downloads."
        )
    prepared: list[dict[str, Any]] = []
    for row in rows:
        proof_path = resolve_data_path(data_dir, row.get("proof_path"))
        if proof_path is None or not proof_path.is_file():
            raise FileNotFoundError(
                f"Prepared proof bundle is missing for {row.get('arxiv_id')}: "
                f"{proof_path}"
            )
        item = dict(row)
        for path_field in (
            "pdf_path",
            "source_archive_path",
            "source_dir",
            "proof_path",
        ):
            resolved = resolve_data_path(data_dir, row.get(path_field))
            item[path_field] = str(resolved.resolve()) if resolved else ""
        item["proof"] = proof_path.read_text(encoding="utf-8")
        # An empty problem is the explicit protocol signal for whole-paper review.
        # The paper's research questions, theorem statements, and proofs all live
        # inside `proof`; no synthetic problem statement or prover call is needed.
        item["problem"] = ""
        # Every benchmark paper is the pre-correction version and is therefore negative
        # under pverify's convention that `True` means a correct proof.
        item["gt_eval"] = False
        item["gt_error_location"] = row.get("Location of Error", "")
        item["dataset_name"] = ARXIV_MATH_GRADING_BENCH
        prepared.append(item)
    return prepared


def split_error_locations(value: str | None) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[,;]|\band\b", value or "", flags=re.IGNORECASE)
        if part.strip()
    ]


def compute_location_tnr(
    verifier_predictions: Iterable[bool | int | float],
    location_matches: Iterable[bool],
) -> dict[str, Any]:
    """Compute paper-level TNR with location-aware true negatives.

    All benchmark papers are known negatives. A paper counts as TN only when
    the verifier rejects it and the location judge confirms that at least one
    reported error matches an annotated location. A pass, an unmatched report,
    or an unparseable judge response counts as FP.
    """

    predictions = [bool(value) for value in verifier_predictions]
    matches = [bool(value) for value in location_matches]
    if len(predictions) != len(matches):
        raise ValueError("prediction and location-match counts differ")
    tn = sum(
        1 for predicted_correct, matched in zip(predictions, matches)
        if not predicted_correct and matched
    )
    fp = len(predictions) - tn
    accuracy = (tn / len(predictions)) if predictions else None
    return {
        "total": len(predictions),
        "tn": tn,
        "fp": fp,
        # Every ArxivMathGradingBench paper is a ground-truth negative, so
        # location-aware accuracy and TNR have the same denominator.
        "accuracy": accuracy,
        "tnr": accuracy,
        "definition": (
            "TN iff the verifier reports an error and the location judge matches "
            "that report to an annotated `Location of Error`; all other papers are FP."
        ),
    }


def compute_iteration_location_metrics(
    prediction_history: Iterable[Iterable[bool | int | float]],
    location_judgments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive cumulative location-aware metrics after every verifier iteration.

    Location reports are judged once at the end. A final matched judgment is
    counted starting from the first iteration whose prediction history marks
    that paper incorrect. This avoids repeated judge calls for the same report.
    """

    judgments = list(location_judgments)
    final_matches = [bool(entry.get("matched", False)) for entry in judgments]
    rows: list[dict[str, Any]] = []
    previous_tn = 0
    for iteration_index, predictions in enumerate(prediction_history, start=1):
        prediction_list = list(predictions)
        if len(prediction_list) != len(judgments):
            raise ValueError(
                "iteration prediction and location-judgment counts differ"
            )
        metrics = compute_location_tnr(prediction_list, final_matches)
        metrics["new_tn_this_iteration"] = metrics["tn"] - previous_tn
        previous_tn = metrics["tn"]
        rows.append({
            "iteration_index": iteration_index,
            "metrics": metrics,
        })
    return rows


class ErrorLocationJudge:
    """Independent agent that aligns verifier error reports with GT locations."""

    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = LLMClient(api_base, api_key, model)

    @staticmethod
    def _messages(
        paper_source: str,
        predicted_report: str,
        ground_truth_location: str,
    ) -> list[dict[str, str]]:
        gt_items = split_error_locations(ground_truth_location)
        numbered_gt = "\n".join(
            f"{index}. {location}" for index, location in enumerate(gt_items)
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an independent evaluation agent for a mathematical "
                    "error-localization benchmark. Judge location alignment only; "
                    "do not re-grade the whole paper."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Determine whether the verifier's reported mathematical error is "
                    "located at any annotated ground-truth error location.\n\n"
                    "A match requires either the same labelled result, or a specific "
                    "step/equation/passage inside that result's statement or proof. "
                    "A different result that merely cites, depends on, or inherits a "
                    "consequence from the annotated result is not a match. Different "
                    "numbers or letters are not matches.\n\n"
                    "Return exactly:\n"
                    "<location_match>true</location_match> or "
                    "<location_match>false</location_match>\n"
                    "<matched_location>the matched annotation, or empty</matched_location>\n"
                    "<rationale>one concise sentence</rationale>\n\n"
                    f"<ground_truth_locations>\n{numbered_gt}\n"
                    "</ground_truth_locations>\n\n"
                    f"<verifier_report>\n{strip_think_simple(predicted_report)}\n"
                    "</verifier_report>\n\n"
                    f"<paper_source>\n{paper_source}\n</paper_source>"
                ),
            },
        ]

    async def judge_async(
        self,
        proofs: list[str],
        verifier_predictions: list[bool | int | float],
        verifier_reports: list[str],
        ground_truth_locations: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if not (
            len(proofs)
            == len(verifier_predictions)
            == len(verifier_reports)
            == len(ground_truth_locations)
        ):
            raise ValueError("location-judge input lengths differ")

        results: list[dict[str, Any]] = [
            {
                "ran": False,
                "matched": False,
                "matched_location": "",
                "rationale": "Verifier did not report an error.",
                "raw_response": "",
            }
            for _ in proofs
        ]
        indices = [
            index
            for index, predicted_correct in enumerate(verifier_predictions)
            if not bool(predicted_correct)
        ]
        if not indices:
            return results

        messages = [
            self._messages(
                proofs[index],
                verifier_reports[index],
                ground_truth_locations[index],
            )
            for index in indices
        ]
        responses = await self.client.infer_batch_async(messages, **kwargs)
        for index, response in zip(indices, responses):
            verdict = extract_xml_content(response, "location_match")
            results[index] = {
                "ran": True,
                "matched": verdict == "true",
                "matched_location": extract_xml_content(
                    response, "matched_location"
                ) or "",
                "rationale": extract_xml_content(response, "rationale") or "",
                "raw_response": response,
                "parse_ok": verdict in {"true", "false"},
            }
        return results

    def __call__(
        self,
        proofs: list[str],
        verifier_predictions: list[bool | int | float],
        verifier_reports: list[str],
        ground_truth_locations: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return ASYNC_LOOP.run(
            self.judge_async(
                proofs,
                verifier_predictions,
                verifier_reports,
                ground_truth_locations,
                **kwargs,
            )
        )
