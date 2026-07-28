import re
import json
import asyncio
import logging
import random
from pathlib import Path
from datetime import datetime, timezone
from litellm import aresponses
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from utils.async_runner import AsyncLoopThread
from math_verify import parse, verify
from concurrent.futures import ProcessPoolExecutor

ASYNC_LOOP = AsyncLoopThread()

def _verify_one_sample(args):
    solution_str, gt_answer = args
    try:
        # math_verify.parse without timeout args relies on signal.alarm
        # which works fine in a separate process (it has its own main thread)
        candidates = parse(solution_str)
        gold_parsed = parse(gt_answer)
        return verify(candidates, gold_parsed)
    except Exception as e:
        # Logging inside process might need configuration, print strictly for debug
        # or just return False
        return False

def batch_check_correctness(solutions: list[str], gt_answers: list[str]) -> list[bool]:
    """
    Check correctness of a batch of solutions against ground truth answers.
    Uses ProcessPoolExecutor to avoid signal handling issues in threaded environments.
    """
    if not solutions:
        return []
    
    # Pack arguments
    args_list = list(zip(solutions, gt_answers))
    
    # Use ProcessPoolExecutor to run verification in separate processes
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_verify_one_sample, args_list))
    
    return results

def extract_xml_content(text: str, tag: str):
    flags = re.DOTALL | 0
    pattern = rf"<{re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{re.escape(tag)}\s*>"

    last_content = None
    for m in re.finditer(pattern, text, flags):
        last_content = m.group(1)

    if last_content is None:
        return None
    return last_content.strip()

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def strip_think_simple(s: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

def get_current_log_path(log_dir: str):
    ts = datetime.now(timezone.utc).strftime("%m%dT%H%M")
    logdir = Path(log_dir) / ts
    return logdir

def _compute_binary_metrics(preds: list[int | None], targets: list[int]) -> dict:
    tp = tn = fp = fn = 0
    total = 0
    correct = 0
    for pred, target in zip(preds, targets):
        if pred is None:
            continue
        gt = int(target)
        total += 1
        if pred == gt:
            correct += 1
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1

    accuracy = (correct / total) if total else None
    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    f1 = ((2 * precision * recall / (precision + recall))
          if (precision is not None and recall is not None and (precision + recall))
          else None)
    return {
        "total": total,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def save_progressive_iteration_samples(
    logdir: Path,
    iteration_logs: list[dict],
    problems: list[str],
    proofs: list[str],
    stripped_proofs: list[str],
    summaries: list[dict] | None = None,
    costs: list[dict] | None = None,
    prefix: str = "progressive",
):
    if not iteration_logs:
        return
    summary_map = {entry.get("iteration_index"): entry for entry in (summaries or [])}
    cost_map = {entry.get("iteration_index"): entry for entry in (costs or [])}
    for entry in iteration_logs:
        iteration_index = entry.get("iteration_index")
        samples_payload = []
        for sample in entry.get("samples", []):
            sample_idx = sample.get("sample_index")
            record = dict(sample)
            if isinstance(sample_idx, int) and 0 <= sample_idx < len(problems):
                record.update({
                    "problem": problems[sample_idx],
                    "proof": proofs[sample_idx],
                    "stripped_proof": stripped_proofs[sample_idx],
                })
            samples_payload.append(record)

        payload = {
            "iteration_index": iteration_index,
            "summary": summary_map.get(iteration_index),
            "cost": cost_map.get(iteration_index),
            "samples": samples_payload,
        }
        out_path = logdir / f"{prefix}_iteration_{iteration_index}_samples.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

def _load_jsonl_problems(jsonl_path: Path, content_keys: tuple[str, ...] = ("markdown_statement",)) -> list[str]:
    logger = logging.getLogger("dataset")
    problems: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s line %d", jsonl_path, i)
                continue
            content = None
            for key in content_keys:
                if key in obj and isinstance(obj[key], str):
                    content = obj[key]
                    break
            if content is None:
                logger.warning("No problem content key %s found at %s line %d", content_keys, jsonl_path, i)
                continue
            problems.append(content)
    logger.info("Loaded %d problems from %s", len(problems), jsonl_path)
    return problems

def _derive_keystream(canary: str, length: int) -> bytes:
    import hashlib
    from math import ceil
    digest = hashlib.sha256(canary.encode("utf-8")).digest()
    if length <= len(digest):
        return digest[:length]
    repeats = ceil(length / len(digest))
    return (digest * repeats)[:length]

def _xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    if len(data) != len(key_stream):
        raise ValueError("Data and keystream must be the same length for XOR.")
    return bytes([a ^ b for a, b in zip(data, key_stream)])

def _deserialize_field(text: str):
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return text


def decrypt_str(input_str, canary):
    import base64
    if input_str == "":
        return ""
    ct = base64.b64decode(input_str)
    ks = _derive_keystream(canary, len(ct))
    pt = _xor_bytes(ct, ks)
    text = pt.decode("utf-8")
    return _deserialize_field(text)

def decrypt_h2eval_sample(example):
    if "canary" not in example:
        raise ValueError("Missing canary field `canary`.")
    canary = example["canary"]
    if not isinstance(canary, str):
        raise ValueError("Canary should be a string.")

    target_fields = ["question", "model_response_by_step", "human_labels", "human_labels_first_error_idx"]
    for k, v in example.items():
        if k in target_fields and isinstance(v, str):
            try:
                example[k] = decrypt_str(v, canary)
            except Exception:
                example[k] = v

    return example

def prepare_dataset(dataset_path, arxiv_data_dir=None):
    logger = logging.getLogger("dataset")
    logger.info("preparing dataset at path: %s", dataset_path)
    if dataset_path == "NP_dataset/train_full.json" or dataset_path == "NP_dataset/train_3000.json" or dataset_path == "NP_dataset/test_hard.json" or dataset_path == "NP_dataset/test_random.json" or dataset_path == "NP_dataset/train_300.json":
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            problems = json.load(f)
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path in {
        "NP_dataset/qz_bench_train.jsonl",
        "NP_dataset/qz_bench_eval.jsonl",
    }:
        problems = _load_jsonl_problems(Path(dataset_path), content_keys=("markdown_statement",))
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path == "NP_dataset/gradingbench.csv":
        ds = load_dataset("csv", data_files=dataset_path)
        ds = ds["train"].select(range(300))
        ds = ds.rename_column("Problem", "problem")
        ds = ds.rename_column("Response", "proof")
        gt_evals = [int(e["Points"]) > 6 for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
    elif dataset_path == "HuggingFaceH4/MATH-500":
        ds = load_dataset(dataset_path)
        ds = ds.remove_columns(["solution"])
        ds = ds["test"]
    elif dataset_path == "AIME24/25":
        ds24 = load_dataset("Maxwell-Jia/AIME_2024")
        ds24 = ds24["train"]
        ds24 = Dataset.from_dict(
            {
                "problem": [e["Problem"] for e in ds24],
                "answer": [str(e["Answer"]) for e in ds24]
            }
        )
        ds25_1 = load_dataset("opencompass/AIME2025", "AIME2025-I")
        ds25_2 = load_dataset("opencompass/AIME2025", "AIME2025-II")
        ds25 = concatenate_datasets([ds25_1['test'], ds25_2['test']])
        ds25 = Dataset.from_dict(
            {
                "problem": [e["question"] for e in ds25],
                "answer": [e["answer"] for e in ds25]
            }
        )
        ds = concatenate_datasets([ds24, ds25])
    elif dataset_path == "Salesforce/Hard2Verify":
        ds = load_dataset(dataset_path, split="test")
        ds = ds.map(decrypt_h2eval_sample)
        ds = ds.rename_column("question", "problem")
        proofs = ["\n".join(e["model_response_by_step"]) for e in ds]
        ds = ds.add_column("proof", proofs)
        gt_evals = [e["human_labels_first_error_idx"] < 0 for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
    elif dataset_path == "MathArena/imo_2025":
        ds = load_dataset(dataset_path, split="train")
        proofs = ["" for e in ds]
        ds = ds.add_column("proof", proofs)
        gt_evals = [False for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
    elif dataset_path == "MathArena/apex_2025":
        ds = load_dataset(dataset_path, split="train")
        # Ensure we have 'problem' and 'gt_answer'
        # The dataset has 'problem' and 'answer'
        ds = ds.rename_column("answer", "gt_answer")
    elif dataset_path == "LukeBailey181Pub/ArxivMathGradingBench":
        # Import lazily because arxiv_bench's location judge uses LLMClient from
        # this module.
        from utils.arxiv_bench import DEFAULT_ARXIV_DATA_DIR, load_prepared_rows

        rows = load_prepared_rows(arxiv_data_dir or DEFAULT_ARXIV_DATA_DIR)
        ds = Dataset.from_list(rows)
    else:
        raise NotImplementedError(f"Unknown dataset name or path: {dataset_path}")

    logger.info("completed preparing dataset at: %s", dataset_path)

    return ds

class LLMClient():
    MAX_RESPONSE_ATTEMPTS = 3

    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.input_tokens = []
        self.comp_tokens = []
        self.last_input_tokens = []
        self.last_comp_tokens = []

    def _supports_enable_thinking(self) -> bool:
        model_name = (self.model or "").lower()
        return "deepseek" in model_name or "qwen" in model_name

    @staticmethod
    def _field(value, name, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _response_output_text(cls, response) -> str:
        output_text = cls._field(response, "output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        text_parts = []
        for item in cls._field(response, "output", []) or []:
            if cls._field(item, "type") != "message":
                continue
            for part in cls._field(item, "content", []) or []:
                if cls._field(part, "type") != "output_text":
                    continue
                text = cls._field(part, "text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)

    @classmethod
    def _validate_completed_response(cls, response):
        if response is None:
            raise RuntimeError("Responses API returned no response")
        if cls._field(response, "status") != "completed":
            error = (
                cls._field(response, "error")
                or cls._field(response, "incomplete_details")
            )
            raise RuntimeError(
                f"Responses API returned a non-completed response: {error}"
            )
        if not cls._response_output_text(response).strip():
            raise RuntimeError(
                "Responses API completed without final output_text"
            )
        return response

    @classmethod
    async def _consume_response_stream(cls, stream):
        completed_response = None
        async for event in stream:
            event_type = cls._field(event, "type")
            if event_type == "response.completed":
                completed_response = cls._field(event, "response")
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                error = cls._field(event, "error")
                response = cls._field(event, "response")
                if error is None and response is not None:
                    error = (
                        cls._field(response, "error")
                        or cls._field(response, "incomplete_details")
                    )
                raise RuntimeError(
                    f"Responses API stream ended with {event_type}: {error}"
                )

        if completed_response is None:
            raise RuntimeError(
                "Responses API stream ended without response.completed"
            )
        return cls._validate_completed_response(completed_response)

    async def _infer_one(self,
                         messages,
                         sem: asyncio.Semaphore,
                         **kwargs):
        logger = logging.getLogger("evaluator")
        backoff = 1.0
        last_error = None
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning = (
            {"effort": reasoning_effort}
            if reasoning_effort is not None
            else None
        )
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        for provider_param in ("thinking", "enable_thinking"):
            if provider_param in kwargs:
                extra_body[provider_param] = kwargs.pop(provider_param)
        for attempt in range(1, self.MAX_RESPONSE_ATTEMPTS + 1):
            try:
                async with sem:
                    response_or_stream = await aresponses(
                        model="openai/"+self.model,
                        input=messages,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        drop_params=True,
                        temperature=1.0,
                        timeout=3600,
                        # Keep retry policy in the outer loop only. Combining
                        # LiteLLM retries with MAX_RESPONSE_ATTEMPTS multiplies
                        # traffic during gateway outages.
                        num_retries=0,
                        stream=True,
                        store=False,
                        reasoning=reasoning,
                        extra_body=extra_body or None,
                        **kwargs)
                    if hasattr(response_or_stream, "__aiter__"):
                        return await self._consume_response_stream(
                            response_or_stream
                        )
                    return self._validate_completed_response(response_or_stream)
            except Exception as e:
                last_error = e
                if attempt >= self.MAX_RESPONSE_ATTEMPTS:
                    break
                logger.warning(
                    "Responses API request failed on attempt %d/%d; retrying: %s",
                    attempt,
                    self.MAX_RESPONSE_ATTEMPTS,
                    e,
                )
                await asyncio.sleep(backoff + random.random() * 0.2)
                backoff = min(backoff * 2, 60)

        raise RuntimeError(
            f"Responses API request failed after {self.MAX_RESPONSE_ATTEMPTS} attempts"
        ) from last_error

    async def infer_batch_async(self,
                                all_messages,
                                concurrency: int = 8,
                                show_progress: bool = True,
                                **kwargs) -> list[str]:
        logger = logging.getLogger("evaluator")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        logger.info(
            "running batch inference on %d samples with concurrency=%d",
            len(all_messages),
            concurrency,
        )
        sem = asyncio.Semaphore(concurrency)
        ALLOWED_PARAM_KEYS = {"reasoning_effort", "thinking"}
        infer_params = {k: v for k, v in kwargs.items() if k in ALLOWED_PARAM_KEYS}
        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is not None and self._supports_enable_thinking():
            infer_params["enable_thinking"] = enable_thinking
        async def _run_one(index: int, messages):
            try:
                r = await self._infer_one(messages, sem, **infer_params)
            except Exception as e:
                r = e
            return index, r

        tasks = [asyncio.create_task(_run_one(i, messages)) for i, messages in enumerate(all_messages)]
        raw_results = [None] * len(all_messages)

        pbar = tqdm(total=len(all_messages), desc="LLM batch", leave=False) if show_progress else None
        try:
            for t in asyncio.as_completed(tasks):
                idx, r = await t
                raw_results[idx] = r
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                raise RuntimeError(f"Task {i} failed") from r
        logger.info("completed batch inference on %d samples",  len(all_messages))
        completions = [self._response_output_text(r) for r in raw_results]
        batch_input_tokens = [
            self._field(self._field(r, "usage"), "input_tokens", 0)
            for r in raw_results
        ]
        batch_comp_tokens = [
            self._field(self._field(r, "usage"), "output_tokens", 0)
            for r in raw_results
        ]
        self.last_input_tokens = batch_input_tokens
        self.last_comp_tokens = batch_comp_tokens
        self.input_tokens.extend(batch_input_tokens)
        self.comp_tokens.extend(batch_comp_tokens)
        return completions

class NaiveProver():
    """
    NaiveProver directly proves the given problem
    """
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    def __call__(self, problems: list[str], **kwargs):
        all_messages = [
            [
                {"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}"}
            ]
            for p in problems
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        return results
