import logging
import random
from utils.common import LLMClient, ASYNC_LOOP, extract_xml_content, strip_think_simple, _compute_binary_metrics


def _completion_text(completion, *, preserve_verbatim: bool = False) -> str:
    text = completion if isinstance(completion, str) else completion[0]["content"]
    return text if preserve_verbatim else strip_think_simple(text)


def _build_full_review_message(problem: str, completion) -> list[dict]:
    """Build one full-input review request.

    An empty problem selects whole-paper mode. In that mode the completion is
    preserved verbatim and the reviewer is explicitly told that the research
    questions and proofs are contained in the paper itself.
    """

    if not (problem or "").strip():
        paper = _completion_text(completion, preserve_verbatim=True)
        return [
            {
                "role": "system",
                "content": (
                    "You are a mathematical referee reviewing a complete research "
                    "paper. The entire paper will be supplied as `paper_source`. "
                    "Its research questions, theorem statements, and proofs are all "
                    "contained in that source; there is intentionally no separate "
                    "problem statement."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review the complete paper below as one document and determine "
                    "whether it contains a substantive mathematical error.\n\n"
                    "Important input protocol:\n"
                    "- `<problem>` is intentionally empty. Do not treat this as "
                    "missing input and do not expect a separate exercise statement.\n"
                    "- `<paper_source>` is the complete paper and is the sole object "
                    "to verify. Read its own definitions, results, and proofs.\n"
                    "- Check the paper globally, including dependencies between "
                    "sections and labelled results.\n\n"
                    "Check mathematical derivations, theorem hypotheses, unsupported "
                    "assumptions, circular reasoning, and whether stated conclusions "
                    "follow. Ignore purely stylistic, formatting, or harmless "
                    "notational issues.\n\n"
                    "If no substantive error is found, reply with "
                    "`<verification>true</verification>` and a concise explanation. "
                    "If an error is found, reply with "
                    "`<verification>false</verification>`, identify its precise "
                    "labelled location (theorem, proposition, lemma, equation, or "
                    "proof passage), and concisely explain the mathematical flaw. "
                    "Do not report missing input: the paper source is the complete "
                    "input by design.\n\n"
                    "<problem></problem>\n\n"
                    f"<paper_source>{paper}</paper_source>"
                ),
            },
        ]

    answer = _completion_text(completion)
    return [
        {"role": "system", "content": (
            "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
        )},
        {"role": "user", "content": (
            "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
            "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
            "2. The solution actually derives the conclusion required by the original problem.\n"
            "3. Every step of calculation and formula derivation in the solution is correct.\n"
            "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
            "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
            "Consistency and error-severity policy (important):\n"
            "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
            "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
            "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
            " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error."
            " Do not include any restatement of the entire solution or problem.\n\n"
            f"<problem>{problem}</problem>\n\n"
            f"<answer>{answer}</answer>"
        )}
    ]


class Verifier():
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    async def verify_async(self, problems, completions, **kwargs):
        all_messages = [
            _build_full_review_message(p, c)
            for (p, c) in zip(problems, completions)
        ]
        results = await self.client.infer_batch_async(all_messages, **kwargs)
        rewards = [1.0 if extract_xml_content(r, "verification") == "true" else 0.0 for r in results]
        
        # Calculate costs
        costs = []
        inputs = getattr(self.client, "last_input_tokens", [])
        outputs = getattr(self.client, "last_comp_tokens", [])
        # Ensure lengths match problems even if some calls failed
        if len(inputs) != len(problems):
             inputs = [0] * len(problems)
        if len(outputs) != len(problems):
             outputs = [0] * len(problems)

        for i in range(len(problems)):
            costs.append({
                "api_calls": 1,
                "input_tokens": inputs[i],
                "output_tokens": outputs[i]
            })

        return rewards, results, costs

    def __call__(self, problems, completions, **kwargs):
        rewards, results, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, **kwargs))
        return rewards, results

class PessimisticVerifier():
    """
    Runs multiple parallel reviews using the same checklist as Verifier.
    Instead of asking a judger, it treats the FIRST review that reports an error
    (`<verification>false</verification>`) as the final verdict for that proof.
    """
    def __init__(self, api_base, api_key, model, review_times: int = 3):
        self.client = LLMClient(api_base, api_key, model)
        self.review_times = max(1, review_times)
        self.last_majority_results: tuple[list[float], list[str]] = ([], [])
        self.stepwise_review_logs: list[dict] = []
        self.majority_step_logs: list[dict] = []
        self.last_request_audit: list[dict] = []

    def _review_messages(self, problems, completions):
        messages = []
        audit = []
        for sample_index, (p, c) in enumerate(zip(problems, completions)):
            base = _build_full_review_message(p, c)
            source = _completion_text(
                c,
                preserve_verbatim=not (p or "").strip(),
            )
            for review_index in range(self.review_times):
                messages.append(base)
                user_content = base[1]["content"]
                audit.append({
                    "sample_index": sample_index,
                    "review_index": review_index + 1,
                    "problem_empty": not bool((p or "").strip()),
                    "proof_characters": len(source),
                    "user_message_characters": len(user_content),
                    "complete_proof_occurrences": (
                        user_content.count(source) if source else 0
                    ),
                    "whole_paper_mode": "<paper_source>" in user_content,
                })
        self.last_request_audit = audit
        return messages

    def _majority_vote(self, reviews, verdicts):
        positives = sum(1 for v in verdicts if v == "true")
        negatives = sum(1 for v in verdicts if v == "false")

        if positives > negatives:
            target_verdict = "true"
        elif negatives > positives:
            target_verdict = "false"
        else:
            # Tie: randomly select one review as the final result
            if reviews:
                chosen = random.choice(reviews)
                return extract_xml_content(chosen, "verification") == "true", chosen
            return False, ""

        for review, verdict in zip(reviews, verdicts):
            if verdict == target_verdict:
                return target_verdict == "true", review

        # Fallback if no matching verdict found (should not happen)
        return (target_verdict == "true"), (reviews[0] if reviews else "")

    def _normalize_ground_truth(self, ground_truth_labels, total_samples: int) -> list[int] | None:
        if ground_truth_labels is None:
            return None
        if len(ground_truth_labels) != total_samples:
            logging.getLogger("pessimistic_verifier").warning(
                "Ground truth label count (%d) does not match sample count (%d); skipping stepwise metrics.",
                len(ground_truth_labels),
                total_samples,
            )
            return None
        normalized: list[int] = []
        for idx, label in enumerate(ground_truth_labels):
            if label is None:
                logging.getLogger("pessimistic_verifier").warning(
                    "Ground truth label at index %d is None; skipping stepwise metrics.",
                    idx,
                )
                return None
            normalized.append(1 if bool(label) else 0)
        return normalized

    def _record_stepwise_logs(
        self,
        verdicts_per_sample: list[list[str | None]],
        ground_truth_labels: list | None,
    ) -> None:
        total_samples = len(verdicts_per_sample)
        if total_samples == 0:
            self.stepwise_review_logs = []
            return

        gt_vector = self._normalize_ground_truth(ground_truth_labels, total_samples)
        step_count = max((len(v) for v in verdicts_per_sample), default=0)
        cumulative_fail = [False] * total_samples
        logs: list[dict] = []

        for step_idx in range(step_count):
            preds: list[int] = []
            for sample_idx, verdicts in enumerate(verdicts_per_sample):
                verdict = verdicts[step_idx] if step_idx < len(verdicts) else None
                if verdict == "false":
                    cumulative_fail[sample_idx] = True
                preds.append(0 if cumulative_fail[sample_idx] else 1)

            metrics = _compute_binary_metrics(preds, gt_vector) if gt_vector is not None else None
            entry = {"step_index": step_idx + 1}
            if metrics is not None:
                entry["metrics"] = metrics
            logs.append(entry)

        self.stepwise_review_logs = logs

    def _majority_label_from_subset(self, verdict_subset: list[str | None]) -> int | None:
        votes = [v for v in verdict_subset if v in {"true", "false"}]
        if not votes:
            return None
        positives = sum(1 for v in votes if v == "true")
        negatives = len(votes) - positives
        if positives > negatives:
            return 1
        if negatives > positives:
            return 0
        chosen = random.choice(votes)
        return 1 if chosen == "true" else 0

    def _record_majority_step_logs(
        self,
        verdicts_per_sample: list[list[str | None]],
        ground_truth_labels: list | None,
    ) -> None:
        total_samples = len(verdicts_per_sample)
        if total_samples == 0:
            self.majority_step_logs = []
            return

        gt_vector = self._normalize_ground_truth(ground_truth_labels, total_samples)
        step_count = max((len(v) for v in verdicts_per_sample), default=0)
        logs: list[dict] = []

        for step_idx in range(step_count):
            preds: list[int | None] = []
            for verdicts in verdicts_per_sample:
                subset = verdicts[:step_idx + 1]
                preds.append(self._majority_label_from_subset(subset))

            metrics = _compute_binary_metrics(preds, gt_vector) if gt_vector is not None else None
            entry = {"step_index": step_idx + 1}
            if metrics is not None:
                entry["metrics"] = metrics
            logs.append(entry)

        self.majority_step_logs = logs

    async def verify_async(self, problems, completions, ground_truth_labels=None, **kwargs):
        # Only perform parallel reviews and take the first error as verdict
        review_messages = self._review_messages(problems, completions)
        all_reviews = await self.client.infer_batch_async(review_messages, **kwargs)
        
        # Calculate costs
        costs = []
        inputs = getattr(self.client, "last_input_tokens", [])
        outputs = getattr(self.client, "last_comp_tokens", [])
        expected_len = len(problems) * self.review_times
        if len(inputs) != expected_len:
             inputs = [0] * expected_len
        if len(outputs) != expected_len:
             outputs = [0] * expected_len
             
        k = self.review_times
        grouped = [all_reviews[i * k:(i + 1) * k] for i in range(len(problems))]
        
        for i in range(len(problems)):
            start = i * k
            end = (i + 1) * k
            costs.append({
                "api_calls": k,
                "input_tokens": sum(inputs[start:end]),
                "output_tokens": sum(outputs[start:end])
            })

        verdicts_per_sample = [
            [extract_xml_content(r, "verification") for r in reviews]
            for reviews in grouped
        ]

        final_reviews = []
        rewards = []
        majority_reviews = []
        majority_rewards = []
        for reviews, verdicts in zip(grouped, verdicts_per_sample):

            # find first negative review for pessimistic verdict
            first_negative = None
            for review, verdict in zip(reviews, verdicts):
                if verdict == "false":
                    first_negative = review
                    break
            if first_negative is not None:
                rewards.append(0.0)
                final_reviews.append(first_negative)
            else:
                rewards.append(1.0)
                final_reviews.append(reviews[0] if reviews else "")

            majority_true, majority_review = self._majority_vote(reviews, verdicts)
            majority_rewards.append(1.0 if majority_true else 0.0)
            majority_reviews.append(majority_review)

        self.last_majority_results = (majority_rewards, majority_reviews)
        self._record_stepwise_logs(verdicts_per_sample, ground_truth_labels)
        self._record_majority_step_logs(verdicts_per_sample, ground_truth_labels)

        return rewards, final_reviews, costs

    def __call__(self, problems, completions, ground_truth_labels=None, **kwargs):
        rewards, final_reviews, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, ground_truth_labels=ground_truth_labels, **kwargs))
        return rewards, final_reviews

class PessimisticPruningVerifier():
    """
    Runs pessimistic verification with iterative pruning.

    Instead of launching all reviews in parallel, it performs 1, 2, 4, 8, ...
    full-proof reviews across successive iterations (truncating the final round
    so the cumulative count never exceeds `--reviews`). After each wave, any
    proof flagged incorrect is removed from future iterations, reducing the
    number of required calls for later rounds while retaining the same
    first-error-wins behavior as PessimisticVerifier.
    """

    def __init__(self, api_base, api_key, model, review_times: int = 3):
        self.client = LLMClient(api_base, api_key, model)
        self.max_reviews = max(1, int(review_times))
        self.iteration_plan = self._build_iteration_plan(self.max_reviews)
        self.last_majority_results: tuple[list[float], list[str]] = ([], [])
        self.stepwise_review_logs: list[dict] = []
        self.majority_step_logs: list[dict] = []
        self.last_review_counts: list[int] = []
        self.iteration_samples_log: list[dict] = []
        self.iteration_summary: list[dict] = []
        self.iteration_prediction_history: list[list[int]] = []
        self.iteration_resolved_predictions: list[list[int | None]] = []
        self.iteration_pending_masks: list[list[bool]] = []
        self.iteration_review_costs: list[dict] = []

    @staticmethod
    def _build_iteration_plan(max_reviews: int) -> list[int]:
        plan: list[int] = []
        used = 0
        chunk = 1
        while used < max_reviews:
            remaining = max_reviews - used
            this_chunk = chunk if chunk <= remaining else remaining
            plan.append(this_chunk)
            used += this_chunk
            if this_chunk < chunk:
                break
            chunk *= 2
        return plan

    def _build_base_message(self, problem: str, completion) -> list[dict]:
        return _build_full_review_message(problem, completion)

    def _majority_vote(self, reviews, verdicts):
        positives = sum(1 for v in verdicts if v == "true")
        negatives = sum(1 for v in verdicts if v == "false")

        if positives > negatives:
            target_verdict = "true"
        elif negatives > positives:
            target_verdict = "false"
        else:
            if reviews:
                chosen = random.choice(reviews)
                return extract_xml_content(chosen, "verification") == "true", chosen
            return False, ""

        for review, verdict in zip(reviews, verdicts):
            if verdict == target_verdict:
                return target_verdict == "true", review

        return (target_verdict == "true"), (reviews[0] if reviews else "")

    def _normalize_ground_truth(self, ground_truth_labels, total_samples: int) -> list[int] | None:
        if ground_truth_labels is None:
            return None
        if len(ground_truth_labels) != total_samples:
            logging.getLogger("pessimistic_pruning_verifier").warning(
                "Ground truth label count (%d) does not match sample count (%d); skipping stepwise metrics.",
                len(ground_truth_labels),
                total_samples,
            )
            return None
        normalized: list[int] = []
        for idx, label in enumerate(ground_truth_labels):
            if label is None:
                logging.getLogger("pessimistic_pruning_verifier").warning(
                    "Ground truth label at index %d is None; skipping stepwise metrics.",
                    idx,
                )
                return None
            normalized.append(1 if bool(label) else 0)
        return normalized

    def _record_stepwise_logs(
        self,
        verdicts_per_sample: list[list[str | None]],
        ground_truth_labels: list | None,
    ) -> None:
        total_samples = len(verdicts_per_sample)
        if total_samples == 0:
            self.stepwise_review_logs = []
            return

        gt_vector = self._normalize_ground_truth(ground_truth_labels, total_samples)
        step_count = max((len(v) for v in verdicts_per_sample), default=0)
        cumulative_fail = [False] * total_samples
        logs: list[dict] = []

        for step_idx in range(step_count):
            preds: list[int] = []
            for sample_idx, verdicts in enumerate(verdicts_per_sample):
                verdict = verdicts[step_idx] if step_idx < len(verdicts) else None
                if verdict == "false":
                    cumulative_fail[sample_idx] = True
                preds.append(0 if cumulative_fail[sample_idx] else 1)

            metrics = _compute_binary_metrics(preds, gt_vector) if gt_vector is not None else None
            entry = {"step_index": step_idx + 1}
            if metrics is not None:
                entry["metrics"] = metrics
            logs.append(entry)

        self.stepwise_review_logs = logs

    def _majority_label_from_subset(self, verdict_subset: list[str | None]) -> int | None:
        votes = [v for v in verdict_subset if v in {"true", "false"}]
        if not votes:
            return None
        positives = sum(1 for v in votes if v == "true")
        negatives = len(votes) - positives
        if positives > negatives:
            return 1
        if negatives > positives:
            return 0
        chosen = random.choice(votes)
        return 1 if chosen == "true" else 0

    def _record_majority_step_logs(
        self,
        verdicts_per_sample: list[list[str | None]],
        ground_truth_labels: list | None,
    ) -> None:
        total_samples = len(verdicts_per_sample)
        if total_samples == 0:
            self.majority_step_logs = []
            return

        gt_vector = self._normalize_ground_truth(ground_truth_labels, total_samples)
        step_count = max((len(v) for v in verdicts_per_sample), default=0)
        logs: list[dict] = []

        for step_idx in range(step_count):
            preds: list[int | None] = []
            for verdicts in verdicts_per_sample:
                subset = verdicts[:step_idx + 1]
                preds.append(self._majority_label_from_subset(subset))

            metrics = _compute_binary_metrics(preds, gt_vector) if gt_vector is not None else None
            entry = {"step_index": step_idx + 1}
            if metrics is not None:
                entry["metrics"] = metrics
            logs.append(entry)

        self.majority_step_logs = logs

    async def verify_async(self, problems, completions, ground_truth_labels=None, **kwargs):
        total = len(problems)
        # Initialize costs
        costs = [{"api_calls": 0, "input_tokens": 0, "output_tokens": 0} for _ in range(total)]
        
        if total == 0:
            self.last_majority_results = ([], [])
            self.stepwise_review_logs = []
            self.majority_step_logs = []
            self.last_review_counts = []
            self.iteration_samples_log = []
            self.iteration_summary = []
            self.iteration_prediction_history = []
            self.iteration_resolved_predictions = []
            self.iteration_pending_masks = []
            self.iteration_review_costs = []
            return [], [], costs

        base_messages = [self._build_base_message(problem, completion) for problem, completion in zip(problems, completions)]
        rewards: list[float | None] = [None] * total
        final_reviews: list[str] = [""] * total
        pending_indices = list(range(total))
        total_review_counts = [0] * total
        reviews_history: list[list[str]] = [[] for _ in range(total)]
        verdicts_history: list[list[str | None]] = [[] for _ in range(total)]
        sample_states = [
            {
                "status": "pending",
                "resolved_iteration": None,
                "final_text": "",
            }
            for _ in range(total)
        ]
        self.iteration_samples_log = []
        self.iteration_summary = []
        self.iteration_prediction_history = []
        self.iteration_resolved_predictions = []
        self.iteration_pending_masks = []
        self.iteration_review_costs = []
        cumulative_reviews = 0

        for iteration, reviews_per_sample in enumerate(self.iteration_plan):
            if not pending_indices:
                break

            batch_messages = []
            iteration_batch_info: list[dict] = []
            for idx in pending_indices:
                for _ in range(reviews_per_sample):
                    batch_messages.append(base_messages[idx])
                iteration_batch_info.append({
                    "sample_index": idx,
                    "num_reviews": reviews_per_sample,
                })
                total_review_counts[idx] += reviews_per_sample

            if not batch_messages:
                break

            iteration_reviews_raw = await self.client.infer_batch_async(batch_messages, **kwargs)
            
            # Accumulate costs
            inputs = getattr(self.client, "last_input_tokens", [])
            outputs = getattr(self.client, "last_comp_tokens", [])
            if len(inputs) != len(batch_messages):
                inputs = [0] * len(batch_messages)
            if len(outputs) != len(batch_messages):
                outputs = [0] * len(batch_messages)
                
            cursor_token = 0
            for info in iteration_batch_info:
                sample_idx = info["sample_index"]
                count = info["num_reviews"]
                # Sum tokens for this sample's reviews in this iteration
                s_inp = sum(inputs[cursor_token : cursor_token + count])
                s_out = sum(outputs[cursor_token : cursor_token + count])
                cursor_token += count
                
                costs[sample_idx]["api_calls"] += count
                costs[sample_idx]["input_tokens"] += s_inp
                costs[sample_idx]["output_tokens"] += s_out

            latest_input_tokens = getattr(self.client, "last_input_tokens", []) or []
            latest_comp_tokens = getattr(self.client, "last_comp_tokens", []) or []
            avg_input_tokens_this_iter = (
                sum(latest_input_tokens) / len(latest_input_tokens)
                if latest_input_tokens else None
            )
            avg_comp_tokens_this_iter = (
                sum(latest_comp_tokens) / len(latest_comp_tokens)
                if latest_comp_tokens else None
            )
            cumulative_input_tokens = getattr(self.client, "input_tokens", []) or []
            cumulative_comp_tokens = getattr(self.client, "comp_tokens", []) or []
            avg_input_tokens_cumulative = (
                sum(cumulative_input_tokens) / len(cumulative_input_tokens)
                if cumulative_input_tokens else None
            )
            avg_comp_tokens_cumulative = (
                sum(cumulative_comp_tokens) / len(cumulative_comp_tokens)
                if cumulative_comp_tokens else None
            )

            cursor = 0
            next_pending = []
            iteration_samples = []
            failed_this_iter = 0
            passed_this_iter = 0
            reviews_this_iter = 0

            for info in iteration_batch_info:
                sample_idx = info["sample_index"]
                count = info["num_reviews"]
                sample_reviews = iteration_reviews_raw[cursor:cursor + count]
                cursor += count
                reviews_this_iter += count
                verdicts = [extract_xml_content(r, "verification") for r in sample_reviews]
                reviews_history[sample_idx].extend(sample_reviews)
                verdicts_history[sample_idx].extend(verdicts)

                first_negative = None
                for review, verdict in zip(sample_reviews, verdicts):
                    if verdict == "false":
                        first_negative = review
                        break
                if first_negative is not None:
                    rewards[sample_idx] = 0.0
                    final_reviews[sample_idx] = first_negative
                    sample_states[sample_idx].update({
                        "status": "failed",
                        "resolved_iteration": iteration + 1,
                        "final_text": first_negative,
                    })
                    status_label = "failed"
                    status_eval = 0.0
                    failed_this_iter += 1
                else:
                    if iteration == len(self.iteration_plan) - 1:
                        rewards[sample_idx] = 1.0
                        final_reviews[sample_idx] = reviews_history[sample_idx][0] if reviews_history[sample_idx] else ""
                        sample_states[sample_idx].update({
                            "status": "passed",
                            "resolved_iteration": iteration + 1,
                            "final_text": final_reviews[sample_idx],
                        })
                        status_label = "passed"
                        status_eval = 1.0
                        passed_this_iter += 1
                    else:
                        next_pending.append(sample_idx)
                        status_label = "pending"
                        status_eval = None

                iteration_samples.append({
                    "sample_index": sample_idx,
                    "status": status_label,
                    "eval": status_eval,
                    "verification": final_reviews[sample_idx] if status_eval is not None else None,
                    "num_reviews": count,
                    "reviews": sample_reviews,
                    "verdicts": verdicts,
                })

            pending_indices = next_pending

            statuses = [state["status"] for state in sample_states]
            preds_if_stop = [0 if status == "failed" else 1 for status in statuses]
            resolved_preds = [
                0 if status == "failed" else 1 if status == "passed" else None
                for status in statuses
            ]
            pending_mask = [status == "pending" for status in statuses]
            cumulative_reviews += reviews_this_iter

            self.iteration_samples_log.append({
                "iteration_index": iteration + 1,
                "samples": iteration_samples,
            })
            self.iteration_summary.append({
                "iteration_index": iteration + 1,
                "reviews_per_sample": reviews_per_sample,
                "reviewed_samples": len(iteration_batch_info),
                "failed_this_iter": failed_this_iter,
                "passed_this_iter": passed_this_iter,
                "pending_after_iteration": len(next_pending),
            })
            self.iteration_prediction_history.append(preds_if_stop)
            self.iteration_resolved_predictions.append(resolved_preds)
            self.iteration_pending_masks.append(pending_mask)
            self.iteration_review_costs.append({
                "iteration_index": iteration + 1,
                "reviews_this_iter": reviews_this_iter,
                "cumulative_reviews": cumulative_reviews,
                "avg_input_tokens_this_iter": avg_input_tokens_this_iter,
                "avg_output_tokens_this_iter": avg_comp_tokens_this_iter,
                "avg_input_tokens_cumulative": avg_input_tokens_cumulative,
                "avg_output_tokens_cumulative": avg_comp_tokens_cumulative,
            })

        for idx in pending_indices:
            if rewards[idx] is None:
                rewards[idx] = 1.0
                final_reviews[idx] = reviews_history[idx][0] if reviews_history[idx] else ""
                sample_states[idx].update({
                    "status": "passed",
                    "resolved_iteration": len(self.iteration_plan),
                    "final_text": final_reviews[idx],
                })

        for i, reward in enumerate(rewards):
            if reward is None:
                rewards[i] = 1.0
            if not final_reviews[i] and reviews_history[i]:
                final_reviews[i] = reviews_history[i][0]

        majority_rewards = []
        majority_reviews = []
        for reviews, verdicts in zip(reviews_history, verdicts_history):
            majority_true, majority_review = self._majority_vote(reviews, verdicts)
            majority_rewards.append(1.0 if majority_true else 0.0)
            majority_reviews.append(majority_review)

        self.last_majority_results = (majority_rewards, majority_reviews)
        self._record_stepwise_logs(verdicts_history, ground_truth_labels)
        self._record_majority_step_logs(verdicts_history, ground_truth_labels)
        self.last_review_counts = total_review_counts

        return rewards, final_reviews, costs

    def __call__(self, problems, completions, ground_truth_labels=None, **kwargs):
        rewards, final_reviews, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, ground_truth_labels=ground_truth_labels, **kwargs))
        return rewards, final_reviews

class VPessimisticVerifier():
    """
    Chunked pessimistic verifier.

    Instead of reviewing the whole proof at once, it splits the proof into
    chunks of `chunk_length` lines. For each chunk, it asks the reviewer to
    focus only on that chunk while still providing the full problem and full
    proof for context. If any chunk is flagged incorrect (`<verification>false</verification>`),
    the final verdict is false. It also aggregates all error reports found.
    """
    def __init__(self, api_base, api_key, model, chunk_length: int = 7):
        self.client = LLMClient(api_base, api_key, model)
        self.chunk_length = max(1, int(chunk_length))
        
        # Constant fallback text when no critical errors are found across all chunks
        self.NO_ERROR_FALLBACK: str = (
            "<verification>true</verification>\n"
            "No critical error found in this proof after chunked review. "
            "All inspected chunks were considered correct overall given the problem and prior steps. "
            "Minor, non-decisive issues (e.g., superficial notation or small slips later corrected) "
            "may exist but do not undermine correctness."
        )

    def _split_into_chunks(self, proof: str) -> list[str]:
        lines = (proof or "").splitlines()
        chunks = []
        for i in range(0, len(lines), self.chunk_length):
            chunk_lines = lines[i:i + self.chunk_length]
            chunks.append("\n".join(chunk_lines))
        if not chunks:
            chunks = [proof or ""]
        return chunks

    def _build_messages_for_one(self, problem: str, full_proof: str) -> list[list[dict]]:
        """Build messages for all chunks of a single (problem, proof)."""
        chunks = self._split_into_chunks(full_proof)
        messages_per_chunk = []
        for idx, chunk in enumerate(chunks, start=1):
            messages_per_chunk.append([
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "We provide the original problem and the complete proposed solution for full context. "
                    "Then we provide a specific chunk from the solution for focused checking. "
                    "Your task: Check ONLY the given chunk for errors while considering the overall context.\n\n"
                    "Checklist:\n"
                    "1. The chunk’s reasoning and calculations adhere to mathematical correctness.\n"
                    "2. Any theorems used in the chunk match their hypotheses and conclusions.\n"
                    "3. The chunk does not rely on assumptions not justified by the problem or earlier proven steps.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the chunk as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness in this chunk (e.g., invalid step, wrong theorem usage without required conditions), treat the chunk as incorrect.\n\n"
                    "Response requirements: If the chunk is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the chunk is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error in the chunk.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<full_answer>{strip_think_simple(full_proof)}</full_answer>\n\n"
                    f"<chunk_index>{idx}</chunk_index>\n"
                    f"<chunk>{chunk}</chunk>"
                )}
            ])
        return messages_per_chunk

    async def verify_async(self, problems, completions, **kwargs):
        """
        For each proof, review all chunks. Any chunk error makes the final verdict false.
        Returns evals (1.0 or 0.0 per proof) and aggregated review texts per proof.
        """
        # Build all chunk messages across the batch
        batch_messages = []
        per_item_chunk_counts = []
        for p, c in zip(problems, completions):
            full_answer = c if isinstance(c, str) else c[0]['content']
            full_answer = strip_think_simple(full_answer)
            msgs = self._build_messages_for_one(p, full_answer)
            per_item_chunk_counts.append(len(msgs))
            batch_messages.extend(msgs)

        # Expose counts for logging/analysis
        self.last_chunk_counts = per_item_chunk_counts[:]

        # Run inference over all chunks
        all_chunk_reviews = await self.client.infer_batch_async(batch_messages, **kwargs)
        
        # Calculate costs
        costs = []
        inputs = getattr(self.client, "last_input_tokens", [])
        outputs = getattr(self.client, "last_comp_tokens", [])
        if len(inputs) != len(batch_messages):
             inputs = [0] * len(batch_messages)
        if len(outputs) != len(batch_messages):
             outputs = [0] * len(batch_messages)

        cursor = 0
        for count in per_item_chunk_counts:
            costs.append({
                "api_calls": count,
                "input_tokens": sum(inputs[cursor:cursor + count]),
                "output_tokens": sum(outputs[cursor:cursor + count])
            })
            cursor += count

        # Group reviews by original sample
        grouped_reviews = []
        cursor = 0
        for count in per_item_chunk_counts:
            grouped_reviews.append(all_chunk_reviews[cursor:cursor + count])
            cursor += count

        # Aggregate verdicts and collect all errors
        evals = []
        final_texts = []
        for reviews in grouped_reviews:
            has_error = False
            errors_text = []
            fallback_text = reviews[0] if reviews else ""
            for r in reviews:
                verdict = extract_xml_content(r, "verification")
                if verdict == "false":
                    has_error = True
                    errors_text.append(strip_think_simple(r))
            if has_error:
                evals.append(0.0)
                # Aggregate all error reports into one text block
                combined = "\n\n".join(errors_text) if errors_text else fallback_text
                final_texts.append(combined)
            else:
                evals.append(1.0)
                # If no errors, return a constant message instead of first review
                final_texts.append(self.NO_ERROR_FALLBACK)

        return evals, final_texts, costs

    def __call__(self, problems, completions, **kwargs):
        rewards, final_reviews, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, **kwargs))
        return rewards, final_reviews

class ProgressivePessimisticVerifier():
    """
    Iteratively applies chunked pessimistic verification with progressively
    finer granularity. It now begins with the same whole-proof prompt used by
    the standard verifier to avoid repeating the solution twice, and then
    doubles the number of chunks (down to min_chunk_size per chunk) for
    still-positive samples until either an error is found or max_iters is
    reached.
    """
    def __init__(self, api_base, api_key, model, max_iters: int = 3, min_chunk_size: int = 6):
        self.client = LLMClient(api_base, api_key, model)
        self.max_iters = max(1, int(max_iters))
        self.min_chunk_size = max(1, int(min_chunk_size))
        self.last_review_counts: list[int] = []
        self.iteration_samples_log: list[dict] = []
        self.iteration_summary: list[dict] = []
        self.iteration_prediction_history: list[list[int]] = []
        self.iteration_resolved_predictions: list[list[int | None]] = []
        self.iteration_pending_masks: list[list[bool]] = []
        self.iteration_review_costs: list[dict] = []

        self.NO_ERROR_FALLBACK: str = (
            "<verification>true</verification>\n"
            "No critical error found in this proof after progressive chunked review. "
            "All passes (from coarse to fine) considered the solution correct overall. "
            "Minor, non-decisive issues may exist but do not undermine correctness."
        )

    def _split_into_chunks(self, proof: str, chunk_length: int) -> list[str]:
        lines = (proof or "").splitlines()
        if not lines:
            return [proof or ""]
        chunks = []
        for i in range(0, len(lines), chunk_length):
            chunk_lines = lines[i:i + chunk_length]
            chunks.append("\n".join(chunk_lines))
        return chunks

    def _build_standard_messages_for_one(self, problem: str, full_proof: str) -> list[list[dict]]:
        stripped_proof = strip_think_simple(full_proof)
        return [[
            {"role": "system", "content": (
                "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
            )},
            {"role": "user", "content": (
                "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
                "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                "2. The solution actually derives the conclusion required by the original problem.\n"
                "3. Every step of calculation and formula derivation in the solution is correct.\n"
                "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                "Consistency and error-severity policy (important):\n"
                "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
                " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error.\n\n"
                f"<problem>{problem}</problem>\n\n"
                f"<answer>{stripped_proof}</answer>"
            )}
        ]]

    def _build_messages_for_one(self, problem: str, full_proof: str, chunk_length: int) -> list[list[dict]]:
        chunks = self._split_into_chunks(full_proof, chunk_length)
        messages_per_chunk = []
        for idx, chunk in enumerate(chunks, start=1):
            messages_per_chunk.append([
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "We provide the original problem and the complete proposed solution for full context. "
                    "Then we provide a specific chunk from the solution for focused checking. "
                    "Your task: Check ONLY the given chunk for errors while considering the overall context.\n\n"
                    "Checklist:\n"
                    "1. The chunk's reasoning and calculations adhere to mathematical correctness.\n"
                    "2. Any theorems used in the chunk match their hypotheses and conclusions.\n"
                    "3. The chunk does not rely on assumptions not justified by the problem or earlier proven steps.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the chunk as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness in this chunk (e.g., invalid step, wrong theorem usage without required conditions), treat the chunk as incorrect.\n\n"
                    "Response requirements: If the chunk is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the chunk is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error in the proof that you found in the chunk.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<full_answer>{full_proof}</full_answer>\n\n"
                    f"<chunk_index>{idx}</chunk_index>\n"
                    f"<chunk>{chunk}</chunk>"
                )}
            ])
        return messages_per_chunk

    def _chunk_length_for_iteration(self, proof: str, iteration: int) -> int:
        lines = (proof or "").splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return self.min_chunk_size
        if iteration == 0:
            return max(num_lines, self.min_chunk_size)
        target_chunks = max(1, 2 ** iteration)
        approx_length = (num_lines + target_chunks - 1) // target_chunks
        return max(self.min_chunk_size, approx_length)

    async def verify_async(self, problems, completions, **kwargs):
        total = len(problems)
        # Initialize costs
        costs = [{"api_calls": 0, "input_tokens": 0, "output_tokens": 0} for _ in range(total)]

        if total == 0:
            self.last_review_counts = []
            self.iteration_samples_log = []
            self.iteration_summary = []
            self.iteration_prediction_history = []
            self.iteration_resolved_predictions = []
            self.iteration_pending_masks = []
            self.iteration_review_costs = []
            return [], [], costs

        proofs = [strip_think_simple(c if isinstance(c, str) else c[0]['content']) for c in completions]
        evals: list[float | None] = [None] * total
        final_texts = [""] * total
        pending_indices = list(range(total))
        total_review_counts = [0] * total
        self.iteration_samples_log = []
        self.iteration_summary = []
        self.iteration_prediction_history = []
        self.iteration_resolved_predictions = []
        self.iteration_pending_masks = []
        self.iteration_review_costs = []
        sample_states = [
            {
                "status": "pending",
                "resolved_iteration": None,
                "final_text": ""
            }
            for _ in range(total)
        ]
        cumulative_reviews = 0

        for iteration in range(self.max_iters):
            if not pending_indices:
                break

            batch_messages = []
            iteration_batch_info: list[dict] = []
            for idx in pending_indices:
                problem = problems[idx]
                proof = proofs[idx]
                chunk_length = self._chunk_length_for_iteration(proof, iteration)
                if iteration == 0:
                    msgs = self._build_standard_messages_for_one(problem, proof)
                    iteration_mode = "standard"
                else:
                    msgs = self._build_messages_for_one(problem, proof, chunk_length)
                    iteration_mode = "chunk"
                iteration_batch_info.append({
                    "sample_index": idx,
                    "chunk_length": chunk_length,
                    "num_chunks": len(msgs),
                    "mode": iteration_mode,
                })
                total_review_counts[idx] += len(msgs)
                batch_messages.extend(msgs)

            if not batch_messages:
                break

            chunk_reviews = await self.client.infer_batch_async(batch_messages, **kwargs)
            
            # Accumulate costs
            inputs = getattr(self.client, "last_input_tokens", [])
            outputs = getattr(self.client, "last_comp_tokens", [])
            if len(inputs) != len(batch_messages):
                inputs = [0] * len(batch_messages)
            if len(outputs) != len(batch_messages):
                outputs = [0] * len(batch_messages)
                
            cursor_token = 0
            for info in iteration_batch_info:
                sample_idx = info["sample_index"]
                count = info["num_chunks"]
                # Sum tokens for this sample's chunks in this iteration
                s_inp = sum(inputs[cursor_token : cursor_token + count])
                s_out = sum(outputs[cursor_token : cursor_token + count])
                cursor_token += count
                
                costs[sample_idx]["api_calls"] += count
                costs[sample_idx]["input_tokens"] += s_inp
                costs[sample_idx]["output_tokens"] += s_out

            latest_input_tokens = getattr(self.client, "last_input_tokens", []) or []
            latest_comp_tokens = getattr(self.client, "last_comp_tokens", []) or []
            avg_input_tokens_this_iter = (
                sum(latest_input_tokens) / len(latest_input_tokens)
                if latest_input_tokens else None
            )
            avg_comp_tokens_this_iter = (
                sum(latest_comp_tokens) / len(latest_comp_tokens)
                if latest_comp_tokens else None
            )
            cumulative_input_tokens = getattr(self.client, "input_tokens", []) or []
            cumulative_comp_tokens = getattr(self.client, "comp_tokens", []) or []
            avg_input_tokens_cumulative = (
                sum(cumulative_input_tokens) / len(cumulative_input_tokens)
                if cumulative_input_tokens else None
            )
            avg_comp_tokens_cumulative = (
                sum(cumulative_comp_tokens) / len(cumulative_comp_tokens)
                if cumulative_comp_tokens else None
            )

            cursor = 0
            next_pending = []
            iteration_samples = []
            failed_this_iter = 0
            passed_this_iter = 0
            chunk_lengths_this_iter = []
            reviews_this_iter = 0

            for info in iteration_batch_info:
                sample_idx = info["sample_index"]
                count = info["num_chunks"]
                sample_reviews = chunk_reviews[cursor:cursor + count]
                cursor += count
                reviews_this_iter += count
                chunk_lengths_this_iter.append(info["chunk_length"])

                chunk_errors: list[str] = []
                for chunk_id, review in enumerate(sample_reviews, start=1):
                    verdict = extract_xml_content(review, "verification")
                    if verdict == "false":
                        formatted = strip_think_simple(review)
                        chunk_errors.append(formatted)

                if chunk_errors:
                    evals[sample_idx] = 0.0
                    combined_errors = "\n\n".join(chunk_errors)
                    final_texts[sample_idx] = combined_errors
                    sample_states[sample_idx].update({
                        "status": "failed",
                        "resolved_iteration": iteration + 1,
                        "final_text": combined_errors,
                    })
                    status_label = "failed"
                    status_eval = 0.0
                    failed_this_iter += 1
                else:
                    if iteration == self.max_iters - 1:
                        evals[sample_idx] = 1.0
                        final_texts[sample_idx] = self.NO_ERROR_FALLBACK
                        sample_states[sample_idx].update({
                            "status": "passed",
                            "resolved_iteration": iteration + 1,
                            "final_text": self.NO_ERROR_FALLBACK,
                        })
                        status_label = "passed"
                        status_eval = 1.0
                        passed_this_iter += 1
                    else:
                        next_pending.append(sample_idx)
                        status_label = "pending"
                        status_eval = None

                iteration_samples.append({
                    "sample_index": sample_idx,
                    "status": status_label,
                    "eval": status_eval,
                    "verification": final_texts[sample_idx] if status_eval is not None else None,
                    "chunk_length": info["chunk_length"],
                    "num_chunks": info["num_chunks"],
                    "chunk_reviews": sample_reviews,
                    "chunk_errors": chunk_errors,
                    "mode": info.get("mode"),
                })

            pending_indices = next_pending

            chunk_length_stats = {}
            if chunk_lengths_this_iter:
                chunk_length_stats = {
                    "min": min(chunk_lengths_this_iter),
                    "max": max(chunk_lengths_this_iter),
                    "mean": sum(chunk_lengths_this_iter) / len(chunk_lengths_this_iter),
                }
            cumulative_reviews += reviews_this_iter

            statuses = [state["status"] for state in sample_states]
            preds_if_stop = [0 if status == "failed" else 1 for status in statuses]
            resolved_preds = [
                0 if status == "failed" else 1 if status == "passed" else None
                for status in statuses
            ]
            pending_mask = [status == "pending" for status in statuses]

            self.iteration_samples_log.append({
                "iteration_index": iteration + 1,
                "samples": iteration_samples,
            })
            self.iteration_summary.append({
                "iteration_index": iteration + 1,
                "reviewed_samples": len(iteration_batch_info),
                "failed_this_iter": failed_this_iter,
                "passed_this_iter": passed_this_iter,
                "pending_after_iteration": len(next_pending),
                "chunk_length_stats": chunk_length_stats,
            })
            self.iteration_prediction_history.append(preds_if_stop)
            self.iteration_resolved_predictions.append(resolved_preds)
            self.iteration_pending_masks.append(pending_mask)
            self.iteration_review_costs.append({
                "iteration_index": iteration + 1,
                "reviews_this_iter": reviews_this_iter,
                "cumulative_reviews": cumulative_reviews,
                "avg_input_tokens_this_iter": avg_input_tokens_this_iter,
                "avg_output_tokens_this_iter": avg_comp_tokens_this_iter,
                "avg_input_tokens_cumulative": avg_input_tokens_cumulative,
                "avg_output_tokens_cumulative": avg_comp_tokens_cumulative,
            })

        # Any remaining samples (e.g., no further iterations but never failed) are treated as passes.
        for idx in pending_indices:
            if evals[idx] is None:
                evals[idx] = 1.0
                final_texts[idx] = self.NO_ERROR_FALLBACK
                sample_states[idx].update({
                    "status": "passed",
                    "resolved_iteration": self.max_iters,
                    "final_text": self.NO_ERROR_FALLBACK,
                })

        # For any sample that never received a review (e.g., empty proof), ensure defaults.
        for i, value in enumerate(evals):
            if value is None:
                evals[i] = 1.0
                final_texts[i] = self.NO_ERROR_FALLBACK
                sample_states[i].update({
                    "status": "passed",
                    "resolved_iteration": self.max_iters,
                    "final_text": self.NO_ERROR_FALLBACK,
                })

        self.last_review_counts = total_review_counts
        return evals, final_texts, costs

    def __call__(self, problems, completions, **kwargs):
        rewards, final_reviews, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, **kwargs))
        return rewards, final_reviews

class NoneVerifier():
    """
    A dummy verifier that always returns True (1.0).
    It maintains the same interface as other verifiers but performs no actual verification.
    """
    def __init__(self, api_base, api_key, model):
        # Initialize client to maintain compatibility if accessed elsewhere,
        # though we won't use it for inference.
        self.client = LLMClient(api_base, api_key, model)

    async def verify_async(self, problems, completions, **kwargs):
        # Always return 1.0 (True)
        rewards = [1.0] * len(problems)
        final_reviews = ["Verification skipped."] * len(problems)
        
        # Zero costs
        costs = []
        for _ in problems:
            costs.append({
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0
            })
        
        return rewards, final_reviews, costs

    def __call__(self, problems, completions, **kwargs):
        rewards, final_reviews, _ = ASYNC_LOOP.run(self.verify_async(problems, completions, **kwargs))
        return rewards, final_reviews
