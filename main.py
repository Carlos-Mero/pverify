import argparse
import json
import logging
from pathlib import Path
import sys

from utils.common import (
    ASYNC_LOOP,
    extract_xml_content,
    strip_think_simple,
    get_current_log_path,
    _compute_binary_metrics,
    save_progressive_iteration_samples,
    prepare_dataset,
    NaiveProver
)
from utils.verifiers import (
    Verifier,
    PessimisticVerifier,
    VPessimisticVerifier,
    ProgressivePessimisticVerifier,
    PessimisticPruningVerifier,
    NoneVerifier,
)
from utils.prover_pipeline import ProverPipeline

def main():
    # Define common arguments in a parent parser
    common_parser = argparse.ArgumentParser(add_help=False)
    # Global/Common arguments
    common_parser.add_argument("-ed", "--eval_dataset", help="the path to the dataset used for evaluation", default="")
    common_parser.add_argument("-pm", "--proof_model", help="model that generates proofs for given problems", default="")
    common_parser.add_argument("-em", "--eval_model", help="the model used for evaluation (if needed)", default="")
    common_parser.add_argument("--log_dir", help="the logging directory path", default="eval_logs")
    common_parser.add_argument("--reasoning_effort", help="the reasoning_effort parameter for some models", default="medium", choices=["minimal", "low", "medium", "high"])
    common_parser.add_argument("--reviewer", default="standard", choices=["standard", "pessimistic", "vpessimistic", "progressive", "ppruning", "none"], help="the reviewer used for evaluation")
    common_parser.add_argument("--reviews", type=int, default=3, help="maximum reviews per sample for multi-review verifiers (pessimistic/ppruning)")
    common_parser.add_argument("--chunk_length", type=int, default=7, help="lines per chunk for vpessimistic reviewer")
    common_parser.add_argument("--progressive_max_iters", type=int, default=3, help="maximum refinement passes for progressive reviewer")
    common_parser.add_argument("--progressive_min_chunk_size", type=int, default=6, help="minimum lines per chunk for progressive reviewer")
    common_parser.add_argument("--prover_base_url", default="", help="the base url for prover")
    common_parser.add_argument("--eval_base_url", default="", help="the base url for evaluator")
    common_parser.add_argument("--prover_api_key", default="", help="the api key for the prover")
    common_parser.add_argument("--eval_api_key", default="", help="the api key for the evaluator")
    common_parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True, help="toggle enable_thinking parameter for models that support reasoning traces")
    common_parser.add_argument(
        "--verifier_samples",
        default="",
        help=(
            "path to a previously generated verifier_samples.json. "
            "When set, uses the same problems, proofs, and golden labels from the file, "
            "skipping new proof generation and ground-truth verification."
        ),
    )

    # Main parser inherits from common_parser
    parser = argparse.ArgumentParser(
        description="RProver",
        parents=[common_parser]
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # 'prove' subcommand inherits from common_parser
    prove_parser = subparsers.add_parser("prove", help="Run the iterative prover pipeline", parents=[common_parser])
    prove_parser.add_argument("--dataset", required=True, help="Dataset to use (overrides -ed if present)")
    prove_parser.add_argument("--prover", required=True, help="Prover model name (overrides -pm if present)")
    prove_parser.add_argument("--refine_iters", type=int, default=3, help="Maximum number of refinement iterations")
    # Note: 'prove' will also use global args like --reviewer, --reviews, etc.

    logger = logging.getLogger("main")
    args = parser.parse_args()

    # Handle 'prove' subcommand
    if args.command == "prove":
        # Override global args with subcommand specific args if needed/provided
        # But here we used unique names mostly. Let's sync them for clarity
        dataset_path = args.dataset
        prover_model = args.prover
        # Set args for compatibility with functions that use them
        args.eval_dataset = dataset_path
        args.proof_model = prover_model
        
        logger.info(f"Starting 'prove' pipeline on {dataset_path} with prover {prover_model}")

        # Resolve API bases
        prover_base_url = args.prover_base_url
        prover_api_key = args.prover_api_key
        eval_base_url = args.eval_base_url or prover_base_url
        eval_api_key = args.eval_api_key or prover_api_key
        
        # Instantiate Pipeline
        pipeline = ProverPipeline(
            args=args,
            dataset_path=dataset_path,
            prover_model=prover_model,
            prover_api_base=prover_base_url,
            prover_api_key=prover_api_key,
            reviewer_model=args.eval_model or prover_model, # Default to prover if eval not set? or check if empty.
            reviewer_api_base=eval_base_url,
            reviewer_api_key=eval_api_key,
            max_refine_iters=args.refine_iters
        )
        
        ASYNC_LOOP.run(pipeline.run())
        return

    # --- Standard 'main' logic (default behavior) ---

    logger.info("start verifying with proof_model: %s", args.proof_model)
    logger.info("using eval model: %s", args.eval_model)

    # If verifier_samples is provided, use it to load problems/proofs and GT labels
    loaded_verifier_samples = None
    preloaded_gt_labels = None
    preloaded_gt_texts = None
    if args.verifier_samples:
        if args.verifier_samples == "Salesforce/Hard2Verify" or args.verifier_samples == "NP_dataset/gradingbench.csv":
            ds = prepare_dataset(args.verifier_samples)
            problems = ds["problem"]
            proofs = ds["proof"]
            # preloaded_gt_texts = [e["human_labels_first_error_idx"] for e in ds]
            preloaded_gt_labels = ds["gt_eval"]
            preloaded_gt_texts = [None] * len(problems)
            logger.info("Loaded %d samples from verifier_samples: %s", len(problems), args.verifier_samples)
        else:
            vs_path = Path(args.verifier_samples) / "samples.json"
            with vs_path.open("r", encoding="utf-8") as f:
                loaded_verifier_samples = json.load(f)
            if not isinstance(loaded_verifier_samples, list):
                raise ValueError("verifier_samples must be a list of sample dicts")
            problems = [s.get("problem", "") for s in loaded_verifier_samples]
            proofs = [s.get("proof", "") for s in loaded_verifier_samples]
            preloaded_gt_labels = [bool(s.get("eval", False)) for s in loaded_verifier_samples]
            preloaded_gt_texts = [s.get("verification", "") for s in loaded_verifier_samples]
            logger.info("Loaded %d samples from verifier_samples: %s", len(problems), vs_path)
    else:
        ds = prepare_dataset(args.eval_dataset)
        problems = [e['problem'] for e in ds]

    # Resolve API bases and keys with fallback: prover -> eval
    prover_base_url = args.prover_base_url
    prover_api_key = args.prover_api_key

    eval_base_url = args.eval_base_url or prover_base_url
    eval_api_key = args.eval_api_key or prover_api_key

    prover = NaiveProver(
        api_base=prover_base_url,
        api_key=prover_api_key,
        model=args.proof_model,
    )

    logdir = get_current_log_path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Collect proofs unless verifier_samples is provided
    if args.verifier_samples:
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("Using preloaded proofs from verifier_samples, skipping prover generation")
    else:
        proofs = prover(
            problems,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("successfully collected %d proofs from %s", len(proofs), args.proof_model)

    if args.reviewer == "pessimistic":
        # Use the new PessimisticVerifier (first error wins, also records majority vote)
        evaluator = PessimisticVerifier(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
    elif args.reviewer == "vpessimistic":
        # Chunked pessimistic verifier (focus per-chunk)
        evaluator = VPessimisticVerifier(eval_base_url, eval_api_key, args.eval_model, chunk_length=args.chunk_length)
    elif args.reviewer == "progressive":
        # Progressive chunking: coarse-to-fine pessimistic verifier
        evaluator = ProgressivePessimisticVerifier(
            eval_base_url,
            eval_api_key,
            args.eval_model,
            max_iters=args.progressive_max_iters,
            min_chunk_size=args.progressive_min_chunk_size,
        )
    elif args.reviewer == "ppruning":
        evaluator = PessimisticPruningVerifier(
            eval_base_url,
            eval_api_key,
            args.eval_model,
            review_times=args.reviews,
        )
    elif args.reviewer == "none":
        evaluator = NoneVerifier(eval_base_url, eval_api_key, args.eval_model)
    else:
        evaluator = Verifier(eval_base_url, eval_api_key, args.eval_model)
    eval_call_kwargs = {
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking": args.enable_thinking,
    }
    if args.reviewer in {"pessimistic", "ppruning"}:
        eval_call_kwargs["ground_truth_labels"] = preloaded_gt_labels

    evals, verifications = evaluator(
        problems,
        striped_proofs,
        **eval_call_kwargs,
    )
    accuracy = sum(evals) / len(evals)
    logger.info(f"Obtained final accuracy: {accuracy}")

    majority_evals = None
    majority_verifications = None
    majority_accuracy = None
    pess_family = args.reviewer in {"pessimistic", "ppruning"}
    if pess_family:
        majority_results = getattr(evaluator, "last_majority_results", None)
        if majority_results:
            majority_evals, majority_verifications = majority_results
            if len(majority_evals) == len(evals) and len(evals) > 0:
                majority_accuracy = sum(majority_evals) / len(majority_evals)
                logger.info(f"Majority voting accuracy from the same reviews: {majority_accuracy}")
        else:
            majority_evals = None
            majority_verifications = None

        if args.reviewer == "pessimistic":
            step_logs = getattr(evaluator, "stepwise_review_logs", None)
            if step_logs:
                step_log_path = logdir / "pessimistic_step_metrics.json"
                with step_log_path.open("w", encoding="utf-8") as f:
                    json.dump(step_logs, f, ensure_ascii=False, indent=2, default=str)
                logger.info("Saved stepwise pessimistic review metrics to %s", step_log_path)

            majority_step_logs = getattr(evaluator, "majority_step_logs", None)
            majority_metrics = None
            if (
                preloaded_gt_labels
                and majority_evals
                and len(majority_evals) == len(preloaded_gt_labels)
            ):
                gt_vector = [1 if bool(x) else 0 for x in preloaded_gt_labels]
                majority_preds = [1 if bool(x) else 0 for x in majority_evals]
                majority_metrics = _compute_binary_metrics(majority_preds, gt_vector)

            if majority_step_logs or majority_metrics:
                majority_log_path = logdir / "pessimistic_majority_metrics.json"
                payload = {}
                if majority_step_logs:
                    payload["steps"] = majority_step_logs
                if majority_metrics:
                    payload["metrics"] = majority_metrics
                with majority_log_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                logger.info("Saved pessimistic majority metrics to %s", majority_log_path)


    logger.info("Evaluating reviewer against ground truth")
    if args.verifier_samples:
        # Use ground-truth labels/texts from the provided verifier_samples file
        gt_labels = preloaded_gt_labels
        gt_texts = preloaded_gt_texts
        logger.info("Using GT labels from verifier_samples; skipping new GT verification")
    else:
        gt_labels = [1] * len(evals)

    preds = [int(x) for x in evals]
    gts = [int(x) for x in gt_labels]
    verifier_eval = _compute_binary_metrics(preds, gts)

    if pess_family and majority_evals and len(majority_evals) == len(preds):
        majority_preds = [int(x) for x in majority_evals]
        verifier_eval["majority_vote_metrics"] = _compute_binary_metrics(majority_preds, gts)

    pred_history = getattr(evaluator, "iteration_prediction_history", [])
    if pred_history:
        resolved_history = getattr(evaluator, "iteration_resolved_predictions", [])
        pending_masks = getattr(evaluator, "iteration_pending_masks", [])
        review_costs = getattr(evaluator, "iteration_review_costs", [])
        iteration_metrics = []
        total_samples = len(gts)
        for idx, preds_if_stop in enumerate(pred_history):
            iteration_index = idx + 1
            metrics_if_stop = _compute_binary_metrics(preds_if_stop, gts)
            resolved_preds = resolved_history[idx] if idx < len(resolved_history) else [None] * total_samples
            resolved_metrics = _compute_binary_metrics(resolved_preds, gts)
            pending_mask = pending_masks[idx] if idx < len(pending_masks) else [False] * total_samples
            pending_samples = sum(1 for flag in pending_mask if flag)
            cost_info = review_costs[idx] if idx < len(review_costs) else {}
            iteration_metrics.append({
                "iteration_index": iteration_index,
                "metrics_if_stopped": metrics_if_stop,
                "resolved_metrics": resolved_metrics,
                "pending_samples": pending_samples,
                "resolved_samples": total_samples - pending_samples,
                "reviews_this_iter": cost_info.get("reviews_this_iter"),
                "cumulative_reviews": cost_info.get("cumulative_reviews"),
                "avg_input_tokens_this_iter": cost_info.get("avg_input_tokens_this_iter"),
                "avg_output_tokens_this_iter": cost_info.get("avg_output_tokens_this_iter"),
                "avg_input_tokens_cumulative": cost_info.get("avg_input_tokens_cumulative"),
                "avg_output_tokens_cumulative": cost_info.get("avg_output_tokens_cumulative"),
            })
        if iteration_metrics:
            key_name = "progressive_iteration_metrics" if args.reviewer == "progressive" else f"{args.reviewer}_iteration_metrics"
            verifier_eval[key_name] = iteration_metrics

        # Save sample-level comparison
        majority_sample_fields = None
        if (
            pess_family
            and majority_evals
            and majority_verifications
            and len(majority_evals) == len(gts)
            and len(majority_verifications) == len(gts)
        ):
            majority_sample_fields = list(zip(
                [bool(int(x)) for x in majority_evals],
                majority_verifications,
            ))

        verifier_samples = []
        for idx, (problem, proof, pred, pred_text, gt, gt_text) in enumerate(
            zip(problems, proofs, preds, verifications, gts, gt_texts)
        ):
            sample_entry = {
                "problem": problem,
                "proof": proof,
                "pred_label": bool(pred),
                "pred_text": pred_text,
                "gt_label": bool(gt),
                "gt_text": gt_text,
            }
            if majority_sample_fields:
                maj_label, maj_text = majority_sample_fields[idx]
                sample_entry["majority_pred_label"] = maj_label
                sample_entry["majority_pred_text"] = maj_text
            verifier_samples.append(sample_entry)

        with open(logdir / "verifier_eval.json", "w", encoding="utf-8") as f:
            json.dump(verifier_eval, f, ensure_ascii=False, indent=2, default=str)
        with open(logdir / "verifier_samples.json", "w", encoding="utf-8") as f:
            json.dump(verifier_samples, f, ensure_ascii=False, indent=2, default=str)

        # Add summary to logs.json payload
        vars_dict_key = "verifier_evaluation"
        # vars_dict defined below; collect into a temporary dict for later merge
        extra_verifier_eval = {vars_dict_key: verifier_eval}


    logger.info("evaluation ended")
    vars_dict = vars(args)
    vars_dict["accuracy"] = accuracy
    if majority_accuracy is not None:
        vars_dict["majority_accuracy"] = majority_accuracy
    # Reviewer cost metrics for post-hoc cost/performance analysis
    reviewer_cost = {"reviewer": args.reviewer}
    num_samples = len(problems)
    if args.reviewer in {"vpessimistic", "progressive", "ppruning"}:
        if args.reviewer == "vpessimistic":
            counts = getattr(evaluator, "last_chunk_counts", []) or []
        else:
            counts = getattr(evaluator, "last_review_counts", []) or []
        total_reviews = sum(counts)
        avg_per_sample = (total_reviews / len(counts)) if counts else 0.0
        reviewer_cost.update({
            "total_reviews": total_reviews,
            "avg_reviews_per_sample": avg_per_sample,
            "min_reviews_per_sample": (min(counts) if counts else 0),
            "max_reviews_per_sample": (max(counts) if counts else 0),
        })
    else:
        if args.reviewer == "standard":
            per_sample = 1
        elif args.reviewer == "pessimistic":
            per_sample = int(args.reviews)
        else:
            per_sample = 1
        reviewer_cost.update({
            "reviews_per_sample": per_sample,
            "total_reviews": per_sample * num_samples,
        })
    vars_dict["reviewer_cost"] = reviewer_cost
    # Token stats: skip prover token stats when using preloaded samples
    if args.verifier_samples:
        average_prover_inp_tokens = None
        average_prover_opt_tokens = None
    else:
        average_prover_inp_tokens = (
            sum(prover.client.input_tokens) / len(prover.client.input_tokens)
            if prover.client.input_tokens else 0.0
        )
        average_prover_opt_tokens = (
            sum(prover.client.comp_tokens) / len(prover.client.comp_tokens)
            if prover.client.comp_tokens else 0.0
        )
    average_eval_inp_tokens = (
        sum(evaluator.client.input_tokens) / len(evaluator.client.input_tokens)
        if evaluator.client.input_tokens else 0.0
    )
    average_eval_opt_tokens = (
        sum(evaluator.client.comp_tokens) / len(evaluator.client.comp_tokens)
        if evaluator.client.comp_tokens else 0.0
    )
    logger.info(f"Average token inputs in prover: {average_prover_inp_tokens}")
    logger.info(f"Average completion tokens in prover: {average_prover_opt_tokens}")
    logger.info(f"Average token inputs in evaluator: {average_eval_inp_tokens}")
    logger.info(f"Average completion inputs in evaluator: {average_eval_opt_tokens}")
    vars_dict["average_prover_inp_tokens"] = average_prover_inp_tokens
    vars_dict["average_prover_opt_tokens"] = average_prover_opt_tokens
    vars_dict["average_eval_inp_tokens"] = average_eval_inp_tokens
    vars_dict["average_eval_opt_tokens"] = average_eval_opt_tokens

    # Merge verifier evaluation summary if available
    try:
        if 'extra_verifier_eval' in locals():
            vars_dict.update(extra_verifier_eval)
    except Exception:
        pass

    with open(logdir / "logs.json", "w", encoding="utf-8") as f:
        json.dump(vars_dict, f, ensure_ascii=False, indent=2, default=str)

    # Prepare sample payload; use placeholder tokens if verifier_samples is provided
    if args.verifier_samples:
        prover_inp_tokens = [None] * len(problems)
        prover_comp_tokens = [None] * len(problems)
    else:
        prover_inp_tokens = prover.client.last_input_tokens
        prover_comp_tokens = prover.client.last_comp_tokens

    samples = [
        {
            "problem": problem,
            "proof": proof,
            "eval": eval,
            "verification": verification,
            "input_tokens": inp_tokens,
            "completion_tokens": comp_tokens
        }
        for (problem, proof, eval, verification, inp_tokens, comp_tokens) in zip(
            problems, proofs, evals, verifications, prover_inp_tokens, prover_comp_tokens
        )
    ]
    if majority_evals and majority_verifications and len(samples) == len(majority_evals):
        for sample, maj_eval, maj_verification in zip(samples, majority_evals, majority_verifications):
            sample["majority_eval"] = maj_eval
            sample["majority_verification"] = maj_verification

    with open(logdir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2, default=str)

    if args.reviewer in {"progressive", "ppruning"}:
        prefix = "progressive" if args.reviewer == "progressive" else "ppruning"
        save_progressive_iteration_samples(
            logdir,
            getattr(evaluator, "iteration_samples_log", []),
            problems,
            proofs,
            striped_proofs,
            getattr(evaluator, "iteration_summary", []),
            getattr(evaluator, "iteration_review_costs", []),
            prefix=prefix,
        )
    logger.info(f"successfully saved logs to path {logdir}")

if __name__ == "__main__":
    LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=DATE_FMT,
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Program Started")
    main()
