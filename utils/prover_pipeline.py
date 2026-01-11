import json
import logging
import asyncio
from pathlib import Path

from utils.common import (
    ASYNC_LOOP,
    LLMClient,
    NaiveProver,
    prepare_dataset,
    get_current_log_path,
    strip_think_simple,
    extract_xml_content,
    batch_check_correctness
)
from utils.verifiers import (
    Verifier,
    PessimisticVerifier,
    VPessimisticVerifier,
    ProgressivePessimisticVerifier,
    PessimisticPruningVerifier,
)

class ProverPipeline:
    def __init__(
        self,
        args,
        dataset_path: str,
        prover_model: str,
        prover_api_base: str,
        prover_api_key: str,
        reviewer_model: str,
        reviewer_api_base: str,
        reviewer_api_key: str,
        max_refine_iters: int = 3,
    ):
        self.args = args
        self.dataset_path = dataset_path
        self.prover_model = prover_model
        self.prover_client = LLMClient(prover_api_base, prover_api_key, prover_model)
        self.max_refine_iters = max_refine_iters
        
        # Initialize Reviewer
        self.reviewer_type = args.reviewer
        if self.reviewer_type == "pessimistic":
            self.reviewer = PessimisticVerifier(reviewer_api_base, reviewer_api_key, reviewer_model, review_times=args.reviews)
        elif self.reviewer_type == "vpessimistic":
            self.reviewer = VPessimisticVerifier(reviewer_api_base, reviewer_api_key, reviewer_model, chunk_length=args.chunk_length)
        elif self.reviewer_type == "progressive":
            self.reviewer = ProgressivePessimisticVerifier(
                reviewer_api_base,
                reviewer_api_key,
                reviewer_model,
                max_iters=args.progressive_max_iters,
                min_chunk_size=args.progressive_min_chunk_size,
            )
        elif self.reviewer_type == "ppruning":
            self.reviewer = PessimisticPruningVerifier(
                reviewer_api_base,
                reviewer_api_key,
                reviewer_model,
                review_times=args.reviews,
            )
        else:
            self.reviewer = Verifier(reviewer_api_base, reviewer_api_key, reviewer_model)
            
        self.log_dir = get_current_log_path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("prover_pipeline")

    async def run(self):
        # 1. Load Dataset
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        ds = prepare_dataset(self.dataset_path)
        problems = [e['problem'] for e in ds]
        gt_answers = ds['gt_answer'] if 'gt_answer' in ds.column_names else None
        
        # Track state
        current_proofs = [""] * len(problems)
        # Statistics per problem
        problem_stats = [
            {"api_calls": 0, "cum_input_tokens": 0, "cum_output_tokens": 0}
            for _ in problems
        ]
        
        accumulated_stats = []
        
        # Helper to run inference using the client directly
        async def generate_proofs(messages):
            return await self.prover_client.infer_batch_async(
                messages,
                reasoning_effort=self.args.reasoning_effort,
                enable_thinking=self.args.enable_thinking
            )

        # 2. Initial Proof Generation
        self.logger.info("Generating initial proofs...")
        
        initial_messages = [
            [{"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}"}]
            for p in problems
        ]
        
        current_proofs = await generate_proofs(initial_messages)
        
        # Update stats for initial generation
        last_inputs = getattr(self.prover_client, "last_input_tokens", [0]*len(problems))
        last_outputs = getattr(self.prover_client, "last_comp_tokens", [0]*len(problems))
        
        # Ensure lists are correct length (handle potential failures/empty returns)
        if len(last_inputs) != len(problems):
             last_inputs = [0] * len(problems)
        if len(last_outputs) != len(problems):
             last_outputs = [0] * len(problems)

        for i in range(len(problems)):
            problem_stats[i]["api_calls"] += 1
            problem_stats[i]["cum_input_tokens"] += last_inputs[i]
            problem_stats[i]["cum_output_tokens"] += last_outputs[i]
        
        # 3. Refinement Loop
        passed_indices = set()
        current_evals = [0.0] * len(problems)
        current_verifications = [""] * len(problems)

        for iteration in range(self.max_refine_iters + 1):
            self.logger.info(f"--- Iteration {iteration} ---")
            
            # Verify proofs
            # Only verify proofs that haven't passed yet
            indices_to_verify = [i for i in range(len(problems)) if i not in passed_indices]
            
            if not indices_to_verify:
                self.logger.info("All proofs have passed previously. Skipping verification.")
            else:
                self.logger.info(f"Verifying {len(indices_to_verify)} proofs...")
                
                subset_problems = [problems[i] for i in indices_to_verify]
                subset_proofs = [current_proofs[i] for i in indices_to_verify]
                
                eval_kwargs = {
                    "reasoning_effort": self.args.reasoning_effort,
                    "enable_thinking": self.args.enable_thinking,
                }
                if self.reviewer_type in {"pessimistic", "ppruning"}:
                     eval_kwargs["ground_truth_labels"] = None

                subset_evals, subset_verifications, subset_costs = await self.reviewer.verify_async(subset_problems, subset_proofs, **eval_kwargs)
                
                # Update state with new verification results
                for local_idx, original_idx in enumerate(indices_to_verify):
                    res = subset_evals[local_idx]
                    current_evals[original_idx] = res
                    current_verifications[original_idx] = subset_verifications[local_idx]
                    
                    if res == 1.0:
                        passed_indices.add(original_idx)
                        
                    # Update problem stats with review costs
                    cost = subset_costs[local_idx]
                    problem_stats[original_idx]["api_calls"] += cost.get("api_calls", 0)
                    problem_stats[original_idx]["cum_input_tokens"] += cost.get("input_tokens", 0)
                    problem_stats[original_idx]["cum_output_tokens"] += cost.get("output_tokens", 0)
            
            pass_count = len(passed_indices)
            self.logger.info(f"Iteration {iteration}: {pass_count}/{len(problems)} passed.")

            # Calculate Ground Truth Accuracy if gt_answers exist
            gt_accuracy = None
            gt_correctness = []
            if gt_answers:
                # Prepare inputs for batch verification
                # We need to strip thinking from proofs
                stripped_proofs = [strip_think_simple(p) for p in current_proofs]
                
                # batch_check_correctness runs in separate processes
                gt_correctness = batch_check_correctness(stripped_proofs, gt_answers)
                
                correct_count = sum(1 for x in gt_correctness if x)
                gt_accuracy = correct_count / len(problems)
                self.logger.info(f"Iteration {iteration}: Ground Truth Accuracy: {gt_accuracy:.2%} ({correct_count}/{len(problems)})")
            
            # Snapshot stats
            iteration_stats = {
                "iteration": iteration,
                "pass_rate": pass_count / len(problems),
                "gt_accuracy": gt_accuracy,
                "gt_correctness": gt_correctness if gt_answers else None,
                "proofs": current_proofs[:], 
                "evals": current_evals[:],
                "verifications": current_verifications[:],
                "problem_stats": [dict(s) for s in problem_stats]
            }
            accumulated_stats.append(iteration_stats)
            self._save_logs(iteration, accumulated_stats)

            # Check termination
            if iteration >= self.max_refine_iters:
                self.logger.info("Max iterations reached.")
                break
            
            if pass_count == len(problems):
                self.logger.info("All problems passed!")
                break
                
            # Prepare for refinement
            refinement_messages = []
            indices_to_refine = []
            
            for i, (p, proof, result, feedback) in enumerate(zip(problems, current_proofs, current_evals, current_verifications)):
                if result == 1.0:
                    continue # Already passed
                
                indices_to_refine.append(i)
                msg = [
                    {"role": "user", "content": f"Problem: {p}\n\nPrevious Proof:\n{proof}\n\nReviewer Feedback:\n{feedback}\n\nPlease provide a corrected and rigorous proof that addresses the reviewer's feedback."}
                ]
                refinement_messages.append(msg)
            
            if not indices_to_refine:
                break
                
            self.logger.info(f"Refining {len(indices_to_refine)} proofs...")
            refined_proofs = await generate_proofs(refinement_messages)
            
            # Update stats for refinement
            last_inputs = getattr(self.prover_client, "last_input_tokens", [0]*len(indices_to_refine))
            last_outputs = getattr(self.prover_client, "last_comp_tokens", [0]*len(indices_to_refine))

            if len(last_inputs) != len(indices_to_refine):
                 last_inputs = [0] * len(indices_to_refine)
            if len(last_outputs) != len(indices_to_refine):
                 last_outputs = [0] * len(indices_to_refine)

            for local_idx, (new_proof, inp, out) in enumerate(zip(refined_proofs, last_inputs, last_outputs)):
                original_idx = indices_to_refine[local_idx]
                current_proofs[original_idx] = new_proof
                problem_stats[original_idx]["api_calls"] += 1
                problem_stats[original_idx]["cum_input_tokens"] += inp
                problem_stats[original_idx]["cum_output_tokens"] += out
        
        self.logger.info("Pipeline finished.")

    def _save_logs(self, iteration, stats):
        log_path = self.log_dir / f"iteration_{iteration}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)