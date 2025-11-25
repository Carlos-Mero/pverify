python main.py \
  --reviewer standard \
  --verifier_samples <path_to_yau_gt_samples> \
  --proof_model gpt-5-mini \
  --eval_model gpt-5-mini \
  --eval_dataset NP_dataset/qz_bench_eval.jsonl \
  --prover_base_url <your_base_url> \
  --prover_api_key <your_api_key> \
