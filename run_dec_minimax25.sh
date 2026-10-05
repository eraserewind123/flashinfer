source .venv/bin/activate
python benchmarks/bench_trtllm_gen_fused_moe_autotuner.py \
  --quant-mode NvFP4xNvFP4 \
  --backends trtllm \
  --num-experts 128 \
  --local-num-experts 128 \
  --num-tokens 4096 \
  --hidden-size 3072 \
  --intermediate-size 768 \
  --top-k 1 \
  --no-use-bias

