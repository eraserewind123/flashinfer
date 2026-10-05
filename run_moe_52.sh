source .venv/bin/activate
cd benchmarks
python bench_trtllm_gen_fused_moe_autotuner.py \
  --quant-mode NvFP4xNvFP4 \
  --backends prims_ts \
  --num-tokens 8192 \
  --hidden-size 6144 \
  --intermediate-size 2048 \
  --num-experts 256 \
  --top-k 8 \
  --backends both
