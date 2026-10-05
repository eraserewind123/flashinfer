source .venv/bin/activate

python -m flashinfer.prims_ts.batched_gemm.tools.bench \
  --runner ts --tokens 8192 \
  --hidden-size 3072 --intermediate-size 768 \
  --num-experts 256 --top-k 8 \
  --variants fp4_fc1_ll_t8_k512

# fp4_fc1_ht_t128_k256_ldgsts,fp4_fc1_ll_t8_k512

#,fp4_fc2_ht_t64_k512,fp4_fc2_ll_t8_k512

#fp4_fc1_ht_t128_k256_ldgsts', 'fp4_fc1_ll_t8_k512', 'fp4_fc2_ht_t64_k512', 'fp4_fc2_ll_t8_k512'
#fp4_fc1_ll_t8_k512 FC1, low-latency
#fp4_fc1_ht_t128_k256_ldgsts FC1, high-throughput
#fp4_fc2_ll_t8_k512 FC2, low-latency (default)
#fp4_fc2_ht_t64_k512 FC2, high-throughputfp4_fc1_ll_t8_k512
