# Billion-scale runs (Deep1B, SIFT1B)

The scripts behind the billion-scale results of the paper, as run for it.

- `prep_sift1b.py`: SIFT1B (BIGANN `base.1B.u8bin`, `bigann_query.bvecs`, `gnd/idx_1000M.ivecs`) as fp16 `.npy` base,
  f32 queries, and int64 ground truth, by sequential buffered reads and writes.
- `big.py`: `python big.py prepare deep1B` writes the fp16 base of Deep1B; the fp16 base and the queries of both
  datasets for the scripts below.
- `cagra_big.py`: CAGRA of cuVS on 8 shards, 2 per GPU on four GPUs:
  `python cagra_big.py build bigann1B --shards 8 --degree 32`, then
  `python cagra_big.py search bigann1B --shards 8 --degree 32 --itopk 32,64,80,96,112,128,160,192,224,256,320,384,512,768,1024 --width 1,2,4`.
- `rabitq_eval.py`, `rabitq_bench.cu`, `run_rabitq.sh`: IVF-RaBitQ of cuVS on 8 shards with an exact rerank:
  `python rabitq_eval.py bigann1B --shards 8 --n-lists 65536 --bits 4,7 --probes 32,48,64,128,256,512 --ratios 2,4`.
  The exact rerank uses `select_rerank2s` of `proto_cuda.py`, the CUDA extension of the research prototype.

The OctEx indexes and fronts of both datasets are built and measured by `bench/paper_build.sh` and
`bench/paper_frontier.sh`.
