#!/bin/bash
# octex runs behind the paper's design-choice and deployment figures (4 GPUs unless noted); JSON lines in
# results/octex/<figure>_<dataset>.jsonl. Run after paper_frontier.sh.
set -e
cd "$(dirname "$0")/.."
B=${B:-build/octex_bench}
I=/home/cc/data/octex
R=results/octex
mkdir -p $R
D100="--placement shard --devices 0,1,2,3 --shard-vectors"
D1B="--placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --final-rerank 32"

# billion scale first, while its files are still in the page cache after the frontier run:
# neighbor table in host memory against HBM (2 waves), then the batch size (the first nq queries as one batch)
$B $I/deep1B --dataset deep1B $D1B --table-on-host --read 16 --starts 128,256,512 --waves 2 --resident --json $R/tablehost_deep1B.jsonl
$B $I/deep1B --dataset deep1B $D1B --read 16 --starts 128,256,512 --waves 2 --resident --json $R/tablehbm_deep1B.jsonl
# exact rerank depth after sketch ranking: 32, 64 and 128 entries
$B $I/deep1B --dataset deep1B --placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank \
  --final-rerank 32,64,128 --read 16 --starts 512,768 --waves 4 --resident --json $R/finalrerank_deep1B.jsonl
for NQ in 100 200 500 1000 2000 5000 10000; do
  $B $I/deep1B --dataset deep1B $D1B --nq $NQ --read 16 --starts 640 --waves 4 --resident --json $R/batch_deep1B.jsonl
done
for NQ in 100 200 500 1000 2000 5000 10000; do
  $B $I/deep100M --dataset deep100M $D100 --nq $NQ --read 64 --starts 64,128 --waves 2 --resident --json $R/batch_deep100M.jsonl
done

# reverse edges at 100M: the same 272 copies with the forward table only and with reverse edges merged
$B $I/deep100M_fwd --dataset deep100M $D100 --read 32,48,64 --starts 128 --waves 1,2 --resident --json $R/reverse_fwd_deep100M.jsonl
$B $I/deep100M --dataset deep100M $D100 --read 32,48,64 --starts 128 --waves 1,2 --resident --json $R/reverse_rev_deep100M.jsonl

# copies at 100M: recall and throughput as the first round reads more copies
for C in 64 128 192 272; do
  $B $I/deep100M --dataset deep100M $D100 --copies-limit $C --read 48 --starts 128 --waves 1,2 --resident --json $R/copies_deep100M.jsonl
done

# GPUs: SIFT1M first round on 1, 2 and 4 GPUs, every GPU with the whole index
for G in 0 0,1 0,1,2,3; do
  $B $I/sift1m --dataset sift1m --placement replicate --devices $G --read 8,12,16,24,32 --waves 0 --rerank 128 --resident --json $R/scaling_sift1m.jsonl
done

# one GPU on Deep100M: 48 copies (6 of each of 8 parts), lists of 32
$B $I/deep100M --dataset deep100M --placement shard --devices 0 --parts 8 --copies-per-part 6 --list-limit 32 \
  --read 16,32 --starts 64,128 --waves 1,2,3 --resident --json $R/onegpu_deep100M.jsonl

