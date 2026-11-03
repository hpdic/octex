#!/bin/bash
cd "$(dirname "$0")/.."
B=build/octex_bench
I=/home/cc/data/octex
R=results/octex
D1B="--placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --final-rerank 32"
$B $I/deep1B --dataset deep1B $D1B --table-on-host --read 16 --starts 128,256,512 --waves 2 --resident --json $R/tablehost_deep1B.jsonl
$B $I/deep1B --dataset deep1B $D1B --read 16 --starts 128,256,512 --waves 2 --resident --json $R/tablehbm_deep1B.jsonl
for NQ in 100 200 500 1000 2000 5000 10000; do
  $B $I/deep1B --dataset deep1B $D1B --nq $NQ --read 16 --starts 640 --waves 4 --resident --json $R/batch_deep1B.jsonl
done
