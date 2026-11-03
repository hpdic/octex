#!/bin/bash
# octex reruns of every configuration behind the paper's recall-QPS fronts (4 GPUs).
# Generated; results as JSON lines in results/octex/frontier_<dataset>.jsonl.
set -e
cd "$(dirname "$0")/.."
B=${B:-build/octex_bench}
I=/home/cc/data/octex
mkdir -p results/octex

$B $I/sift1m --dataset sift1m --placement replicate --devices 0,1,2,3 --read 8,12,16,24,32 --waves 0 --rerank 128 --resident --json results/octex/frontier_sift1m.jsonl
$B $I/sift1m --dataset sift1m --placement shard --devices 0,1,2,3 --read 4,8,16 --starts 16,32,64 --waves 1,2 --rerank 128 --wave-rerank 128 --resident --json results/octex/frontier_sift1m.jsonl

$B $I/gist --dataset gist --placement shard --devices 0,1,2,3 --read 8,16,32 --starts 16,32,64,128 --waves 1,2 --rerank 128 --wave-rerank 128 --resident --json results/octex/frontier_gist.jsonl

$B $I/dinov2 --dataset dinov2 --placement replicate --devices 0,1,2,3 --read 4,8,12,16 --waves 0 --rerank 128,256 --resident --json results/octex/frontier_dinov2.jsonl
$B $I/dinov2 --dataset dinov2 --placement shard --devices 0,1,2,3 --read 8,16 --starts 32,64,128 --waves 1,2 --rerank 256 --wave-rerank 128 --resident --json results/octex/frontier_dinov2.jsonl

$B $I/msmarco --dataset msmarco --placement shard --devices 0,1,2,3 --read 4,8,16,32,64 --starts 16,32,64,128,192,256 --waves 1,2,3 --rerank 128 --wave-rerank 128 --final-rerank 32 --resident --json results/octex/frontier_msmarco.jsonl

$B $I/deep10M --dataset deep10M --placement shard --devices 0,1,2,3 --read 8,16,32 --starts 32,64,128,192 --waves 1,2 --rerank 128 --wave-rerank 128 --resident --json results/octex/frontier_deep10M.jsonl

$B $I/deep100M --dataset deep100M --placement shard --devices 0,1,2,3 --copies-limit 128 --shard-vectors --read 16,32,64 --starts 32,64,128 --waves 1,2 --rerank 128 --wave-rerank 128 --resident --json results/octex/frontier_deep100M.jsonl
$B $I/deep100M --dataset deep100M --placement shard --devices 0,1,2,3 --copies-limit 224 --shard-vectors --read 32,48,64 --starts 64,128 --waves 1,2 --rerank 128 --wave-rerank 128 --resident --json results/octex/frontier_deep100M.jsonl
$B $I/deep100M --dataset deep100M --placement shard --devices 0,1,2,3 --shard-vectors --read 32,48,64 --starts 128,192,256,320 --waves 2,3 --rerank 128 --wave-rerank 128,256 --resident --json results/octex/frontier_deep100M.jsonl

$B $I/bigann100M --dataset bigann100M --placement shard --devices 0,1,2,3 --shard-vectors --read 4,8,16,32,64 --starts 16,32,64,128,192,256 --waves 1,2,3 --rerank 128 --wave-rerank 128 --final-rerank 32 --resident --json results/octex/frontier_bigann100M.jsonl

$B $I/dinov2_exact --dataset dinov2 --placement shard --devices 0,1,2,3 --read 4,8,16 --starts 16,32 --waves 1,2 --rerank 256 --wave-rerank 128 --resident --json results/octex/frontier_dinov2.jsonl

$B $I/deep1B --dataset deep1B --placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --read 16 --starts 64,128,192,256,320,384,512,640,768,896,1024 --waves 1,2,3,4,5,6 --rerank 128 --wave-rerank 128 --final-rerank 32 --resident --json results/octex/frontier_deep1B.jsonl
$B $I/bigann1B --dataset bigann1B --placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --read 16 --starts 64,128,192,256,320,384,512,640,768,896,1024 --waves 1,2,3,4,5,6 --rerank 128 --wave-rerank 128 --final-rerank 32 --resident --json results/octex/frontier_bigann1B.jsonl
