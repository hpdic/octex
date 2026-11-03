#!/bin/bash
# The billion-scale measurements of the paper on the indexes of octex_build --fast (deep1B_member, bigann1B_member):
# frontiers, design-choice and deployment runs, the list-length index, the GPU-metrics and timeline profiles. JSON
# lines in results/octex as paper_frontier.sh and paper_figures.sh write them. Run after both --fast builds.
set -e
cd "$(dirname "$0")/.."
B=${B:-build/octex_bench}
OB=${OB:-build/octex_build}
I=/home/cc/data/octex
R=results/octex
P=/home/cc/tmp/prof_billion
D1B="--placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --final-rerank 32"
STARTS=64,128,192,256,320,384,512,640,768,896,1024
mkdir -p $R $P
export TMPDIR=${TMPDIR:-/home/cc/.cache/tmp}
# FROM=table|ablation|profile|listlen resumes at that stage
FROM=${FROM:-frontier}
stage() {  # true from the stage FROM on
  local order=(frontier table ablation profile listlen) seen=0
  for x in "${order[@]}"; do
    [ "$x" = "$FROM" ] && seen=1
    [ "$x" = "$1" ] && { [ $seen = 1 ]; return; }
  done
}

if stage frontier; then
echo "== frontiers $(date)"
rm -f $R/frontier_deep1B.jsonl $R/frontier_bigann1B.jsonl
$B $I/deep1B_member --dataset deep1B $D1B --read 16 --starts $STARTS --waves 1,2,3,4,5,6 --rerank 128 \
  --wave-rerank 128 --resident --json $R/frontier_deep1B.jsonl
$B $I/bigann1B_member --dataset bigann1B $D1B --read 16 --starts $STARTS --waves 1,2,3,4,5,6 --rerank 128 \
  --wave-rerank 128 --resident --json $R/frontier_bigann1B.jsonl

fi
if stage table; then
echo "== table placement, rerank depth, batch size $(date)"
rm -f $R/tablehost_deep1B.jsonl $R/tablehbm_deep1B.jsonl $R/finalrerank_deep1B.jsonl $R/batch_deep1B.jsonl
$B $I/deep1B_member --dataset deep1B $D1B --table-on-host --read 16 --starts 128,256,512 --waves 2 --resident \
  --json $R/tablehost_deep1B.jsonl
$B $I/deep1B_member --dataset deep1B $D1B --read 16 --starts 128,256,512 --waves 2 --resident \
  --json $R/tablehbm_deep1B.jsonl
$B $I/deep1B_member --dataset deep1B --placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard \
  --sketch-rank --final-rerank 32,64,128 --read 16 --starts 512,768 --waves 4 --resident \
  --json $R/finalrerank_deep1B.jsonl
for NQ in 100 200 500 1000 2000 5000 10000; do
  $B $I/deep1B_member --dataset deep1B $D1B --nq $NQ --read 16 --starts 640 --waves 4 --resident \
    --json $R/batch_deep1B.jsonl
done

fi
if stage ablation; then
echo "== kernel ablation, Deep1B rows $(date)"
python3 - $R/ablation_system.jsonl <<'EOF'
import json, sys
p = sys.argv[1]
rows = [l for l in open(p) if json.loads(l)["dataset"] != "deep1B"]
open(p, "w").writelines(rows)
EOF
for F in "" --no-fused-merge --no-graph; do
  $B $I/deep1B_member --dataset deep1B $D1B --read 16 --starts 512 --waves 4 $F --resident \
    --json $R/ablation_system.jsonl
done

fi
if stage profile; then
echo "== GPU metrics and timelines $(date)"
# GPU performance counters need root (ERR_NVGPUCTRPERM)
sudo -n env TMPDIR=$TMPDIR /usr/local/cuda/bin/nsys profile -t cuda --gpu-metrics-devices=all \
  --gpu-metrics-set=gh100 --gpu-metrics-frequency=20000 -f true -o $P/d1b_metrics \
  $B $I/deep1B_member --dataset deep1B $D1B --read 16 --starts 512 --waves 4 --reps 30 --resident \
  > $P/d1b_metrics.log 2>&1
sudo -n chown -R "$(id -u):$(id -g)" $P
/usr/local/cuda/bin/nsys export --type sqlite -f true -o $P/d1b_metrics.sqlite $P/d1b_metrics.nsys-rep > /dev/null
python3 bench/profile_metrics.py $P/d1b_metrics.sqlite $R/monitor_deep1B.json
for v in graph nograph; do
  F=""; [ $v = nograph ] && F="--no-graph"
  /usr/local/cuda/bin/nsys profile -t cuda --cuda-graph-trace=node -f true -o $P/d1b_$v \
    $B $I/deep1B_member --dataset deep1B $D1B --read 16 --starts 512 --waves 4 --reps 3 $F --resident \
    > $P/d1b_$v.log 2>&1
  /usr/local/cuda/bin/nsys stats --report cuda_gpu_trace --format csv -o $P/d1b_$v $P/d1b_$v.nsys-rep > /dev/null
done
python3 bench/profile_timeline.py $P/d1b_graph_cuda_gpu_trace.csv $P/d1b_nograph_cuda_gpu_trace.csv \
  $R/profile_deep1B.json

fi
if stage listlen; then
echo "== list length 32 $(date)"
rm -rf $I/deep1B_l32_member && mkdir -p $I/deep1B_l32_member
$OB /home/cc/tmp/big/deep1B/base_fp16.npy $I/deep1B_l32_member --copies 256 --list-len 32 --min-cell 100 \
  --sketch-width 96 --table-len 0 --member-lists --link-vectors > $R/build_deep1B_l32.log 2>&1
cp $I/deep1B_member/table.npy $I/deep1B_l32_member/
for f in queries groundtruth; do ln -sf $I/deep1B/$f.npy $I/deep1B_l32_member/$f.npy; done
python3 - $I/deep1B_l32_member/meta.json <<'EOF'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
m["table_len"] = 32
json.dump(m, open(p, "w"), indent=1)
EOF
rm -f $R/listlen_deep1B.jsonl
$B $I/deep1B_l32_member --dataset deep1B $D1B --copies-limit 128 --read 16,32 --starts 512 --waves 4 --resident \
  --json $R/listlen_deep1B.jsonl
$B $I/deep1B_l32_member --dataset deep1B $D1B --copies-limit 128 --list-limit 16 --read 16 --starts 512 --waves 4 \
  --resident --json $R/listlen_deep1B.jsonl
fi
echo "== done $(date)"
