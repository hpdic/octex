#!/bin/bash
# Deep1B with lists of 32 entries next to the evaluated lists of 16 (Section 4.5), with the neighbor rows of the
# evaluated index, and the list-length runs on it. Run after paper_build.sh.
cd "$(dirname "$0")/.."
B=${B:-build/octex_build}
I=/home/cc/data/octex
R=results/octex
G="--devices 0,1,2,3"
swap() {  # name: replace $I/<name> by $I/new_<name>
  local name=$1
  cp $I/$name/queries.npy $I/$name/groundtruth.npy $I/new_$name/ || return 1
  rm -rf $I/old_$name && mv $I/$name $I/old_$name && mv $I/new_$name $I/$name && rm -rf $I/old_$name
}

# Deep1B: lists of 32 entries next to the evaluated lists of 16, with the neighbor rows of the evaluated index
echo "== deep1B_l32 $(date)"
rm -rf $I/new_deep1B_l32
OCTEX_VERBOSE=1 $B /home/cc/tmp/big/deep1B/base_fp16.npy $I/new_deep1B_l32 $G --copies 256 --list-len 32 \
  --min-cell 100 --sketch-width 96 --table-len 0 > $R/build_deep1B_l32.log 2>&1 || { echo "== deep1B_l32 FAILED"; exit 1; }
cp $I/deep1B/table.npy $I/new_deep1B_l32/ && python3 - $I/new_deep1B_l32/meta.json <<'EOF' || exit 1
import json, sys
m = json.load(open(sys.argv[1]))
m["table_len"] = 32
json.dump(m, open(sys.argv[1], "w"), indent=1)
EOF
swap deep1B_l32 || exit 1
# list length at one billion vectors: 128 copies with all 32 entries per list and with the first 16
BB=${BB:-build/octex_bench}
D1B="--placement shard --devices 0,1,2,3 --vectors-on-host --sketch shard --sketch-rank --final-rerank 32"
rm -f $R/listlen_deep1B.jsonl
$BB $I/deep1B_l32 --dataset deep1B $D1B --copies-limit 128 --read 16,32 --starts 512 --waves 4 --resident \
  --json $R/listlen_deep1B.jsonl
$BB $I/deep1B_l32 --dataset deep1B $D1B --copies-limit 128 --list-limit 16 --read 16 --starts 512 --waves 4 --resident \
  --json $R/listlen_deep1B.jsonl
echo "== all done $(date)"
