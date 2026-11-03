#!/bin/bash
# Builds every index that the paper evaluates with the octex builder on all four GPUs, from the fp16 vectors of
# the dataset, with the configuration of Section 5.1. Each index goes to $I/new_<name>, receives the queries and
# ground truth of the index it replaces, and then replaces $I/<name>. Phase times and peak memory in
# results/octex/build_<name>.log. Stops at the first failure and leaves the old index in place. Deep1B with lists of
# 32 entries: paper_build_l32.sh.
cd "$(dirname "$0")/.."
B=${B:-build/octex_build}
I=/home/cc/data/octex
R=results/octex
G="--devices 0,1,2,3"
mkdir -p $R

swap() {  # name: replace $I/<name> by $I/new_<name>
  local name=$1
  cp $I/$name/queries.npy $I/$name/groundtruth.npy $I/new_$name/ || return 1
  rm -rf $I/old_$name && mv $I/$name $I/old_$name && mv $I/new_$name $I/$name && rm -rf $I/old_$name
}
run() {  # name, input, flags...
  local name=$1 src=$2
  shift 2
  echo "== $name $(date)"
  rm -rf $I/new_$name
  OCTEX_VERBOSE=1 $B $src $I/new_$name $G "$@" > $R/build_$name.log 2>&1 || { echo "== $name FAILED"; exit 1; }
  swap $name || { echo "== $name swap FAILED"; exit 1; }
  grep -a "^copies:\|^table:\|^build:\|peak host" $R/build_$name.log
}

run sift1m $I/sift1m/vectors.npy --copies 200 --list-len 32 --exact-lists --sketch-width 64 --table-len 64 --no-reverse
run gist $I/gist/vectors.npy --copies 400 --list-len 64 --exact-lists --sketch-width 128 --table-len 64 --no-reverse
run dinov2 $I/dinov2/vectors.npy --copies 600 --list-len 32 --min-cell 2 --sketch-width 128 --table-len 64
run dinov2_exact $I/dinov2/vectors.npy --copies 600 --list-len 32 --min-cell 2 --sketch-width 128 --table-len 64 \
  --no-reverse
run msmarco $I/msmarco/vectors.npy --copies 256 --list-len 64 --min-cell 2 --sketch-width 128 --table-len 64
run deep10M $I/deep10M/vectors.npy --copies 320 --list-len 64 --min-cell 2 --sketch-width 64 --table-len 64
run deep100M $I/deep100M/vectors.npy --copies 272 --list-len 64 --min-cell 10 --sketch-width 64 --table-len 64

# the neighbor rows of Deep100M without reverse edges, over the arrays of the new Deep100M index
echo "== deep100M_fwd $(date)"
rm -rf $I/new_deep100M_fwd && mkdir -p $I/new_deep100M_fwd
for f in $I/deep100M/*.npy; do
  case $(basename $f) in table.npy|queries.npy|groundtruth.npy) ;; *) ln -s $f $I/new_deep100M_fwd/ ;; esac
done
OCTEX_TABLE_ONLY=1 OCTEX_VERBOSE=1 $B $I/deep100M/vectors.npy $I/new_deep100M_fwd $G --copies 272 --list-len 64 \
  --min-cell 10 --sketch-width 64 --table-len 64 --no-reverse > $R/build_deep100M_fwd.log 2>&1 \
  || { echo "== deep100M_fwd FAILED"; exit 1; }
swap deep100M_fwd || exit 1

run bigann100M $I/bigann100M/vectors.npy --copies 224 --list-len 64 --min-cell 10 --sketch-width 64 --table-len 64
# the two billion-scale indexes from their fp16 vectors (bench/billion/prep_sift1b.py, bench/billion/big.py prepare
# for Deep1B), with the configuration of Section 5.1
run deep1B /home/cc/tmp/big/deep1B/base_fp16.npy --copies 256 --list-len 16 --min-cell 100 --sketch-width 96 --table-len 32
run bigann1B /home/cc/tmp/big/bigann1B/base_fp16.npy --copies 256 --list-len 16 --min-cell 100 --sketch-width 96 \
  --table-len 32
echo "== all done $(date)"
