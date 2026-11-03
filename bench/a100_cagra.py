"""CAGRA of cuVS on one whole GPU, the baseline of the single-A100 figures: graphs of degree 32, 64, and 96
(intermediate degree twice the degree) over the base vectors of an octex dataset directory (vectors.npy, fp16, searched
as fp32 and as fp16), every itopk_size and search_width of the grid, one batch of the whole query set (at most 10,000)
resident on the GPU, 5 timed searches after a warm-up. QPS from the best and the median search time, recall at 10
against groundtruth.npy, index memory as vectors plus graph.

    python bench/a100_cagra.py sift1m [--data /home/cc/data/octex] [--degrees 32,64,96] [--fp16 0,1]

Writes results/baselines/a100_cuda/cagra_<name>.json ({"args": ..., "rows": [...]}) after every graph.
"""

import argparse
import json
import os
import time

import cupy as cp
import numpy as np
from cuvs.common import Resources
from cuvs.neighbors import cagra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 10


def ints(s):
    return [int(v) for v in s.split(",") if v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--data", default="/home/cc/data/octex")
    ap.add_argument("--degrees", default="32,64,96")
    ap.add_argument("--fp16", default="0,1")
    ap.add_argument("--itopk", default="32,64,128,256,512,768,1024")
    ap.add_argument("--width", default="1,2,4")
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "baselines", "a100_cuda"))
    args = ap.parse_args()

    d = os.path.join(args.data, args.dataset)
    base = np.load(os.path.join(d, "vectors.npy"))  # a whole-file read, not a map
    Qh = np.load(os.path.join(d, "queries.npy")).astype(np.float32)
    G = np.load(os.path.join(d, "groundtruth.npy"))[:, :K]
    n, dim = base.shape
    nq = len(Qh)
    assert nq <= args.batch, "one batch of the whole query set"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"cagra_{args.dataset}.json")
    out = {
        "args": {
            **vars(args),
            "n": n,
            "dim": dim,
            "nq": nq,
            "k": K,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuvs": __import__("cuvs").__version__,
        },
        "rows": [],
    }
    res = Resources()
    for fp16 in ints(args.fp16):
        dt = cp.float16 if fp16 else cp.float32
        X = cp.empty((n, dim), dtype=dt)
        for i in range(0, n, 1 << 20):  # the fp16 base in chunks, as fp32 or as is
            X[i : i + (1 << 20)] = cp.asarray(base[i : i + (1 << 20)]).astype(dt)
        Q = cp.asarray(Qh).astype(dt)
        I = cp.empty((nq, K), dtype=cp.uint32)
        D = cp.empty((nq, K), dtype=cp.float32)
        for deg in ints(args.degrees):
            t0 = time.time()
            idx = cagra.build(
                cagra.IndexParams(metric="sqeuclidean", graph_degree=deg, intermediate_graph_degree=2 * deg),
                X,
                resources=res,
            )
            res.sync()
            build_s = time.time() - t0
            gib = (n * dim * (2 if fp16 else 4) + n * deg * 4) / 2**30
            print(f"{args.dataset} degree {deg} fp16 {fp16}: built in {build_s:.1f} s, {gib:.2f} GiB", flush=True)
            for itopk in ints(args.itopk):
                for width in ints(args.width):
                    sp = cagra.SearchParams(itopk_size=itopk, search_width=width)

                    def run():
                        cagra.search(sp, idx, Q, K, neighbors=I, distances=D, resources=res)  # noqa: B023, F821 (called at once)
                        res.sync()

                    run()
                    times = []
                    for _ in range(args.reps):
                        t = time.perf_counter()
                        run()
                        times.append(time.perf_counter() - t)
                    ids = cp.asnumpy(I)
                    recall = float(np.mean([len(np.intersect1d(ids[q], G[q])) for q in range(nq)]) / K)
                    row = {
                        "degree": deg,
                        "fp16": fp16,
                        "itopk": itopk,
                        "width": width,
                        "recall": recall,
                        "qps": nq / min(times),
                        "qps_median": nq / float(np.median(times)),
                        "gib": gib,
                        "build_s": build_s,
                        "times": times,
                    }
                    out["rows"].append(row)
                    print(
                        json.dumps({k: row[k] for k in ("degree", "fp16", "itopk", "width", "recall", "qps_median")}),
                        flush=True,
                    )
            del idx
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
        del X, Q
        cp.get_default_memory_pool().free_all_blocks()
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
