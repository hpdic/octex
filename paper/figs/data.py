"""Measured recall-QPS points of OctEx and the baselines on 4x H100 (recall at 10, 10k queries unless noted),
collected from results/*.json and results/*.jsonl.

frontier(points) keeps the Pareto front: the best QPS among all measured configurations at each recall.
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "baselines")  # baseline measurements
OURS = os.path.join(ROOT, "results", "octex")  # octex_bench JSON lines
DIR = {"sift1m": "sift1m", "gist-960-euclidean": "gist", "imagenet-dinov2": "dinov2", "msmarco": "msmarco",
       "deep10M": "deep10M", "deep100M": "deep100M", "bigann100M": "bigann100M", "deep1B": "deep1B",
       "bigann1B": "bigann1B"}
DATASETS = ["sift1m", "gist-960-euclidean", "imagenet-dinov2", "msmarco", "deep10M", "deep100M", "bigann100M",
            "deep1B", "bigann1B"]
LABEL = {"sift1m": "SIFT1M", "gist-960-euclidean": "GIST1M", "imagenet-dinov2": "DINOv2-1.28M", "deep10M": "Deep10M",
         "deep100M": "Deep100M", "deep1B": "Deep1B", "msmarco": "MS MARCO", "bigann100M": "SIFT100M",
         "bigann1B": "SIFT1B"}
SHORT = {"sift1m": "SIFT", "gist-960-euclidean": "GIST", "imagenet-dinov2": "DINOv2", "deep10M": "D-10M",
         "deep100M": "D-100M", "deep1B": "D-1B", "msmarco": "MARCO", "bigann100M": "S-100M",
         "bigann1B": "S-1B"}


# one timed batch only, and its 0.94 ms (10.7M QPS at recall 0.936) is the outlier noted in HANDOFF finding 64
EXCLUDE = {"scale_ripple_deep100M_C176_v2.json"}
FUSED = os.environ.get("PAPER_FUSED", "1") == "1"  # the paper uses the fused merge kernels (PAPER_FUSED=0: before)


def _rows(pattern, gpus=4, single=False):
    """(recall, QPS) per configuration; QPS = the median over the timed repetitions where they were recorded."""
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        if os.path.basename(f) in EXCLUDE or ("_1gpu" in f) != single:  # single-GPU runs are not 4-GPU points
            continue
        if "_fused" in f and not FUSED:  # runs with the fused kernels enter the paper only once adopted
            continue
        if FUSED and "_fused" in f and "_fused2" not in f:  # first fused version (merge only): superseded
            continue
        for r in json.load(open(f))["rows"]:
            if (single or r.get("gpus", gpus) == gpus) and "recall" in r and "qps" in r:
                out.append((r["recall"], r.get("qps_median", r["qps"])))
    return out



def ours(ds, kind="frontier", gpus=4):
    """(recall, QPS) of every octex_bench configuration on ds (QPS: median over the timed repetitions)."""
    out = []
    path = os.path.join(OURS, f"{kind}_{DIR[ds]}.jsonl")
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if len(r.get("devices", [0] * gpus)) == gpus:
                out.append((r["recall"], r.get("qps_median", r["qps"])))
    return out


def cagra(ds):
    if ds in ("deep1B", "bigann1B"):  # 8 shards of 125M, 2 per GPU
        return _rows(f"cagra_big_{ds}_search.json")
    return _rows(f"scale_cagra_{ds}*.json")


def ivfpq(ds):
    return _rows(f"scale_ivfpq_{ds}*.json")


def rabitq(ds):
    if ds == "deep1B":
        return _rows("rabitq_deep1B_b4.json") + _rows("rabitq_deep1B_b7.json") + _rows("rabitq_deep1B_lowprobe.json")
    return _rows(f"rabitq_{ds}.json")


def ggnn(ds):
    """GGNN on 4 GPUs: sharded (one shard per GPU, measured) and full index per GPU (one GPU measured, credited
    with ideal 4x scaling of a query split, as CAGRA up to 100M vectors)."""
    path = os.path.join(RES, f"ggnn_{ds}.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    return ([(r["recall"], r["qps"]) for r in d["sharded_4gpu"]] + [(r["recall"], 4 * r["qps"]) for r in d["full_1gpu"]])


def flowann(ds, gpus=4):
    """FlowANN is a single-GPU system: measured on one GPU; on 4 GPUs credited with ideal 4x scaling."""
    path = os.path.join(RES, f"flowann_{ds}.json")
    if not os.path.exists(path):
        return []
    return [(r["recall"], r["qps"] * (gpus if gpus == 4 else 1)) for r in json.load(open(path))["rows"]]


def pathweaver(ds):
    """PathWeaver on 4 GPUs (its multi-GPU mode, one shard per GPU), end-to-end time."""
    path = os.path.join(RES, f"pathweaver_{ds}.json")
    return [(r["recall"], r["qps"]) for r in json.load(open(path))["rows"]] if os.path.exists(path) else []


def ours_1gpu(ds):
    return ours(ds, kind="onegpu", gpus=1)


def runs(name):
    """Every octex_bench JSON line of results/octex/<name>.jsonl as a dict."""
    path = os.path.join(OURS, f"{name}.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def batch_study(ds="deep100M", nqs=(100, 200, 500, 1000, 2000, 5000, 10000)):
    """Batch time (ms, median over the timed repetitions) against the batch size, the first nq queries as the batch,
    near recall 0.99. deep100M: OctEx 128 start points, 2 waves, read 64 and CAGRA degree 64 fp16 itopk 320 width 4.
    deep1B: OctEx 640 start points, 4 waves, sketch ranking and CAGRA on 8 shards, itopk 512 width 4."""
    import numpy as np
    starts = 128 if ds == "deep100M" else 640
    mine = {r["nq"]: r for r in runs(f"batch_{DIR[ds]}") if r["starts"] == starts}
    if ds == "deep1B":
        rows = {r["nq"]: r for r in json.load(open(os.path.join(RES, "cagra_big_deep1B_batch.json")))["rows"]}
    out_n, ours, cag = [], [], []
    for nq in nqs:
        if nq not in mine:
            continue
        out_n.append(nq)
        ours.append(1e3 * float(np.median(mine[nq]["times"])))
        if ds == "deep1B":
            c = rows[nq]
        else:
            c = [r for r in json.load(open(os.path.join(RES, f"scale_cagra_{ds}_batch{nq}.json")))["rows"]
                 if r["itopk"] == 320][0]
        cag.append(1e3 * nq / c["qps_median"])
    return out_n, ours, cag


def frontier(points):
    """Pareto front: sorted by recall, each kept point has higher QPS than every point of higher recall."""
    best, out = -1.0, []
    for rec, qps in sorted(points, key=lambda p: (-p[0], -p[1])):
        if qps > best:
            out.append((rec, qps))
            best = qps
    return sorted(out)


def qps_at(points, r):
    """The best QPS among configurations with recall >= r (None if none reaches r)."""
    q = [qps for rec, qps in points if rec >= r]
    return max(q) if q else None


if __name__ == "__main__":
    for ds in DATASETS:
        print(ds)
        for name, fn in (("ours", ours), ("cagra", cagra), ("ggnn", ggnn), ("flowann", flowann), ("pathweaver", pathweaver), ("ivfpq", ivfpq), ("rabitq", rabitq)):
            f = frontier(fn(ds))
            if f:
                print(f"  {name:7s}", " ".join(f"{r:.4f}@{q / 1e3:.0f}k" for r, q in f))
