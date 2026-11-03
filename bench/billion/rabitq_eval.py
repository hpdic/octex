"""IVF-RaBitQ baseline (cuVS 26.10 C++, ../baselines/rabitq_bench) on 4 GPUs, with an exact fp16 rerank on the GPU.

    python rabitq_eval.py deep100M [--n-lists 16384] [--bits 4,8] [--probes 32,64,128,256] [--ratios 2,4,8]
    python rabitq_eval.py deep1B --shards 8 ...

<= 100M: every GPU builds the full index and answers a quarter of the queries. 1B: the base in --shards row ranges,
GPU g builds and searches shards g, g + 4, ... with all queries (one after the other). RaBitQ returns k x ratio
candidates per query and shard; our select_rerank2 kernel reranks them exactly (fp16 rows: on the GPU at <= 100M,
in host memory through its registered pointer at 1B). Time = slowest GPU's search + slowest GPU's rerank; the
RaBitQ search time is the benchmark's own (search only, the second of two passes).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("BENCH_LOADERS", os.path.expanduser("~/datasets/src")))  # dataset loaders (shared.py)
BENCH = os.path.join(HERE, "run_rabitq.sh")
RESULTS = os.path.join(os.path.dirname(HERE), "results")
DEEP = os.path.expanduser("~/hpdic/datasets/deep1b")
K = 10
_REG = {}  # dataset -> (fp16 base in anonymous pinned host memory, host pointer)


def gt_and_data(name):
    """(base .fbin path, N, D, queries float32 [nq, D] (cpu), ground truth int64 [nq, 10] (cpu))."""
    import big
    if name == "bigann1B":  # fp32 .fbin written once from the fp16 base by bigann_fbin.py
        Q, G = big.queries("bigann1B", "cpu")
        return "/home/cc/tmp/rabitq_runs/bigann1B_base.fbin", 1_000_000_000, 128, Q, G
    base = os.path.join(DEEP, "base.1B.fbin")
    Q, G = big.queries("deep1B", "cpu")
    if name == "deep1B":
        return base, 1_000_000_000, 96, Q, G
    from shared import load  # deep<n>M: prefix of the base with its exact ground truth
    X, Qn, Gn = load(name)
    if name.startswith("deep"):
        return base, int(name[4:-1]) * 1_000_000, 96, Qn.cpu(), Gn.cpu()
    # other datasets: the loader's base (normalized where the loader normalizes) written once as .fbin
    path = os.path.join("/home/cc/tmp/rabitq_runs", f"{name}_base.fbin")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as fh:
            np.array(X.shape, dtype=np.int32).tofile(fh)
            for n0 in range(0, X.shape[0], 1 << 20):
                np.ascontiguousarray(X[n0:n0 + (1 << 20)], dtype=np.float32).tofile(fh)
        os.replace(path + ".tmp", path)
    return path, X.shape[0], X.shape[1], Qn.cpu(), Gn.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n-lists", type=int, default=16384)
    ap.add_argument("--bits", default="4,8")
    ap.add_argument("--probes", default="32,64,128,256")
    ap.add_argument("--ratios", default="2,4,8")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ng = args.gpus
    base, N, D, Q, G = gt_and_data(args.dataset)
    nq = Q.shape[0]
    out_dir = os.path.join("/home/cc/tmp/rabitq_runs", args.dataset + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    qfile = os.path.join(out_dir, "queries.fbin")
    with open(qfile, "wb") as fh:  # the dataset's own queries (deep<n>M uses its cached query set)
        np.array([nq, D], dtype=np.int32).tofile(fh)
        Q.numpy().astype(np.float32).tofile(fh)
    rows = []
    for bits in (int(v) for v in args.bits.split(",")):
        # one benchmark process per (GPU, shard): <= 100M: shard = all rows, queries split; 1B: queries all
        jobs = []
        for g in range(ng):
            if args.shards == 1:
                q0, q1 = nq * g // ng, nq * (g + 1) // ng
                jobs.append((g, [(0, N, q0, q1 - q0)]))
            else:
                jobs.append((g, [(N * s // args.shards, N * (s + 1) // args.shards - N * s // args.shards, 0, nq)
                                 for s in range(args.shards) if s % ng == g]))
        procs = []
        for g, work in jobs:
            cmd = " && ".join(f"{BENCH} {base} {b0} {n} {D} {qfile} {q0} {m} {K} {args.n_lists} {bits} "
                              f"{args.probes} {args.ratios} {out_dir}" for b0, n, q0, m in work)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), RABITQ_QBATCH=os.environ.get("RABITQ_QBATCH", "256"))  # bounds its search workspace
            procs.append(subprocess.Popen(["bash", "-c", cmd], env=env, stdout=subprocess.PIPE, text=True,
                                          start_new_session=True))  # own process group: killed as a whole
        search_s = {}  # (probes, ratio) -> per GPU summed search seconds
        for g, p in enumerate(procs):
            out, _ = p.communicate()
            if p.returncode != 0:  # stop the other GPUs' benchmarks first: no orphans left running
                import signal
                for q in procs:
                    if q.poll() is None:
                        os.killpg(q.pid, signal.SIGKILL)
                raise SystemExit(f"rabitq_bench failed on GPU {g}")
            for line in out.splitlines():
                ev = json.loads(line)
                if ev["event"] == "build":
                    print(f"[gpu {g}] build {ev['build_s']:.1f}s, index {ev['index_gpu_bytes'] / 2**30:.1f} GiB", flush=True)
                elif ev["event"] == "search":
                    key = (ev["n_probes"], ev["refine_ratio"])
                    search_s.setdefault(key, [0.0] * ng)[g] += ev["search_s"]
        rows += rerank(args, bits, jobs, search_s, out_dir, N, D, Q, G, ng)
        with open(os.path.join(RESULTS, f"rabitq_{args.dataset}{args.tag}.json"), "w") as fh:
            json.dump({"args": vars(args), "rows": rows}, fh, indent=1)


def rerank(args, bits, jobs, search_s, out_dir, N, D, Q, G, ng):
    """Exact rerank of every (probes, ratio) candidate set on the GPUs; rows with recall and QPS."""
    import big
    from proto_cuda import ext  # the exact-rerank kernels (proto_cuda.py)
    E = ext()
    nq = Q.shape[0]
    if args.dataset in ("deep1B", "bigann1B"):
        if args.dataset not in _REG:  # once per process: the fp16 base in anonymous pinned memory
            _REG[args.dataset] = big.register_half(args.dataset)
        Hreg, hptr = _REG[args.dataset]
    else:
        if args.dataset.startswith("deep"):
            Xh = torch.from_numpy(np.ascontiguousarray(big.base_map("deep1B")[:N])).to("cuda:0").half()
        else:  # the .fbin that gt_and_data wrote
            fb = np.memmap(os.path.join("/home/cc/tmp/rabitq_runs", f"{args.dataset}_base.fbin"), dtype=np.float32,
                           mode="r", offset=8, shape=(N, D))
            Xh = torch.empty((N, D), dtype=torch.half, device="cuda:0")
            for n0 in range(0, N, 1 << 20):
                Xh[n0:n0 + (1 << 20)] = torch.from_numpy(np.ascontiguousarray(fb[n0:n0 + (1 << 20)])).to("cuda:0").half()
    out = []
    for (pr, ra), ss in sorted(search_s.items()):
        kp = K * ra
        parts, times = [], []
        for g in range(ng):
            dev = f"cuda:{g}"
            torch.cuda.set_device(g)
            q0, q1 = nq * g // ng, nq * (g + 1) // ng  # this GPU reranks a quarter of the queries
            cands = []
            for gg, work in jobs:
                for b0, n, wq0, m in work:
                    a = np.fromfile(os.path.join(out_dir, f"ids_b{b0}_q{wq0}_p{pr}_r{ra}.i64"), dtype=np.int64)
                    a = a.reshape(m, kp)
                    lo, hi = max(q0, wq0), min(q1, wq0 + m)
                    if lo < hi:
                        cands.append((lo, a[lo - wq0:hi - wq0]))
            per_q = {}
            for lo, a in cands:
                for i in range(a.shape[0]):
                    per_q.setdefault(lo + i, []).append(a[i])
            C = np.stack([np.concatenate(per_q[q]) for q in range(q0, q1)])
            m, W = C.shape
            assert W <= 512, f"{W} candidates per query: beyond the rerank kernel's 512"
            cand = torch.from_numpy(C.astype(np.int32)).to(dev)
            dist = torch.zeros((m, W), device=dev)
            n = torch.full((m,), W, dtype=torch.int32, device=dev)
            rowof = torch.arange(m, dtype=torch.int32, device=dev)
            Qd = Q[q0:q1].to(dev).float().contiguous()
            bi = torch.full((m, K), -1, dtype=torch.long, device=dev)
            bd = torch.full((m, K), float("inf"), device=dev)
            if args.dataset in ("deep1B", "bigann1B"):
                X0 = torch.zeros((1, D), dtype=torch.half, device=dev)
                xp = torch.tensor([hptr], dtype=torch.int64, device=dev)
                run = lambda: E.select_rerank2s(cand, dist, n, rowof, m, X0, xp, N, Qd, W, bi, bd)  # noqa: E731
            else:
                Xg = Xh if g == 0 else Xh.to(dev)
                run = lambda: E.select_rerank2(cand, dist, n, rowof, m, Xg, Qd, W, bi, bd)  # noqa: E731
            for rep in range(3):
                bi.fill_(-1)
                bd.fill_(float("inf"))
                torch.cuda.synchronize(g)
                s0 = torch.cuda.Event(enable_timing=True)
                s1 = torch.cuda.Event(enable_timing=True)
                s0.record()
                run()
                s1.record()
                torch.cuda.synchronize(g)
            times.append(s0.elapsed_time(s1) / 1e3)
            parts.append(bi.cpu())
        I = torch.cat(parts)
        rec = float((I[:, :, None] == G[:, None, :]).any(-1).float().mean())
        total = max(ss) + max(times)
        row = {"bits": bits, "probes": pr, "ratio": ra, "recall": rec, "search_s": max(ss), "rerank_s": max(times),
               "qps": nq / total}
        out.append(row)
        print(args.dataset, f"rabitq bits={bits} probes={pr} ratio={ra} recall={rec:.4f} search={max(ss) * 1e3:.2f}ms",
              f"rerank={max(times) * 1e3:.2f}ms qps={nq / total:.0f}", flush=True)
    return out


if __name__ == "__main__":
    main()
