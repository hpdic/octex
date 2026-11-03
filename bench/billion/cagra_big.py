"""CAGRA baseline at billion scale on 4 GPUs: the fp16 base in --shards row ranges, 2 shards per GPU (graph degree 32,
dataset fp16 on the device), every query searched on every shard, the per-shard top 10 merged by distance.

    python cagra_big.py build deep1B [--shards 8] [--degree 32]
    python cagra_big.py search deep1B [--itopk 64,128,256] [--width 1,4] [--reps 5]

Shards are saved with their dataset to /home/cc/tmp/cagra_big/<dataset>_s<i>of<n>_d<degree>.bin. Timing: one
process per GPU, a barrier before and after each batch; the batch time is the slowest GPU (all of its shards).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/cc/tmp/cagra_big"
RESULTS = os.path.join(os.path.dirname(HERE), "results")
K = 10


def shard_path(args, s):
    return os.path.join(OUT, f"{args.dataset}_s{s}of{args.shards}_d{args.degree}.bin")


def worker(rank, ng, args, bar, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    from cuvs.neighbors import cagra
    import big
    H = big.half_map(args.dataset)
    N = H.shape[0]
    mine = [s for s in range(args.shards) if s % ng == rank]
    if args.cmd == "build":
        os.makedirs(OUT, exist_ok=True)
        for s in mine:
            if os.path.exists(shard_path(args, s)):
                continue
            a, b = N * s // args.shards, N * (s + 1) // args.shards
            t0 = time.time()
            X = torch.from_numpy(np.ascontiguousarray(H[a:b])).cuda()
            idx = cagra.build(cagra.IndexParams(metric="sqeuclidean", graph_degree=args.degree,
                                                intermediate_graph_degree=2 * args.degree), X)
            cagra.save(shard_path(args, s) + ".tmp", idx, include_dataset=True)
            os.replace(shard_path(args, s) + ".tmp", shard_path(args, s))
            del idx, X
            torch.cuda.empty_cache()
            print(f"[gpu {rank}] shard {s} ({b - a} rows) built and saved, {time.time() - t0:.0f}s", flush=True)
        return
    Q, _ = big.queries(args.dataset)
    Qh = Q.half()
    idxs, keep = [], []
    for s in mine:
        ix = cagra.Index()
        ds = cagra.Dataset()  # without out_dataset, load keeps only the graph (index dimension 0)
        cagra.load(ix, shard_path(args, s), out_dataset=ds)
        keep.append(ds)
        idxs.append((N * s // args.shards, ix))
    nqs = [int(v) for v in args.nqs.split(",")] if args.nqs else [Qh.shape[0]]
    for nq in nqs:  # batch-size study: the first nq queries as the batch (shards loaded once)
        Qb = Qh[:nq].contiguous()
        for it in (int(v) for v in args.itopk.split(",")):
            for w in (int(v) for v in args.width.split(",")):
                sp = cagra.SearchParams(itopk_size=it, search_width=w)

                def run():
                    out = []
                    for a, ix in idxs:
                        d, i = cagra.search(sp, ix, Qb, K)
                        out.append((torch.as_tensor(d, device="cuda"), torch.as_tensor(i, device="cuda").long() + a))
                    torch.cuda.synchronize()
                    return out
                run()
                times = []
                for _ in range(args.reps):
                    bar.wait()
                    t = time.time()
                    out = run()
                    bar.wait()
                    times.append(time.time() - t)
                D = torch.cat([d for d, _ in out], 1).cpu()
                I = torch.cat([i for _, i in out], 1).cpu()
                q.put((rank, nq, it, w, times, D.numpy(), I.numpy()))  # numpy: torch tensors go by fd, lost once the worker exits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "search"])
    ap.add_argument("dataset")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--itopk", default="64,128,256")
    ap.add_argument("--width", default="1,4")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nqs", default="", help="batch sizes: the first nq queries as the batch, e.g. 10000,1000,100")
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    bar = ctx.Barrier(args.gpus)
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, args.gpus, args, bar, q)) for r in range(args.gpus)]
    for p in ps:
        p.start()
    if args.cmd == "search":
        sys.path.insert(0, HERE)
        import torch
        import big
        _, G = big.queries(args.dataset, "cpu")
        n_cfg = len(args.itopk.split(",")) * len(args.width.split(",")) * (len(args.nqs.split(",")) if args.nqs else 1)
        got = {}
        rows = []
        for _ in range(n_cfg * args.gpus):
            rank, nq, it, w, times, D, I = q.get()
            got.setdefault((nq, it, w), []).append((times, D, I))
            if len(got[(nq, it, w)]) == args.gpus:
                parts = got.pop((nq, it, w))
                D = torch.cat([torch.from_numpy(p[1]) for p in parts], 1)
                I = torch.cat([torch.from_numpy(p[2]) for p in parts], 1)
                top = torch.gather(I, 1, torch.topk(D, K, 1, largest=False).indices)
                rec = float((top[:, :, None] == G[:nq, None, :]).any(-1).float().mean())
                per = [max(p[0][r] for p in parts) for r in range(args.reps)]
                best, med = min(per), sorted(per)[len(per) // 2]
                rows.append({"nq": nq, "itopk": it, "width": w, "recall": rec, "qps": nq / best, "qps_median": nq / med})
                print(args.dataset, f"cagra-{args.shards}shards nq={nq} itopk={it} width={w} recall={rec:.4f}",
                      f"time={best * 1e3:.2f}ms qps={nq / best:.0f} (median {nq / med:.0f})", flush=True)
        with open(os.path.join(RESULTS, f"cagra_big_{args.dataset}{args.tag}.json"), "w") as fh:
            json.dump({"args": vars(args), "rows": rows}, fh, indent=1)
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
