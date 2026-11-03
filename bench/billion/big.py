"""Billion-scale data preparation (Deep1B): one streaming pass over the fp32 base on disk writes the fp16 vectors
(exact rerank, kept in host memory at search time) and the int8 sketch (the hot path, split over the GPUs) to
/home/cc/tmp/big/<dataset>/, with the PCA and the sketch scales from a 10M-row sample.

    python big.py prepare deep1B [--sketch 64] [--sample 10000000]

Nothing at 1B fits one GPU: fp32 base 384 GB, fp16 192 GB, sketch 64 GB. Every pass streams row chunks through a
GPU; the outputs are host memmaps.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

DEEP1B = os.path.expanduser("~/hpdic/datasets/deep1b")
OUT = "/home/cc/tmp/big"
K = 10


def base_map(name):
    """The fp32 base on disk as a read-only memmap [N, D]."""
    assert name.startswith("deep1B"), name
    path = os.path.join(DEEP1B, "base.1B.fbin")
    n, d = (int(x) for x in np.fromfile(path, dtype=np.int32, count=2))
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))


def queries(name, dev="cuda"):
    """(queries float32 [nq, D] on dev, ground truth int64 [nq, 10] on dev): the public 10K set of Deep1B, or the 10K
    queries and 1B ground truth of BIGANN-1B as prepared next to its fp16 base."""
    if name == "bigann1B":
        d = paths(name)["dir"]
        Q = np.load(os.path.join(d, "queries.npy"))
        G = np.load(os.path.join(d, "groundtruth.npy"))[:, :K]
        return torch.from_numpy(Q).to(dev), torch.from_numpy(G.astype(np.int64)).to(dev)
    assert name == "deep1B", name
    q = os.path.join(DEEP1B, "query.public.10K.fbin")
    qn, qd = (int(x) for x in np.fromfile(q, dtype=np.int32, count=2))
    Q = np.fromfile(q, dtype=np.float32, offset=8).reshape(qn, qd)
    g = os.path.join(DEEP1B, "groundtruth.public.10K.ibin")
    gn, gk = (int(x) for x in np.fromfile(g, dtype=np.int32, count=2))
    G = np.fromfile(g, dtype=np.int32, offset=8, count=gn * gk).reshape(gn, gk)[:, :K]
    return torch.from_numpy(Q).to(dev), torch.from_numpy(G.astype(np.int64)).to(dev)


def paths(name):
    d = os.path.join(OUT, name)
    return {"dir": d, "half": os.path.join(d, "base_fp16.npy"), "sketch": os.path.join(d, "sketch_int8.npy"),
            "meta": os.path.join(d, "meta.pt")}


def prepare(name, S=64, sample=10_000_000, chunk=1 << 22, dev="cuda", limit=0):
    """One pass: fp16 copy and int8 sketch of the base, PCA and sketch scales from a sample."""
    p = paths(name)
    os.makedirs(p["dir"], exist_ok=True)
    X = base_map(name)
    N, D = X.shape
    if limit > 0:
        N = min(N, limit)
        sample = min(sample, N)
    t0 = time.time()
    nb = 100  # the sample as 100 evenly spaced contiguous blocks (strided rows would be ~1e7 random page reads)
    blk = sample // nb
    smp = torch.cat([torch.from_numpy(np.ascontiguousarray(X[s:s + blk])) for s in
                     np.linspace(0, N - blk, nb).astype(np.int64)]).to(dev)
    mu = smp.mean(0)
    V0 = torch.linalg.eigh((smp - mu).double().T @ (smp - mu).double())[1].flip(-1).float()
    Vp = V0[:, :S].contiguous()
    amax = ((smp - mu) @ Vp).abs().amax(0)  # sample range; rows beyond it saturate at +-127
    sc = (amax / 127).contiguous()
    del smp
    print(f"{name}: N={N} D={D}, PCA and scales from {sample} rows, {time.time() - t0:.0f}s", flush=True)
    with open(p["half"] + ".tmp", "wb") as f_h, open(p["sketch"] + ".tmp", "wb") as f_p:
        np.lib.format.write_array_header_1_0(
            f_h, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                  "fortran_order": False, "shape": (N, D)}
        )
        np.lib.format.write_array_header_1_0(
            f_p, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.int8)),
                  "fortran_order": False, "shape": (N, S)}
        )
        for n0 in range(0, N, chunk):
            x = torch.from_numpy(np.ascontiguousarray(X[n0:n0 + chunk])).to(dev)
            f_h.write(x.half().cpu().numpy().tobytes())
            f_p.write(torch.clamp(torch.round(((x - mu) @ Vp) / sc), -127, 127).to(torch.int8).cpu().numpy().tobytes())
            if (n0 // chunk) % 25 == 0:
                print(f"  {n0 + x.shape[0]} / {N} rows, {time.time() - t0:.0f}s", flush=True)
    os.replace(p["half"] + ".tmp", p["half"])
    os.replace(p["sketch"] + ".tmp", p["sketch"])
    torch.save({"mu": mu.cpu(), "V0": V0.cpu(), "Vp": Vp.cpu(), "sc": sc.cpu(), "N": N, "D": D, "S": S}, p["meta"])
    with open(os.path.join(p["dir"], "prepare.json"), "w") as fh:
        json.dump({"N": N, "D": D, "S": S, "sample": sample, "seconds": time.time() - t0}, fh)
    print(f"{name}: fp16 and sketch written, {time.time() - t0:.0f}s", flush=True)


def half_map(name):
    """The fp16 base written by prepare() as a read-only memmap [N, D], from its copy in host shared memory (the
    disk reads ~300 MB/s: one pass over 192 GB would take ~10 minutes; from /dev/shm a pass is GPU-transfer bound)."""
    import shutil
    shm = f"/dev/shm/octex_{name}_fp16.npy"
    if not os.path.exists(shm):
        shutil.copyfile(paths(name)["half"], shm + ".tmp")
        os.replace(shm + ".tmp", shm)
    return np.load(shm, mmap_mode="r")


def register_half(name):
    """The fp16 base loaded into anonymous pinned host memory (torch pinned tensor):
    kernels read it through its host pointer (UVA over PCIe), no memory-map pinning."""
    src = paths(name)["half"]
    with open(src, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported npy version: {version}")
        N, D = shape
        pinned_t = torch.empty((N, D), dtype=torch.float16, pin_memory=True)
        chunk_rows = 1 << 22
        mv = memoryview(pinned_t.numpy())
        for n0 in range(0, N, chunk_rows):
            n_rows = min(chunk_rows, N - n0)
            f.readinto(mv[n0:n0 + n_rows])
    m = pinned_t.numpy()
    return m, pinned_t.data_ptr()


def sketch_map(name, S=0):
    """The int8 sketch [N, S] as a read-only memmap from its copy in host shared memory (copied there once); S = 0:
    the 64-wide sketch of prepare(), else the full-dimension one of prepare_full(S)."""
    import shutil
    src = paths(name)["sketch"] if not S else os.path.join(paths(name)["dir"], f"sketch{S}_int8.npy")
    shm = f"/dev/shm/octex_{name}_sketch{S or ''}.npy"
    if not os.path.exists(shm):
        shutil.copyfile(src, shm + ".tmp")
        os.replace(shm + ".tmp", shm)
    return np.load(shm, mmap_mode="r")


def projections(meta, C, top=64, blocks=4, copy_set=0, dev="cuda"):
    """Per copy: U [C, D, 8b] in projected-sd units (scales from the sample variance of the PCA directions) and the
    random shift [C, 8b]; the same recipe as ripple_cuda.setup, from the prepared PCA."""
    V = meta["V0"][:, :top].to(dev)
    g = torch.Generator().manual_seed(copy_set)
    S8 = 8 * blocks
    Rot = torch.linalg.qr(torch.randn(C, top, S8, generator=g))[0].to(dev)
    U0 = torch.einsum("dp,cps->cds", V, Rot)
    # PCA eigen-directions: projected sd of a copy = sqrt(mean eigenvalue over its rotated span); estimated below
    return U0, (torch.rand(C, S8, generator=g) * 2).to(dev)


def cells(name, copies=1, probes="2.0,2.5,3.2,4.0,5.0", pops="1,2,10,30,100,300", chunk=1 << 22, dev="cuda"):
    """Cells per level and copy at full scale: how many cells hold >= t vectors, and which share of the data (= the
    query landing probability) they cover; one streaming pass per level over the fp16 base."""
    sys_path()
    from nest import e8b_point, phash
    meta = torch.load(paths(name)["meta"])
    H = half_map(name)
    N = H.shape[0]
    mu = meta["mu"].to(dev)
    U0, shift = projections(meta, copies)
    smp = torch.from_numpy(np.ascontiguousarray(H[: 1 << 20])).to(dev).float() - mu
    sd = torch.einsum("nd,cds->cns", smp, U0).std(1).mean(1)
    U = U0 / sd[:, None, None]
    ts = [int(v) for v in pops.split(",")]
    t0 = time.time()
    for c in range(copies):
        for f in (float(v) for v in probes.split(",")):
            h = torch.empty(N, dtype=torch.long, device=dev)
            for n0 in range(0, N, chunk):
                x = torch.from_numpy(np.ascontiguousarray(H[n0:n0 + chunk])).to(dev).float()
                h[n0:n0 + x.shape[0]] = phash(e8b_point(((x - mu) @ U[c] + shift[c]) / f))
            _, pop = torch.unique(h, return_counts=True)
            del h
            row = [f"copy {c} f={f}: cells {pop.numel()}"]
            for t in ts:
                k = pop >= t
                row.append(f">={t}: {int(k.sum())} cells, {float(pop[k].sum()) / N:.3f} of data")
            print("; ".join(row), f"({time.time() - t0:.0f}s)", flush=True)
            del pop
            torch.cuda.empty_cache()


def sys_path():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.environ.get("BENCH_LOADERS", os.path.expanduser("~/datasets/src"))):
        if p not in sys.path:
            sys.path.insert(0, p)


def prepare_full(name, S=128, sample_rows=10_000_000, chunk=1 << 22, dev="cuda"):
    """A sketch of ALL the PCA dimensions (int8, zero padded to width S, a multiple of 16 times a power of two for the
    gather kernels): distances on it are exact up to int8 rounding, so seeds can be chosen and merged without
    reading the fp16 rows (in host memory at 1B). Written from the fp16 copy in /dev/shm; scales from a sample."""
    p = paths(name)
    meta = torch.load(p["meta"])
    H = half_map(name)
    N, D = H.shape
    mu = meta["mu"].to(dev)
    V = meta["V0"].to(dev)  # [D, D], all directions
    Vp = torch.zeros((D, S), device=dev)
    Vp[:, :D] = V
    nb = 100
    blk = sample_rows // nb
    smp = torch.cat([torch.from_numpy(np.ascontiguousarray(H[s0:s0 + blk])) for s0 in
                     np.linspace(0, N - blk, nb).astype(np.int64)]).to(dev).float()
    amax = ((smp - mu) @ Vp).abs().amax(0)
    sc = torch.where(amax > 0, amax / 127, torch.ones_like(amax)).contiguous()
    del smp
    out = os.path.join(p["dir"], f"sketch{S}_int8.npy")
    t0 = time.time()
    with open(out + ".tmp", "wb") as f_p:
        np.lib.format.write_array_header_1_0(
            f_p, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.int8)),
                  "fortran_order": False, "shape": (N, S)}
        )
        for n0 in range(0, N, chunk):
            x = torch.from_numpy(np.ascontiguousarray(H[n0:n0 + chunk])).to(dev).float()
            f_p.write(torch.clamp(torch.round(((x - mu) @ Vp) / sc), -127, 127).to(torch.int8).cpu().numpy().tobytes())
            if (n0 // chunk) % 50 == 0:
                print(f"  {n0 + x.shape[0]} / {N} rows, {time.time() - t0:.0f}s", flush=True)
    os.replace(out + ".tmp", out)
    torch.save({"Vp": Vp.cpu(), "sc": sc.cpu(), "S": S}, os.path.join(p["dir"], f"meta{S}.pt"))
    print(f"{name}: full sketch S={S} written, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "cells", "prepare_full"])
    ap.add_argument("dataset")
    ap.add_argument("--sketch", type=int, default=64)
    ap.add_argument("--sample", type=int, default=10_000_000)
    ap.add_argument("--limit", type=int, default=0, help="limit N rows (reduced-scale test)")
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--pops", default="1,2,10,30,100,300")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.dataset, args.sketch, args.sample, limit=args.limit)
    elif args.cmd == "prepare_full":
        prepare_full(args.dataset, args.sketch)
    elif args.cmd == "cells":
        cells(args.dataset, args.copies, pops=args.pops)


if __name__ == "__main__":
    main()
