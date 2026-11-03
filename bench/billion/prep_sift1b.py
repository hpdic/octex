"""BIGANN-1B for octex: fp16 base (.npy), f32 queries, int64 ground truth. Sequential buffered reads and writes in
chunks; no file mappings."""
import numpy as np, os, time
SRC = "/home/cc/hpdic/datasets/bigann"
DST = "/home/cc/tmp/big/bigann1B"
# base: u8bin = int32 n, int32 d, then n*d uint8
with open(f"{SRC}/base.1B.u8bin", "rb") as f:
    n, d = np.fromfile(f, dtype=np.int32, count=2)
    n, d = int(n), int(d)
    print("base", n, d, flush=True)
    out = f"{DST}/base_fp16.npy"
    with open(out + ".tmp", "wb") as g:
        np.lib.format.write_array_header_1_0(g, {"descr": "<f2", "fortran_order": False, "shape": (n, d)})
        chunk = 1 << 24
        t0 = time.time()
        for r0 in range(0, n, chunk):
            m = min(chunk, n - r0)
            a = np.fromfile(f, dtype=np.uint8, count=m * d)
            assert a.size == m * d
            g.write(a.astype(np.float16).tobytes())
            if (r0 // chunk) % 8 == 0:
                print(f"  {r0 + m}/{n} rows, {time.time() - t0:.0f} s", flush=True)
    os.replace(out + ".tmp", out)
# queries: bvecs = per row int32 dim + dim uint8
raw = np.fromfile(f"{SRC}/bigann_query.bvecs", dtype=np.uint8)
dq = int(raw[:4].view(np.int32)[0])
q = raw.reshape(-1, 4 + dq)[:, 4:].astype(np.float32)
np.save(f"{DST}/queries.npy", q)
print("queries", q.shape, flush=True)
# ground truth: ivecs = per row int32 k + k int32 ids
gt = np.fromfile(f"{SRC}/gnd/idx_1000M.ivecs", dtype=np.int32)
k = int(gt[0])
gt = gt.reshape(-1, 1 + k)[:, 1:].astype(np.int64)
np.save(f"{DST}/groundtruth.npy", gt)
print("groundtruth", gt.shape, flush=True)
