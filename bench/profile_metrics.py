"""GPU metrics of one average batch for the monitor figure: from the sqlite export of an `nsys profile
--gpu-metrics-devices=all --gpu-metrics-set=gh100` run of octex_bench, the SM activity, HBM read bandwidth, NVLink
received user data, and PCIe received throughput of every GPU (percent of peak), aligned at the start of each timed
batch and averaged over the batches and the GPUs on a 50-us grid; and the raw samples of one batch on GPU 0.

    python bench/profile_metrics.py run.sqlite results/octex/monitor_deep1B.json
"""

import json
import sqlite3
import sys

import numpy as np

METRICS = {"sm_active": (3,), "hbm_read": (18,), "nvlink_rx": (21, 23), "pcie_rx": (28,)}


def main(src, dst, step_ms=0.05):
    c = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    graphs = np.array(
        c.execute("select deviceId, start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall(),
        dtype=float,
    )
    ng = int(graphs[:, 0].max()) + 1
    starts = graphs[graphs[:, 0] == 0][0::2, 1]  # each batch is two graphs per GPU: first round, then the waves
    length = np.median(starts[1:] - starts[:-1]) / 1e6
    base = c.execute("select min(typeId) from TARGET_INFO_GPU_METRICS").fetchone()[0]
    t0, t1 = int(graphs[0, 1]) - 5_000_000, int(graphs[-1, 2]) + 5_000_000
    ids = tuple(sorted({m for ms in METRICS.values() for m in ms}))
    raw = {}
    for ts, ty, m, v in c.execute(
        f"select timestamp, typeId, metricId, value from GPU_METRICS where timestamp between "
        f"{t0} and {t1} and metricId in {ids} and typeId between {base} and {base + ng - 1}"
    ):
        raw.setdefault((ty - base, m), []).append((ts, v))
    grid = np.arange(0, length, step_ms)
    out = {
        "batch_ms": round(float(length), 3),
        "batches": len(starts) - 2,
        "gpus": ng,
        "step_ms": step_ms,
        "t_ms": [round(float(t), 3) for t in grid],
    }
    for name, ms in METRICS.items():
        rows = []
        for g in range(ng):
            series = [np.array(sorted(raw[(g, m)]), dtype=float) for m in ms]
            t = series[0][:, 0]
            y = sum(np.interp(t, s[:, 0], s[:, 1]) for s in series)
            for s0 in starts[1:-1]:  # the first and the last batch border the warm-up and the end of the run
                rows.append(np.interp(grid, (t - s0) / 1e6, y))
        out[name] = [round(float(v), 2) for v in np.mean(rows, axis=0)]
        # one batch as sampled: GPU 0, the middle timed batch, raw samples from its start, no averaging
        s0 = starts[len(starts) // 2]
        series = [np.array(sorted(raw[(0, m)]), dtype=float) for m in ms]
        t = series[0][:, 0]
        y = sum(np.interp(t, s[:, 0], s[:, 1]) for s in series)
        keep = (t >= s0) & (t < s0 + length * 1e6)
        out["raw_" + name] = [
            [round(float((a - s0) / 1e6), 3), round(float(v), 1)] for a, v in zip(t[keep], y[keep], strict=False)
        ]
    json.dump(out, open(dst, "w"), indent=0)
    print("wrote", dst, out["batches"], "batches of", out["batch_ms"], "ms")


if __name__ == "__main__":
    main(*sys.argv[1:3])
