"""The kernels of the last timed batch on every GPU, from the cuda_gpu_trace CSV of `nsys stats`, as JSON for the
timeline figure: per GPU, (start ms from the first kernel of the batch on any GPU, duration ms, kernel name).

    python bench/profile_timeline.py graph.csv nograph.csv results/octex/profile_deep1B.json
"""

import csv
import json
import sys


def batch(path):
    rows = [r for r in csv.DictReader(open(path)) if "Host-to-Device" not in r["Name"]]
    per = {}
    for r in rows:
        per.setdefault(int(r["Device"].split("(")[-1].rstrip(")")), []).append(r)
    out, t0 = {}, None
    for dev, rs in sorted(per.items()):
        rs.sort(key=lambda r: int(r["Start (ns)"]))
        first = max(i for i, r in enumerate(rs) if "center_kernel" in r["Name"])  # the last batch starts here
        last = max(i for i, r in enumerate(rs) if "final_rerank_kernel" in r["Name"])
        out[dev] = rs[first : last + 1]
        s = int(rs[first]["Start (ns)"])
        t0 = s if t0 is None else min(t0, s)
    short = lambda n: n.split("(")[0].split("<")[0].split("::")[-1]  # noqa: E731
    return {
        str(d): [[(int(r["Start (ns)"]) - t0) / 1e6, int(r["Duration (ns)"]) / 1e6, short(r["Name"])] for r in rs]
        for d, rs in out.items()
    }


if __name__ == "__main__":
    graph, nograph, dst = sys.argv[1:4]
    json.dump({"graph": batch(graph), "nograph": batch(nograph)}, open(dst, "w"), indent=0)
    print("wrote", dst)
