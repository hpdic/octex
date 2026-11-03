"""Numbers that the paper's figures take from run logs, as JSON in results/: the logs themselves stay out of the
repository. Run after paper_build.sh and after new baseline runs.

    python bench/summarize_logs.py
"""

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "baselines")
OURS = os.path.join(ROOT, "results", "octex")


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", os.path.relpath(path, ROOT))


def octex_builds():
    """Phase times of the octex builder per index: lists (copies) and neighbor rows (table), in seconds."""
    out = {}
    for f in sorted(glob.glob(os.path.join(OURS, "build_*.log"))):
        name = re.match(r"build_(.*)\.log", os.path.basename(f)).group(1)
        base = name[: -len("_table")] if name.endswith("_table") else name
        txt = open(f, errors="replace").read()
        for key, pat in (("copies_s", r"^copies: ([0-9.]+) s"), ("table_s", r"^table: ([0-9.]+) s")):
            m = re.search(pat, txt, re.M)
            if m:
                out.setdefault(base, {})[key] = float(m.group(1))
    dump(os.path.join(OURS, "build_times.json"), out)


def cagra_deep1b():
    """CAGRA on Deep1B over 8 shards: the search sweeps (recall, median QPS) and the build time per GPU."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "logs", "cagra_big_search_deep1B*.log"))):
        for line in open(f, errors="replace"):
            m = re.search(r"cagra-8shards .*recall=([0-9.]+) .*\(median ([0-9]+)\)", line)
            if m:
                cfg = dict((k, int(v)) for k, v in re.findall(r"(itopk|width)=(\d+)", line))
                rows.append(
                    {
                        "itopk": cfg.get("itopk"),
                        "search_width": cfg.get("width"),
                        "recall": float(m.group(1)),
                        "qps": float(m.group(2)),
                        "qps_median": float(m.group(2)),
                        "gpus": 4,
                        "log": os.path.basename(f),
                    }
                )
    dump(os.path.join(RES, "cagra_big_deep1B_search.json"), {"rows": rows})
    per_gpu = {}
    path = os.path.join(RES, "logs", "cagra_big_build_deep1B.log")
    for g, sec in re.findall(r"\[gpu (\d)\] shard \d+ .* built and saved, (\d+)s", open(path).read()):
        per_gpu[g] = per_gpu.get(g, 0) + int(sec)
    dump(os.path.join(RES, "cagra_big_deep1B_build.json"), {"seconds_per_gpu": per_gpu})


if __name__ == "__main__":
    octex_builds()
    cagra_deep1b()
