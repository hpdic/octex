"""All figures of the OctEx paper from the measured results (python figs.py -> paper/figures/*.pdf).

opener.pdf     (a) QPS over CAGRA at recall >= 0.95 / 0.99 on six datasets, (b) Deep1B recall-QPS
frontier.pdf   recall-QPS fronts of OctEx, CAGRA, cuVS IVF-PQ, and IVF-RaBitQ on six datasets
ablation.pdf   start points at 1B, reverse edges at 100M, load-balanced gathers at 100M
placement.pdf  per-GPU memory at 1B, table placement at 1B, GPU scaling on SIFT1M
build.pdf      build time on four H100 GPUs and memory of the loaded index
portability.pdf  speedup over CAGRA on one A100 GPU by recall target
arch.pdf       the search path and the memory tiers
monitor.pdf    one Deep1B batch: phases and GPU metrics (SM, HBM, NVLink, PCIe)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import data

FIG = os.environ.get("PAPER_FIGDIR", os.path.join(data.ROOT, "paper", "figures"))
SYS = os.environ.get("PAPER_SYSNAME", "OctEx")
OURS, OURS2, BASE, REF = "#5B3B9E", "#9B7FD1", "#A67C00", "#E76F51"
BASE2, BASE3 = "#D4A017", "#7A5C00"
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42, "hatch.linewidth": 0.6})
METHODS = (("ours", SYS, OURS, "-", "o"), ("cagra", "CAGRA", BASE, "--", "s"),
           ("flowann", "FlowANN", "#B8860B", (0, (5, 1, 1, 1)), "P"), ("ggnn", "GGNN", "#8C6D1F", (0, (1, 1)), "v"),
           ("pathweaver", "PathWeaver", "#C8A951", (0, (3, 1, 1, 1, 1, 1)), "X"),
           ("ivfpq", "IVF-PQ", BASE3, ":", "^"), ("rabitq", "IVF-RaBitQ", BASE2, "-.", "D"))


HBM = 93.1  # GiB of one H100 as CUDA reports it (cudaMemGetInfo total)

NINES = (lambda r: -np.log10(np.clip(1 - np.asarray(r, float), 1e-6, None)),
         lambda x: 1 - 10 ** (-np.asarray(x, float)))  # recall axis by its number of nines: 0.9, 0.99, 0.999 equally spaced
RTICKS = [0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]


def inlegend(ax, *args, **kw):
    """The legend of a panel, one style everywhere: no frame, icons and text inside the axes, away from its edges."""
    style = dict(frameon=False, fontsize=5.6, borderaxespad=0.6, borderpad=0.0, handlelength=1.0, handleheight=0.8,
                 handletextpad=0.4, labelspacing=0.3, columnspacing=0.8)
    style.update(kw)
    return ax.legend(*args, **style)


def even_gaps(fig, axes, gap=None, shrink=()):
    """Equal blank space between neighboring panels of one row: from the right frame of one panel to the y-axis
    label of the next. The row keeps its extent. Without gap, the frames keep their widths and the gap follows;
    with gap (a fraction of the figure width), the panels in shrink give up the width it takes, in proportion."""
    for _ in range(3):  # the ticks and labels do not move with the frames; a few passes settle the positions
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        fr = [a.get_position() for a in axes]
        dl = [f.x0 - inv.transform_bbox(a.yaxis.get_tightbbox(r)).x0 for a, f in zip(axes, fr)]  # y labels
        free = fr[-1].x1 - fr[0].x0 - sum(f.width for f in fr) - sum(dl[1:])
        n = len(axes) - 1
        wd = [f.width for f in fr]
        if gap is not None:
            take = gap * n - free  # width the shrinking panels give up
            tot = sum(wd[i] for i in shrink)
            wd = [w - take * w / tot if i in shrink else w for i, w in enumerate(wd)]
        g = gap if gap is not None else free / n
        x = fr[0].x0
        for i, (a, f) in enumerate(zip(axes, fr)):
            a.set_position([x, f.y0, wd[i], f.height])
            x += wd[i] + g + (dl[i + 1] if i < n else 0)
        if gap is not None:
            gap, shrink = None, ()  # the later passes only even out the label shifts


def uniform(fig, tick=6.2, label=6.6, title=7.5, legend=5.6):
    """One set of font sizes for every data figure: tick labels (major and minor), axis titles, panel titles, legends."""
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=tick)
        for t in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_xticklabels(minor=True)
                  + ax.get_yticklabels(minor=True)):
            t.set_fontsize(tick)
        ax.xaxis.label.set_size(label)
        ax.yaxis.label.set_size(label)
        ax.title.set_size(title)
        legs = [ax.get_legend()] if ax.get_legend() else []
    for leg in [a.get_legend() for a in fig.axes if a.get_legend()] + list(fig.legends):
        for t in leg.get_texts():
            t.set_fontsize(legend)
        if leg.get_title().get_text():
            leg.get_title().set_fontsize(legend)


def nines_axis(a, lo=0.9, hi=0.999, ticks=None):
    a.set_xscale("function", functions=NINES)
    a.set_xlim(lo, hi)
    t = [v for v in (ticks or RTICKS) if lo <= v <= hi]
    a.set_xticks(t)
    a.set_xticklabels([f"{v:g}" for v in t], fontsize=7)
    minor = [v for v in (0.98, 0.995, 0.998) if lo <= v <= hi and v not in t]
    a.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(minor))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def front(name, ds):
    return data.frontier(getattr(data, name)(ds))


def opener():
    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.62), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.3]})
    a = ax[0]
    for name, lab, col, ls, mk in METHODS:
        f = [p for p in front(name, "deep1B") if p[0] >= 0.95]
        if f:
            a.plot([p[0] for p in f], [p[1] / 1e3 for p in f], color=col, ls=ls, marker=mk, ms=1.8, lw=1.0, label=lab)
    a.set_ylim(0, 820)
    a.set_xlabel("Recall@10", fontsize=7, labelpad=1)
    a.set_ylabel("Throughput (kQPS)", fontsize=7, labelpad=1)
    nines_axis(a, 0.95, 0.9965, ticks=[0.95, 0.99, 0.995])
    a.tick_params(labelsize=6.3)
    inlegend(a, loc="upper right")
    a.set_title("(a) Deep1B", fontsize=7.2, pad=2)

    b = ax[1]
    x = np.arange(len(data.DATASETS))
    w = 0.4
    for i, (r, col, hatch) in enumerate(((0.95, OURS2, None), (0.99, OURS, "////"))):
        sp = []
        for ds in data.DATASETS:
            o, c = data.qps_at(front("ours", ds), r), data.qps_at(front("cagra", ds), r)
            sp.append(o / c if o and c else 0)
        bars = b.bar(x + (i - 0.5) * w, sp, w, color=col, hatch=hatch, edgecolor="white" if hatch else col, lw=0.2,
                     label=f"recall $\\geq$ {r}")
        for bar, v in zip(bars, sp):
            b.text(bar.get_x() + bar.get_width() / 2 + (i - 0.5) * 0.08, v * 1.06, f"{v:.1f}" if v else "",
                   ha="center", va="bottom", fontsize=4.2, rotation=90)  # the pair of labels spread apart
    b.axhline(1, color=BASE, lw=0.8, ls="--")
    b.set_yscale("log")
    b.set_ylim(0.8, 200)
    b.set_yticks([1, 2, 5, 10, 20, 50])
    b.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xticks(x)
    b.set_xticklabels([data.SHORT[ds] for ds in data.DATASETS], fontsize=5.6, rotation=60, ha="right",
                      rotation_mode="anchor")
    b.tick_params(axis="y", labelsize=6.3)
    b.tick_params(axis="x", pad=1)
    b.set_ylabel("QPS over CAGRA", fontsize=7, labelpad=1)
    inlegend(b, loc="upper left", ncol=1)
    b.set_title("(b) Speedup over CAGRA", fontsize=7.2, pad=2)
    uniform(fig)
    fig.savefig(os.path.join(FIG, "opener.pdf"), bbox_inches="tight", pad_inches=0.02)


def frontier():
    sets = [d for d in data.DATASETS if d != "deep1B"]  # Deep1B: Figure 1(a)
    nc = 3 if len(sets) <= 6 else 4  # seven datasets: 2 x 4, the legend in the spare cell; eight: the legend on top
    fig, ax = plt.subplots(2, nc, figsize=(7.0, 3.3), gridspec_kw={"hspace": 0.5, "wspace": 0.32 if nc == 3 else 0.42})
    for k, ds in enumerate(sets):
        a = ax[k // nc][k % nc]
        lo = 0.95 if ds.endswith("1B") else 0.9
        for name, lab, col, ls, mk in METHODS:
            f = [p for p in front(name, ds) if p[0] >= lo]
            if f:
                a.plot([p[0] for p in f], [p[1] / 1e3 for p in f], color=col, ls=ls, marker=mk, ms=2.3, lw=1.2,
                       label=lab)
        if ds.endswith("1B"):  # linear: the gaps at high recall are tens of percent, which a log axis flattens
            a.set_ylim(0, None)
        else:
            a.set_yscale("log")
            a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())  # decades only: no 3 x 10^2 labels
        top = max(p[0] for name, *_ in METHODS for p in front(name, ds))
        nines_axis(a, lo, min(0.9995, 1 - (1 - top) * 0.75),
                   ticks=[0.95, 0.98, 0.995] if ds.endswith("1B") else
                   [0.9, 0.99, 0.999] if top >= 0.999 else [0.9, 0.97, 0.99, 0.999])
        a.set_title(data.LABEL[ds])
        if k % nc == 0:
            a.set_ylabel("Throughput (kQPS)")
        if k >= len(sets) - nc:
            a.set_xlabel("Recall@10")
    h = [plt.Line2D([], [], color=col, ls=ls, marker=mk, ms=2.5, lw=1.2) for _, _, col, ls, mk in METHODS]
    if len(sets) < 2 * nc:
        for k in range(len(sets), 2 * nc):
            ax[k // nc][k % nc].axis("off")
        leg = ax[1][nc - 1].legend(h, [m[1] for m in METHODS], loc="center", frameon=True, fontsize=7.5,
                                   fancybox=False, borderpad=0.6)
        leg.get_frame().set(facecolor="#F4F2F8", edgecolor="#6A5A8E", linewidth=0.6)
    else:
        leg = fig.legend(h, [m[1] for m in METHODS], loc="upper center", ncol=7, frameon=True, fontsize=7,
                         bbox_to_anchor=(0.5, 1.045), handlelength=3.2, columnspacing=1.2, handletextpad=0.5,
                         fancybox=False, borderpad=0.4)
        leg.get_frame().set(facecolor="#F4F2F8", edgecolor="#6A5A8E", linewidth=0.6)
    uniform(fig)
    fig.savefig(os.path.join(FIG, "frontier.pdf"), bbox_inches="tight", pad_inches=0.02)


def ablation():
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.88), gridspec_kw={"wspace": 0.55, "width_ratios": [1.18, 1.1, 1.1, 0.92, 0.86]})
    # (a) start points at 1B: 256 copies, read 16, waves 3..5
    a = ax[2]
    rows = [r for r in data.runs("frontier_deep1B") if r["starts"] >= 192 and 3 <= r["waves"] <= 5]
    starts = sorted({r.get("starts", r.get("seeds")) for r in rows})
    cmap = plt.get_cmap("Purples")
    for i, s in enumerate(starts):
        pts = sorted((r["waves"], r["recall"], r["qps_median"]) for r in rows if r.get("starts", r.get("seeds")) == s)
        a.plot([p[1] for p in pts], [p[2] / 1e3 for p in pts], marker="os^vD<>"[i % 7], ms=2.8, lw=1.2,
               color=cmap(0.35 + 0.65 * i / max(1, len(starts) - 1)), label=str(s))
    c = data.frontier(data.cagra("deep1B"))
    c = [p for p in c if p[0] >= 0.955]
    a.plot([p[0] for p in c], [p[1] / 1e3 for p in c], color=BASE, ls="--", marker="s", ms=2.3, lw=1.2)
    a.text(0.9957, 64, "CAGRA", color=BASE, fontsize=6.2, ha="right", va="center")  # left of the lower end of its line
    a.set_yscale("log")
    nines_axis(a, 0.955, 0.9965, ticks=[0.96, 0.99])
    a.set_ylim(50, 2500)  # headroom for the legend above the lines
    a.set_xlabel("Recall@10")
    a.set_ylabel("Throughput (kQPS)")
    inlegend(a, loc="upper right", ncol=3, columnspacing=0.2, handletextpad=0.15, handlelength=0.6, labelspacing=0.15, title="start points")
    a.set_yticks([100, 300, 1000])  # one set of labeled ticks, no labeled minor ticks
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_title("(c) Start points", fontsize=7.5)
    # (a) reverse edges at 100M: neighbor rows of forward neighbors only against rows with reverse entries of the
    # same length; 272 copies, 128 start points, two waves, read depth 32 / 48 / 64
    b = ax[0]
    fwd = {r["read"]: r for r in data.runs("reverse_fwd_deep100M") if r["waves"] == 2}
    revr = {r["read"]: r for r in data.runs("reverse_rev_deep100M") if r["waves"] == 2}
    reads = sorted(fwd)
    x = np.arange(len(reads))
    w = 0.3
    b.bar(x - w / 2, [100 * (1 - revr[r]["recall"]) for r in reads], w, color=OURS)
    b.bar(x + w / 2, [100 * (1 - fwd[r]["recall"]) for r in reads], w, color="white", edgecolor=OURS, hatch="//",
          lw=1.0)
    b.set_xticks(x)
    b.set_xticklabels([str(r) for r in reads])
    b.set_xlabel("Read depth (2 waves)", labelpad=1)
    b.set_ylabel("Missed neighbors (%)")
    b.set_ylim(0, 3.2)
    b.set_yticks([0, 1, 2])
    inlegend(b, [SYS, "no reverse edges"], loc="upper left", handlelength=0.8, handleheight=1.2, handletextpad=0.3)
    b.set_title("(a) Reverse edges", fontsize=7.5)
    # (c) copies at 100M: recall and throughput as the first round reads more copies (read 48, 128 start points)
    cc = ax[3]
    rows = data.runs("copies_deep100M")
    for wv, col, mk in ((1, "#1B9E8A", "o"), (2, OURS, "s")):  # contrasting hues: teal and dark purple
        pts = sorted((r["copies_limit"] or r["copies"], r["recall"], r["qps_median"]) for r in rows if r["waves"] == wv)
        cc.plot([p[1] for p in pts], [p[2] / 1e6 for p in pts], color=col, marker=mk, ms=2.8, lw=1.2,
                label=f"{wv} wave" + ("s" if wv > 1 else ""))
        # copies: right of the first point, left of the others; 2 waves only at its two ends (its points are close)
        for c_, rc, q in pts if wv == 1 else (pts[0], pts[-1]):
            off, ha = ((4, 0), "left") if c_ == pts[0][0] else ((-4, 0), "right")
            cc.annotate(str(c_), (rc, q / 1e6), textcoords="offset points", xytext=off, ha=ha, va="center",
                        fontsize=5.8, color=col, fontweight="bold")
    cc.set_xticks([0.93, 0.96, 0.99])
    cc.set_xlim(0.92, 1.006)  # room right of the 2-wave line for its labels
    cc.set_xlabel("Recall@10")
    cc.set_ylabel("Throughput (MQPS)")
    cc.set_ylim(1.68, 3.5)  # room above the 1-wave line for the legend
    cc.set_yticks([2.0, 2.5, 3.0, 3.5])
    inlegend(cc, loc="upper right")
    cc.set_title("(d) Copies", fontsize=7.5)
    # (d) where the time of a batch goes: the fastest configuration with recall >= 0.99 per dataset (slowest GPU)
    dd = ax[4]
    import glob as _glob
    labs, fr, wv, ot = [], [], [], []
    for ds, lab in (("sift1m", "SIFT"), ("gist-960-euclidean", "GIST"), ("imagenet-dinov2", "DINOv2"),
                    ("deep10M", "D-10M"), ("deep100M", "D-100M"), ("deep1B", "D-1B")):
        best = None
        for r in data.runs(f"frontier_{data.DIR[ds]}"):
            if r.get("split_ms") and r.get("times") and r["placement"] == "shard" and r["recall"] >= 0.99:
                if best is None or r["qps_median"] > best["qps_median"]:
                    best = r
        if best is None:
            continue
        tot = float(np.median(best["times"])) * 1e3
        f1 = max(g["first_round_ms"] for g in best["split_ms"])
        w1 = max(g["waves_ms"] for g in best["split_ms"])
        labs.append(lab)
        fr.append(100 * f1 / tot)
        wv.append(100 * w1 / tot)
        ot.append(max(0.0, 100 - 100 * (f1 + w1) / tot))
    x = np.arange(len(labs))
    dd.bar(x, fr, 0.6, color=OURS2, label="first round")
    dd.bar(x, wv, 0.6, bottom=fr, color=OURS, hatch="////", edgecolor="white", lw=0.3, label="waves")
    dd.bar(x, ot, 0.6, bottom=np.array(fr) + np.array(wv), color="white", edgecolor=OURS, hatch="..", lw=0.8,
           label="other")
    dd.set_xticks(x)
    dd.set_xticklabels(labs, rotation=90, fontsize=7)
    dd.set_ylabel("Batch time (%)")
    dd.set_ylim(0, 150)
    dd.set_yticks([0, 25, 50, 75, 100])
    inlegend(dd, ["first round", "waves", "other"], loc="upper left")
    dd.set_title("(e) Batch time", fontsize=7.5)
    # (b) system optimizations, ablated: throughput without each one relative to the full search (ablation_system)
    sy = ax[1]
    rows = data.runs("ablation_system")
    sets = (("sift1m", "SIFT"), ("deep100M", "D-100M"), ("deep1B", "D-1B"))
    def qps(ds, fused, graph):
        return next(r["qps_median"] for r in rows if r["dataset"] == ds and r["fused_merge"] == fused
                    and r["use_graph"] == graph)
    fus = [qps(d, False, True) / qps(d, True, True) for d, _ in sets]
    gra = [qps(d, True, False) / qps(d, True, True) for d, _ in sets]
    x = np.arange(len(sets))
    w = 0.27
    sy.bar(x - w, [1.0] * len(sets), w, color=OURS, label=SYS)
    sy.bar(x, gra, w, color="white", edgecolor=OURS, hatch="\\\\\\", lw=1.0, label="no CUDA graph")
    sy.bar(x + w, fus, w, color="white", edgecolor=OURS, hatch="//", lw=1.0, label="no fused merge")
    sy.set_xticks(x)
    sy.set_xticklabels([lab for _, lab in sets], rotation=45, ha="right", rotation_mode="anchor")
    sy.set_ylabel("Relative throughput")
    sy.set_ylim(0, 1.55)
    sy.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    inlegend(sy, loc="upper right", handlelength=0.8, handleheight=1.2, handletextpad=0.3)
    sy.set_title("(b) Search kernels", fontsize=7.5)
    for axis in ax:
        axis.yaxis.label.set_size(6.4)
        axis.xaxis.label.set_size(6.6)
        axis.yaxis.labelpad = 1.0
        axis.tick_params(axis="both", labelsize=6.2, pad=1.5)
        axis.tick_params(axis="both", which="minor", labelsize=6.2)
        for t in (axis.get_xticklabels() + axis.get_yticklabels() + axis.get_xticklabels(minor=True)
                  + axis.get_yticklabels(minor=True)):  # one size, whatever a panel set before
            t.set_fontsize(6.2)
    uniform(fig)
    even_gaps(fig, ax, gap=0.009, shrink=(3, 4))  # (d) and (e) have room left of their legends
    fig.savefig(os.path.join(FIG, "ablation.pdf"), bbox_inches="tight", pad_inches=0.02)


def placement():
    small = {"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    ctx = plt.rc_context(small)
    ctx.__enter__()
    # one row of five panels across both columns
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.6), gridspec_kw={"wspace": 0.92, "width_ratios": [1.1, 1, 0.7, 0.9, 1.3]})
    # (a) device memory per GPU at 1B (256 copies, 64 per GPU): parts as loaded (HANDOFF findings 91, 99, 100)
    a = ax[0]
    r1b = data.runs("frontier_deep1B")
    m0 = r1b[0]["memory"][0]
    buf = max(g["search_setup_delta"] for r in r1b for g in r["memory"]) + m0["other"]
    gib = lambda v: v / 2**30  # noqa: E731
    parts = [("sketch", gib(m0["sketch"]), OURS), ("copies", gib(m0["copies"]), OURS2),
             ("rows", gib(m0["table"]), BASE2), ("buffers", gib(buf), REF)]
    bottom = 0
    for lab, v, col in parts:
        a.bar([0], [v], 0.3, bottom=bottom, color=col)
        a.text(0.2, bottom + v / 2, f"{lab} {v:.1f}", fontsize=6.2, va="center")
        bottom += v
    a.axhline(HBM, color=REF, lw=1.0, ls=":")
    a.text(-0.2, HBM + 2, "HBM capacity", fontsize=6.2, color=REF, va="bottom")  # above the line, inside the frame
    vec = r1b[0]["n"] * r1b[0]["dim"] * 2 / 2**30
    a.set_xlim(-0.25, 1.75)
    a.set_xticks([0])
    a.set_xticklabels(["one GPU"])
    a.set_ylabel("GiB")
    a.set_ylim(0, 108)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_title("(a) HBM, D-1B")
    # (b) neighbor table in host memory vs HBM at 1B (findings 89 and 91; 2 waves)
    b = ax[1]
    th = sorted(data.runs("tablehost_deep1B"), key=lambda r: r["starts"])
    tb = sorted(data.runs("tablehbm_deep1B"), key=lambda r: r["starts"])
    host = [(r["recall"], r["qps_median"] / 1e3) for r in th]
    hbm = [(r["recall"], r["qps_median"] / 1e3) for r in tb]
    lab = [str(r["starts"]) for r in th]
    x = np.arange(len(lab))
    w = 0.38
    b.bar(x - w / 2, [p[1] for p in host], w, color="white", edgecolor=OURS, hatch="..", lw=1.0,
          label="table in host memory")
    b.bar(x + w / 2, [p[1] for p in hbm], w, color=OURS, label="table in HBM")
    b.set_xticks(x)
    b.set_xticklabels(lab, fontsize=6.2)
    b.set_xlabel("Start points", labelpad=1)
    b.set_ylabel("Throughput (kQPS)")
    b.set_ylim(0, 1500)
    b.set_yticks([0, 500, 1000])
    inlegend(b, ["in host", "in HBM"], loc="upper right")
    b.set_title("(b) Rows, D-1B")
    # (c) 1, 2, 4 GPUs on SIFT1M (stored ripples, full index per GPU; CAGRA degree 64)
    c = ax[2]
    ours = [{"gpus": len(r["devices"]), "read": r["read"], "qps": r["qps_median"]} for r in data.runs("scaling_sift1m")]
    cag = json.load(open(os.path.join(data.RES, "scale_cagra_sift1m.json")))["rows"]
    for rows, key, val, col, ls, mk, lab in ((ours, "read", 24, OURS, "-", "o", SYS),
                                             (cag, "itopk", 64, BASE, "--", "s", "CAGRA")):
        pts = sorted((r["gpus"], r["qps"]) for r in rows if r.get(key) == val)
        c.plot([p[0] for p in pts], [p[1] / 1e6 for p in pts], color=col, ls=ls, marker=mk, ms=3, lw=1.3, label=lab)
        if col == OURS:  # top left, above the highest point of its line, for a name of any length
            c.text(1.1, 9.95, lab, color=col, fontsize=6.5, ha="left", va="bottom")
        else:
            c.text(3.85, pts[-1][1] / 1e6 + 0.3, lab, color=col, fontsize=6.5, ha="right", va="bottom", linespacing=1.0)
    c.set_xticks([1, 2, 4])
    c.set_xlabel("GPUs")
    c.set_ylabel("Throughput (MQPS)")
    c.set_ylim(0, 11)
    c.set_title("(c) GPUs, SIFT")
    # (d) one GPU on Deep100M: OctEx (48 copies, lists of 32) against FlowANN, a single-GPU system
    d = ax[3]
    for pts, col, ls, mk, lab in ((data.frontier(data.ours_1gpu("deep100M")), OURS, "-", "o", SYS),
                                  (data.frontier(data.flowann("deep100M", gpus=1)), "#B8860B", (0, (5, 1, 1, 1)), "P",
                                   "FlowANN")):
        pts = [p for p in pts if p[0] >= 0.9]
        d.plot([p[0] for p in pts], [p[1] / 1e3 for p in pts], color=col, ls=ls, marker=mk, ms=3, lw=1.3, label=lab)
    d.set_yscale("log")
    nines_axis(d, 0.9, 0.99, ticks=[0.95, 0.99])
    d.set_xlabel("Recall@10")
    d.set_ylabel("Throughput (kQPS)")
    d.set_ylim(25, 6000)
    d.text(0.93, 2600, SYS, color=OURS, fontsize=6.8)
    d.text(0.902, 210, "FlowANN", color="#B8860B", fontsize=6.8)
    d.set_title("(d) 1 GPU, D-100M")
    # (e) batch latency against the batch size on Deep100M near recall 0.993 (batch_study*.sh: the first nq queries)
    e = ax[4]
    for ds, tag, mk_o, mk_c, alpha in (("deep100M", "100M", "o", "s", 1.0), ("deep1B", "1B", "D", "^", 0.55)):
        nqs, ours_ms, cag_ms = data.batch_study(ds)
        e.plot(nqs, ours_ms, color=OURS, marker=mk_o, ms=2.6, lw=1.2, alpha=alpha, label=f"{SYS} {tag}")
        e.plot(nqs, cag_ms, color=BASE, ls="--", marker=mk_c, ms=2.6, lw=1.2, alpha=alpha, label=f"CAGRA {tag}")
    e.set_xscale("log")
    e.set_yscale("log")
    e.set_xticks([100, 1000, 10000])
    e.set_xticklabels(["100", "1k", "10k"])
    e.set_ylim(0.15, 2500)  # headroom: the legend sits above the curves
    e.set_yticks([0.3, 1, 3, 10, 30, 100, 300])
    e.set_yticklabels(["0.3", "1", "3", "10", "30", "100", "300"])
    e.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    e.set_xlabel("Batch size")
    e.set_ylabel("Latency (ms)")
    inlegend(e, loc="upper left")
    e.set_title("(e) Batch latency")
    uniform(fig)
    fig.savefig(os.path.join(FIG, "placement.pdf"), bbox_inches="tight", pad_inches=0.02)
    ctx.__exit__(None, None, None)


def build():
    """Build time and loaded memory of the two billion-vector indexes on four H100 GPUs. Build: wall time of
    octex_build --fast, split by phase from its log (results/octex/build_billion.json, bench/build/build_breakdown.py);
    CAGRA: 8 shards, 2 per GPU, wall time of the slowest GPU (results/baselines/cagra_big_*_build.json). Memory as
    loaded for the measured fronts."""
    sets = ("deep1B", "bigann1B")
    ph = json.load(open(os.path.join(data.OURS, "build_billion.json")))

    def parts(d):  # minutes: lists, the two passes of the neighbor rows, and the rest (read, sketch, writes)
        r = ph[f"{d}_fast"]
        p1 = sum(r.get(f"table_v1_{x}", 0) for x in ("load", "search", "reverse")) / 60
        p2 = sum(v for k, v in r.items() if k.startswith("table_v") and not k.startswith("table_v1_")) / 60
        return [r["copies"] / 60, p1, p2, r["wall"] / 60 - r["copies"] / 60 - p1 - p2]

    cagra = {d: max(json.load(open(os.path.join(data.RES, f"cagra_big_{d}_build.json")))["seconds_per_gpu"].values())
             / 60 for d in sets}
    mem = {}
    for d in sets:  # GiB over the four GPUs, without search buffers; host: the fp16 vectors
        r = max((r for r in data.runs(f"frontier_{data.DIR[d]}") if r["placement"] == "shard"),
                key=lambda r: sum(g["copies"] for g in r["memory"]))
        g = lambda f: sum(x[f] for x in r["memory"]) / 2**30  # noqa: E731
        mem[d] = (g("copies"), g("sketch") + g("table") + g("other"), r["n"] * r["dim"] * 2 / 2**30)

    fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.45), gridspec_kw={"wspace": 0.42, "width_ratios": [1.25, 1]})
    x = np.arange(len(sets))
    w = 0.36
    a = ax[0]
    a.bar(x - w / 2, [cagra[d] for d in sets], w, color=BASE, edgecolor="white", lw=0.3, label="CAGRA")
    names = ("lists", "rows, pass 1", "rows, pass 2", "other")
    cols = (OURS, OURS2, OURS2, REF)
    hats = (None, None, "////", None)
    bottom = np.zeros(len(sets))
    for k, (lab, c, h) in enumerate(zip(names, cols, hats)):
        v = np.array([parts(d)[k] for d in sets])
        a.bar(x + w / 2, v, w, bottom=bottom, color=c, hatch=h, edgecolor="white" if h is None else "#2B1A4F",
              lw=0.3, label=f"{SYS}: {lab}")
        bottom += v
    for xi, v in zip(x - w / 2, [cagra[d] for d in sets]):
        a.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6)
    for xi, v in zip(x + w / 2, bottom):
        a.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.6)
    a.set_ylim(0, 112)
    a.set_yticks([0, 20, 40, 60])
    a.set_ylabel("Build time (min)", fontsize=7, labelpad=1)
    a.set_title("(a) Build on 4 H100", fontsize=7.2, pad=2)
    inlegend(a, loc="upper left", ncol=1)
    a.set_xticks(x)
    a.set_xticklabels([data.SHORT[d] for d in sets])

    a = ax[1]
    names = ("copies", "sketch, rows", "host vectors")
    cols = (OURS, OURS2, REF)
    hats = (None, None, "////")
    bottom = np.zeros(len(sets))
    for k, (lab, c, h) in enumerate(zip(names, cols, hats)):
        v = np.array([mem[d][k] for d in sets])
        a.bar(x, v, 0.55, bottom=bottom, color=c, hatch=h, edgecolor="white" if h is None else "#2B1A4F", lw=0.3,
              label=lab)
        bottom += v
    a.axhline(4 * HBM, color=BASE3, lw=0.9, ls="--", label="HBM capacity")
    a.set_ylim(0, 1150)
    a.set_yticks([0, 200, 400, 600])
    a.set_ylabel("Memory (GiB)", fontsize=7, labelpad=1)
    a.set_title("(b) Loaded index", fontsize=7.2, pad=2)
    inlegend(a, loc="upper left", ncol=1)
    a.set_xticks(x)
    a.set_xticklabels([data.SHORT[d] for d in sets])
    uniform(fig)
    fig.savefig(os.path.join(FIG, "build.pdf"), bbox_inches="tight", pad_inches=0.02)


def portability():
    """Speedup over CAGRA on one A100 GPU (whole GPU; results/octex/a100 and results/baselines/a100_cuda): median QPS,
    best throughput at or above each recall target on both fronts; a target that one of the two systems does not
    reach has no bar."""
    targets = (0.95, 0.97, 0.98, 0.99, 0.995)
    sets = (("sift1m", "SIFT"), ("gist", "GIST"), ("dinov2", "DINOv2"), ("msmarco", "MARCO"), ("deep10M", "D-10M"))

    def best(p, t):
        q = [v for r, v in p if r >= t]
        return max(q) if q else None

    fig, a = plt.subplots(figsize=(3.35, 1.2))
    w = 0.16
    cmap = plt.get_cmap("Purples")
    for j, t in enumerate(targets):
        sp = []
        for ds, _ in sets:
            o = best([(r["recall"], r["qps_median"]) for r in data.runs(f"a100/frontier_{ds}")], t)
            c = best([(r["recall"], r["qps_median"])
                      for r in json.load(open(os.path.join(data.RES, "a100_cuda", f"cagra_{ds}.json")))["rows"]], t)
            sp.append(o / c if o and c else np.nan)
        hatch = ("", "////", "....", "xxxx", "\\\\\\\\")[j]  # fill patterns: the targets stay apart in grayscale
        a.bar(np.arange(len(sets)) + (j - 2) * w, sp, w, color=cmap(0.25 + 0.7 * j / 4), hatch=hatch,
              edgecolor="#2B1A4F" if j < 3 else "white", lw=0.3, label=f"{t:g}")
    a.axhline(1, color=BASE, lw=1.0, ls="--")
    a.set_yscale("log")
    a.set_ylim(0.8, 60)  # headroom for the legend inside the axes
    a.set_yticks([1, 2, 5, 10, 20])
    a.set_yticklabels(["1", "2", "5", "10", "20"])
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks(np.arange(len(sets)))
    a.set_xticklabels([s[1] for s in sets], fontsize=7)
    a.set_ylabel("QPS over CAGRA", fontsize=7.5)
    a.tick_params(axis="y", labelsize=7)
    leg = inlegend(a, loc="upper right", ncol=5, handlelength=1.4, handleheight=1.1)
    uniform(fig)
    fig.canvas.draw()  # "recall" to the left of the legend row
    lb = leg.get_window_extent().transformed(a.transAxes.inverted())
    a.text(lb.x0 - 0.015, (lb.y0 + lb.y1) / 2, "recall", transform=a.transAxes, fontsize=5.6, ha="right", va="center")
    fig.savefig(os.path.join(FIG, "portability.pdf"), bbox_inches="tight", pad_inches=0.02)


def arch():
    from matplotlib.patches import Circle, RegularPolygon
    fig = plt.figure(figsize=(3.35, 2.75))
    # the explosion in one projected copy: a hexagonal lattice stands in for the E8 cells
    a = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    a.set_xlim(-2.35, 2.35)
    a.set_ylim(-1.85, 1.85)
    a.set_aspect("equal")
    a.axis("off")
    h = 0.95
    hex_r = h / 1.732
    for i in range(-5, 6):
        for j in range(-5, 6):
            cx, cy = h * (i + 0.5 * (j % 2)), h * 0.866 * j
            a.add_patch(RegularPolygon((cx, cy), 6, radius=hex_r, orientation=0, fill=False, ec="#D8CFEA",
                                       lw=0.6))
    a.add_patch(RegularPolygon((0, 0), 6, radius=hex_r, orientation=0, fc="#EFEAF8", ec=OURS, lw=1.3, zorder=1))

    c0 = np.array([0.0, 0.0])
    query_col = "#0066CC"
    q = np.array([hex_r * 0.42, hex_r * 0.38])

    starts = np.array([
        [hex_r * 0.25, -hex_r * 0.16],  # start 0 (lower)
        [hex_r * 0.60, hex_r * 0.06],   # start 1 (right)
    ])

    pool_other = np.array([
        [-hex_r * 0.35, hex_r * 0.38],   # top-left (target of ripple arrow)
        [-hex_r * 0.60, hex_r * 0.22],   # upper-left, the dot the ripple label points at
        [-hex_r * 0.08, hex_r * 0.52],   # top
        [-hex_r * 0.40, -hex_r * 0.26],  # lower-left
        [-hex_r * 0.12, -hex_r * 0.55],  # bottom
    ])
    all_purple = np.vstack([starts, pool_other])

    r1 = hex_r * 0.78  # the waves stay clear of the left half, where the labels point
    r2 = hex_r * 0.74

    # Wave 1 dots: downward sector (angles -50 to -82)
    d1_1 = starts[0] + r1 * np.array([np.cos(np.radians(-50)), np.sin(np.radians(-50))])
    d1_2 = starts[0] + r1 * np.array([np.cos(np.radians(-82)), np.sin(np.radians(-82))])
    d1_in = starts[0] + (r1 - hex_r * 0.24) * np.array([np.cos(np.radians(-66)), np.sin(np.radians(-66))])

    # Wave 2 dots: upward sector (angles 8 to 55)
    d2_1 = starts[1] + r2 * np.array([np.cos(np.radians(8)), np.sin(np.radians(8))])
    d2_2 = starts[1] + r2 * np.array([np.cos(np.radians(55)), np.sin(np.radians(55))])
    d2_in = starts[1] + (r2 - hex_r * 0.22) * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])

    second_pts = np.vstack([d1_1, d1_2, d1_in, d2_1, d2_2, d2_in])

    a.add_patch(Circle(starts[0], r1, fill=False, ec=REF, lw=1.0, ls=(0, (3, 2)), zorder=2))
    a.add_patch(Circle(starts[1], r2, fill=False, ec=REF, lw=1.0, ls=(0, (3, 2)), zorder=2))

    rng = np.random.default_rng(999)
    bg_pts = rng.uniform(-2.35, 2.35, (220, 2))
    bg_pts = [p for p in bg_pts if np.linalg.norm(p - c0) > hex_r * 2.2]
    a.scatter(all_purple[:, 0], all_purple[:, 1], s=14, c=OURS, zorder=4, lw=0)
    a.scatter(second_pts[:, 0], second_pts[:, 1], s=16, c=REF, zorder=4, lw=0)
    a.scatter(starts[:, 0], starts[:, 1], s=52, facecolors="none", edgecolors=REF, lw=1.2, zorder=5)
    a.scatter([q[0]], [q[1]], s=95, marker="*", c=query_col, zorder=6, lw=0)
    a.scatter([0], [0], s=30, marker="x", c=OURS, zorder=6, lw=1.6)

    kw = dict(fontsize=6.6, va="center")
    a.annotate("query\n(blue star)", q, xytext=(0.85, 1.25), arrowprops=dict(arrowstyle="->", lw=0.6, color=query_col),
               color=query_col, **kw)

    hex_edge = np.array([-h / 2, -0.15])
    a.annotate("fragment of the query\n(purple hexagon)", hex_edge, xytext=(-2.3, -1.05),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=OURS), color=OURS, **kw)
    a.annotate("fragment center\n(purple cross)", (0, 0), xytext=(-2.3, 0.20),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=OURS), color=OURS, **kw)
    target_candidate = pool_other[1]
    a.annotate("fragment ripple\n(purple dots)", target_candidate, xytext=(-2.3, 1.25),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=OURS), color=OURS, **kw)
    wave_pt = starts[1] + r2 * np.array([np.cos(np.radians(32)), np.sin(np.radians(32))])  # upper edge, clear of the rings
    a.annotate("vector ripples\n(dashed orange circles)", wave_pt, xytext=(0.95, 0.68),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=REF, shrinkA=0, shrinkB=0), color=REF, **kw)
    # the tip ends on the outer edge of the ring (marker of diameter sqrt(52) pt, 1.2 pt edge), toward the label
    lab = np.array([1.05, -0.45])
    u = (lab - starts[1]) / np.linalg.norm(lab - starts[1])
    inch_per_unit = 3.35 / 4.7  # equal aspect: the 4.7 data units across set the scale
    ring_r = (np.sqrt(52) / 2 + 0.6) / 72 / inch_per_unit  # scatter size s is the squared marker diameter
    a.annotate("start points\n(orange rings)", starts[1] + ring_r * u, xytext=tuple(lab),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=REF, shrinkA=0, shrinkB=0), color=REF, **kw)
    a.annotate("neighbors found by the waves\n(orange dots)", d1_1, xytext=(0.55, -1.30),
               arrowprops=dict(arrowstyle="->", lw=0.6, color=REF), color=REF, **kw)
    # background vectors, none under a label (the labels have no background)
    fig.canvas.draw()
    inv = a.transData.inverted()
    boxes = [inv.transform(matplotlib.text.Text.get_window_extent(t).expanded(1.04, 1.15).get_points())  # text only
             for t in a.texts]
    rest_pts = np.array([p for p in bg_pts if not any(b0[0] <= p[0] <= b1[0] and b0[1] <= p[1] <= b1[1]
                                                      for b0, b1 in boxes)])
    a.scatter(rest_pts[:, 0], rest_pts[:, 1], s=4, c="#C9B98A", zorder=3, lw=0)
    fig.savefig(os.path.join(FIG, "arch.pdf"), bbox_inches="tight", pad_inches=0.02)


def monitor():
    """One batch of 10,000 queries on Deep1B on GPU 0 of the four H100 GPUs, as nsys samples its counters at 20 kHz:
    the raw samples of one timed batch, without averaging or smoothing (results/octex/monitor_deep1B.json,
    bench/profile_metrics.py). 512 start points, four waves, read depth 16. The phase boundaries come from the kernel
    trace of the same configuration (results/octex/profile_deep1B.json, bench/profile_timeline.py): the fragment
    ripples until the first start-point pick, one wave from each pick to the next, and the final rerank."""
    mon = json.load(open(os.path.join(data.OURS, "monitor_deep1B.json")))
    ks = json.load(open(os.path.join(data.OURS, "profile_deep1B.json")))["graph"]["0"]
    spans, w = {}, 0
    for t, d, n in ks:  # (start, end) of every phase on GPU 0
        if n == "pick_starts_kernel":
            w += 1
        p = "first" if w == 0 and n != "final_rerank_kernel" else "final" if n == "final_rerank_kernel" else w
        a0, a1 = spans.get(p, (t, t + d))
        spans[p] = (min(a0, t), max(a1, t + d))
    order = ["first", 1, 2, 3, 4, "final"]
    edges = [0.0] + [spans[p][1] for p in order[:-1]] + [mon["batch_ms"]]
    rows = [("SM", "sm_active", 100, OURS), ("HBM", "hbm_read", 40, "#1B9E8A"),
            ("NVLink", "nvlink_rx", 80, BASE), ("PCIe", "pcie_rx", 16, REF)]  # utilization, % of peak
    fig, ax = plt.subplots(len(rows) + 1, 1, figsize=(3.35, 1.95), sharex=True,
                           gridspec_kw={"hspace": 0.45, "height_ratios": [0.55] + [1] * len(rows)})
    bar = ax[0]
    labels = {"first": "fragment\nripples", 1: "wave 1", 2: "wave 2", 3: "3", 4: "4", "final": "rerank"}
    for k, p in enumerate(order):
        bar.text((edges[k] + edges[k + 1]) / 2, 0.5, labels[p], ha="center", va="center", fontsize=5.6,
                 linespacing=0.9)
    bar.set_ylim(0, 1)
    bar.set_yticks([])
    for sp in bar.spines.values():
        sp.set_visible(False)
    bar.set_ylabel("Phase", rotation=0, ha="right", va="center", labelpad=4)
    bar.tick_params(axis="x", length=0)
    for a, (lab, key, top, col) in zip(ax[1:], rows):
        r = np.array(mon["raw_" + key])  # every 50-us sample of the batch as measured
        a.fill_between(r[:, 0], 0, r[:, 1], step="post", color=col, alpha=0.3, lw=0)
        a.step(r[:, 0], r[:, 1], where="post", color=col, lw=0.6)
        a.set_ylim(0, top)
        a.set_yticks([top // 2, top])  # each row on its own scale, in the color of its metric
        a.set_yticklabels([f"{top // 2}%", f"{top}%"])
        a.tick_params(axis="y", colors=col, length=2, pad=1)
        a.spines["left"].set_color(col)
        a.set_ylabel(lab, rotation=0, ha="right", va="center", color=col, labelpad=4)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    for a in ax:  # the phases, cut by vertical lines through every row
        for e in edges[1:-1]:
            a.axvline(e, color="#666666", lw=0.5, ls=(0, (2, 1.5)), zorder=0)
    ax[-1].set_xlim(0, mon["batch_ms"])
    ax[-1].set_xlabel("Time in one batch on one GPU (ms)")
    uniform(fig)
    fig.align_ylabels(ax)
    fig.savefig(os.path.join(FIG, "monitor.pdf"), bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    opener()
    frontier()
    ablation()
    placement()
    build()
    portability()
    arch()
    monitor()
    print("figures written to", FIG)
