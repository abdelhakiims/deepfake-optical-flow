"""Figures of EXPERIMENTS.md.

The illustrations are drawn (no dataset images are reproduced); the head-motion
figure uses the project's own compensate_motion on a constructed flow field; the
result charts use the numbers of EXPERIMENTS.md, section 10.

    python docs/make_figures.py            # English, into docs/figures/
    python docs/make_figures.py --lang fr  # French, into docs/figures/fr/
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepfake_flow.flow import compensate_motion  # noqa: E402

LANG = "fr" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "fr" else "en"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", *(["fr"] if LANG == "fr" else []))
BLUE, ORANGE, GREEN, RED, GREY = "#2f6db5", "#e08a1e", "#2e8b57", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def to_french(fig):
    """Translate every text of the figure, including tick labels and legends."""
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter

    from figure_text_fr import translate
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            old = axis.get_major_formatter()
            axis.set_major_formatter(FuncFormatter(lambda x, pos, old=old: translate(old(x, pos))))
    for t in fig.findobj(Text):
        t.set_text(translate(t.get_text()))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    if LANG == "fr":
        fig.canvas.draw()
        to_french(fig)
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def face(ax, x, y, r=1.0, yaw=0.0, color="#f2c9a0", label=None):
    """Cartoon face; yaw in [-1, 1] moves the eyes towards one side (0 = frontal)."""
    ax.add_patch(Circle((x, y), r, fc=color, ec="black", lw=1))
    for side in (-1, 1):
        ex = x + r * (0.38 * side * (1 - abs(yaw)) + 0.45 * yaw)
        ax.add_patch(Circle((ex, y + 0.25 * r), 0.09 * r, fc="black"))
    ax.add_patch(Ellipse((x + 0.45 * yaw * r, y - 0.4 * r), 0.45 * r * (1 - 0.5 * abs(yaw)), 0.12 * r,
                         fc="#a0522d"))
    if label:
        ax.text(x, y - 1.45 * r, label, ha="center", va="top", fontsize=9)


def box(ax, x, y, w, h, color, lw=2, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=color, lw=lw, ls=ls))


def blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------------- #
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.2))
    blank(ax, (0, 22), (0, 4))
    steps = ["Video", "Face tracking\n(YuNet, 8/s)", "Clips of 21\nconsecutive crops",
             "RAFT optical\nflow (20 maps)", "Remove head\nmotion (affine)",
             "ResNet18 flow +\nResNet18 image", "LSTM", "Real / Fake"]
    for i, s in enumerate(steps):
        x = 0.2 + i * 2.75
        ax.add_patch(FancyBboxPatch((x, 1), 2.3, 2, boxstyle="round,pad=0.05",
                                    fc="#eaf1fb" if i not in (0, 7) else "#f4f4f4", ec=BLUE))
        ax.text(x + 1.15, 2, s, ha="center", va="center", fontsize=7.6)
        if i < len(steps) - 1:
            ax.annotate("", (x + 2.7, 2), (x + 2.35, 2), arrowprops=dict(arrowstyle="->", lw=1.3))
    save(fig, "pipeline.png")


def fig_fixed_vs_tracked():
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6))
    xs = [2.2, 6.0, 9.8, 13.6, 17.4]
    for ax, title, tracked in ((axes[0], "Before: one fixed box for the whole clip", False),
                               (axes[1], "After: the box follows the tracked face", True)):
        blank(ax, (0, 22.5), (0, 4.6))
        ax.set_title(title, loc="left", fontsize=10)
        for k, x in enumerate(xs):
            ax.add_patch(Rectangle((x - 1.6, 0.3), 3.2, 4, fc="#fafafa", ec=GREY, lw=0.8))
            face(ax, x + (k - 2) * 0.35, 2.3, 0.8)
            cx = x + (k - 2) * 0.35 if tracked else x - 0.7
            box(ax, cx - 1.05, 1.25, 2.1, 2.1, GREEN if tracked else RED)
            ax.text(x, 0.45, f"frame {k * 5 + 1}", ha="center", fontsize=8)
    axes[0].text(19.3, 2.3, "face slides\nout of the box", color=RED, ha="left", fontsize=9)
    axes[1].text(19.3, 2.3, "face stays\ncentred", color=GREEN, ha="left", fontsize=9)
    save(fig, "crop_fixed_vs_tracked.png")


def fig_multi_person():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    titles = ["Interview: guest facing the camera,\ninterviewer seen from the side",
              "Small face in the background",
              "Two people facing the camera"]
    verdicts = [("main subject = the guest", GREEN), ("used: the other face is small", GREEN),
                ("ambiguous: frame not used", RED)]
    for ax, t, (v, c) in zip(axes, titles, verdicts):
        blank(ax, (0, 10), (0, 7))
        ax.add_patch(Rectangle((0.2, 0.8), 9.6, 6, fc="#fafafa", ec=GREY))
        ax.set_title(t, fontsize=9)
        ax.text(5, 0.2, v, ha="center", color=c, fontsize=9.5, weight="bold")
    face(axes[0], 6.5, 3.8, 1.3)
    box(axes[0], 5.0, 2.3, 3.0, 3.0, GREEN)
    face(axes[0], 2.3, 3.6, 1.2, yaw=-0.9)
    box(axes[0], 0.9, 2.2, 2.8, 2.8, GREY, ls="--")
    axes[0].text(2.3, 1.2, "frontality e.g. 0.07", ha="center", fontsize=8, color=GREY)
    axes[0].text(6.5, 1.2, "frontality e.g. 0.4", ha="center", fontsize=8, color=GREEN)
    face(axes[1], 4.0, 3.8, 1.5)
    box(axes[1], 2.3, 2.1, 3.4, 3.4, GREEN)
    face(axes[1], 8.2, 5.2, 0.6)
    axes[1].text(8.2, 4.1, "area < 50%", ha="center", fontsize=8, color=GREY)
    face(axes[2], 3.0, 3.8, 1.3)
    face(axes[2], 7.0, 3.8, 1.25)
    box(axes[2], 1.5, 2.3, 3.0, 3.0, RED)
    box(axes[2], 5.55, 2.35, 2.9, 2.9, RED)
    axes[2].text(5, 1.4, "who is wearing the mask?", ha="center", fontsize=8.5, color=RED)
    save(fig, "multi_person.png")


def fig_frontality():
    fig, ax = plt.subplots(figsize=(9, 2.9))
    blank(ax, (0, 15), (0, 4.3))
    for x, yaw, val, ok in ((2.5, 0.0, "0.38", True), (7.5, 0.5, "0.19 to 0.30", None),
                            (12.5, 0.95, "0.05", False)):
        face(ax, x, 2.5, 1.2, yaw=yaw)
        col = GREEN if ok else (RED if ok is False else ORANGE)
        ax.text(x, 0.65, f"eye distance / width ~ {val}", ha="center", fontsize=9, color=col)
    ax.text(2.5, 0.15, "frontal: used", ha="center", color=GREEN, fontsize=9, weight="bold")
    ax.text(7.5, 0.15, "three-quarter: used if >= 0.25", ha="center", color=ORANGE, fontsize=9,
            weight="bold")
    ax.text(12.5, 0.15, "profile: not used", ha="center", color=RED, fontsize=9, weight="bold")
    save(fig, "frontality.png")


def fig_frame_skipping():
    t = np.arange(0, 60)
    rng = np.random.default_rng(3)
    head = 3 * np.sin(t / 9.0)                       # smooth head movement
    wobble = np.zeros_like(head, dtype=float)
    wobble[[14, 15, 31, 46, 47]] = [0.9, -0.8, 1.0, -0.9, 0.8]   # the mask slips for one frame
    pos = head + wobble + rng.normal(0, 0.05, len(t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, step, title in ((axes[0], 1, "Consecutive frames: every slip is visible"),
                            (axes[1], 6, "One frame in 6 (5 per second): the slips disappear")):
        ax.plot(t, head, color=GREY, lw=1, ls="--", label="head movement")
        idx = t[::step]
        ax.plot(idx, pos[idx], "-o", color=BLUE, ms=3.5, lw=1.2, label="frames used")
        for k in np.nonzero(wobble)[0]:
            ax.axvline(k, color=RED, alpha=0.18, lw=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame (25 per second)")
    axes[0].set_ylabel("position of a point of the face")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].text(58, 2.9, "red bands = frames where\nthe mask slips", color=RED, ha="right", fontsize=8.5)
    save(fig, "frame_skipping.png")


def fig_head_motion():
    n = 24
    ys, xs = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    rigid_dx = 0.8 - 0.4 * ys + 0.15 * xs                  # move right + rotate + zoom
    rigid_dy = 0.2 + 0.4 * xs + 0.15 * ys
    mouth = np.exp(-(((xs) / 0.25) ** 2 + ((ys - 0.5) / 0.15) ** 2))   # small local motion at the mouth
    local_dx, local_dy = 0.0 * mouth, 0.9 * mouth
    flow = np.stack([rigid_dx + local_dx, rigid_dy + local_dy])[None].astype(np.float32)
    residual = compensate_motion(flow, "affine")[0]
    fitted = flow[0] - residual
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    s = 2
    for ax, (u, v), title, col in ((axes[0], flow[0], "Raw flow: head moves, turns, comes closer\n"
                                    "(+ a small mouth movement)", BLUE),
                                   (axes[1], fitted, "Affine fit: the head's rigid motion", GREY),
                                   (axes[2], residual, "Residual = raw - fit:\nonly the mouth movement is left",
                                    RED)):
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#d9b38c", lw=6, alpha=0.6))
        ax.quiver(xs[::s, ::s], -ys[::s, ::s], u[::s, ::s], -v[::s, ::s], color=col, angles="xy",
                  scale_units="xy", scale=8 if ax is not axes[2] else 3, width=0.006)
        ax.set_title(title, fontsize=9.5)
        ax.text(0, -1.3, "(brown circle: the face; arrows: flow)", ha="center", fontsize=7.5, color=GREY)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.4, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, "head_motion_removal.png")


def fig_smoothing():
    rng = np.random.default_rng(0)
    t = np.arange(40)
    true = 100 + 25 * np.sin(t / 12)
    detected = true + rng.normal(0, 3, len(t))
    k = 5
    pad = np.concatenate([np.repeat(detected[:1], 2), detected, np.repeat(detected[-1:], 2)])
    smooth = np.convolve(pad, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, detected, "-o", ms=3, color=RED, lw=1, label="box centre from the detector (shaky)")
    ax.plot(t, smooth, color=GREEN, lw=2, label="after the 5-frame average (steady)")
    ax.plot(t, true, color=GREY, ls="--", lw=1, label="real head position")
    ax.set_xlabel("frame")
    ax.set_ylabel("box centre x (pixels)")
    ax.legend(fontsize=8)
    save(fig, "box_smoothing.png")


def fig_margin():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for ax, m, title, col in ((axes[0], 1.0, "Box = detected face (x 1.0)\nthe jaw-line seam is cut off", RED),
                              (axes[1], 1.3, "Box x 1.3 (used)\nthe seam around the face is kept", GREEN)):
        blank(ax, (-2.2, 2.2), (-2.2, 2.2))
        face(ax, 0, 0, 1.2)
        ax.add_patch(Circle((0, 0), 1.32, fill=False, ec=ORANGE, lw=2, ls=":"))
        half = 1.15 * m
        box(ax, -half, -half, 2 * half, 2 * half, col, lw=2.5)
        ax.set_title(title, fontsize=9.5)
    axes[1].text(0, -2.1, "orange dotted line = where a swapped face is blended in", ha="center", fontsize=8,
                 color=ORANGE)
    save(fig, "crop_margin.png")


def fig_leaky_split():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [1.6, 1]})
    blank(axes[0], (0, 27), (-1.2, 7.6))
    blank(axes[1], (0, 27), (0, 3))
    axes[0].set_title("First version: windows of 20 flow maps, a new one every 10 maps, split at random",
                      loc="left", fontsize=10)
    for f in range(0, 81, 10):                       # frame axis of one video
        axes[0].plot([1 + f * 0.3] * 2, [-0.2, 7.2], color="#e5e5e5", lw=0.8, zorder=0)
        axes[0].text(1 + f * 0.3, -0.7, str(f), ha="center", fontsize=7.5, color=GREY)
    parts = ["train", "train", "test", "train", "train", "test", "train"]
    for i, part in enumerate(parts):
        y = 6.4 - i * 0.95
        axes[0].add_patch(Rectangle((1 + i * 3, y), 6, 0.6, fc=BLUE if part == "train" else RED, alpha=0.8))
        axes[0].text(1 + i * 3 + 6.3, y + 0.3, f"window {i + 1}: {part}", va="center", fontsize=8,
                     color=BLUE if part == "train" else RED)
    axes[0].text(1, -1.2, "flow map index in one video. Window 3 (test) shares maps 20-29 with window 2 and "
                 "maps 30-39 with window 4, both in training.", fontsize=8.5)
    axes[1].set_title("Current version: split by person; up to 6 non-overlapping clips per video",
                      loc="left", fontsize=10)
    for i in range(6):
        axes[1].add_patch(Rectangle((1 + i * 2.1, 1.4), 1.8, 0.5, fc=BLUE, alpha=0.8))
        axes[1].add_patch(Rectangle((14 + i * 2.0, 1.4), 1.7, 0.5, fc=RED, alpha=0.8))
    axes[1].text(1, 0.6, "videos of person A: training only", fontsize=8.5, color=BLUE)
    axes[1].text(14, 0.6, "videos of person B: test only", fontsize=8.5, color=RED)
    save(fig, "leaky_split.png")


def fig_network():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    blank(ax, (0, 23), (0, 6.4))

    def block(x, y, w, h, text, fc="#eaf1fb", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))

    block(0.2, 4.0, 3.6, 1.6, "20 flow maps\n2 x 128 x 128\n(head motion removed)", fc="#f4f4f4", ec=GREY)
    block(0.2, 0.8, 3.6, 1.6, "20 face images\n3 x 128 x 128", fc="#f4f4f4", ec=GREY)
    block(4.8, 4.0, 3.6, 1.6, "ResNet18\n2-channel first layer\n(from ImageNet filters)")
    block(4.8, 0.8, 3.6, 1.6, "ResNet18\nImageNet weights")
    block(9.4, 2.4, 2.6, 1.6, "join\n20 x 1,024")
    block(13.0, 2.4, 3.0, 1.6, "LSTM\n2 layers, 128")
    block(17.0, 2.4, 2.6, 1.6, "dropout 0.3\nLinear 128 -> 2")
    block(20.4, 2.4, 2.4, 1.6, "real / fake", fc="#f4f4f4", ec=GREY)
    arrow(3.85, 4.8, 4.75, 4.8)
    arrow(3.85, 1.6, 4.75, 1.6)
    ax.text(8.75, 5.75, "20 x 512", fontsize=8, ha="center")
    ax.text(8.75, 0.55, "20 x 512", fontsize=8, ha="center")
    arrow(8.45, 4.8, 9.6, 4.05)
    arrow(8.45, 1.6, 9.6, 2.35)
    arrow(12.05, 3.2, 12.95, 3.2)
    arrow(16.05, 3.2, 16.95, 3.2)
    arrow(19.65, 3.2, 20.35, 3.2)
    ax.text(14.5, 1.7, "last hidden state\n(or mean over the 20 steps)", ha="center", fontsize=7.5, color=GREY)
    save(fig, "network.png")


def fig_tracking():
    """Detections on 6 scan frames, linked into one track per person."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    blank(ax, (0, 24), (0, 7.2))
    xs = [2, 5.8, 9.6, 13.4, 17.2, 21]
    guest = [(1.0, 4.2), (1.25, 4.25), (1.5, 4.2), (1.7, 4.1), None, (2.0, 4.0)]   # None = detector missed
    other = [(2.6, 4.5), (2.6, 4.5), (2.55, 4.5), (2.55, 4.5), (2.5, 4.5), (2.5, 4.5)]
    for k, x in enumerate(xs):
        ax.add_patch(Rectangle((x - 1.7, 2.4), 3.4, 3.8, fc="#fafafa", ec=GREY, lw=0.8))
        ax.text(x, 6.45, f"scan {k + 1}", ha="center", fontsize=8.5)
        g = guest[k]
        if g:
            face(ax, x - 1.7 + g[0], g[1], 0.55)
            box(ax, x - 1.7 + g[0] - 0.65, g[1] - 0.65, 1.3, 1.3, GREEN, lw=1.6)
        else:
            ax.text(x - 0.5, 4.2, "missed", ha="center", fontsize=7.5, color=RED)
        o = other[k]
        face(ax, x - 1.7 + o[0], o[1] - 1.3, 0.35, yaw=-0.9)
        box(ax, x - 1.7 + o[0] - 0.42, o[1] - 1.72, 0.84, 0.84, ORANGE, lw=1.4)
    gx = [x - 1.7 + g[0] for x, g in zip(xs, guest) if g]
    gy = [g[1] for g in guest if g]
    ax.plot(gx, gy, color=GREEN, lw=1.2, ls="--", zorder=0)
    ax.plot([x - 1.7 + o[0] for x, o in zip(xs, other)], [o[1] - 1.3 for o in other], color=ORANGE, lw=1.2,
            ls="--", zorder=0)
    ax.text(12, 1.6, "green track = person A (large, frontal: the main subject).  "
            "orange track = person B (small, seen from the side).", ha="center", fontsize=9)
    ax.text(12, 0.8, "At scan 5 the detector missed person A; the track continues because A is found again "
            "close to the same place within 3 scans.", ha="center", fontsize=9)
    save(fig, "tracking.png")


def fig_clips():
    """Where the clips go in a video of 250 frames."""
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(-5, 255)
    ax.set_ylim(0, 4)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    scan = np.arange(0, 250, 3)
    usable = ((scan >= 20) & (scan <= 110)) | ((scan >= 150) & (scan <= 235))
    ax.scatter(scan, np.full(len(scan), 3.2), c=[GREEN if u else RED for u in usable], s=10)
    ax.text(-5, 3.55, "scan frames (about 8 per second): green = main face usable, red = not usable "
            "(turned away, second person, ...)", fontsize=8.5)
    starts = [21, 54, 87, 152, 183, 214]
    for i, s0 in enumerate(starts):
        ax.add_patch(Rectangle((s0, 1.4), 21, 0.9, fc=BLUE, alpha=0.8))
        ax.text(s0 + 10.5, 1.85, f"clip {i + 1}", ha="center", va="center", fontsize=7.5, color="white")
    ax.text(-5, 0.6, "6 clips of 21 consecutive frames, only where every scan frame is green, "
            "spread over the video, never overlapping", fontsize=8.5)
    ax.set_xlabel("frame number (a 10-second video at 25 frames per second = 250 frames)")
    save(fig, "clips.png")


def fig_flow_map():
    """Two frames of a face and the optical flow between them."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, dx, mouth, title in ((axes[0], 0.0, 0.10, "frame t"), (axes[1], 0.25, 0.22, "frame t+1")):
        blank(ax, (-1.6, 1.6), (-1.6, 1.6))
        ax.add_patch(Circle((dx, 0), 1.2, fc="#f2c9a0", ec="black"))
        for side in (-1, 1):
            ax.add_patch(Circle((dx + 0.45 * side, 0.35), 0.1, fc="black"))
        ax.add_patch(Ellipse((dx, -0.5), 0.6, mouth, fc="#a0522d"))
        ax.set_title(title, fontsize=10)
    axes[1].text(0.25, -1.5, "the head moved right, the mouth opened", ha="center", fontsize=8.5)
    n = 13
    ys, xs = np.meshgrid(np.linspace(-1.2, 1.2, n), np.linspace(-1.2, 1.2, n), indexing="ij")
    inside = xs ** 2 + ys ** 2 <= 1.2 ** 2
    u = np.where(inside, 0.25, 0.0)
    v = np.where(inside, -0.15 * np.exp(-((xs / 0.35) ** 2 + ((ys + 0.5) / 0.15) ** 2)), 0.0)
    ax = axes[2]
    ax.add_patch(Circle((0, 0), 1.2, fill=False, ec="#d9b38c", lw=5, alpha=0.6))
    ax.quiver(xs, ys, u, v * 2.5, color=BLUE, angles="xy", scale_units="xy", scale=1.3, width=0.006)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("flow map t -> t+1: one arrow per point\n(dx = left/right, dy = up/down)", fontsize=10)
    save(fig, "flow_map.png")


# ----------------------------------------------------------------------------- #
def fig_results():
    folds = [0.984, 0.958, 0.963, 0.989, 0.945, 0.977, 0.990]
    celeb = [0.780, 0.767, 0.808, 0.742, 0.743, 0.822, 0.805]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(1, 8)
    ax.bar(x - 0.2, folds, 0.4, color=BLUE, label="DFD, unseen actors")
    ax.bar(x + 0.2, celeb, 0.4, color=ORANGE, label="Celeb-DF (unseen dataset)")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(0.45, 0.51, "chance", color=GREY, fontsize=8, ha="left")
    ax.set_xticks(x, [f"fold {i}" for i in x])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("video AUC")
    ax.set_title("Models trained on DFD (7-fold by actor)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.set_ylim(0.4, 1.08)
    save(fig, "dfd_folds_and_celebdf.png")

    rows = ["Full model\n(affine removal)", "Images only", "Flow only", "No motion\nremoval",
            "Mean motion\nremoval", "Mean over time\n(no LSTM)"]
    vals = {"FaceForensics++ test": [0.967, 0.939, 0.980, 0.971, 0.957, 0.986],
            "DFD (unseen)": [0.820, 0.798, 0.681, 0.824, 0.870, 0.790],
            "Celeb-DF (unseen)": [0.719, 0.738, 0.596, 0.687, 0.739, 0.748]}
    err = {"FaceForensics++ test": [(0.955, 0.977), (0.920, 0.955), (0.970, 0.988), (0.960, 0.981),
                                    (0.942, 0.970), (0.979, 0.992)],
           "DFD (unseen)": [(0.799, 0.842), (0.772, 0.825), (0.647, 0.711), (0.799, 0.849),
                            (0.850, 0.891), (0.763, 0.814)],
           "Celeb-DF (unseen)": [(0.672, 0.764), (0.692, 0.785), (0.549, 0.646), (0.639, 0.734),
                                 (0.691, 0.786), (0.700, 0.790)]}
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    x = np.arange(len(rows))
    for j, (name, col) in enumerate(zip(vals, (BLUE, GREEN, ORANGE))):
        v = np.array(vals[name])
        lo = v - np.array([e[0] for e in err[name]])
        hi = np.array([e[1] for e in err[name]]) - v
        ax.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=name, yerr=[lo, hi], capsize=3)
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(x, rows)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("video AUC (95% interval)")
    ax.set_title("Trained on FaceForensics++: the six versions (one seed each)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "ffpp_ablations.png")


# ----------------------------------------------------------------------------- #
def fig_training_curves():
    """Per-epoch curves copied from the history.json files of the runs (docs/data)."""
    import json
    h = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                    "training_histories.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    run = h["ffpp_two_stream"]
    ep = [e["epoch"] for e in run]
    ax.plot(ep, [e["train_loss"] for e in run], "o-", color=BLUE, ms=3, label="training loss")
    ax.plot(ep, [e["val_loss"] for e in run], "o-", color=ORANGE, ms=3, label="validation loss")
    best = max(run, key=lambda e: e["val_video_auc"])["epoch"]
    ax.axvline(best, color=GREY, ls=":", lw=1)
    ax.text(best + 0.2, 0.55, "best epoch", color=GREY, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Full model, FaceForensics++", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    names = [("ffpp_two_stream", "Full model"), ("ffpp_rgb", "Images only"), ("ffpp_flow", "Flow only"),
             ("ffpp_nomotion", "No motion removal"), ("ffpp_meanmotion", "Mean motion removal"),
             ("ffpp_meantemp", "Mean over time")]
    colors = [BLUE, ORANGE, GREEN, RED, "#7b4fa0", GREY]
    for (key, label), c in zip(names, colors):
        run = h[key]
        ax.plot([e["epoch"] for e in run], [e["val_video_auc"] for e in run], "-", color=c, lw=1.4,
                label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation video AUC")
    ax.set_title("FaceForensics++ validation, six versions", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    for k in range(1, 8):
        run = h[f"dfd_fold{k}"]
        ep = [e["epoch"] for e in run]
        ax.plot(ep, [e["train_loss"] for e in run], "-", color=BLUE, lw=1, alpha=0.7,
                label="training loss" if k == 1 else None)
        ax.plot(ep, [e["val_loss"] for e in run], "-", color=ORANGE, lw=1, alpha=0.7,
                label="validation loss" if k == 1 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("DFD, the 7 folds", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "training_curves.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_tracking()
    fig_clips()
    fig_flow_map()
    fig_leaky_split()
    fig_fixed_vs_tracked()
    fig_multi_person()
    fig_frontality()
    fig_frame_skipping()
    fig_smoothing()
    fig_margin()
    fig_head_motion()
    fig_network()
    fig_results()
    fig_training_curves()
