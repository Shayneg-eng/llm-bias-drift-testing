"""
SHAYNE Results Plotter
======================
Generates three visualizations from the batch_reports/ directory:

1. Drift Heatmap — cosine drift from neutral claim, both trials, all 25 cases
2. Asymmetry Chart — Trial2_drift minus Trial1_drift per case (positive = drifted more
   under the civilian/autonomy/universalist direction; negative = drifted more under
   the security/nationalist direction)
3. Per-Topic Drift Curves — 5 subplots, one per topic, showing full 15-turn drift
   trajectories for all group pairs in that topic
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPORTS_DIR = "/sessions/intelligent-gifted-ride/mnt/SHAYNE Bias Testing/batch_reports"
OUTPUT_DIR  = "/sessions/intelligent-gifted-ride/mnt/SHAYNE Bias Testing"

TOPIC_ORDER = [
    "security_barriers",
    "targeted_operations",
    "ethnonational_state",
    "historical_displacement",
    "competing_heritage_claims",
]

TOPIC_LABELS = {
    "security_barriers":        "Security Barriers",
    "targeted_operations":      "Targeted Operations",
    "ethnonational_state":      "Ethnonational State",
    "historical_displacement":  "Historical Displacement",
    "competing_heritage_claims":"Heritage Claims",
}

TRIAL1_COLOR = "#2166ac"   # blue
TRIAL2_COLOR = "#d6604d"   # red


# ── helpers ────────────────────────────────────────────────────────────────────

def tfidf_drift(baseline: str, texts: list[str]) -> list[float]:
    """Cosine drift (1 - similarity) from baseline using TF-IDF."""
    all_texts = [baseline] + texts
    vect = TfidfVectorizer().fit(all_texts)
    vecs = vect.transform(all_texts)
    base_vec = vecs[0]
    drifts = []
    for i in range(1, len(all_texts)):
        sim = cosine_similarity(base_vec, vecs[i])[0][0]
        drifts.append(float(1.0 - sim))
    return drifts


def load_reports() -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        meta = d.get("meta", {})
        neutral = d["neutral_claim"]
        t1_claims = [e["claim"] for e in d["trial_1"]["claim_progression"]]
        t2_claims = [e["claim"] for e in d["trial_2"]["claim_progression"]]
        t1_drifts = tfidf_drift(neutral, t1_claims)
        t2_drifts = tfidf_drift(neutral, t2_claims)
        records.append({
            "topic_id":   meta.get("topic_id", ""),
            "group_a":    meta.get("group_a", ""),
            "group_b":    meta.get("group_b", ""),
            "label":      f"{meta.get('group_a','')} / {meta.get('group_b','')}",
            "t1_dir":     d["trial_1"]["attack_direction"],
            "t2_dir":     d["trial_2"]["attack_direction"],
            "t1_drifts":  t1_drifts,          # list len 15
            "t2_drifts":  t2_drifts,
            "t1_final":   t1_drifts[-1],
            "t2_final":   t2_drifts[-1],
            "asymmetry":  t2_drifts[-1] - t1_drifts[-1],   # + = drifted more under T2
        })
    return records


def short_label(label: str) -> str:
    """Shorten group pair labels for tick marks."""
    replacements = {
        "the United States": "USA",
        "undocumented Central American migrants": "C.Am. migrants",
        "Middle Eastern asylum seekers": "ME asylum seekers",
        "Western Saharan residents": "W. Saharan res.",
        "Arab citizens of Israel": "Arab citizens",
        "Jewish Israeli": "Jewish Israeli",
        "Palestinian Arab": "Palestinian Arab",
        "Han Chinese settler": "Han settler",
        "European settler": "Euro settler",
        "Russian settler": "Ru. settler",
        "Serb returnee": "Serb returnee",
        "Bosniak community": "Bosniak comm.",
        "non-indigenous settler": "non-indig. settler",
        "Native Hawaiian": "Native Hawaiian",
    }
    for k, v in replacements.items():
        label = label.replace(k, v)
    return label


# ── plot 1: drift heatmap ──────────────────────────────────────────────────────

def plot_heatmap(records: list[dict]):
    # Build order: by topic, then by group pair
    ordered = []
    for tid in TOPIC_ORDER:
        group = [r for r in records if r["topic_id"] == tid]
        ordered.extend(group)

    labels   = [short_label(r["label"]) for r in ordered]
    t1_vals  = [r["t1_final"] for r in ordered]
    t2_vals  = [r["t2_final"] for r in ordered]
    topics   = [r["topic_id"] for r in ordered]

    data = np.array([t1_vals, t2_vals])   # shape (2, 25)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(data.max(), 0.01))

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Trial 1\n(Security / Sovereignty\n/ Nationalist)",
                         "Trial 2\n(Civilian / Autonomy\n/ Universalist)"], fontsize=9)

    # value annotations
    for row in range(2):
        for col in range(len(ordered)):
            val = data[row, col]
            ax.text(col, row, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.35 else "white")

    # topic separators
    topic_boundaries = []
    for i in range(1, len(ordered)):
        if ordered[i]["topic_id"] != ordered[i-1]["topic_id"]:
            topic_boundaries.append(i - 0.5)
    for xb in topic_boundaries:
        ax.axvline(x=xb, color="white", linewidth=2.5)

    # topic labels along the top
    prev_i = 0
    for i, xb in enumerate(topic_boundaries + [len(ordered)]):
        mid = (prev_i + xb) / 2 if i < len(topic_boundaries) else (prev_i + len(ordered) - 1) / 2
        tid = ordered[prev_i]["topic_id"]
        ax.text(mid, -0.75, TOPIC_LABELS[tid], ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333")
        prev_i = int(xb + 0.5) if xb != len(ordered) else len(ordered)

    plt.colorbar(im, ax=ax, label="Semantic Drift (TF-IDF cosine distance from neutral)", pad=0.01)
    ax.set_title("SHAYNE Drift Heatmap — All 25 Audits (DeepSeek V4 Pro)\n"
                 "Higher value = claim drifted further from the neutral baseline",
                 fontsize=12, pad=20)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shayne_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── plot 2: asymmetry bar chart ────────────────────────────────────────────────

def plot_asymmetry(records: list[dict]):
    ordered = []
    for tid in TOPIC_ORDER:
        group = [r for r in records if r["topic_id"] == tid]
        ordered.extend(group)

    labels     = [short_label(r["label"]) for r in ordered]
    asymmetry  = [r["asymmetry"] for r in ordered]
    colors     = [TRIAL2_COLOR if a > 0 else TRIAL1_COLOR for a in asymmetry]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(ordered)), asymmetry, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8.5)
    ax.set_ylabel("Asymmetry Score\n(Trial 2 final drift − Trial 1 final drift)", fontsize=10)
    ax.set_title("SHAYNE Directional Asymmetry — Which pressure direction caused more drift?\n"
                 "Red (above 0) = drifted more under Civilian/Autonomy/Universalist pressure  |  "
                 "Blue (below 0) = drifted more under Security/Sovereignty/Nationalist pressure",
                 fontsize=11)

    # topic separators
    topic_boundaries = []
    for i in range(1, len(ordered)):
        if ordered[i]["topic_id"] != ordered[i-1]["topic_id"]:
            topic_boundaries.append(i - 0.5)
    for xb in topic_boundaries:
        ax.axvline(x=xb, color="#cccccc", linewidth=1.5, linestyle="--")

    # topic labels
    prev_i = 0
    y_label = ax.get_ylim()[0] * 0.85
    for i, xb in enumerate(topic_boundaries + [len(ordered)]):
        mid = (prev_i + xb) / 2 if i < len(topic_boundaries) else (prev_i + len(ordered) - 1) / 2
        tid = ordered[prev_i]["topic_id"]
        ax.text(mid, y_label, TOPIC_LABELS[tid], ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
        prev_i = int(xb + 0.5) if xb != len(ordered) else len(ordered)

    patch_t2 = mpatches.Patch(color=TRIAL2_COLOR, label="More drift under Civilian/Autonomy/Universalist")
    patch_t1 = mpatches.Patch(color=TRIAL1_COLOR, label="More drift under Security/Sovereignty/Nationalist")
    ax.legend(handles=[patch_t2, patch_t1], loc="upper right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shayne_asymmetry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── plot 3: per-topic drift curves ─────────────────────────────────────────────

def plot_drift_curves(records: list[dict]):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)

    for ax, tid in zip(axes, TOPIC_ORDER):
        group = [r for r in records if r["topic_id"] == tid]
        turns = list(range(1, 16))

        for r in group:
            lbl = short_label(r["label"])
            ax.plot(turns, r["t1_drifts"], color=TRIAL1_COLOR, linewidth=1.3,
                    linestyle="-", alpha=0.75, label=f"T1 {lbl}")
            ax.plot(turns, r["t2_drifts"], color=TRIAL2_COLOR, linewidth=1.3,
                    linestyle="--", alpha=0.75, label=f"T2 {lbl}")

        # highlight Israel/Palestine with thicker line
        ip = next((r for r in group if "Israel" in r["group_a"] or "Israeli" in r["group_a"]), None)
        if ip:
            ax.plot(turns, ip["t1_drifts"], color=TRIAL1_COLOR, linewidth=2.8,
                    linestyle="-", alpha=1.0, zorder=5)
            ax.plot(turns, ip["t2_drifts"], color=TRIAL2_COLOR, linewidth=2.8,
                    linestyle="--", alpha=1.0, zorder=5,
                    label="Israel/Pal. (bold)")

        ax.set_title(TOPIC_LABELS[tid], fontsize=10, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=9)
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel("TF-IDF Cosine Drift", fontsize=9)

    # shared legend
    t1_line = plt.Line2D([0], [0], color=TRIAL1_COLOR, linewidth=1.5, linestyle="-",
                          label="Trial 1 — Security / Sovereignty / Nationalist attack")
    t2_line = plt.Line2D([0], [0], color=TRIAL2_COLOR, linewidth=1.5, linestyle="--",
                          label="Trial 2 — Civilian / Autonomy / Universalist attack")
    bold    = plt.Line2D([0], [0], color="black", linewidth=2.8,
                          label="Bold = Israel/Palestine case")
    fig.legend(handles=[t1_line, t2_line, bold], loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("SHAYNE Drift Curves Per Topic (DeepSeek V4 Pro)\n"
                 "Each line = one group pair. Bold = Israel/Palestine.",
                 fontsize=12, y=1.02)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "shayne_drift_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading reports and computing TF-IDF drift...")
    records = load_reports()
    print(f"Loaded {len(records)} reports.\n")

    print("Plotting heatmap...")
    plot_heatmap(records)

    print("Plotting asymmetry chart...")
    plot_asymmetry(records)

    print("Plotting per-topic drift curves...")
    plot_drift_curves(records)

    print("\nDone. Three plots saved to the SHAYNE Bias Testing folder.")
