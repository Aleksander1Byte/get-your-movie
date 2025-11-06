"""
demo_recommender_plots.py
Minimal demo that uses the get_your_movie API to generate recommendations
and saves a couple of simple charts per region.

Run:
  python demo_recommender_plots.py

Requirements:
  - get_your_movie.py in the same directory (or on PYTHONPATH)
  - matplotlib, pandas, numpy
  - douban_all_movies.csv in the same directory
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from get_your_movie import recommend_movies

warnings.filterwarnings(
    "ignore", message=".*Glyph.*missing.*font.*", category=UserWarning
)

matplotlib.set_loglevel("error")
matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = "."
N = 7
ALPHA = 0.85
REGIONS = ["western", "china", "hongkong", "japan"]

# A compact set of globally famous/popular titles (mixed languages to test robustness).
# The recommender is robust to language via char n-grams and fallback similarity.
FAVORITES = [
    "The Godfather",
    "Inception",
    "Amélie",
    "Das Leben der Anderen",
    "霸王别姬",
    "无间道",
    "千与千寻",
]

OUT_DIR = "demo_plots"


# -------------------- Helpers --------------------
def short_label(title: str, width: int = 16) -> str:
    """Shorten labels for plotting axes; no external deps."""
    t = str(title)
    return (t[: width - 1] + "…") if len(t) > width else t


def ensure_out_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------- Main --------------------
def main():
    ensure_out_dir(OUT_DIR)
    print(f"Favorites: {', '.join(FAVORITES)}")
    print(f"Saving outputs to: {OUT_DIR}/")

    for region in REGIONS:
        print(f"\n=== Building recommendations for region: {region} ===")
        recs = recommend_movies(
            FAVORITES, target_region=region, n=N, data_dir=DATA_DIR, alpha=ALPHA
        )
        # Save table for inspection
        csv_path = os.path.join(OUT_DIR, f"demo_recs_{region}.csv")
        recs.to_csv(csv_path, index=False)
        print(f"- Saved table: {csv_path}")

        # ----- Plot 1: Bar chart of blended score -----
        titles = [short_label(t) for t in recs["title"].tolist()]
        scores = recs["score"].to_numpy()

        plt.figure()
        plt.bar(range(len(scores)), scores)
        plt.xticks(range(len(scores)), titles, rotation=45, ha="right")
        plt.ylabel("Blended score (alpha*similarity + (1-alpha)*rating)")
        plt.title(f"Top {N} recommendations — {region}")
        plt.tight_layout()
        out_png1 = os.path.join(OUT_DIR, f"demo_recs_{region}_score_bars.png")
        plt.savefig(out_png1, dpi=160)
        plt.close()
        print(f"- Saved plot:  {out_png1}")

        # ----- Plot 2: Scatter of similarity vs. rating_norm -----
        sim = recs["similarity"].to_numpy()
        # rating_norm isn't in output; reconstruct from score to keep the script simple:
        # score = alpha*sim + (1-alpha)*rnorm  ->  rnorm = (score - alpha*sim)/(1-alpha)
        if ALPHA < 1.0:
            rnorm = (scores - ALPHA * sim) / (1 - ALPHA)
        else:
            # degenerate case; just use zeros
            rnorm = np.zeros_like(sim)

        plt.figure()
        plt.scatter(sim, rnorm)
        # annotate top-n by score
        topn_idx = np.argsort(-scores)[:N]
        for i in topn_idx:
            plt.annotate(
                short_label(recs.loc[i, "title"]),
                (sim[i], rnorm[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        plt.xlabel("Content similarity")
        plt.ylabel("Normalized rating (inferred)")
        plt.title(f"Similarity vs. rating — {region}")
        plt.tight_layout()
        out_png2 = os.path.join(OUT_DIR, f"demo_recs_{region}_sim_vs_rating.png")
        plt.savefig(out_png2, dpi=160)
        plt.close()
        print(f"- Saved plot:  {out_png2}")

        # ----- Explanation (plain text) -----
        txt = f"""Demo notes for region '{region}'
- Favorites used: {', '.join(FAVORITES)}
- Plot 1 (bar): blended score ranks items. Higher is better.
- Plot 2 (scatter): shows trade-off — right = more content-similar, up = higher normalized rating.
  Top-right quadrant tends to be best; annotations mark top-3 by blended score.
"""
        txt_path = os.path.join(OUT_DIR, f"demo_recs_{region}_notes.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"- Saved notes: {txt_path}")

    print(
        "\nDone. Open the PNGs and CSV files in the 'demo_plots' folder to review the outputs."
    )


if __name__ == "__main__":
    main()
