# Get Your Movie 🎬

A **content-based movie recommender** for a Douban-style dataset.  
You give it your favorite movies and a target region, it recommends top-N titles from that region — fast and explainably. ✨

---

## What’s in the box

- `get_your_movie.py` — the recommender (CLI + importable API)
- `main.py` — minimal API usage example
- `demo_recommender_plots.py` — super-simple demo that makes a couple of charts
- `requirements.txt` — dependencies
- `douban_all_movies.csv` — the master dataset (required)

> ℹ️ The code looks **only** for the master dataset file `douban_all_movies.csv` in your working directory.

---

## Install

```bash
# (optional) create and activate a virtual env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt
```

Recommended Python: **3.9+**

---

## Dataset

Place **`douban_all_movies.csv`** next to the scripts.  
Expected columns (the more, the better):

```
title, rating, total_ratings, directors, actors, screenwriters,
release_date, genres, countries, languages, runtime, summary, link, tags
```

At minimum, `title` and `countries` should be present for best results.

---

## Quick Start

### 1) CLI (one-liner)

Recommend 10 **Western** movies for a user who likes *In Bruges* and *Amélie*:

```bash
python get_your_movie.py \
  --favorites "In Bruges; Amélie" \
  --region western \
  --n 10 \
  --data_dir .
```

Other valid regions: `china`, `hongkong`, `japan`.

Tune the score blend (content vs rating):

```bash
# make it more content-driven
python get_your_movie.py --favorites "Amélie; The Lives of Others" --region japan --n 12 --alpha 0.9
```

Help:

```bash
python get_your_movie.py -h
```

### 2) Python API (via `main.py`)

`main.py` (example):

```python
from get_your_movie import recommend_movies

recs = recommend_movies(
    favorite_titles=["Amélie", "Das Leben der Anderen", "La vita è bella"],
    target_region="hongkong",
    n=10,
    alpha=0.85
)

print(recs[["title","rating","genres","countries","similarity","score"]])
```

Run:

```bash
python main.py
```

---

## Demo (plots) 📊

Generate a small report with **two charts per region** and CSVs:

```bash
python demo_recommender_plots.py
```

Outputs go to the `demo_plots/` folder:
- `*_score_bars.png` — blended score bars
- `*_sim_vs_rating.png` — similarity vs normalized rating (top-3 annotated)
- `*.csv` — the raw recommendation tables
- `*.txt` — short notes

---

## How it works (short)

- **Features:** concatenation of text fields (title, genres, people, summary, tags) → **TF-IDF** (character n-grams 2–5)  
- **User profile:** average TF-IDF vector of favorites (L2-normalized)  
- **Scoring:** `score = alpha * cosine_similarity + (1 - alpha) * normalized_rating`  
- **Region filter:** inferred from `countries` (supports Mainland China, Hong Kong, Japan, Western hints)

---

## Parameters you may care about

- `--favorites` (CLI) / `favorite_titles` (API): semicolon-separated titles (any mix of languages)
- `--region` / `target_region`: `western | china | hongkong | japan`
- `--n`: number of recommendations
- `--alpha`: weight for content similarity (use `0.8–0.95` for more “taste-based” results)
- `--data_dir`: folder containing `douban_all_movies.csv`

---

Happy recommending! 🍿
