# Get Your Movie 🎬

A **content-based movie recommender** for a Douban-style dataset.  
Give it your favorite movies and a target region — it returns top‑N titles from that region, fast and explainably. ✨

---

## What’s in the box

- `get_your_movie.py` — the recommender (CLI + importable API)
- `main.py` — minimal API usage example
- `demo_recommender_plots.py` — tiny demo that makes a couple of charts
- `requirements.txt` — dependencies
- `douban_all_movies.csv` — the master dataset (required)

> ℹ️ The code looks **only** for `douban_all_movies.csv` in your working directory.

---

## Install

```bash
# create and activate a virtual env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### (Optional) Neural boost
To enable the neural part of the hybrid recommender:
```bash
pip install sentence-transformers torch
```
If these aren’t installed, the system gracefully falls back to the classic (non-neural) pipeline.

---

## Mainland China setup (Hugging Face mirror) 🇨🇳

If you’re in China and model downloads from `huggingface.co` are blocked, set a mirror before the **first** run that uses sentence-transformers:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
# now run your script, e.g.:
python main.py
```

Windows (PowerShell):
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python main.py
```

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

### 1) CLI

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
python get_your_movie.py --favorites "教父; The godfather 2" --region japan --n 12 --alpha 0.9
```

Help:
```bash
python get_your_movie.py -h
```

### 2) Python API (via `main.py`)

```python
from get_your_movie import recommend_movies

recs = recommend_movies(
    favorite_titles=["Amélie", "Das Leben der Anderen", "La vita è bella"],
    target_region="hongkong",
    n=10,
    data_dir=".",
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
Outputs go to `demo_plots/`:
- `*_score_bars.png` — blended score bars
- `*_sim_vs_rating.png` — similarity vs normalized rating (top‑3 annotated)
- `*.csv` — raw recommendation tables
- `*.txt` — short notes

> 🈶 If you see “missing glyph” warnings or squares for CJK titles, install CJK fonts (e.g. *Noto Sans CJK*) and set a CJK font family in Matplotlib.

---

## How it works (short)

- **Classic side:** concatenate textual fields → TF‑IDF (char n‑grams 2–5 and token features) → **LSA** → L2‑normalize.  
- **Neural side (optional):** multilingual sentence embeddings on `title | genres | tags | summary | …`.  
- **Hybrid similarity:**  
  `sim_hybrid = (1 - w) * sim_svd + w * sim_neural` (if neural available; otherwise `sim_svd`),  
  `score = alpha * sim_hybrid + (1 - alpha) * rating_norm`.

---

## Parameters you may care about

- `--favorites` (CLI) / `favorite_titles` (API): semicolon‑separated titles (any language mix)
- `--region` / `target_region`: `western | china | hongkong | japan`
- `--n`: number of recommendations
- `--alpha`: weight for content similarity (use `0.8–0.95` for more “taste‑based” results)
- `--data_dir`: folder containing `douban_all_movies.csv`

---

## Troubleshooting 🧰

- **Model downloads blocked in China** → set the mirror:
  ```bash
  export HF_ENDPOINT="https://hf-mirror.com"
  ```
- **“No movies detected for region=…”** → check `countries` formatting in the CSV (Hong Kong is often `中国香港` / `香港`).
- **“None of the favorite titles matched the dataset.”** → try local titles or aliases; with the neural boost installed, similarity should remain meaningful even with different languages.
- **Weird characters / squares on plots** → install a CJK-capable font and set it in Matplotlib.

---

Happy recommending! 🍿
