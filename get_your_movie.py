"""
get_your_movie.py — content-based recommender (hybrid: scikit-learn + optional neural embeddings)

- Input: user's favorite movies (any region) and a target region: western / china / hongkong / japan.
- Output: N recommendations from the selected region.
- Pipeline (always available):
    ColumnTransformer( multiple TF-IDF vectorizers per field )
    -> TruncatedSVD (LSA)
    -> Normalizer (unit vectors -> cosine via dot product)
- Optional neural boost (if 'sentence-transformers' is installed):
    SentenceTransformer (multilingual) embeddings over text fields,
    combined with SVD similarity into a single hybrid score.
"""

import argparse
import os
import re
from typing import List, Optional
from unicodedata import normalize as ucnorm

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

# --------------------------- helpers ---------------------------


def _safe_str(x) -> str:
    """Return a safe string representation (empty string for NaN)."""
    return "" if pd.isna(x) else str(x)


def normalize_title(t: str) -> str:
    """Normalize movie titles for robust matching: NFKC -> lower -> strip punctuation -> single spaces."""
    if t is None:
        return ""
    t = ucnorm("NFKC", str(t)).lower()
    t = re.sub(r"\s+", " ", t).strip()
    # keep latin/digits/CJK, replace others with space
    t = re.sub(
        r"[^0-9a-z\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\uac00-\ud7af ]", " ", t
    )
    return re.sub(r"\s+", " ", t).strip()


def split_countries(s: str):
    """Split the `countries` cell by common separators used on Douban."""
    if not isinstance(s, str):
        return []
    return [
        p.strip()
        for p in re.split(r"\s*/\s*|,|、|\||；|;|\s{2,}", s)
        if p and p.strip()
    ]


# Region detector patterns (priority: Hong Kong → Japan → Mainland China → Western)
_HK_PAT = re.compile(r"(中国香港|中國香港|香港|hong\s*kong|\bhk\b)", re.I)
_JP_PAT = re.compile(r"(日本|japan)", re.I)
_CN_PAT = re.compile(
    r"(中国大陆|中國大陸|中国内地|中國內地|中国|china|mainland\s*china|prc|大陆)", re.I
)
_WEST_PAT = re.compile(
    r"(美国|united\s*states|usa|uk|united\s*kingdom|england|scotland|wales|ireland|"
    r"france|法国|germany|德国|italy|意大利|spain|西班牙|canada|加拿大|australia|澳大利亚|"
    r"new\s*zealand|新西兰|sweden|瑞典|norway|挪威|denmark|丹麦|finland|芬兰|iceland|冰岛|"
    r"austria|奥地利|switzerland|瑞士|portugal|葡萄牙|netherlands|荷兰|belgium|比利时|"
    r"greece|希腊|poland|波兰|czech|捷克|hungary|匈牙利|romania|罗马尼亚)",
    re.I,
)


def detect_region_group(country_cell: str) -> Optional[str]:
    """
    Infer region group from a `countries` cell.
    Priority: Hong Kong -> Japan -> Mainland China -> Western hints.
    Returns one of {'china','hongkong','japan','western'} or None if unknown.
    """
    if not isinstance(country_cell, str):
        return None
    tokens = split_countries(country_cell)
    for t in tokens:
        if _HK_PAT.search(t):
            return "hongkong"
        if _JP_PAT.search(t):
            return "japan"
        if _CN_PAT.search(t):
            return "china"
    if _WEST_PAT.search(country_cell):
        return "western"
    return None


def build_region_mask(df: pd.DataFrame, target_region: str) -> np.ndarray:
    """
    Build a boolean mask for rows that belong to `target_region` based on the countries column.
    If no countries-like column is found, no filtering is applied (all True).
    """
    target_region = target_region.lower()
    col = "countries" if "countries" in df.columns else None
    if col is None:
        for c in ("country", "region", "regions"):
            if c in df.columns:
                col = c
                break
    if col is None:
        return np.ones(len(df), dtype=bool)
    inferred = df[col].apply(detect_region_group)
    return (inferred == target_region).fillna(False).values


def compose_text_features(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate available textual columns into a single feature string per movie.
    """
    cols = [
        c
        for c in (
            "title",
            "genres",
            "directors",
            "actors",
            "screenwriters",
            "languages",
            "summary",
            "tags",
        )
        if c in df.columns
    ]
    if not cols:
        cols = ["title"]
    return (
        df[cols]
        .astype(str)
        .apply(lambda row: " | ".join(_safe_str(x) for x in row), axis=1)
    )


# --------------------------- model ---------------------------


class MovieRecommender:
    def __init__(self, df: pd.DataFrame):
        """
        Prepare a scikit-learn Pipeline with multi-field TF-IDF + LSA + unit normalization.
        If available, also build SentenceTransformer embeddings for a neural similarity boost.
        """
        self.df = df.copy()

        # Ensure text fields contain valid strings (no NaN) before TF-IDF / embeddings
        _text_fields = [
            "title",
            "genres",
            "tags",
            "directors",
            "actors",
            "screenwriters",
            "languages",
            "summary",
        ]
        for c in [c for c in _text_fields if c in self.df.columns]:
            self.df[c] = self.df[c].fillna("").astype(str)

        # Normalized rating [0,1]
        if "rating" in self.df.columns:
            r = pd.to_numeric(self.df["rating"], errors="coerce")
            rmin, rmax = r.min(), r.max()
            if pd.isna(rmin) or pd.isna(rmax) or rmax == rmin:
                self.df["_rating_norm"] = pd.Series(0.0, index=self.df.index)
            else:
                self.df["_rating_norm"] = ((r - rmin) / (rmax - rmin)).fillna(0.0)
        else:
            self.df["_rating_norm"] = 0.0

        # Normalized titles for fast lookup
        self.df["_norm_title"] = (
            self.df["title"].apply(normalize_title)
            if "title" in self.df.columns
            else ""
        )

        # ---------- Classical Feature Pipeline (always available) ----------
        char_ngram_title = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.99,
            sublinear_tf=True,
        )
        char_ngram_summary = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.99,
            sublinear_tf=True,
        )
        token_vec = TfidfVectorizer(
            token_pattern=r"[^,;/\|\s]+", lowercase=True, min_df=1, sublinear_tf=True
        )

        columns = self.df.columns

        transformers = []
        if "title" in columns:
            transformers.append(("title", char_ngram_title, "title"))
        if "genres" in columns:
            transformers.append(("genres", token_vec, "genres"))
        if "tags" in columns:
            transformers.append(("tags", token_vec, "tags"))
        if "directors" in columns:
            transformers.append(("directors", token_vec, "directors"))
        if "actors" in columns:
            transformers.append(("actors", token_vec, "actors"))
        if "screenwriters" in columns:
            transformers.append(("screenwriters", token_vec, "screenwriters"))
        if "languages" in columns:
            transformers.append(("languages", token_vec, "languages"))
        if "summary" in columns:
            transformers.append(("summary", char_ngram_summary, "summary"))

        self.coltx = ColumnTransformer(
            transformers=transformers, remainder="drop", verbose=False
        )

        # Pipeline: features -> SVD (LSA) -> L2-normalized vectors
        self.pipeline = Pipeline(
            steps=[
                ("features", self.coltx),
                ("svd", TruncatedSVD(n_components=128, random_state=42)),
                ("norm", Normalizer(copy=False)),
            ]
        )

        # Fit and transform the whole corpus to latent space (dense, L2-normalized)
        self.X = self.pipeline.fit_transform(self.df)

        # ---------- Optional Neural Embeddings (SentenceTransformer) ----------
        self.use_sbert = False
        self.sbert = None
        self.E = None  # (N, d) normalized embeddings if available
        self.NEURAL_WEIGHT = 0.65  # weight for neural similarity in hybrid (0..1)

        try:
            from sentence_transformers import SentenceTransformer

            model_name = os.environ.get(
                "GYM_SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.sbert = SentenceTransformer(model_name)
            # Build embed text from several columns to give context beyond the title.
            embed_cols = [
                c
                for c in ("title", "genres", "tags", "summary", "directors", "actors")
                if c in self.df.columns
            ]
            if not embed_cols:
                embed_cols = ["title"]
            self.df["_embed_text"] = (
                self.df[embed_cols].astype(str).agg(" | ".join, axis=1)
            )
            # Encode corpus
            emb = self.sbert.encode(
                self.df["_embed_text"].tolist(),
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Ensure float32 for memory; already L2-normalized
            self.E = emb.astype(np.float32, copy=False)
            self.use_sbert = True
        except Exception:
            # If sentence-transformers or torch is not available, we silently fall back to SVD-only.
            self.use_sbert = False
            self.sbert = None
            self.E = None

    def _match_title_to_index(self, user_title: str) -> Optional[int]:
        """
        Map a user-provided title to a dataset row index.
        Cascade: exact match on normalized title -> substring match -> similarity against pipeline features.
        """
        if "title" not in self.df.columns:
            return None
        nt = normalize_title(user_title)
        if not nt:
            return None

        # Exact match)
        exact = self.df.index[self.df["_norm_title"] == nt].tolist()
        if exact:
            return exact[0]
        # Substring match
        contains = self.df.index[
            self.df["_norm_title"].str.contains(re.escape(nt), na=False)
        ].tolist()
        if contains:
            return contains[0]
        # Fallback: build pseudo-row with only 'title' filled, transform via pipeline
        row = {c: "" for c in self.df.columns}
        row["title"] = user_title
        df_one = pd.DataFrame([row])[self.df.columns]
        vec = self.pipeline.transform(df_one)  # (1, k), L2-normalized
        sims = (self.X @ vec.T).ravel()  # cosine via dot product
        idx = int(np.argmax(sims))
        return idx if np.max(sims) > 0 else None

    def _encode_favorites(self, favorite_titles: List[str]) -> dict:
        """
        Encode favorites into both latent (SVD) vector and, if available, neural embedding.
        Returns dict with keys 'svd' (np.ndarray) and 'neural' (np.ndarray or None).
        """
        rows = []
        for t in favorite_titles:
            r = {c: "" for c in self.df.columns}
            r["title"] = t
            rows.append(r)
        df_fav = pd.DataFrame(rows)[self.df.columns]
        V = self.pipeline.transform(df_fav)  # (m, k), already normalized
        svd_pref = V.mean(axis=0)
        svd_norm = np.linalg.norm(svd_pref)
        if svd_norm > 0:
            svd_pref = svd_pref / svd_norm

        # Neural-side: encode just the raw favorite titles (robust across languages)
        neural_pref = None
        if self.use_sbert and self.sbert is not None:
            try:
                neural_pref = self.sbert.encode(
                    favorite_titles,
                    batch_size=16,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                neural_pref = neural_pref.mean(axis=0)
                nrm = np.linalg.norm(neural_pref)
                if nrm > 0:
                    neural_pref = neural_pref / nrm
            except Exception:
                neural_pref = None

        return {"svd": svd_pref, "neural": neural_pref}

    def recommend(
        self,
        favorite_titles: List[str],
        target_region: str,
        n: int = 10,
        alpha: float = 0.85,
        diversify_by_year: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend `n` movies from `target_region` given a list of favorite titles.

        - Encode favorites via SVD (always) and SentenceTransformer (if available).
        - Compute cosine similarities for candidates in each space; combine into a hybrid:
              sim_hybrid = (1 - NEURAL_WEIGHT) * sim_svd + NEURAL_WEIGHT * sim_neural   (if neural available)
              sim_hybrid = sim_svd                                                     (otherwise)
        - Final score: score = alpha * sim_hybrid + (1 - alpha) * normalized_rating.
        - Optional light diversification by year.
        """
        target_region = target_region.lower().strip()
        if target_region not in {"western", "china", "hongkong", "japan"}:
            raise ValueError(
                "target_region must be one of: western, china, hongkong, japan"
            )

        # Region candidates
        mask = build_region_mask(self.df, target_region)
        candidate_idx = np.where(mask)[0]
        if len(candidate_idx) == 0:
            raise RuntimeError(
                f"No movies detected for region='{target_region}'. Check your 'countries' data."
            )

        # Try to locate favorites to exclude them if they fall into candidates
        matched_indices = []
        for t in favorite_titles:
            idx = self._match_title_to_index(t)
            if idx is not None:
                matched_indices.append(idx)

        # Encode favorites (svd + optional neural)
        pref = self._encode_favorites(favorite_titles)
        pref_svd = pref["svd"]
        pref_neural = pref["neural"]

        # Similarity in SVD space
        Xcand = self.X[candidate_idx]  # (Ncand, k)
        sims_svd = (Xcand @ pref_svd).ravel()

        # Similarity in neural space (if available)
        if self.use_sbert and self.E is not None and pref_neural is not None:
            Ecand = self.E[candidate_idx]  # (Ncand, d), already normalized
            sims_neural = (Ecand @ pref_neural).ravel()
            sims = (
                1.0 - self.NEURAL_WEIGHT
            ) * sims_svd + self.NEURAL_WEIGHT * sims_neural
        else:
            sims = sims_svd

        # Blend with normalized rating
        rnorm = self.df["_rating_norm"].values[candidate_idx]
        score = alpha * sims + (1.0 - alpha) * rnorm

        # Exclude favorites themselves
        exclude = set(matched_indices)
        order = np.argsort(-score)
        ranked_idx = [
            candidate_idx[i] for i in order if candidate_idx[i] not in exclude
        ]

        # Light diversification by year
        if diversify_by_year and "release_date" in self.df.columns:
            YEAR_BUCKET = 5
            MAX_PER_BUCKET = 2
            years = []
            kept = []
            for ridx in ranked_idx:
                y = None
                s = str(self.df.at[ridx, "release_date"])
                m = re.search(r"(\d{4})", s)
                if m:
                    y = int(m.group(1))
                if y is None:
                    kept.append(ridx)
                else:
                    bucket = (y // YEAR_BUCKET) * YEAR_BUCKET
                    if years.count(bucket) < MAX_PER_BUCKET:
                        years.append(bucket)
                        kept.append(ridx)
                if len(kept) >= n * 3:
                    break
            ranked_idx = kept

        top_idx = ranked_idx[:n]

        out_cols = [
            c
            for c in (
                "title",
                "rating",
                "total_ratings",
                "genres",
                "countries",
                "directors",
                "actors",
                "release_date",
                "languages",
                "runtime",
                "summary",
                "link",
                "tags",
            )
            if c in self.df.columns
        ]

        # Transparency metrics
        idx_to_sim = {
            candidate_idx[i]: float(sims[i]) for i in range(len(candidate_idx))
        }
        idx_to_score = {
            candidate_idx[i]: float(score[i]) for i in range(len(candidate_idx))
        }

        result = self.df.loc[top_idx, out_cols].copy()
        result.insert(0, "score", [idx_to_score.get(i, np.nan) for i in top_idx])
        result.insert(1, "similarity", [idx_to_sim.get(i, np.nan) for i in top_idx])
        return result.reset_index(drop=True)


# --------------------------- IO ---------------------------


def load_datasets(base_dir: str = ".") -> pd.DataFrame:
    """Load the master dataset `douban_all_movies.csv` from `base_dir`."""
    p = os.path.join(base_dir, "douban_all_movies.csv")
    df = pd.read_csv(p, encoding="utf-8")
    if df is None or len(df) == 0:
        raise FileNotFoundError("douban_all_movies.csv not found or empty.")
    return df


# --------------------------- public API ---------------------------


def recommend_movies(
    favorite_titles: List[str],
    target_region: str,
    n: int = 10,
    data_dir: str = ".",
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Public API: load data, build recommender, and return top-N for the given region."""
    df = load_datasets(data_dir)
    rec = MovieRecommender(df)
    return rec.recommend(favorite_titles, target_region, n=n, alpha=alpha)


# --------------------------- CLI ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Get-Your-Movie: content-based recommender (hybrid: SVD + optional neural)."
    )
    parser.add_argument(
        "--favorites",
        type=str,
        required=True,
        help="Semicolon-separated list: 'In Bruges; Amélie; Un prophète'",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        choices=["western", "china", "hongkong", "japan"],
        help="Which region to recommend from.",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="How many movies to recommend."
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Directory with douban_all_movies.csv"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Blend weight: alpha*similarity + (1-alpha)*rating.",
    )
    args = parser.parse_args()

    favs = [f.strip() for f in args.favorites.split(";") if f.strip()]
    df_out = recommend_movies(
        favs, args.region, n=args.n, data_dir=args.data_dir, alpha=args.alpha
    )

    pd.set_option("display.max_colwidth", 80)
    print("=" * 88)
    print(
        f"Recommendations ({args.region}, top {args.n}) for favorites: {', '.join(favs)}"
    )
    print("=" * 88)
    cols = [
        c
        for c in (
            "title",
            "rating",
            "genres",
            "countries",
            "directors",
            "actors",
            "release_date",
            "summary",
            "link",
            "similarity",
            "score",
        )
        if c in df_out.columns
    ]
    print(df_out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
