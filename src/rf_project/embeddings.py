from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .io_utils import read_table, save_table


def run_news_embeddings(cfg: dict) -> Path | None:
    """Кодирует уникальные тексты из news_clean в векторы (sentence-transformers). Возвращает путь или None если выключено."""
    emb = cfg.get("embeddings") or {}
    if not emb.get("enabled", False):
        return None

    interim = Path(cfg["paths"]["interim_dir"])
    news_path = interim / "news_clean.parquet"
    out_path = interim / "news_embeddings_wide.parquet"

    news = read_table(news_path)
    model_id = emb.get("model_id", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    batch = int(emb.get("batch_size", 16))
    tfidf_dim = int(emb.get("tfidf_svd_dim", 64))

    texts = news["text_clean"].astype(str).tolist()
    ids = news["news_id"].astype(str).tolist()

    embeddings = None
    dim = 0
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        dim = model.get_sentence_embedding_dimension()
        embeddings = model.encode(texts, batch_size=batch, show_progress_bar=False)
    except Exception as e:
        print(f"Embedding fallback to TFIDF+SVD: {e}", flush=True)
        vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(texts)
        dim = min(tfidf_dim, max(8, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=42)
        embeddings = svd.fit_transform(X)

    cols = {f"emb_{i}": embeddings[:, i] for i in range(dim)}
    out = pd.DataFrame({"news_id": ids, **cols})
    save_table(out, out_path)
    return out_path


def load_news_embedding_columns(interim: Path) -> pd.DataFrame | None:
    path = interim / "news_embeddings_wide.parquet"
    if not path.exists():
        return None
    return read_table(path)
