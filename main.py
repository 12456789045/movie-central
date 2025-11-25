

import json
import os
import io
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie-central")

# --- Config ---
MOVIES_JSON = "new.json"   # place your 100-movie JSON here (list of movies)
EMBED_MODEL = "all-MiniLM-L6-v2"
SYNTHETIC_HISTORY_LEN = 5     # number of historical ratings to synthesize if missing
RATING_MIN, RATING_MAX = 0.0, 10.0

# --- FastAPI app ---
app = FastAPI(title="Movie Central — Backend (semantic + prediction + graphs)")

# --- Load / initialize movies ---
def load_movies(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please place your movies.json in the backend folder.")
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    # Ensure fields and add synthetic rating history if missing
    rng = np.random.RandomState(42)
    for i, m in enumerate(movies):
        # normalize minimal keys
        m.setdefault("id", i + 1)
        m.setdefault("title", f"Untitled #{m['id']}")
        m.setdefault("genre", "unknown")
        m.setdefault("description", "")
        m.setdefault("trailer", "")
        # current_rating fallback
        if "current_rating" not in m:
            # generate a reasonable base rating 6.0-8.5 depending on genre random
            m["current_rating"] = round(float(6.0 + rng.rand() * 2.5), 2)
        # rating_history fallback (list of floats)
        if "rating_history" not in m or not isinstance(m["rating_history"], list) or len(m["rating_history"]) < 2:
            # create synthetic history around current_rating with small noise
            base = m["current_rating"]
            hist = (base + rng.randn(SYNTHETIC_HISTORY_LEN) * 0.35).clip(RATING_MIN, RATING_MAX)
            # make it smoother / monotonic-ish by cumulative sum of small changes
            hist = np.round(hist, 2).tolist()
            m["rating_history"] = hist
    return movies

try:
    movies = load_movies(MOVIES_JSON)
    logger.info(f"Loaded {len(movies)} movies from {MOVIES_JSON}")
except FileNotFoundError as e:
    # Surface a clear message for the user, re-raise as HTTPException at runtime.
    logger.error(str(e))
    movies = []


# --- Load embedding model and precompute embeddings ---
logger.info("Loading sentence-transformer model (this may take a moment)...")
model = SentenceTransformer(EMBED_MODEL)

def build_corpus_embeddings(movie_list: List[dict]):
    texts = [m.get("description", "") for m in movie_list]
    # encode as numpy arrays (convert_to_numpy)
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

if movies:
    movie_embeddings = build_corpus_embeddings(movies)
else:
    movie_embeddings = None


# --- Utility: predict next rating from rating_history ---
def predict_next_rating(rating_history: List[float]) -> float:
    """
    Simple linear forecast: fit a degree-1 polynomial (line) on rating history indices and predict next index.
    If prediction is outside [RATING_MIN, RATING_MAX], clamp it.
    Returns a float rounded to 2 decimals.
    """
    if not rating_history or len(rating_history) < 2:
        # not enough data: return last known or a default
        return round(float(rating_history[-1]) if rating_history else 6.0, 2)
    y = np.array(rating_history, dtype=float)
    x = np.arange(len(y))
    # Fit line y = a*x + b
    coeffs = np.polyfit(x, y, deg=1)
    next_x = len(y)
    pred = np.polyval(coeffs, next_x)
    pred = float(np.clip(pred, RATING_MIN, RATING_MAX))
    return round(pred, 2)


# --- Pydantic request models ---
class RecommendRequest(BaseModel):
    user_text: str
    top_k: Optional[int] = 10
    genre: Optional[str] = None   # optional filter


# --- Endpoints ---


@app.get("/genres")
def get_genres():
    """Return available genres present in the dataset."""
    genre_set = sorted({m.get("genre", "unknown") for m in movies})
    return {"genres": genre_set}


@app.get("/movies")
def get_movies(genre: Optional[str] = Query(None, description="Filter by genre")):
    """Return list of movies, optionally filtered by genre. Returns first 50 by default to keep responses reasonable."""
    if not movies:
        raise HTTPException(status_code=500, detail=f"{MOVIES_JSON} not found on server.")
    results = movies
    if genre:
        results = [m for m in movies if str(m.get("genre", "")).lower() == genre.lower()]
    # return up to 100 (or all)
    return {"count": len(results), "movies": results[:200]}


@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    """Get details for a single movie by id."""
    for m in movies:
        if int(m.get("id")) == int(movie_id):
            # include predicted next rating
            predicted = predict_next_rating(m.get("rating_history", []))
            m_copy = dict(m)
            m_copy["predicted_next_rating"] = predicted
            return m_copy
    raise HTTPException(status_code=404, detail="Movie not found")


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Recommend movies based on semantic similarity between user's text and movie descriptions.
    Optional: filter by genre and limit top_k.
    Returns similarity (0..1), predicted next rating, and movie metadata.
    """
    if not movies:
        raise HTTPException(status_code=500, detail=f"{MOVIES_JSON} not found on server.")
    if not req.user_text or not req.user_text.strip():
        raise HTTPException(status_code=400, detail="user_text is empty")

    # Filter by genre if provided
    candidates = movies
    candidate_indices = np.arange(len(movies))
    if req.genre:
        mask = [str(m.get("genre", "")).lower() == req.genre.lower() for m in movies]
        indices = [i for i, v in enumerate(mask) if v]
        if not indices:
            return {"recommended_movies": []}
        candidates = [movies[i] for i in indices]
        candidate_indices = np.array(indices)

    # compute embedding for user_text
    q_emb = model.encode([req.user_text], convert_to_tensor=True)
    # compute cosine similarities with precomputed movie_embeddings (but only for candidates)
    # extract candidate embeddings
    if movie_embeddings is None:
        raise HTTPException(status_code=500, detail="Embeddings not initialized.")
    cand_embeddings = movie_embeddings[candidate_indices.tolist()]
    sims = util.cos_sim(q_emb, cand_embeddings).numpy().flatten()  # shape (n_candidates,)

    # build results list
    results = []
    for idx_local, sim in enumerate(sims):
        global_idx = int(candidate_indices[idx_local])
        movie = movies[global_idx]
        predicted = predict_next_rating(movie.get("rating_history", []))
        results.append({
            "movie_id": movie.get("id"),
            "title": movie.get("title"),
            "genre": movie.get("genre"),
            "description": movie.get("description"),
            "trailer": movie.get("trailer"),
            "current_rating": float(movie.get("current_rating", 0.0)),
            "predicted_next_rating": predicted,
            "similarity": float(sim)   # 0..1
        })

    # sort by similarity desc and take top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[: req.top_k if req.top_k and req.top_k > 0 else 10]

    return {"query": req.user_text, "recommended_movies": top}


@app.post("/recommend/graph")
def recommend_graph(req: RecommendRequest):
  
    rec = recommend(req)
    top = rec.get("recommended_movies", [])
    if not top:
        raise HTTPException(status_code=400, detail="No recommendations to graph.")

    titles = [f"{r['title'][:30]}" for r in top]
    scores = [r["similarity"] for r in top]

    # create bar chart
    plt.figure(figsize=(max(6, len(titles)*0.6), 4))
    bars = plt.barh(range(len(titles))[::-1], scores[::-1])  # horizontal, highest on top
    plt.yticks(range(len(titles)), labels=titles[::-1])
    plt.xlabel("Similarity (0..1)")
    plt.xlim(0, 1)
    plt.title("Top movie similarities")

    # annotate bars with percent
    for i, b in enumerate(bars):
        w = b.get_width()
        plt.text(w + 0.01, b.get_y() + b.get_height()/2, f"{w:.2f}", va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/top-trailers")
def top_trailers(genre: Optional[str] = None, limit: int = 10):
    """Return top trailers (first N) optionally filtered by genre."""
    results = movies
    if genre:
        results = [m for m in movies if str(m.get("genre","")).lower() == genre.lower()]
    return {"trailers": [{"id": m["id"], "title": m["title"], "trailer": m.get("trailer")} for m in results[:limit]]}


# Simple health root
@app.get("/")
def root():
    return {"status": "Movie Central API running (semantic + prediction + graphs)"}


# --- Start ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

