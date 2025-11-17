from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from surprise import dump
import pandas as pd
import os

MODEL_PATH = "/mnt/models/checkpoint/svd_latest.pkl"
USER_MAP_PATH = "/mnt/datasets/prepared/user_map.parquet"
MOVIE_MAP_PATH = "/mnt/datasets/prepared/movie_map.parquet"
MOVIE_TITLES_PATH = "/mnt/datasets/raw/movie_titles.csv"

# ========== FASTAPI APP ==========
app = FastAPI(title="AI Movie Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD MODEL ==========
try:
    _, algo = dump.load(MODEL_PATH)
    user_map_df = pd.read_parquet(USER_MAP_PATH)
    movie_map_df = pd.read_parquet(MOVIE_MAP_PATH)
    movie_df = pd.read_csv(
        MOVIE_TITLES_PATH, encoding="latin1", usecols=[0, 1, 2], header=None
    )
    movie_df.columns = ["movie_id", "year", "name"]

    user_map = dict(zip(user_map_df.index.astype(str), user_map_df["index"]))
    user_map_rev = {v: k for k, v in user_map.items()}
    movie_map = dict(zip(movie_map_df.index.astype(str), movie_map_df["index"]))
    movie_map_rev = {v: k for k, v in movie_map.items()}
    all_movie_indices = list(movie_map.values())
    print("Model and mappings loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    algo = None

# ========== API MODEL ==========
class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if algo is None:
        return {"error": "Model not loaded"}

    user_id = req.user_id
    k = req.top_k

    if user_id not in user_map:
        return {"user_id": user_id, "recommendations": []}

    u_idx = user_map[user_id]
    preds = [(m_idx, algo.predict(u_idx, m_idx).est) for m_idx in all_movie_indices]
    top_movies_idx = sorted(preds, key=lambda x: x[1], reverse=True)[:k]

    recommendations = []
    for m_idx, score in top_movies_idx:
        movie_id = movie_map_rev[m_idx]
        title_row = movie_df[movie_df["movie_id"] == int(movie_id)]
        title = title_row["name"].values[0] if not title_row.empty else "Unknown"
        recommendations.append({"movie_id": movie_id, "title": title, "score": score})

    return {"user_id": user_id, "recommendations": recommendations}

# ========== SERVE FRONTEND ==========
@app.get("/")
def serve_index():
    return FileResponse("app/static/index.html")
