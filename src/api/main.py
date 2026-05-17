from fastapi import FastAPI
from pydantic import BaseModel
from rapidfuzz import process

import pandas as pd
import numpy as np
import faiss


app = FastAPI()


# =========================
# LOAD DATA
# =========================

movies_df = pd.read_csv(
    "data/processed/enriched_movies.csv"
)

embeddings = np.load(
    "models/movie_embeddings.npy"
)

embeddings = embeddings.astype("float32")


# =========================
# BUILD FAISS INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# =========================
# REQUEST SCHEMA
# =========================

class RecommendationRequest(BaseModel):

    liked_movies: list[str]

    top_k: int = 5


def find_closest_movie(movie_name):

    titles = movies_df["title"].tolist()

    match = process.extractOne(
        movie_name,
        titles
    )

    if match:

        return match[0]

    return movie_name
    
# =========================
# API ENDPOINT
# =========================

@app.post("/recommend")

def recommend_movies(request: RecommendationRequest):

    liked_indices = []

    for movie in request.liked_movies:

        corrected_movie = find_closest_movie(movie)

        matches = movies_df[
            movies_df["title"].str.lower()
            == corrected_movie.lower()
        ]

        if not matches.empty:

            liked_indices.append(
                matches.index[0]
            )

    if not liked_indices:

        return {
            "error": "No valid movies found."
        }

    user_embedding = np.mean(
        embeddings[liked_indices],
        axis=0
    ).reshape(1, -1)

    distances, indices = index.search(
        user_embedding,
        request.top_k + len(liked_indices)
    )

    recommendations = []

    already_liked = set(
        request.liked_movies
    )

    for idx in indices[0]:

        title = movies_df.iloc[idx]["title"]

        if title not in already_liked:

            recommendations.append(title)

    return {
        "recommendations":
        recommendations[:request.top_k]
    }