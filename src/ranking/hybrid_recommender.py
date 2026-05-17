import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from surprise import Dataset
from surprise import Reader
from surprise import SVD

# =========================
# LOAD ENRICHED MOVIES
# =========================

print("Loading enriched movies...")

movies_df = pd.read_csv(
    "data/processed/enriched_movies.csv"
)

movies_df["tags"] = movies_df["tags"].fillna("")


# =========================
# LOAD EMBEDDINGS
# =========================

print("Loading semantic embeddings...")

embeddings = np.load(
    "models/movie_embeddings.npy"
)

embeddings = embeddings.astype("float32")


# =========================
# CREATE FAISS INDEX
# =========================

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Indexed {index.ntotal} movies.")


# =========================
# LOAD RATINGS DATA
# =========================

print("Loading MovieLens ratings...")

ratings = pd.read_csv(
    "data/external/ml-latest-small/ratings.csv"
)

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset = data.build_full_trainset()


# =========================
# TRAIN SVD MODEL
# =========================

print("Training collaborative filtering model...")

svd_model = SVD()

svd_model.fit(trainset)

print("SVD training completed.")


# =========================
# HYBRID RECOMMENDER
# =========================

def hybrid_recommend(movie_title, top_k=5):

    movie_matches = movies_df[
        movies_df["title"].str.lower()
        == movie_title.lower()
    ]

    if movie_matches.empty:

        print("Movie not found.")

        return

    movie_idx = movie_matches.index[0]

    query_vector = embeddings[
        movie_idx
    ].reshape(1, -1)

    distances, indices = index.search(
        query_vector,
        top_k * 3
    )

    recommendations = []

    for idx in indices[0][1:]:

        candidate_movie = movies_df.iloc[idx]

        candidate_title = candidate_movie["title"]

        semantic_score = 1 / (
            1 + distances[0][
                list(indices[0]).index(idx)
            ]
        )

        # Dummy user ID for demo
        user_id = 1

        movie_id = candidate_movie["movie_id"]

        try:

            collaborative_score = svd_model.predict(
                user_id,
                movie_id
            ).est

        except:

            collaborative_score = 0

        hybrid_score = (
            0.7 * semantic_score
            +
            0.3 * collaborative_score
        )

        recommendations.append({
            "title": candidate_title,
            "semantic_score": semantic_score,
            "collaborative_score": collaborative_score,
            "hybrid_score": hybrid_score
        })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    print(f"\nHybrid Recommendations for {movie_title}:\n")

    for rec in recommendations[:top_k]:

        print(
            f"{rec['title']} "
            f"(Hybrid Score: {rec['hybrid_score']:.4f})"
        )


# =========================
# TEST
# =========================

hybrid_recommend("Interstellar")