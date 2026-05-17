import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================
# LOAD DATA
# =========================

print("Loading enriched movie dataset...")

movies_df = pd.read_csv(
    "data/processed/enriched_movies.csv"
)

movies_df["tags"] = movies_df["tags"].fillna("")


# =========================
# LOAD EMBEDDINGS
# =========================

print("Loading embeddings...")

embeddings = np.load(
    "models/movie_embeddings.npy"
)

embeddings = embeddings.astype("float32")


# =========================
# BUILD FAISS INDEX
# =========================

print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Indexed {index.ntotal} movies.")


# =========================
# CREATE USER PROFILE
# =========================

liked_movies = [
    "Interstellar",
    "Inception",
    "Blade Runner 2049"
]

print("\nCreating user profile...")


liked_indices = []

for movie in liked_movies:

    matches = movies_df[
        movies_df["title"].str.lower()
        == movie.lower()
    ]

    if not matches.empty:

        liked_indices.append(
            matches.index[0]
        )


user_embedding = np.mean(
    embeddings[liked_indices],
    axis=0
).reshape(1, -1)


# =========================
# SEARCH SIMILAR MOVIES
# =========================

print("Generating personalized recommendations...")

distances, indices = index.search(
    user_embedding,
    10
)

print("\nRecommended Movies:\n")

recommended_titles = set()

for idx in indices[0]:

    title = movies_df.iloc[idx]["title"]

    if title not in liked_movies:

        if title not in recommended_titles:

            recommended_titles.add(title)

            print(title)