import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("Loading movies...")

df = pd.read_csv("data/raw/movies.csv")

df["overview"] = df["overview"].fillna("")

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Generating embeddings...")

embeddings = model.encode(
    df["overview"].tolist(),
    show_progress_bar=True
)

print("Creating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Indexed {index.ntotal} movies.")


def recommend(movie_title, top_k=5):

    movie_idx = df[
        df["title"] == movie_title
    ].index[0]

    query_vector = embeddings[movie_idx].reshape(1, -1)

    distances, indices = index.search(
        query_vector,
        top_k + 1
    )

    print(f"\nRecommendations for {movie_title}:\n")

    for idx in indices[0][1:]:

        print(df.iloc[idx]["title"])


recommend("Interstellar")