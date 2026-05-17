import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading enriched dataset...")

df = pd.read_csv(
    "data/processed/enriched_movies.csv"
)

# Fill missing tags
df["tags"] = df["tags"].fillna("")

print(df.head())

print("\nLoading transformer model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("\nGenerating rich semantic embeddings...")

embeddings = model.encode(
    df["tags"].tolist(),
    show_progress_bar=True
)

print("\nEmbedding shape:")
print(embeddings.shape)

# Save embeddings
np.save(
    "models/movie_embeddings.npy",
    embeddings
)

print("\nEmbeddings saved successfully.")