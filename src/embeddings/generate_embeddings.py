import pandas as pd
from sentence_transformers import SentenceTransformer

# Load movie dataset
df = pd.read_csv("data/raw/movies.csv")

# Fill missing overviews
df["overview"] = df["overview"].fillna("")

# Load transformer model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Generate embeddings
embeddings = model.encode(
    df["overview"].tolist(),
    show_progress_bar=True
)

print("Embedding shape:")
print(embeddings.shape)