import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD

from surprise.model_selection import train_test_split

from surprise import accuracy

print("Loading ratings dataset...")

ratings = pd.read_csv(
    "data/external/ml-latest-small/ratings.csv"
)

print(ratings.head())

# Define rating scale
reader = Reader(
    rating_scale=(0.5, 5.0)
)

# Load dataset into Surprise
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

# Train-test split
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("\nTraining SVD model...")

model = SVD()

model.fit(trainset)

print("\nGenerating predictions...")

predictions = model.test(testset)

print("\nEvaluating model...")

rmse = accuracy.rmse(predictions)

print(f"\nFinal RMSE: {rmse}")