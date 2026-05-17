import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

# Load movies dataset
movies_df = pd.read_csv(
    "data/raw/movies.csv"
)

# Handle missing overviews
movies_df["overview"] = movies_df["overview"].fillna("")


def fetch_movie_details(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}"

    params = {
        "api_key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    return response.json()


def fetch_movie_credits(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}/credits"

    params = {
        "api_key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    return response.json()


def fetch_movie_keywords(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}/keywords"

    params = {
        "api_key": API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    return response.json()


enriched_movies = []

for idx, row in movies_df.iterrows():

    movie_id = row["movie_id"]

    print(f"Processing: {row['title']}")

    # Fetch TMDB data
    details = fetch_movie_details(movie_id)

    credits = fetch_movie_credits(movie_id)

    keywords_data = fetch_movie_keywords(movie_id)

    # Genres
    genres = [
        genre["name"]
        for genre in details.get("genres", [])
    ]

    # Top 3 cast members
    cast = [
        actor["name"]
        for actor in credits.get("cast", [])[:3]
    ]

    # Director
    director = ""

    for crew_member in credits.get("crew", []):

        if crew_member.get("job") == "Director":

            director = crew_member.get("name", "")

            break

    # Keywords
    keywords = [
        keyword["name"]
        for keyword in keywords_data.get("keywords", [])
    ]

    # Create rich semantic tags safely
    tags = " ".join([
        str(row["overview"]),
        " ".join(map(str, genres)),
        " ".join(map(str, cast)),
        str(director),
        " ".join(map(str, keywords))
    ])

    enriched_movies.append({
        "movie_id": movie_id,
        "title": row["title"],
        "tags": tags
    })


# Create dataframe
enriched_df = pd.DataFrame(enriched_movies)

# Save enriched dataset
enriched_df.to_csv(
    "data/processed/enriched_movies.csv",
    index=False
)

print("\nEnriched dataset saved successfully.")