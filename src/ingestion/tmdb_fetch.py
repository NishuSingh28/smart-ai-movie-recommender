import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API key
API_KEY = os.getenv("TMDB_API_KEY")

# TMDB base URL
BASE_URL = "https://api.themoviedb.org/3"


def fetch_popular_movies(page=1):

    url = f"{BASE_URL}/movie/popular"

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": page
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(
            f"TMDB API Error: {response.status_code}"
        )

    return response.json()


def create_movies_dataframe(num_pages=5):

    all_movies = []

    for page in range(1, num_pages + 1):

        data = fetch_popular_movies(page)

        for movie in data["results"]:

            movie_data = {
                "movie_id": movie["id"],
                "title": movie["title"],
                "overview": movie["overview"],
                "popularity": movie["popularity"],
                "release_date": movie["release_date"],
                "vote_average": movie["vote_average"],
                "vote_count": movie["vote_count"]
            }

            all_movies.append(movie_data)

    df = pd.DataFrame(all_movies)

    return df


if __name__ == "__main__":

    df = create_movies_dataframe(num_pages=10)

    print(df.head())

    df.to_csv(
        "data/raw/movies.csv",
        index=False
    )

    print(f"\nSaved {len(df)} movies successfully.")