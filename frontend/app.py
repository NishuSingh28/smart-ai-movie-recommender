import streamlit as st
import requests
import os
import pandas as pd

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Load movie dataset
movies_df = pd.read_csv(
    "data/processed/enriched_movies.csv"
)

movie_titles = sorted(
    movies_df["title"].unique().tolist()
)

st.set_page_config(
    page_title="Smart AI Movie Recommender",
    layout="wide"
)

st.title("🎬 Smart AI Movie Recommender")

st.write(
    "Select movies you like and get AI-powered recommendations."
)

# Movie selector
movies = st.multiselect(
    "Select Movies You Like",
    movie_titles
)

# Number of recommendations
top_k = st.slider(
    "Number of Recommendations",
    1,
    10,
    5
)

if st.button("Recommend"):

    liked_movies = movies

    response = requests.post(
        "http://127.0.0.1:8000/recommend",
        json={
            "liked_movies": liked_movies,
            "top_k": top_k
        }
    )

    data = response.json()

    st.subheader("Recommended Movies")

    if "recommendations" in data:

        cols = st.columns(5)

        for idx, movie in enumerate(
            data["recommendations"]
        ):

            poster_response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "query": movie
                }
            )

            poster_data = poster_response.json()

            poster_path = None

            if poster_data["results"]:

                poster_path = poster_data[
                    "results"
                ][0].get("poster_path")

            with cols[idx % 5]:

                if poster_path:

                    st.image(
                        TMDB_IMAGE_BASE + poster_path
                    )

                st.write(movie)

    else:

        st.error(data["error"])