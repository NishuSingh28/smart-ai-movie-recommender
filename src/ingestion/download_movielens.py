import os
import zipfile
import requests

DATASET_URL = (
    "https://files.grouplens.org/datasets/"
    "movielens/ml-latest-small.zip"
)

ZIP_PATH = "data/external/ml-latest-small.zip"

EXTRACT_PATH = "data/external/"


def download_dataset():

    os.makedirs(
        "data/external",
        exist_ok=True
    )

    print("Downloading MovieLens dataset...")

    response = requests.get(DATASET_URL)

    with open(ZIP_PATH, "wb") as file:

        file.write(response.content)

    print("Download completed.")


def extract_dataset():

    print("Extracting dataset...")

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_ref:

        zip_ref.extractall(EXTRACT_PATH)

    print("Extraction completed.")


if __name__ == "__main__":

    download_dataset()

    extract_dataset()

    print("\nMovieLens dataset ready.")