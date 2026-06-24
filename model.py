print("🚀 Script started")

import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📂 Loading datasets...")

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

print("🔗 Merging datasets...")

movies = movies.merge(credits, on="title")

movies = movies[
    [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
]

movies.dropna(inplace=True)

print("🛠️ Processing data...")


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L


def convert_cast(text):
    L = []
    counter = 0

    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i["name"])
            counter += 1

    return L


def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])

    return L


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)


def collapse(L):
    return [i.replace(" ", "") for i in L]


movies["genres"] = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"] = movies["cast"].apply(collapse)
movies["crew"] = movies["crew"].apply(collapse)

# Save readable genres
movies["genres_text"] = movies["genres"].apply(
    lambda x: ", ".join(x)
)

movies["overview"] = movies["overview"].apply(
    lambda x: x.split()
)

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

print("🧠 Building recommendation features...")

new_df = movies[
    [
        "movie_id",
        "title",
        "genres_text",
        "overview",
        "tags"
    ]
].copy()

new_df["tags"] = new_df["tags"].apply(
    lambda x: " ".join(x)
)

# Convert overview back to text
new_df["overview"] = new_df["overview"].apply(
    lambda x: " ".join(x)
)

print("📊 Creating vectors...")

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(
    new_df["tags"]
).toarray()

print("🔍 Calculating similarity matrix...")

similarity = cosine_similarity(vectors)

print("💾 Saving files...")

pickle.dump(
    new_df,
    open("movies.pkl", "wb")
)

pickle.dump(
    similarity,
    open("similarity.pkl", "wb")
)

print("✅ Done successfully!")
print(f"🎬 Movies processed: {new_df.shape[0]}")