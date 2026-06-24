import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# LOAD MOVIES
# -----------------------------
@st.cache_resource
def load_data():

    movies = pickle.load(
        open("movies.pkl", "rb")
    )

    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(
        movies["tags"]
    ).toarray()

    similarity = cosine_similarity(
        vectors
    )

    return movies, similarity


movies, similarity = load_data()

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie_name, n=5):

    movie_index = movies[
        movies["title"] == movie_name
    ].index[0]

    distances = list(
        enumerate(
            similarity[movie_index]
        )
    )

    movie_list = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    recommendations = []

    for movie in movie_list:

        row = movies.iloc[movie[0]]

        recommendations.append(
            {
                "title": row["title"],
                "genres": row["genres_text"],
                "overview": row["overview"],
                "score": round(movie[1] * 100, 2)
            }
        )

    return recommendations


# -----------------------------
# HEADER
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover similar movies using Machine Learning and Content-Based Filtering."
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    3,
    10,
    5
)

show_scores = st.sidebar.checkbox(
    "Show Similarity Scores",
    True
)

# -----------------------------
# MOVIE SELECTION
# -----------------------------
selected_movie = st.selectbox(
    "🔍 Search or Select a Movie",
    movies["title"].values
)

# -----------------------------
# SELECTED MOVIE
# -----------------------------
selected_data = movies[
    movies["title"] == selected_movie
].iloc[0]

st.subheader("Selected Movie")

st.markdown(
    f"### 🎬 {selected_data['title']}"
)

st.write(
    f"🎭 {selected_data['genres_text']}"
)

overview = selected_data["overview"]

if len(overview) > 300:
    overview = overview[:300] + "..."

st.write(overview)

st.divider()

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
if st.button("🎥 Get Recommendations"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.subheader(
        "🎬 Recommended Movies"
    )

    for movie in recommendations:

        with st.container(border=True):

            st.markdown(
                f"### 🎬 {movie['title']}"
            )

            st.write(
                f"🎭 {movie['genres']}"
            )

            if show_scores:

                st.progress(
                    min(
                        movie["score"] / 100,
                        1.0
                    )
                )

                st.write(
                    f"⭐ Similarity Score: {movie['score']}%"
                )

            overview = movie["overview"]

            if len(overview) > 250:
                overview = overview[:250] + "..."

            st.write(
                overview
            )

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.divider()

st.sidebar.success(
    "Built with Python, Streamlit and Scikit-Learn"
)