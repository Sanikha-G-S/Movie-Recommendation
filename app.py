import streamlit as st
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1200px;
}

.movie-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #444;
    margin-bottom: 15px;
}

.genre {
    color: #8ecae6;
    font-weight: bold;
}

.score {
    color: #90ee90;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie_name, n=5):

    movie_index = movies[movies["title"] == movie_name].index[0]

    distances = list(enumerate(similarity[movie_index]))

    movie_list = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    recommendations = []

    for movie in movie_list:

        row = movies.iloc[movie[0]]

        recommendations.append({
            "title": row["title"],
            "genres": row["genres_text"],
            "overview": row["overview"],
            "score": round(movie[1] * 100, 2)
        })

    return recommendations

# -----------------------------
# HEADER
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.markdown("""
Discover similar movies using Machine Learning and Content-Based Filtering.
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=10,
    value=5
)

show_scores = st.sidebar.checkbox(
    "Show Similarity Scores",
    value=True
)

# -----------------------------
# MOVIE SELECTION
# -----------------------------
selected_movie = st.selectbox(
    "🔍 Search or Select a Movie",
    movies["title"].values
)

# -----------------------------
# RECOMMEND BUTTON
# -----------------------------
if st.button("🎥 Get Recommendations"):

    recommendations = recommend(
        selected_movie,
        num_recommendations
    )

    st.subheader("🎬 Recommended Movies")

    for movie in recommendations:

        with st.container(border=True):

            st.markdown(
                f"### 🎬 {movie['title']}"
            )

            st.markdown(
                f"**🎭 Genres:** {movie['genres']}"
            )

            if show_scores:

                st.progress(
                    min(movie["score"] / 100, 1.0)
                )

                st.markdown(
                    f"**⭐ Similarity Score:** {movie['score']}%"
                )

            overview = movie["overview"]

            if len(overview) > 250:
                overview = overview[:250] + "..."

            st.write(overview)

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")

st.sidebar.success(
    "Built using Python, Streamlit, Scikit-Learn and Cosine Similarity."
)