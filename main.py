import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

path = "D:\\"
credits_df = pd.read_csv(path + "tmdb_credits.csv")
movies_df = pd.read_csv(path + "tmdb_movies.csv")

print(movies_df.head())

print(credits_df.columns)

credits_df.columns = ['id', 'tittle', 'cast', 'crew']
movies_df = movies_df.merge(credits_df, on="id")

print(movies_df.head)

# Demographic Filtering
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

print("C: ", C)
print("m: ", m)

new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print(new_movies_df.shape)


def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v / (v + m) * R) + (m / (v + m) * C)


new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)

new_movies_df = new_movies_df.sort_values('score', ascending=False)

print(new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10))


# Plot top 10 movies
def plot_top():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(18, 12))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 popular movies")
    plt.xlabel("Popularity")
    plt.show()


plot_top()


# Plot Least Popular Movie
def plot_least():
    popularity = movies_df.sort_values("popularity", ascending=True)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="#ffcccb")
    plt.title("Least  10 popular movies")
    plt.xlabel("Popularity")
    plt.show()


plot_least()


def plot_top_ratings():
    score = new_movies_df.sort_values('score', ascending=True)
    plt.figure(figsize=(12, 6))
    plt.barh(score["title"].head(10), score['score'].head(10), align="center", color="#00008b")
    plt.gca().invert_yaxis()
    plt.title("Top 10 rated Movies")
    plt.xlabel("Score")
    plt.show()


plot_top_ratings()


def plot_least_ratings():
    score = new_movies_df.sort_values('score', ascending=False)
    plt.figure(figsize=(18, 12))
    plt.barh(score["title"].head(10), score['score'].head(10), align="center", color="#8b0000")
    plt.gca().invert_yaxis()
    plt.title("least 10 rated Movies ")
    plt.xlabel("Score")
    plt.show()


plot_least_ratings()

# Content based Filtering
print(movies_df["overview"].head(5))

tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
print(indices.head())


def get_recommendations(title, cosine_sim=cosine_sim):
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    # (a, b) where a is id of movie, b is sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies


print("################ Content Based Filtering - plot#############")
print()
print("Recommendations for The Dark Knight Rises")
print(get_recommendations("The Dark Knight Rises"))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers"))

features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

movies_df[features].head(10)


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names

    return []


movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

print(movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head())


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


movies_df["soup"] = movies_df.apply(create_soup, axis=1)
print(movies_df["soup"].head())

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim2.shape)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])

print("################ Content Based System - metadata #############")
print(
    "============================Recommendations for The Dark Knight Rises==================================================")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
print()
print(
    "============================Recommendations for Avengers=================================================================")
print(get_recommendations("The Avengers", cosine_sim2))

# The Godfather