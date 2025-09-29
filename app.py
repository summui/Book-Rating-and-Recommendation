import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your dataset (replace with your actual dataset)
df = pd.read_csv(r'C:\Users\SUMAIYA FATIMA\Desktop\BOOKS\goodreads_data.csv')  # should have columns: Book, Author, Genres, Description, Avg_Rating

# Load your trained model (e.g., pickled KNN regressor)
with open('best_model_knn.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app layout
st.title("📚 Book Rating Prediction and Recommendation")

# Genre multiselect (extract unique genres from dataset, assuming list in 'Genres' col)
all_genres = sorted(set(g for sublist in df['Genres'].apply(eval) for g in sublist))
selected_genres = st.multiselect("Select Genres:", all_genres)

# User description input (describe what kind of book you want)
description_input = st.text_area("Describe what kind of book you want:", height=120)

if st.button("Predict & Recommend"):
    if not selected_genres or not description_input.strip():
        st.warning("Please select genres and enter a description!")
    else:
        # Mock rating prediction by averaging genre ratings
        genre_avg = df[df['Genres'].apply(lambda x: any(g in eval(x) for g in selected_genres))]['Avg_Rating'].mean()
        predicted_rating = genre_avg

        st.success(f"Predicted average rating based on genres: {round(predicted_rating, 2)} / 5 ⭐")

        # Recommend books similar to user description
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))

        input_vec = tfidf.transform([description_input])
        cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()

        top_n = 5
        top_indices = cosine_sim.argsort()[-top_n:][::-1]

        st.subheader("Top Recommended Books for Your Description:")
        for idx in top_indices:
            book = df.iloc[idx]['Book']
            book_rating = df.iloc[idx]['Avg_Rating']
            st.write(f"- **{book}** (Avg Rating: {book_rating})")
        if cosine_sim[top_indices[0]] == 0:
            st.info("No similar books found based on your description.")
