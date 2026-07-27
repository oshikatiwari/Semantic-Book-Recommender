import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Semantic Book Recommender",
    layout="wide"
)

books = pd.read_csv("./books_with_emotions.csv")

st.title("📚 Semantic Book Recommender")

st.write(
    "AI-powered book recommendation system using semantic search"
)

st.write("Total books loaded:", len(books))
