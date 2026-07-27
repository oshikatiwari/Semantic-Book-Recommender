import streamlit as st
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Semantic Book Recommender",
    layout="wide"
)


# ----------------------------
# Load books dataset
# ----------------------------
@st.cache_data
def load_books():
    return pd.read_csv("books_with_emotions.csv")


books = load_books()


# ----------------------------
# Create vector database
# ----------------------------
@st.cache_resource
def create_vector_db():

    raw_documents = TextLoader(
        "tagged_description.txt"
    ).load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1,
        chunk_overlap=0,
        separator="\n"
    )

    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return db


db_books = create_vector_db()


# ----------------------------
# Recommendation function
# ----------------------------
def retrieve_semantic_recommendations(
    query,
    category="All",
    tone="All",
    initial_top_k=50,
    final_top_k=16
):

    recs = db_books.similarity_search_with_score(
        query,
        k=initial_top_k
    )

    books_list = [
        int(rec[0].page_content.strip('"').split()[0])
        for rec in recs
    ]


    book_recs = books[
        books["isbn13"].isin(books_list)
    ].drop_duplicates(
        subset=["isbn13"]
    )


    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]


    if tone == "Happy":
        book_recs = book_recs.sort_values(
            by="joy_x",
            ascending=False
        )

    elif tone == "Surprising":
        book_recs = book_recs.sort_values(
            by="surprise_x",
            ascending=False
        )

    elif tone == "Angry":
        book_recs = book_recs.sort_values(
            by="anger_x",
            ascending=False
        )

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(
            by="fear_x",
            ascending=False
        )

    elif tone == "Sad":
        book_recs = book_recs.sort_values(
            by="sadness_x",
            ascending=False
        )


    return book_recs.head(final_top_k)



# ----------------------------
# Streamlit UI
# ----------------------------

st.title("📚 Semantic Book Recommender")

st.write(
    "Find books using AI-powered semantic search"
)


query = st.text_input(
    "Describe the type of book you want:"
)


categories = [
    "All"
] + sorted(
    books["simple_categories"]
    .dropna()
    .unique()
    .tolist()
)


category = st.selectbox(
    "Choose category",
    categories
)


tone = st.selectbox(
    "Choose emotional tone",
    [
        "All",
        "Happy",
        "Surprising",
        "Angry",
        "Suspenseful",
        "Sad"
    ]
)



if st.button("Recommend Books"):

    if query:

        results = retrieve_semantic_recommendations(
            query,
            category,
            tone
        )


        st.subheader("Recommended Books")


        for _, row in results.iterrows():

            col1, col2 = st.columns([1,4])


            with col1:
                if pd.notna(row["large_thumbnail"]):
                    st.image(
                        row["large_thumbnail"],
                        width=120
                    )


            with col2:
                st.write(
                    "### " + row["title"]
                )

                st.write(
                    "Author:",
                    row["authors"]
                )

                st.write(
                    row["description"][:300] + "..."
                )

                st.divider()

    else:
        st.warning(
            "Please enter a description."
        )
