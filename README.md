# 📚 Semantic Book Recommender

An AI-powered semantic book recommendation system that recommends books based on natural language descriptions using embeddings, vector search, and emotion-based filtering.

## 🚀 Features

- 🔍 Semantic search using HuggingFace embeddings
- 🧠 Vector similarity search with ChromaDB
- 📂 Category-based filtering
- 😊 Emotion/tone-based recommendations
- 🖼️ Book cover visualization
- 🌐 Interactive Gradio dashboard

## 🛠️ Tech Stack

- Python
- LangChain
- HuggingFace Transformers
- Sentence Transformers
- ChromaDB
- Gradio
- Pandas

## ⚙️ How It Works

1. Book descriptions are converted into embeddings using HuggingFace models.
2. Embeddings are stored in a Chroma vector database.
3. User enters a description of a book they want.
4. The system retrieves semantically similar books.
5. Results are ranked based on selected category and emotional tone.

## 📸 Demo

(Add your Gradio screenshot here)

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
