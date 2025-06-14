import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import requests
import os
import shutil

# --- Category Descriptions ---
category_descriptions = {
    '1': "Game Strategy: Statements discussing game tactics, player lineups, formations, or strategic decisions.",
    '2': "Player Performance: Comments evaluating an individual player’s or team’s performance, including strengths, weaknesses, and statistical assessments.",
    '3': "Injury Updates: Mentions of player injuries, recovery status, or impact on team availability.",
    '4': "Post-Game Analysis: Statements reflecting on the outcome of a game, discussing key moments, player contributions, or the final result.",
    '5': "Team Morale: Discuss team spirit, motivation, chemistry, or internal team challenges.",
    '6': "Upcoming Matches: Predictions, expectations, or discussions about future games, including schedules and potential outcomes.",
    '7': "Off-Game Matters: Topics unrelated to gameplay, such as trades, community involvement, player personal life, or off-field controversies.",
    '8': "Controversies: Statements involving disputes, questionable referee decisions, rule violations, or other forms of conflict in sports."
}

# --- Load Necessary Assets ---
@st.cache_resource
def load_classification_model():
    try:
        vectorizer = joblib.load('Models/tfidf_vectorizer.joblib')
        classifier = joblib.load('Models/logreg_model.joblib')
        return vectorizer, classifier
    except FileNotFoundError:
        st.error("Classification model files not found in 'models/'.")
        return None, None

@st.cache_data
def load_embeddings_data():
    try:
        return pd.read_csv("data/umap_embeddings.csv")
    except FileNotFoundError:
        st.error("UMAP embeddings data not found in 'data/umap_embeddings.csv'.")
        return None

@st.cache_resource
def load_generation_pipeline():
    try:
        download_large_file("https://huggingface.co/kavya-gee-class-works/nlp-project/resolve/main/model.safetensors?download=true", "./sports_interview_gpt2/model.safetensors")
        tokenizer = GPT2Tokenizer.from_pretrained("./sports_interview_gpt2")
        model = GPT2LMHeadModel.from_pretrained("./sports_interview_gpt2").to('cpu')
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned GPT-2 model from './sports_interview_gpt2': {e}")
        return None, None

def download_large_file(url, local_filename=None):
    """
    Downloads a large file from a given URL and saves it to the project folder.

    Args:
        url (str): The URL of the file to download.
        local_filename (str, optional): The name to save the file as.
                                         If None, it will use the filename from the URL.
                                         Defaults to None.

    Returns:
        str: The path to the downloaded file if successful, None otherwise.
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]
        if not local_filename:
            local_filename = "downloaded_file"

    project_folder = os.getcwd()
    file_path = os.path.join(project_folder, local_filename)

    try:
        print(f"Starting download from: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192  # 8KB
            downloaded_size = 0

            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"Downloaded: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({progress:.2f}%)", end='\r')
                        else:
                            print(f"Downloaded: {downloaded_size / (1024*1024):.2f} MB (Total size unknown)", end='\r')
            print(f"\nDownload complete! File saved as: {file_path}")
            return file_path
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {r.status_code} {r.reason}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error connecting: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request: {req_err}")
    except IOError as io_err:
        print(f"File I/O error: {io_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Clean up partially downloaded file if an error occurred
    if os.path.exists(file_path) and downloaded_size < (total_size if 'total_size' in locals() else float('inf')):
        try:
            os.remove(file_path)
            print(f"Partially downloaded file {local_filename} removed.")
        except OSError as e:
            print(f"Error removing partially downloaded file: {e}")
    return None

vectorizer, classifier = load_classification_model()
embeddings_df = load_embeddings_data()
generation_model, generation_tokenizer = load_generation_pipeline()

# --- Streamlit App Layout ---
st.title("AI Systems for Sports Interview Analysis")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Transcript Classification", "Q&A Generator", "Embedding Visualizer"])

# ----------------------------
# 1. Transcript Classification
# ----------------------------
if page == "Transcript Classification":
    st.title("🎯 Transcript Classification")
    st.markdown("Analyzes interview transcripts to predict the main topic category.")
    user_input = st.text_area("Enter the interview transcript:")
    if user_input and st.button("Classify"):
        if vectorizer and classifier:
            cleaned_input = user_input.lower()
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = classifier.predict(vectorized_input)[0]
            st.success(f"Predicted Category: {prediction} - {category_descriptions.get(str(prediction), 'Description not available')}")
        else:
            st.warning("Classification models not loaded.")

# ----------------------------
# 2. Question and Answer
# ----------------------------
elif page == "Q&A Generator":
    st.title("💬 Question and Answer Generator")
    st.markdown("Select a category and ask a question; an AI will generate a response.")
    if generation_model and generation_tokenizer and embeddings_df is not None and 'Labels' in embeddings_df.columns:
        unique_labels = sorted(embeddings_df['Labels'].unique().astype(str).tolist())
        selected_cat_num = st.selectbox("Select an Interview Category:", unique_labels)
        st.info(f"Category Description: {category_descriptions.get(selected_cat_num, 'Description not available')}")
        question = st.text_input("Ask a question relevant to the selected category:")
        if st.button("Generate Answer") and question:
            prompt = f"Category: {category_descriptions.get(selected_cat_num).split(':')[0] if category_descriptions.get(selected_cat_num) else 'Unknown'}. Question: {question} Answer:"
            input_ids = generation_tokenizer.encode(prompt, return_tensors="pt").to(generation_model.device if hasattr(generation_model, 'device') else 'cpu')
            try:
                output = generation_model.generate(
                    input_ids,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=generation_tokenizer.eos_token_id,
                    no_repeat_ngram_size=3, # Helps prevent repetitive phrases
                    top_k=50,             # Limits the next token selection to the top k probabilities
                    top_p=0.95,           # Nucleus sampling - considers tokens within the top p probability mass
                    early_stopping=True   # Stop generation if no new meaningful tokens are produced
                )
                response = generation_tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()
                st.markdown(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"Error during text generation: {e}")
    else:
        st.warning("Text generation model or embedding data not loaded correctly.")

# ----------------------------
# 3. Interactive Embedding Visualizer
# ----------------------------
elif page == "Embedding Visualizer":
    st.title("📊 Transcript Embedding Visualizer")
    st.markdown("""
        Visualizes interview transcripts in a 2D space based on content similarity.
        Closer points indicate more similar transcripts. Colors represent categories.
        Hover over points to see the transcript text and category.
    """)
    if embeddings_df is not None and 'PCA1' in embeddings_df.columns and 'PCA2' in embeddings_df.columns and 'Labels' in embeddings_df.columns and 'Cleaned_Text' in embeddings_df.columns:
        fig = px.scatter(
            embeddings_df,
            x="PCA1", y="PCA2",
            color="Labels",
            hover_data=["Cleaned_Text", "Labels"],
            title="UMAP Projection of Transcripts"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Embedding data not loaded or missing necessary columns.")

# --- Requirements File ---
with open("requirements.txt", "w") as f:
    f.write("streamlit\n")
    f.write("pandas\n")
    f.write("joblib\n")
    f.write("numpy\n")
    f.write("plotly\n")
    f.write("scikit-learn\n")
    f.write("transformers\n")
    f.write("umap-learn\n")