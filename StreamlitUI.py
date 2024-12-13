"""
Streamlit Web application for Document Retrieval and Question Answering

This Streamlit app provides an interactive dashboard with three clickable cards that allow users to choose a model 
for document retrieval and question answering. The available models are TFIDF + KNN, SBERT + FAISS, and BM25.
Once a model is selected, users can enter a question, and the app retrieves the most relevant document to the 
question from a pre-loaded corpus and provides an answer from the document using the BioBERT model.

"""

# Import the necessary packages.
import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


print("Loaded")

# Streamlit page configuration
st.set_page_config(page_title="Card and Chat UI", layout="wide")

# Initialize session state variables if not already initialized
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""
    
if "question" not in st.session_state:
    st.session_state.question = ""
    
# Load pre-trained models and embeddings into Streamlit session state.
if "TFIDFEmbedding" not in st.session_state:
    st.session_state.TFIDFEmbedding = load("TFIDFEmbedding.joblib")

if "KNNModel" not in st.session_state:
    st.session_state.KNNModel = load("KNNModel.joblib")

if "sbert" not in st.session_state:
    st.session_state.sbert = load("sbert.joblib")
    st.session_state.sbert.tokenizer.pad_token = "[UNK]"
    st.session_state.sbert.tokenizer.pad_token_id = st.session_state.sbert.tokenizer.convert_tokens_to_ids("[UNK]")

if "faiss" not in st.session_state:
    st.session_state.faiss = load("faiss.joblib")

if "bm25" not in st.session_state:
    st.session_state.bm25 = load("bm25.joblib")

if "docs" not in st.session_state:
    st.session_state.docs = load("docs.joblib")

if "tokenizer" not in st.session_state:
    qa_model_name = "dmis-lab/biobert-large-cased-v1.1-squad"
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    st.session_state.model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

if "model" not in st.session_state:
    st.session_state.model = ""

# Methods to set the model based on the user's selection.
def knn_model():
    st.session_state.model = "knn"

def faiss_model():
    st.session_state.model = "faiss"

def bm25():
    st.session_state.model = "bm25"

# Function to display a clickable card
def clickable_card(title, description, key, func):
    if st.button(f"{title}\n\n{description}", key=key, use_container_width=True):
        st.session_state.clicked_card = title  # Update which card was clicked
        func()

# Layout for cards
st.write("## Dashboard")
col1, col2, col3 = st.columns(3)

# Display cards in three columns and make them clickable.
with col1:
    clickable_card("Card 1", "TFIDF + KNN", "card_1", knn_model)
with col2:
    clickable_card("Card 2", "SBERT + FAISS", "card_2", faiss_model)
with col3:
    clickable_card("Card 3", "BM25", "card_3" , bm25)

# Display action or message based on the clicked card.
if "clicked_card" in st.session_state and st.session_state.clicked_card:
    st.success(f"You clicked on {st.session_state.clicked_card}")

# Divider line between cards and chat.
st.divider()

# Chat window section.
st.write("## Chat")

# Display chat history using a placeholder container.
chat_placeholder = st.container()


# Continuously display chat history without page reload.
with chat_placeholder:
    for message in st.session_state.chat_history:
        st.write(message)

# Input for user to enter their query.
st.text_input(label = "Query" , placeholder = "enter query here" , key="question_param", on_change=lambda : save_ques())

# Save the user's question from the input into session state.
def save_ques():
    st.session_state.question = st.session_state.question_param
    st.session_state.question_param = ''

#Loads the pre-trained BioBERT model for question answering.
def get_model():
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    return tokenizer, model

# Retrieves an answer for a question given a context using BioBERT.
def get_answer(question, context):
    tokenizer, model = get_model()
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    answer = qa_pipeline({'question': question, 'context': context})
    return answer

# Retrieves the most relevant document from a list of indices.
def get_most_relevant_doc(indices):
    relevant_docs = [st.session_state.docs[i] for i in indices]
    best_doc = relevant_docs[0]
    return best_doc

# Retrieves an answer using the TFIDF + KNN model.
def knn_answer(question):
    tfidf = st.session_state.TFIDFEmbedding.transform([question])
    knn_ans = st.session_state.KNNModel.kneighbors(tfidf , return_distance=False)
    knn_ans = knn_ans.flatten().tolist()
    best_doc = get_most_relevant_doc(knn_ans)
    answer = get_answer(question, best_doc)
    return answer['answer']

# Retrieves an answer using the SBERT + FAISS model.
def faiss_answer(question):
    sbert_embedding = st.session_state.sbert.encode([question]).astype('float32')
    dustabces, indices = st.session_state.faiss.search(sbert_embedding , 10)
    indices = indices.flatten().tolist()
    best_doc = get_most_relevant_doc(indices)
    answer = get_answer(question, best_doc)
    return answer['answer']

# Retrieves an answer using the BM25 model.
def bm25_answer(question):
    bm_tokens = question.lower().split()
    bm_scores = np.argsort(st.session_state.bm25.get_scores(bm_tokens))[::-1][:10]
    best_doc = get_most_relevant_doc(bm_scores)
    answer = get_answer(question, best_doc)
    return answer['answer']


# Logic to handle user input.
if st.session_state.question:
    if st.session_state.model:
        st.write(f"Q : {st.session_state.question}")
        answer = ""
        if st.session_state.model == "knn":
            answer = knn_answer(st.session_state.question)
        elif st.session_state.model == "faiss":
            answer = faiss_answer(st.session_state.question)
        elif st.session_state.model == "bm25":
            answer = bm25_answer(st.session_state.question)
        st.write(f"A : {answer}")
    else:
        st.write(f"Please select a model first.")
    st.session_state.question = ""

# Print to console to debug errors.
print(f"model : {st.session_state.model}")


