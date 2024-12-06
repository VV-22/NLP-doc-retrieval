import streamlit as st
from joblib import load
import numpy as np


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
    
if "TFIDFEmbedding" not in st.session_state:
    st.session_state.TFIDFEmbedding = load("TFIDFEmbedding.joblib")

if "KNNModel" not in st.session_state:
    st.session_state.KNNModel = load("KNNModel.joblib")

if "sbert" not in st.session_state:
    st.session_state.sbert = load("sbert.joblib")
    # st.session_state.sbert.tokenizer.pad_token = st.session_state.sbert.tokenizer.unk_token
    print(f"tokens : {st.session_state.sbert.tokenizer}")

if "faiss" not in st.session_state:
    st.session_state.faiss = load("faiss.joblib")

if "bm25" not in st.session_state:
    st.session_state.bm25 = load("bm25.joblib")


if "model" not in st.session_state:
    st.session_state.model = ""

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

# Display cards in three columns and make them clickable
with col1:
    clickable_card("Card 1", "TFIDF + KNN", "card_1", knn_model)
with col2:
    clickable_card("Card 2", "SBERT + FAISS", "card_2", faiss_model)
with col3:
    clickable_card("Card 3", "BM25", "card_3" , bm25)

# Display action or message based on the clicked card
if "clicked_card" in st.session_state and st.session_state.clicked_card:
    st.success(f"You clicked on {st.session_state.clicked_card}")

# Divider line between cards and chat
st.divider()

# Chat window section
st.write("## Chat")

# Display chat history using a placeholder container
chat_placeholder = st.container()


# Continuously display chat history without page reload
with chat_placeholder:
    for message in st.session_state.chat_history:
        st.write(message)

st.text_input(label = "Query" , placeholder = "enter query here" , key="question_param", on_change=lambda : save_ques())


def save_ques():
    st.session_state.question = st.session_state.question_param
    st.session_state.question_param = ''

def knn_answer(question):
    tfidf = st.session_state.TFIDFEmbedding.transform([question])
    knn_ans = st.session_state.KNNModel.kneighbors(tfidf , return_distance=False)
    return knn_ans

def faiss_answer(question):
    sbert_embedding = st.session_state.sbert.encode([question]).astype('float32')
    dustabces, indices = st.session_state.faiss.search(sbert_embedding , 10)
    return indices

def bm25_answer(question):
    bm_tokens = question.lower().split()
    bm_scores = np.argsort(st.session_state.bm25.get_scores(bm_tokens))[::-1][:10]
    return bm_scores

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

    
print(f"model : {st.session_state.model}")


