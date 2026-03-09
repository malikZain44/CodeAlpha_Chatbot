import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- UI Configuration & Styling ---
st.set_page_config(page_title="CodeAlpha AI Assistant", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-title {
        font-size: 50px;
        font-weight: 800;
        background: -webkit-linear-gradient(#1e293b, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        animation: fadeIn 2s;
    }

    /* Floating AI Emoji Animation */
    .floating-ai {
        display: inline-block;
        font-size: 26px;
        animation: float 3s ease-in-out infinite;
        margin-right: 12px;
        vertical-align: middle;
    }

    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.95) !important;
        backdrop-filter: blur(10px);
        color: white;
    }

    .stChatMessage {
        border-radius: 20px !important;
        padding: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">✨ CodeAlpha AI Assistant</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Powered by Natural Language Processing</p>", unsafe_allow_html=True)

# --- Data Loading ---
try:
    df = pd.read_csv('faqs.csv')
    df['Question'] = df['Question'].str.strip().str.lower()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- AI Logic ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['Question'])

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2044/2044809.png", width=100) # AI Bot Icon
    st.title("Internship Portal")
    st.markdown("---")
    st.success(f"👤 **User:** Zain Ali Awan")
    st.info("🎓 **Role:** AI Intern")
    st.markdown("---")
    st.write("📊 **Overall Progress:**")
    st.progress(50)
    st.write("✅ **Project 1:** FAQ Chatbot")
    st.write("⏳ **Project 2:** Object Detection")

# --- Chat Functionality ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask me about AI or your Internship..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": f"👤 {prompt}"})
    with st.chat_message("user"):
        st.markdown(f"👤 {prompt}")

    # Logic
    query_vec = vectorizer.transform([prompt.lower().strip()])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    best_match_idx = similarity.argmax()
    confidence = similarity[0][best_match_idx]

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            time.sleep(0.4)
            
        if confidence > 0.4:
            ans = df.iloc[best_match_idx]['Answer']
            # AI Themed Emojis
            response = f'<span class="floating-ai">🤖</span> **AI:** {ans} ✨'
        else:
            response = '<span class="floating-ai">🧠</span> **AI:** I am not quite sure. Try asking "What is AI?" or check the internship PDF! 💡'
        
        st.markdown(response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})