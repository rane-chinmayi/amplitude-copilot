import streamlit as st
import json
import os
import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else os.getenv('GOOGLE_API_KEY')

if not api_key:
    st.error("Error: GEMINI_API_KEY not found in Streamlit secrets or .env file")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Amplitude AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Global CSS styling
st.markdown("""
<style>
    :root {
        --bg: #0F1117;
        --card: #1A1D2E;
        --border: #2E3250;
        --accent: #7C3AED;
        --text: #F0F0F0;
        --muted: #8B8FA8;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {
        background: var(--bg) !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
    }

    [data-testid="stAppViewBlockContainer"] {
        padding: 0 !important;
    }

    .main {
        background: var(--bg) !important;
    }

    /* Container wrapper */
    .stMainBlockContainer > div:first-child {
        max-width: 800px !important;
        margin: 0 auto !important;
        padding: 0 20px !important;
    }

    /* Remove extra spacing */
    p, div {
        color: var(--text) !important;
    }

    /* Header Section */
    .header-wrapper {
        text-align: center;
        padding: 40px 0 30px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 40px;
    }

    .header-emoji {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 0.95rem;
        color: var(--muted);
        margin-bottom: 15px;
        font-weight: 400;
    }

    .header-badge {
        display: inline-block;
        padding: 8px 14px;
        border: 1px solid var(--accent);
        border-radius: 20px;
        font-size: 0.75rem;
        color: var(--accent);
        font-weight: 500;
        letter-spacing: 0.3px;
        background: rgba(124, 58, 237, 0.05);
    }

    /* Example Chips Container */
    .chips-wrapper {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }

    /* Input Box */
    [data-testid="stTextInput"] {
        margin-bottom: 20px;
    }

    [data-testid="stTextInput"] input {
        width: 100% !important;
        background: var(--card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        font-size: 0.95rem !important;
        color: var(--text) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: var(--muted) !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
        outline: none !important;
    }

    /* Buttons - Example Chips */
    [data-testid="stButton"] button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        border-radius: 18px !important;
        padding: 8px 14px !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
    }

    [data-testid="stButton"] button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(124, 58, 237, 0.08) !important;
    }

    /* Answer Card */
    .answer-wrapper {
        background: var(--card);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 24px;
        margin: 40px 0 30px;
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
    }

    .answer-header {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: var(--accent);
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .answer-content {
        font-size: 0.95rem;
        line-height: 1.7;
        color: var(--text);
        margin-bottom: 16px;
    }

    .source-badge {
        display: inline-block;
        padding: 6px 12px;
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        font-size: 0.8rem;
        color: var(--accent);
    }

    /* Feedback Section */
    .feedback-wrapper {
        text-align: center;
        margin: 30px 0;
    }

    .feedback-label {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 15px;
    }

    .feedback-buttons {
        display: flex;
        gap: 12px;
        justify-content: center;
    }

    [data-testid="stButton"] button {
        min-width: 100px;
    }

    /* Layout width */
    .main .block-container {
        max-width: 860px;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Footer */
    .footer-wrapper {
        text-align: center;
        padding: 40px 0 30px;
        border-top: 1px solid var(--border);
        margin-top: 60px;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 400;
    }

    /* Status messages */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* Remove default spacing */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== RENDER HEADER AT TOP =====
st.markdown("""
<div class="header-wrapper">
    <span class="header-emoji">🤖</span>
    <div class="header-title">Amplitude AI Assistant</div>
    <div class="header-subtitle">Get instant answers from Amplitude documentation</div>
    <div class="header-badge">✦ Powered by Gemini AI + FAISS</div>
</div>
""", unsafe_allow_html=True)

# Load resources with caching
@st.cache_resource
def load_resources():
    """Load chunks, index, and initialize client"""
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

        index = faiss.read_index("amplitude_index.faiss")
        client = genai.Client(api_key=api_key)

        return chunks, index, client
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found - {e}")
        st.stop()

chunks, index, client = load_resources()


def get_answer(query):
    """Generate an answer using RAG pipeline"""

    # Embed the query
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)

    # Search FAISS index for top 3 chunks
    distances, indices = index.search(query_embedding, 3)

    # Build context from retrieved chunks
    context = "Amplitude Documentation:\n\n"
    top_source = None

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
        if i == 0:
            top_source = chunk['source']

    # Build the prompt for Gemini
    system_prompt = """You are an expert Amplitude analytics assistant. Answer questions based ONLY on the provided Amplitude documentation.

Instructions:
- Answer only using information from the provided documentation
- Keep answers concise and practical
- If the answer isn't in the provided documentation, respond with: "I couldn't find this in the Amplitude documentation. Try searching docs.amplitude.com directly."
- Provide actionable and clear answers"""

    user_prompt = f"""{system_prompt}

{context}

User Question: {query}

Please answer based on the documentation above."""

    # Generate answer with Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt
    )

    answer = response.text

    return {
        "answer": answer,
        "source": top_source
    }


def log_feedback(query, answer, feedback):
    """Log feedback to Google Sheet"""
    try:
        # Load credentials from Streamlit secrets
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client_gs = gspread.authorize(creds)

        # Open the sheet
        sheet = client_gs.open("Amplitude Copilot Logs").sheet1

        # Append row
        timestamp = datetime.now().isoformat()
        answer_preview = answer[:500]
        sheet.append_row([timestamp, query, answer_preview, feedback])

        return True
    except KeyError:
        return False
    except Exception as e:
        return False


# ===== UI LAYOUT =====

# EXAMPLE CHIPS
st.markdown('<div class="chips-wrapper">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="small")

example_questions = [
    "How do I build a funnel?",
    "What is a cohort?",
    "How do I track retention?"
]

with col1:
    if st.button(example_questions[0], key="chip1", use_container_width=True):
        st.session_state.query = example_questions[0]
        st.rerun()

with col2:
    if st.button(example_questions[1], key="chip2", use_container_width=True):
        st.session_state.query = example_questions[1]
        st.rerun()

with col3:
    if st.button(example_questions[2], key="chip3", use_container_width=True):
        st.session_state.query = example_questions[2]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 3. INPUT BOX
query = st.text_input(
    "Ask a question",
    placeholder="🔍 Ask about funnels, cohorts, retention, dashboards...",
    label_visibility="collapsed",
    value=st.session_state.get("query", "")
)

# Clear session query after input
if "query" in st.session_state and query:
    st.session_state.query = ""

# 4. ANSWER CARD & FEEDBACK (only shown after query)
if query:
    try:
        with st.status("Processing your question...", expanded=True) as status:
            status.update(label="🔍 Searching documentation...", state="running")

            # Embed and search
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query
            )
            query_embedding = np.array(response.embeddings[0].values).astype("float32").reshape(1, -1)
            distances, indices = index.search(query_embedding, 3)

            # Build context
            context = "Amplitude Documentation:\n\n"
            top_source = None
            for i, idx in enumerate(indices[0]):
                chunk = chunks[idx]
                context += f"Document {i + 1} (from {chunk['source']}):\n{chunk['text']}\n\n"
                if i == 0:
                    top_source = chunk['source']

            status.update(label="🤖 Generating answer...", state="running")

            # Generate answer
            system_prompt = """You are an expert Amplitude analytics assistant. Answer questions based ONLY on the provided Amplitude documentation.

Instructions:
- Answer only using information from the provided documentation
- Keep answers concise and practical
- If the answer isn't in the provided documentation, respond with: "I couldn't find this in the Amplitude documentation. Try searching docs.amplitude.com directly."
- Provide actionable and clear answers"""

            user_prompt = f"""{system_prompt}

{context}

User Question: {query}

Please answer based on the documentation above."""

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=user_prompt
            )
            answer = response.text

            status.update(label="✓ Done", state="complete")

        # Display answer card
        st.markdown(f"""
        <div class="answer-wrapper">
            <div class="answer-header">Answer</div>
            <div class="answer-content">{answer}</div>
            <div class="source-badge">📄 {top_source}</div>
        </div>
        """, unsafe_allow_html=True)

        # Feedback buttons
        st.markdown('<div class="feedback-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="feedback-label">Was this helpful?</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="small")

        with col1:
            if st.button("👍 Yes", key="feedback_yes", use_container_width=True):
                log_feedback(query, answer, "positive")
                st.success("Thank you for your feedback!")

        with col2:
            if st.button("👎 No", key="feedback_no", use_container_width=True):
                log_feedback(query, answer, "negative")
                st.info("We appreciate your feedback.")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        error_str = str(e)
        if "503" in error_str or "ServerError" in error_str or "high demand" in error_str:
            st.warning("⚠️ Gemini is experiencing high demand. Please try again in a moment.")
        else:
            st.error(f"An error occurred: {e}")

# 6. FOOTER
st.markdown("""
<div class="footer-wrapper">
Built with Amplitude Docs · Gemini AI · FAISS
</div>
""", unsafe_allow_html=True)
