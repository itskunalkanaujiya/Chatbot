import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="YT RAG — Ask Your Video", page_icon="🎬", layout="centered")

# ── Styling — only safe, non-breaking overrides ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar / main background */
section.main > div { padding-top: 2rem; }

/* Hero header */
.yt-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;
}
.yt-hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF0000 0%, #ff6b6b 50%, #ffd93d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.yt-hero p {
    color: #888;
    font-size: 1rem;
    margin: 0;
}

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #FF0000, #ff4444);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 0.6rem;
}

/* Section title */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: inherit;
}

/* Thumbnail card */
.thumb-card {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    margin-bottom: 1rem;
    position: relative;
}
.thumb-card img {
    width: 100%;
    display: block;
    border-radius: 14px;
}
.thumb-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 1.2rem 1rem 0.8rem;
    background: linear-gradient(transparent, rgba(0,0,0,0.75));
    border-radius: 0 0 14px 14px;
}
.thumb-overlay .vid-id {
    color: #fff;
    font-size: 0.8rem;
    font-weight: 500;
    opacity: 0.85;
    font-family: monospace;
}

/* Success / info banners */
.banner {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.banner-green { background: #e6f9f0; color: #1a7a4a; border-left: 4px solid #22c55e; }
.banner-red   { background: #fff0f0; color: #b91c1c; border-left: 4px solid #ef4444; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #FF0000, #ff4444);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 80%;
    font-size: 0.93rem;
    font-weight: 500;
    width: fit-content;
    margin-left: auto;
}
.chat-bot {
    background: #f3f4f6;
    color: #1f2937;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem auto 0.5rem 0;
    max-width: 85%;
    font-size: 0.93rem;
    line-height: 1.65;
    border-left: 3px solid #FF0000;
}

/* Divider */
.yt-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    margin: 1.8rem 0;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.stat-pill {
    flex: 1;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
    font-size: 0.78rem;
    color: #6b7280;
}
.stat-pill strong {
    display: block;
    font-size: 1.1rem;
    color: #111827;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str):
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def get_video_meta(video_id: str):
    """Return title + best thumbnail URL via oEmbed (no API key needed)."""
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("title", ""), data.get("thumbnail_url", "")
    except Exception:
        pass
    return "", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


@st.cache_resource(show_spinner=False)
def build_chain(video_id: str):
    api = YouTubeTranscriptApi()
    transcript_list = None
    for lang in (["en"], ["hi"], ["en-IN"]):
        try:
            transcript_list = api.fetch(video_id=video_id, languages=lang)
            break
        except Exception:
            continue
    if transcript_list is None:
        raise TranscriptsDisabled(video_id)

    transcript = " ".join(chunk.text for chunk in transcript_list)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation")
    model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | model
        | StrOutputParser()
    )
    return chain, len(chunks)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("chain", None), ("history", []), ("video_meta", {}), ("chunk_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="yt-hero">
    <h1>🎬 Ask Your YouTube Video</h1>
    <p>Paste any YouTube link · Load the transcript · Ask questions using AI</p>
</div>
""", unsafe_allow_html=True)


# ── Step 1: Load ──────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">▶ Step 1 — Load a Video</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enter a YouTube URL</div>', unsafe_allow_html=True)

url_input = st.text_input(
    label="url",
    label_visibility="collapsed",
    placeholder="https://www.youtube.com/watch?v=...",
    key="url_input",
)

col1, col2 = st.columns([3, 1])
with col2:
    load_clicked = st.button("🚀 Load Video", use_container_width=True)

if load_clicked:
    if not url_input.strip():
        st.warning("Please enter a URL first.")
    else:
        vid_id = extract_video_id(url_input.strip())
        if not vid_id:
            st.markdown('<div class="banner banner-red">❌ Couldn\'t find a video ID — check the URL.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Fetching transcript and building index…"):
                try:
                    chain, n_chunks = build_chain(vid_id)
                    title, thumb_url = get_video_meta(vid_id)
                    st.session_state.chain = chain
                    st.session_state.chunk_count = n_chunks
                    st.session_state.history = []
                    st.session_state.video_meta = {
                        "id": vid_id, "title": title, "thumb": thumb_url
                    }
                    st.rerun()
                except TranscriptsDisabled:
                    st.markdown('<div class="banner banner-red">❌ No captions available for this video.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="banner banner-red">❌ Error: {e}</div>', unsafe_allow_html=True)


# ── Thumbnail + meta (shown after load) ───────────────────────────────────────
if st.session_state.chain and st.session_state.video_meta:
    meta = st.session_state.video_meta

    st.markdown('<div class="yt-divider"></div>', unsafe_allow_html=True)

    # Thumbnail
    st.markdown(f"""
    <div class="thumb-card">
        <img src="{meta['thumb']}" alt="thumbnail"/>
        <div class="thumb-overlay">
            <div class="vid-id">▶ {meta['id']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Title
    if meta.get("title"):
        st.markdown(f"**📹 {meta['title']}**")

    # Stats
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-pill"><strong>✅</strong>Transcript Loaded</div>
        <div class="stat-pill"><strong>{st.session_state.chunk_count}</strong>Text Chunks</div>
        <div class="stat-pill"><strong>FAISS</strong>Vector Store</div>
        <div class="stat-pill"><strong>Qwen 72B</strong>LLM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 2: Ask ───────────────────────────────────────────────────────────
    st.markdown('<div class="yt-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">💬 Step 2 — Ask a Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What do you want to know?</div>', unsafe_allow_html=True)

    question = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. Summarize the main project discussed in this video.",
        key="question_input",
    )

    q_col1, q_col2 = st.columns([3, 1])
    with q_col2:
        ask_clicked = st.button("🔍 Ask", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching transcript and generating answer…"):
                try:
                    answer = st.session_state.chain.invoke(question.strip())
                    st.session_state.history.append((question.strip(), answer))
                    st.rerun()
                except Exception as e:
                    st.session_state.history.append((question.strip(), f"⚠ Error: {e}"))
                    st.rerun()

    # ── Conversation ──────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown('<div class="yt-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💬 Conversation</div>', unsafe_allow_html=True)

        for q, a in reversed(st.session_state.history):
            st.markdown(f'<div class="chat-user">🧑 {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🤖 {a}</div>', unsafe_allow_html=True)

        if st.button("🗑 Clear conversation"):
            st.session_state.history = []
            st.rerun()