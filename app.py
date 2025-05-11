import streamlit as st
import re
import warnings

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

warnings.filterwarnings("ignore")

api_key = 'AIzaSyC7op0zN_EESTSFhXGtzHRYImp5nhvDz-c'  # Replace with your Gemini API key

# === CUSTOM CSS STYLING ===
st.markdown("""
<style>
/* Global Reset for Smooth Transitions */
.stTextInput, .stButton>button, .stMarkdown {
    transition: all 0.5s ease-in-out;
}     


/* Aggressively Remove Streamlit's Default Header, Toolbar, and Any Space Above Title */
.stApp > header, .stApp > header::before, .stApp > header::after {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    visibility: hidden !important;
}

/* Target Streamlit's Toolbar Specifically */
[data-testid="stToolbar"], [data-testid="stToolbar"] * {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    visibility: hidden !important;
}

/* Target Streamlit's Header Specifically */
[data-testid="stHeader"], [data-testid="stHeader"] * {
    display: none !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    visibility: hidden !important;
}

/* Extra suppression of the first top-level block container that sometimes causes empty space */
.stApp > div:first-child > div:first-child {
    padding: 0 !important;
    margin: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}


/* Ensure No Padding or Margin Above Title in the Main App Container */
.stApp {
    padding-top: 0 !important;
    margin-top: 0 !important;
    padding: 0 !important;
    background-image: url('background.jpg'); 
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Remove Any Extra Space Around the Main Container */
main, [data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Ensure the First Element (Title) Has No Extra Space Above */
.stTitle, .stTitle * {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Enhanced Title with Gradient, Full-Width, Transparent Background */
.gradient-title {
    font-size: 3.5rem;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    text-shadow:
        0 0 5px rgba(255, 255, 255, 0.2),
        0 0 10px rgba(138, 43, 226, 0.4),
        0 0 20px rgba(138, 43, 226, 0.6),
        0 0 40px rgba(138, 43, 226, 0.8);
    background-color: transparent;
    margin: 2rem auto;
    padding: 0.5rem 1rem; /* reduce vertical padding */
    width: 100%;
    max-width: 90vw; /* increase horizontal span */
    display: block;
}







@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 600% 50%; }
    100% { background-position: 0% 50%; }
}

/* Input Fields with Enhanced Gradient Borders and Glow */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.1);
    border: 3px solid transparent;
    border-image: linear-gradient(45deg, #ff00ff, #00b7eb, #8a2be2) 1;
    border-radius: 12px;
    padding: 14px;
    color: #ffffff;
    font-size: 1.2rem;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.5), 0 0 40px rgba(0, 183, 235, 0.3);
    animation: pulseGlow 2s ease-in-out infinite;
}

.stTextInput > div > div > input:focus {
    outline: none;
    box-shadow: 0 0 30px rgba(138, 43, 226, 0.8), 0 0 60px rgba(0, 183, 235, 0.5);
    transform: scale(1.03);
}

@keyframes glowPulse {
    0% {
        text-shadow:
            0 0 8px #ff00ff,
            0 0 16px #00b7eb,
            0 0 24px #8a2be2,
            0 0 32px #ff00ff;
    }
    50% {
        text-shadow:
            0 0 16px #ff00ff,
            0 0 32px #00b7eb,
            0 0 48px #8a2be2,
            0 0 64px #ff00ff;
    }
    100% {
        text-shadow:
            0 0 8px #ff00ff,
            0 0 16px #00b7eb,
            0 0 24px #8a2be2,
            0 0 32px #ff00ff;
    }
}


@keyframes pulseGlow {
    0% { box-shadow: 0 0 20px rgba(138, 43, 226, 0.5), 0 0 40px rgba(0, 183, 235, 0.3); }
    50% { box-shadow: 0 0 30px rgba(138, 43, 226, 0.8), 0 0 60px rgba(0, 183, 235, 0.5); }
    100% { box-shadow: 0 0 20px rgba(138, 43, 226, 0.5), 0 0 40px rgba(0, 183, 235, 0.3); }
}

/* Buttons with Enhanced Gradient, Glow, and Shine */
.stButton > button {
    background: linear-gradient(45deg, #ff00ff, #00b7eb, #8a2be2);
    background-size: 400%;
    border: none;
    border-radius: 30px;
    padding: 14px 28px;
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
    cursor: pointer;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.5), 0 0 40px rgba(0, 183, 235, 0.3);
    animation: buttonShine 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: shine 2s infinite;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 30px rgba(138, 43, 226, 0.8), 0 0 60px rgba(0, 183, 235, 0.5);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.3), 0 0 30px rgba(0, 183, 235, 0.2);
}

@keyframes buttonShine {
    0% { background-position: 0% 50%; }
    50% { background-position: 400% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes shine {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Chat History Messages with Gradient Borders and Fade Animation */
.stMarkdown {
    padding: 18px;
    margin: 12px 0;
    border-radius: 12px;
    border: 2px solid transparent;
    border-image: linear-gradient(45deg, #ff00ff, #00b7eb, #8a2be2) 1;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.3);
    animation: fadeIn 0.7s ease-in;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Differentiate User and Bot Messages */
.stMarkdown:has(strong:contains("You:")) {
    background: rgba(138, 43, 226, 0.15);
}

.stMarkdown:has(strong:contains("Bot:")) {
    background: rgba(0, 183, 235, 0.15);
}

/* Error and Success Messages */
.stAlert {
    border-radius: 12px;
    border: 2px solid transparent;
    border-image: linear-gradient(45deg, #ff00ff, #00b7eb, #8a2be2) 1;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.3);
    animation: fadeIn 0.7s ease-in;
}


/* Dark Theme Compatibility */
@media (prefers-color-scheme: dark) {
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.4);
    }
}
            
            /* AI Message Styling */
div[data-testid="chat-message-ai"] {
    background-color: black !important;
    color: white !important;
    padding: 16px;
    border-radius: 10px;
    border: 2px solid #ffffff30;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
}

</style>
""", unsafe_allow_html=True)

# === GLOBAL CHAT STATE ===
if "chatbot_globals" not in st.session_state:
    st.session_state.chatbot_globals = {
        "chat_history": [],
        "chat_active": False,
        "final_chain": None,
        "url_used": None,
    }

state = st.session_state.chatbot_globals  # Shortcut

# === UI HEADER ===
# st.title("🎬 YouTube ChatBot with Gemini + LangChain")
st.markdown("""
<h1 class="gradient-title">🌟RAG-GenAI-YouTube ChatBot</h1>
""", unsafe_allow_html=True)



if not state["chat_active"]:
    url = st.text_input("Paste your YouTube Video Link:")
    start = st.button("Start Chat")
else:
    url = state["url_used"]
    start = False

# === UTILS ===
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            transcript = next((t for t in transcript_list if t.is_generated), None)
            if not transcript:
                raise NoTranscriptFound("No transcript found in any language.")

        transcript_data = transcript.fetch()
        return " ".join([entry.text for entry in transcript_data])

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        raise RuntimeError(f"⚠️ Error loading transcript: {e}")

# === START CHAT WORKFLOW ===
if (start and url) or state["chat_active"]:
    if start:
        st.info("⏳ Hang tight! Loading the transcript might take a moment...")
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL.")
            st.stop()

        try:
            transcript = fetch_transcript(video_id)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        chunks = splitter.create_documents([transcript])

        # Embedding & Retrieval
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})

        # Prompt + model
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant answering questions from a YouTube video.
You should also remember and use details shared by the user earlier in the conversation, such as their name or preferences.
If something is unclear or not in the transcript or chat history, say "I don't know."

Transcript context: {context}
"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chat_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash-exp")
        parser = StrOutputParser()

        parallel_chain = RunnableParallel({
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda x: format_docs(retriever.get_relevant_documents(x["question"]))),
        })

        def get_history(_):
            return state["chat_history"]

        final_chain = (
            RunnablePassthrough.assign(chat_history=RunnableLambda(get_history))
            | parallel_chain
            | chat_prompt
            | chat_model
            | parser
        )

        state["final_chain"] = final_chain
        state["chat_active"] = True
        state["url_used"] = url
        state["chat_history"] = []
        st.rerun()

    # === CHAT UI ===
    st.markdown("### 💬 Chat History")
    for msg in state["chat_history"]:
        if isinstance(msg, HumanMessage):
            st.markdown(f"🧑‍💬 **You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"🤖 **Bot:** {msg.content}")

    question = st.text_input("Ask a question:", value="", label_visibility="visible")
    col1, col2 = st.columns([1, 1])
    ask_btn = col1.button("Ask")
    end_btn = col2.button("End Chat")

    if ask_btn and question.strip():
        query = {"question": question.strip()}
        response = state["final_chain"].invoke(query)

        state["chat_history"].append(HumanMessage(content=question.strip()))
        state["chat_history"].append(AIMessage(content=response))
        st.rerun()

    if end_btn:
        state["chat_active"] = False
        state["chat_history"] = []
        state["final_chain"] = None
        state["url_used"] = None
        st.success("🧹 Chat ended.")
        st.rerun()




import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image filename
add_bg_from_local('bg5.jpg')

