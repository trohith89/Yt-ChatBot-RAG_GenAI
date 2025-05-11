# 🔗 [Rag-YouTube-GenAi Application🌐 ](https://rag-youtube-genai.streamlit.app)

# 🎬 RAG-Gemini YouTube ChatBot

An interactive chatbot built with **LangChain** and **Google Gemini**, allowing you to chat with **YouTube videos** using their transcript!

---

## 🚀 Features

- 🔗 Paste any YouTube link and start chatting with the video content
- 🧠 Uses **Gemini Flash (`gemini-2.0-flash-exp`)** for fast and coherent answers
- 📚 RAG pipeline using:
  - `youtube-transcript-api` for fetching transcripts
  - `RecursiveCharacterTextSplitter` for chunking
  - `FAISS` with `MMR` for semantic retrieval
- 💡 Custom prompts and **conversational memory**
- 🌈 Modern glowing UI using **Streamlit custom CSS**

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | Frontend UI |
| `LangChain` | RAG + Memory Pipeline |
| `Google Generative AI` | Answer Generation |
| `FAISS` | Vector storage + retrieval |
| `youtube-transcript-api` | Transcript fetching |

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/yt-chatbot-rag_genai
cd yt-chatbot-rag_genai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


```
api_key = "YOUR_API_KEY"
```


🙌 Credits
LangChain

Google Generative AI SDK

YouTube Transcript API

```
### 🧠 Tech Stack

- **LangChain** – Document processing, chains, and memory management
- **FAISS** – Facebook AI Similarity Search vector database for storing embeddings
- **Google Gemini** – LLM inference via ChatGoogleGenerativeAI
- **GoogleGenrativeAIEmbeddings** – Text embeddings using Gemini Embeddings
- **Streamlit** – Frontend UI framework for interactive applications
- **YoutubLoader & YputubeAPITranscript** – YT video transcript extraction
