import base64
import os
import hashlib
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
from typing import Any, List

import openai
import pinecone
import tiktoken

from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.retrievers.base import BaseRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# --------------------- Configuration Constants ---------------------

EMBEDDING_MODEL = "text-embedding-3-large"  # Updated embedding model
EMBEDDING_DIMENSION = 4096  # Update this value based on your embedding model's actual dimension

# Pinecone index names
INDEX_VIDEO = "youtube-video"
INDEX_CHANNEL = "youtube-channel"


# --------------------- Load Environment Variables ---------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")

required_env_vars = [
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENV,
    YOUTUBE_API_KEY,
    LANGSMITH_API_KEY,
    LANGSMITH_TRACING
]

if not all(required_env_vars):
    raise ValueError("🚨 Missing one or more required environment variables. Check your .env file!")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create or connect to Pinecone indexes
existing_indexes = pinecone.list_indexes()
if INDEX_VIDEO not in existing_indexes:
    pinecone.create_index(INDEX_VIDEO, dimension=EMBEDDING_DIMENSION, metric="cosine")  # Adjusted dimension
if INDEX_CHANNEL not in existing_indexes:
    pinecone.create_index(INDEX_CHANNEL, dimension=EMBEDDING_DIMENSION, metric="cosine")  # Adjusted dimension

pinecone_video_index = pinecone.Index(INDEX_VIDEO)
pinecone_channel_index = pinecone.Index(INDEX_CHANNEL)


# -------------------------- Helper Functions --------------------------

def extract_video_id(youtube_link: str) -> str:
    """
    Extracts the video ID from a YouTube link.
    """
    if "youtu.be/" in youtube_link:
        return youtube_link.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com" in youtube_link and "v=" in youtube_link:
        return youtube_link.split("v=")[1].split("&")[0]
    return ""


def fetch_youtube_transcript(video_id: str, lang: str = "en") -> List[dict]:
    """
    Fetches transcript from YouTube for a given video ID.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            if transcript.language_code.startswith(lang):
                return transcript.fetch()
        return transcript_list[0].fetch()
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return []


def clean_transcript_with_gpt(transcript: List[dict]) -> str:
    """
    Cleans and formats the transcript using OpenAI's GPT model.
    """
    if not transcript:
        return ""

    transcript_text = "\n".join(item["text"] for item in transcript)

    chat = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=3000,
        openai_api_key=OPENAI_API_KEY
    )

    system_msg = (
        "You are a meticulous Transcript Cleaner, whose job is to polish a messy transcript. "
        "You will: Correct grammar, spelling, and punctuation without altering the original meaning. "
        "Preserve all technical jargon and domain-specific terms exactly as they appear (e.g., scientific phrases, brand names, code snippets). "
        "Add speaker labels in the format: Speaker1: (dialogue line); Speaker2: (dialogue line). Continue incrementing speaker numbers if there are more than two voices. "
        "Retain chronology and context as best you can from the transcript. Do not add any new information or commentary of your own. Use a concise yet clear style. "
        "If a speaker identity is not obvious, label them generically (e.g., Speaker3)—never invent a name. "
        "Keep an eye out for paragraphs that appear to be a single person talking, but only create new speaker segments if there is an obvious shift or a new voice. "
        "You are thorough, consistent, and direct—like a no-nonsense editor. If something is unclear, you keep it minimal: do not guess or fill in. "
        "You are there to turn chaos into clarity, nothing more."
    )

    cleaned_text = []
    CHUNK_SIZE = 2000
    encoding = tiktoken.get_encoding("cl100k_base")  # Updated to a commonly used encoding
    tokens = encoding.encode(transcript_text)
    chunks = [encoding.decode(tokens[i: i + CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

    for chunk in chunks:
        human_msg = {"role": "user", "content": chunk}
        messages = [
            {"role": "system", "content": system_msg},
            human_msg
        ]
        response = chat(messages)
        cleaned_text.append(response['choices'][0]['message']['content'].strip())

    return "\n".join(cleaned_text)


def generate_text_hash(text: str) -> str:
    """
    Generates a unique SHA-256 hash for a given text chunk.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_and_embed_text(cleaned_text: str, index_obj: pinecone.Index, user_id: str, chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Splits the cleaned text into chunks and embeds them into the specified Pinecone index.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(cleaned_text)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    embedded_chunks = embeddings.embed_documents(chunks)

    vectors = [
        (
            generate_text_hash(chunk),
            embedded_chunks[i],
            {"text": chunk, "user": user_id}
        )
        for i, chunk in enumerate(chunks)
    ]

    index_obj.upsert(vectors=vectors)


def get_video_ids_from_channel(channel_id: str, max_results: int = 40) -> List[str]:
    """
    Fetches all video IDs from a given YouTube channel.
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=min(max_results - len(video_ids), 50),  # YouTube API allows max 50 per request
            pageToken=next_page_token,
            type="video"
        )
        response = request.execute()

        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids[:max_results]


def process_channel_transcripts(channel_id: str, index_obj: pinecone.Index, user_id: str, max_results: int = 40):
    """
    Fetches and processes transcripts for all videos in a YouTube channel.
    """
    video_ids = get_video_ids_from_channel(channel_id, max_results)
    st.write(f"Found {len(video_ids)} videos in channel {channel_id}.")

    for video_id in video_ids:
        st.write(f"Processing video: {video_id}")
        transcript = fetch_youtube_transcript(video_id)

        if not transcript:
            st.warning(f"No transcript available for video {video_id}. Skipping...")
            continue

        cleaned_text = clean_transcript_with_gpt(transcript)
        chunk_and_embed_text(cleaned_text, index_obj, user_id)

    st.success("All transcripts processed and embedded.")


# --------------------- Custom Pinecone Retriever ---------------------

class PineconeRetriever(BaseRetriever, BaseModel):
    """
    Custom retriever that queries the Pinecone index for relevant chunks.
    """
    pinecone_index: pinecone.Index
    embedding_model: OpenAIEmbeddings
    top_k: int = 8

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        results = self.pinecone_index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )

        docs = []
        for match in results.matches:
            text_content = match.metadata.get("text", "")
            # Assuming 'source' is not set; adjust if needed
            docs.append(Document(page_content=text_content, metadata={"source": "Unknown"}))
        return docs

    async def aget_relevant_documents(self, query: str):
        """
        Async retrieval is not implemented.
        """
        raise NotImplementedError("Async retrieval is not implemented.")


# -------------------------- QA Agent Builder --------------------------

def build_qa_agent(index_obj: pinecone.Index) -> Any:
    """
    Creates a conversational retrieval-based QA agent using the provided Pinecone index.
    """
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0.1
    )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    retriever = PineconeRetriever(pinecone_index=index_obj, embedding_model=embeddings, top_k=8)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    tools = [
        Tool(
            name="Knowledge Base",
            func=qa_chain.run,
            description=(
                "You are a Factual Q&A Agent with a single directive: answer user questions based exclusively on the provided transcripts (or sources). "
                "Strictly no fabrication; if the information is not present in the transcripts, respond with: 'The transcript does not specify.' "
                "You never invent details, interpret beyond the text, or assume. You do not guess. "
                "If needed, cite the source in square brackets, e.g., [Video 123], [Channel XYZ], or any relevant location in the provided transcripts. "
                "Keep your tone professional, with blunt honesty. If the transcript does not have the answer, say so plainly—no sugarcoating. "
                "If the answer is in the transcript, provide it; if not, respond accordingly. Keep your responses precise and focused: only the facts that appear in the transcripts. "
                "Deliver a full, complete answer with as many details as possible, yet referencing only known content from the transcript."
            )
        )
    ]

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=8,
        return_messages=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=conversational_memory,
        verbose=True,
        max_iterations=3
    )
    return agent


def validate_answer(answer: str, sources: List[Document]) -> bool:
    """
    Validates if the answer directly follows from the provided sources.
    """
    checker_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)
    prompt = (
        f"Does this answer directly follow from these sources?\n"
        f"ANSWER: {answer}\n"
        f"SOURCES: {', '.join([doc.page_content for doc in sources])}\n"
        f"Respond with 'Yes' or 'No'."
    )
    response = checker_llm(prompt)
    return response['choices'][0]['message']['content'].strip().lower() == "yes"


def delete_user_vectors(index_obj: pinecone.Index, user_id: str):
    """
    Deletes all vectors uploaded by a specific user in Pinecone.
    """
    try:
        # Pinecone's filter syntax uses the metadata fields
        query_filter = {"user": {"$eq": user_id}}
        index_obj.delete(filter=query_filter)
        st.success("Your session data has been deleted.")
    except Exception as e:
        st.error(f"Failed to delete vectors: {e}")


# -------------------------- Streamlit Frontend --------------------------

def set_background(image_path: str):
    """
    Sets the provided image as a background in Streamlit.
    """
    if not os.path.isfile(image_path):
        st.warning(f"Background image not found at {image_path}. Skipping background setting.")
        return

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def run_streamlit_app():
    """
    Runs the Streamlit application.
    """
    # Set background image
    set_background("images/background.png")

    st.title("Ask YouTube!")

    # Generate a unique user session ID
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid4())

    user_id = st.session_state["user_id"]

    approach = st.radio("I have questions about:", ["a video", "a channel"])

    if approach == "a video":
        youtube_link = st.text_input("YouTube video URL:")
        if st.button("Load video"):
            if not youtube_link:
                st.error("Please provide a valid YouTube link.")
            else:
                with st.spinner("Getting your video..."):
                    video_id = extract_video_id(youtube_link)
                    if not video_id:
                        st.error("Invalid YouTube URL provided.")
                        return
                    transcript = fetch_youtube_transcript(video_id)
                with st.spinner("This might take a few minutes..."):
                    cleaned_text = clean_transcript_with_gpt(transcript)
                with st.spinner("Almost there..."):
                    chunk_and_embed_text(cleaned_text, pinecone_video_index, user_id)
                st.success("Got it!")

    else:  # approach == "A channel"
        # Add a clickable link to the SEOStudio Channel ID tool
        st.markdown(
            "Don't know the channel ID? "
            "[Click here](https://seostudio.tools/youtube-channel-id) "
            "to retrieve it!"
        )

        channel_id = st.text_input(
            "channel ID (Use the link above to find it):"
        )

        if st.button("Load videos"):
            if not channel_id:
                st.error("Please provide a channel ID.")
            else:
                with st.spinner("This might take a few minutes..."):
                    process_channel_transcripts(channel_id, pinecone_channel_index, user_id, max_results=40)
                st.success("Got them!")

    st.markdown("---")
    st.subheader("What would you like to know?")

    if st.button("Wake up Scott, /n the ChatBott"):                 
        if approach == "a video":
            st.session_state['agent'] = build_qa_agent(pinecone_video_index)
        else:
            st.session_state['agent'] = build_qa_agent(pinecone_channel_index)

    if 'agent' in st.session_state:
        user_q = st.text_input("Enter your question:")
        if st.button("Send question"):
            if not user_q:
                st.error("Please enter a question.")
            else:
                with st.spinner("Hmm...let me think..."):
                    try:
                        answer = st.session_state['agent'].run(user_q)
                        st.write( answer)
                    except Exception as e:
                        st.error(f"Oops! Something happened while formulating your answer: {e}")

    if st.button("End session (clear data)"):
        delete_user_vectors(pinecone_video_index, user_id)
        delete_user_vectors(pinecone_channel_index, user_id)
        st.session_state.clear()  # Ensures session resets
        st.success("Session ended and vectors deleted.")


# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    run_streamlit_app()
