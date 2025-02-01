import os
import base64
import hashlib
import uuid

from dotenv import load_dotenv

import openai
import tiktoken
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from pinecone import Pinecone, Index

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory

from pydantic import BaseModel
from typing import List, Any

# Import message types for ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.docstore.document import Document

# ========= 1) Load Environment Variables =========
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, YOUTUBE_API_KEY]):
    raise ValueError("Missing one or more required environment variables (OpenAI, Pinecone, YouTube).")

# ========= 2) Setup APIs =========
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone client with a clear name
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Define index names and retrieve them from Pinecone
INDEX_VIDEO = "youtube-video"
INDEX_CHANNEL = "youtube-channel"
pinecone_video_index = pinecone_client.Index(INDEX_VIDEO)
pinecone_channel_index = pinecone_client.Index(INDEX_CHANNEL)

# ========= 3) Custom CSS & Background Setup =========
def apply_custom_css():
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@300;400&display=swap');

    /* Base font settings */
    body {
        font-family: 'Orbitron', sans-serif;
    }

    /* Futuristic styled container that wraps all content */
    .futuristic-box {
        background: rgba(0, 0, 0, 0.6); /* Semi-transparent background */
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0px 0px 15px rgba(161, 200, 255, 1);
        font-size: 26px;
        color: #a1c8ff; /* Ensure all text inside uses this color */
        font-family: 'Orbitron', sans-serif;
        text-shadow: 2px 2px 5px rgba(161, 200, 255, 0.8);
        max-width: 600px;
        margin: 20px auto;
    }

    /* Styling for input fields and buttons inside the futuristic box */
    .futuristic-input input, .stButton>button {
        background: rgba(0, 0, 0, 0.6);
        color: #a1c8ff;
        border: 1px solid #98F2FF;
        font-family: 'Roboto Mono', monospace;
        text-shadow: 1px 1px 5px rgba(161, 200, 255, 0.8);
        max-width: 500px;
        margin: 10px auto;
        display: block;
    }

    .stTextInput input, .stButton>button {
        color: #a1c8ff;
        font-family: 'Roboto Mono', monospace;
    }

    /* Center the main container nicely */
    .main > div > div > div > div {
        max-width: 600px;
        margin: 0 auto;
        padding-top: 50px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def set_background(image_path: str):
    """
    Set a background image for the Streamlit app so that it sits on top of the solid
    background color without blending or covering it entirely.
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
            background-color: #5E0101;
            background-image: url("data:image/gif;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ========= 4) Helper Functions =========

def extract_video_id(youtube_link: str) -> str:
    """
    Extract the video ID from a YouTube link.
    """
    if "youtu.be/" in youtube_link:
        return youtube_link.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com" in youtube_link and "v=" in youtube_link:
        return youtube_link.split("v=")[1].split("&")[0]
    return ""

def fetch_youtube_transcript(video_id: str, lang="en") -> list:
    """
    Fetch the transcript for a YouTube video ID.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            if transcript.language_code.startswith(lang):
                return transcript.fetch()
        # Fallback to the first available transcript if none match the language
        return transcript_list[0].fetch()
    except Exception as e:
        print(f"Error fetching transcript for video {video_id}: {e}")
        return []

def clean_transcript(transcript: list) -> str:
    """
    Clean and format the transcript using OpenAI's GPT model.
    """
    if not transcript:
        return ""
    
    raw_text = "\n".join(item["text"] for item in transcript)
    
    chat = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=3000,
        openai_api_key=OPENAI_API_KEY
    )

    system_msg = SystemMessage(
        content=(
            "You are a meticulous Transcript Cleaner. Correct grammar, spelling, and punctuation "
            "without altering the original meaning. Preserve technical jargon and domain-specific terms exactly. "
            "Add speaker labels in the format: Speaker 1 Name: (dialogue line); Speaker 2 Name: (dialogue line), "
            "and so on. Do NOT add extra information or commentary. Keep the style concise and clear. "
            "If a speaker is unknown, use a generic label (e.g., Speaker 3). Do not guess or fill in missing parts."
        )
    )

    cleaned_parts = []
    CHUNK_SIZE = 2000
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(raw_text)
    chunks = [encoding.decode(tokens[i: i + CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

    for chunk in chunks:
        human_msg = HumanMessage(content=chunk)
        response = chat([system_msg, human_msg])
        cleaned_parts.append(response.content.strip())

    return "\n".join(cleaned_parts)

def generate_text_hash(text: str) -> str:
    """
    Generate a unique SHA-256 hash for the given text.
    """
    return hashlib.sha256(text.encode()).hexdigest()

def embed_text(cleaned_text: str, index_obj, user_id: str, video_id: str = None, channel_id: str = None,
               chunk_size=2000, chunk_overlap=400):
    """
    Split the cleaned text into chunks and embed them into the given Pinecone index.
    """
    if not cleaned_text.strip():
        return

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(cleaned_text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )

    vectors = []
    for chunk in chunks:
        chunk_hash = generate_text_hash(chunk)
        embedded = embeddings.embed_query(chunk)
        source = f"Video: {video_id}" if video_id else f"Channel: {channel_id}"
        vectors.append((chunk_hash, embedded, {"text": chunk, "user": user_id, "source": source}))

    index_obj.upsert(vectors=vectors)

def get_video_ids_from_channel(channel_id: str, max_results=40) -> list:
    """
    Retrieve video IDs from a given YouTube channel.
    """
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=min(max_results - len(video_ids), 50),
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

def process_channel_transcripts(channel_id: str, index_obj, user_id: str, max_results=40):
    """
    Process and embed transcripts for all videos in a YouTube channel.
    """
    video_ids = get_video_ids_from_channel(channel_id, max_results)
    print(f"Found {len(video_ids)} videos in channel {channel_id}.")

    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        transcript = fetch_youtube_transcript(video_id)
        if not transcript:
            continue  # Skip if no transcript
        cleaned_text = clean_transcript(transcript)
        embed_text(cleaned_text, index_obj, user_id, video_id=video_id)

    print("All transcripts processed and embedded.")

# ========= 5) Pinecone Retriever and QA Agent =========

class PineconeRetriever(BaseRetriever, BaseModel):
    """
    Custom retriever that queries the Pinecone index for relevant text chunks.
    """
    pinecone_index: Index
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
            source = match.metadata.get("source", "Unknown")
            docs.append(Document(page_content=text_content, metadata={"source": source}))
        return docs

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError("Async retrieval is not implemented.")

def build_qa_agent(index_obj: Index) -> Any:
    """
    Create a conversational QA agent using the given Pinecone index.
    """
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",
        temperature=0.1
    )

    EMBEDDING_MODEL = "text-embedding-ada-002"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    retriever = PineconeRetriever(pinecone_index=index_obj, embedding_model=embeddings, top_k=8)

    system_prompt = (
        "You are a Factual Q&A Agent. Answer questions strictly based on the provided transcripts. "
        "Do not fabricate, guess, or add details beyond what is given. "
        "If the transcript does not contain the answer, reply: 'The transcript does not specify.' "
        "Cite the source if needed (e.g., [Video 123] or [Channel XYZ]). "
        "Keep your response precise and factual.\n\nContext: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    combine_documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_documents_chain)

    # Test the retriever to ensure it returns documents
    test_docs = retriever.get_relevant_documents("Test query")
    if not test_docs:
        raise ValueError("Retriever is not returning any documents. Check your index setup!")

    tools = [
        Tool(
            name="Knowledge Base",
            func=lambda x: retrieval_chain.invoke({"input": x}),
            description=(
                "Answers user queries based strictly on the provided transcripts. "
                "If the transcript lacks the answer, it responds: 'The transcript does not specify.'"
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
        max_iterations=3,
        handle_parsing_errors=True
    )

    return agent

def validate_answer(answer: str, sources: List[Document]) -> bool:
    """
    Validate if the answer directly follows from the provided sources.
    """
    checker_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0)
    prompt = (
        f"Does this answer directly follow from these sources?\n"
        f"ANSWER: {answer}\n"
        f"SOURCES: {', '.join([doc.page_content for doc in sources])}\n"
        "Respond with 'Yes' or 'No'."
    )
    response = checker_llm.invoke(prompt)
    return response.content.strip().lower() == "yes"

def delete_user_vectors(index_obj, user_id: str):
    """
    Delete all vectors associated with a specific user from the given Pinecone index.
    """
    try:
        query_filter = {"user": {"$eq": user_id}}
        index_obj.delete(filter=query_filter)
    except Exception as e:
        print(f"Failed to delete vectors: {e}")

# ========= 6) Streamlit Frontend =========

def run_streamlit_app():
      # Apply the custom CSS and set the background image
    apply_custom_css()
    set_background("images/background.gif")
    
    # Create a unique session user ID if not already set.
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    user_id = st.session_state["user_id"]

    # Use a flag to determine if transcript processing has completed.
    if "transcribed" not in st.session_state:
        st.session_state["transcribed"] = False

    approach = st.radio("I have questions about:", ["A video", "A channel"])

    if approach == "A video":
        youtube_link = st.text_input("YouTube video URL:")
        if st.button("Process video"):
            if not youtube_link:
                st.error("Please provide a valid YouTube link.")
            else:
                with st.spinner("Fetching transcript..."):
                    video_id = extract_video_id(youtube_link)
                    transcript = fetch_youtube_transcript(video_id)
                if not transcript:
                    st.error("No transcript available for this video.")
                else:
                    with st.spinner("Cleaning transcript..."):
                        cleaned_text = clean_transcript(transcript)
                    with st.spinner("Embedding transcript into Pinecone..."):
                        embed_text(cleaned_text, pinecone_video_index, user_id, video_id=video_id)
                    st.success("Video transcript processed successfully!")
                    # Reveal question-related UI after a successful transcript.
                    st.session_state["transcribed"] = True
    else:  # Channel approach
        st.markdown(
            "Need help finding your Channel ID? "
            "[Click here](https://seostudio.tools/youtube-channel-id) to retrieve it!"
        )
        channel_id = st.text_input("Channel ID (not a custom URL; use the link above to find it):")
        if st.button("Process channel"):
            if not channel_id:
                st.error("Please provide a Channel ID.")
            else:
                with st.spinner("Processing channel transcripts..."):
                    process_channel_transcripts(channel_id, pinecone_channel_index, user_id, max_results=40)
                st.success("Channel transcripts processed successfully!")
                st.session_state["transcribed"] = True

    # Only reveal the QA (question) UI if transcript processing was successful.
    if st.session_state.get("transcribed"):
        st.markdown("---")
        st.subheader("Ask a Question")
        
        if st.button("Initialize QA Agent"):
            if approach == "A video":
                st.session_state['agent'] = build_qa_agent(pinecone_video_index)
            else:
                st.session_state['agent'] = build_qa_agent(pinecone_channel_index)
            st.success("QA Agent is ready to answer questions!")

        if 'agent' in st.session_state:
            user_question = st.text_input("Your question:")
            if st.button("Ask"):
                with st.spinner("Thinking..."):
                    try:
                        # Pass the question as a dictionary.
                        answer = st.session_state['agent'].run({"input": user_question})
                        st.write("**Answer:**", answer)
                    except Exception as e:
                        st.error(f"An error occurred while fetching the answer: {e}")

        if st.button("End Session (Clear Data)"):
            delete_user_vectors(pinecone_video_index, user_id)
            delete_user_vectors(pinecone_channel_index, user_id)
            st.session_state.clear()
            st.success("Session ended and data deleted.")
    else:
        st.info("Please process a video or channel to unlock the Q&A features.")
    

if __name__ == "__main__":
    run_streamlit_app()
