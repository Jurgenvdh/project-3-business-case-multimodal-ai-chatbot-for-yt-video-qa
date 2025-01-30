import os
import re
import openai
import base64
import tiktoken
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from uuid import uuid4
from dotenv import load_dotenv

# YouTube and Whisper
from youtube_transcript_api import YouTubeTranscriptApi
import whisper

# LangChain Imports
from typing import List, Any
from pydantic import BaseModel
from pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, Document, BaseRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool

# Streamlit
import streamlit as st

# Load Environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Validate API Keys
if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing required API keys in .env file.")

openai.api_key = OPENAI_API_KEY

# YouTube Transcript Functions
def extract_video_id(youtube_link: str) -> str:
    """Extracts the video ID from a YouTube link."""
    if "youtu.be/" in youtube_link:
        return youtube_link.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com" in youtube_link and "v=" in youtube_link:
        return youtube_link.split("v=")[1].split("&")[0]
    return ""

def fetch_youtube_transcript(video_id: str, lang="en") -> list:
    """Fetch transcript from YouTube."""
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    for candidate in transcript_list:
        if candidate.language_code.startswith(lang):
            return candidate.fetch()
    return transcript_list[0].fetch()

def fallback_whisper_transcribe(video_id: str) -> list:
    """Fallback for transcription with Whisper."""
    return [{"text": "Whisper transcription not implemented.", "start": 0, "duration": 0}]

def get_transcript_or_whisper(youtube_link: str, fallback=True) -> list:
    """Retrieve YouTube transcript or fallback to Whisper."""
    video_id = extract_video_id(youtube_link)
    if not video_id:
        raise ValueError("Invalid YouTube link.")
    try:
        return fetch_youtube_transcript(video_id)
    except Exception as e:
        if fallback:
            return fallback_whisper_transcribe(video_id)
        raise ValueError(f"Transcript unavailable. Error: {e}")

# Transcript Cleaning
def clean_transcript_with_gpt(transcript: list) -> str:
    """
    Sends the transcript (list of dicts) to GPT-3.5 in chunks for cleaning,
    grammar, speaker identification, etc. Returns a single cleaned string.
    """
    raw_text = "\n".join(item["text"] for item in transcript)

    # Initialize tiktoken encoding for GPT-3.5-turbo
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(raw_text)

    # Chunk the tokens into manageable sizes
    CHUNK_SIZE = 600
    chunks = [
        encoding.decode(tokens[i:i + CHUNK_SIZE])
        for i in range(0, len(tokens), CHUNK_SIZE)
    ]

    # Set up the GPT-3.5 API
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=800,
        openai_api_key=OPENAI_API_KEY
    )

    # Prepare system message for cleaning
    system_msg = SystemMessage(
        content="You are a helpful assistant who cleans transcripts. "
                "Fix grammar, identify speakers if possible, and maintain readability. "
                "No extraneous commentary."
    )

    cleaned_chunks = []
    for chunk in chunks:
        human_msg = HumanMessage(content=chunk)
        response = chat([system_msg, human_msg])
        cleaned_chunks.append(response.content.strip())

    cleaned_text = "\n".join(cleaned_chunks)
    return cleaned_text

# Pinecone Integration
def chunk_and_embed_text(cleaned_text: str, index_name="youtube-video"):
    """Split, embed, and upsert text into Pinecone."""
  
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(cleaned_text)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    vectors_to_upsert = [
        (f"doc-{i}", embeddings.embed_query(chunk), {"text": chunk})
        for i, chunk in enumerate(chunks)
    ]

    index.upsert(vectors=vectors_to_upsert)
    return index

class PineconeRetriever(BaseRetriever, BaseModel):
    pinecone_index: Any
    embedding_model: Any
    top_k: int = 3

    class Config:
        arbitrary_types_allowed = True  # Allow non-primitive fields like Pinecone index

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        response = self.pinecone_index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True
        )
        docs = [
            Document(
                page_content=result.metadata["text"],
                metadata={"source": result.metadata.get("source", "")}
            )
            for result in response.matches
        ]
        return docs

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError("Async not implemented yet.")

# QA Agent
def build_qa_agent(pinecone_index):
    """
    Creates a conversational RAG agent using the Pinecone index for retrieval.
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )

    # Create the embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create the custom Pinecone retriever
    retriever = PineconeRetriever(
        pinecone_index=pinecone_index,
        embedding_model=embeddings,
        top_k=3
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever
    )

    # Wrap into tools
    tools = [
        Tool(
            name="Knowledge Base",
            func=qa_chain.run,
            description="You are a helpful assistant who uses this tool to answer user questions based on the YouTube video content. You will answer correctly, in full and as detailed as possible. If no answers to the question can be found, you'll apologize and say so. Under no circumstances will you provide answers that are not 100% truthful."
        )
    ]

    # Set up conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",
        memory=conversational_memory,
        verbose=True,
        max_iterations=3
    )

    return agent

# Streamlit App
# Function to set the background image
def set_background(image_path):
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

    # Set images
    set_background("images/background.png")

    # Apply font style dynamically
    st.markdown(
        f"""
        <style>
        h1, h2, h3, h4, h5, h6, p, .stApp {{
            font-family: 'Josefin Sans', monospace;
            color: #24012F;
        }}
        </style>
        """,
    unsafe_allow_html=True
    )

    # Main app content
    st.title("YouTube ChatBoT")
    youtube_link = st.text_input("YouTube Link", "")
    
    if st.button("Let's go!") and youtube_link:
        try:
            with st.spinner("Fetching transcript..."):
                transcript = get_transcript_or_whisper(youtube_link)
            with st.spinner("Just a minute..."):
                cleaned_text = clean_transcript_with_gpt(transcript)
            with st.spinner("Almost there..."):
                pinecone_index = chunk_and_embed_text(cleaned_text)
                st.success("Got it!")

            agent = build_qa_agent(pinecone_index)
            st.session_state['agent'] = agent
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    if 'agent' in st.session_state:
        user_question = st.text_input("What ya wanna know?", "")
        if st.button("Ask"):
            with st.spinner("Hmmm..let me think... "):
                response = st.session_state['agent'].run(user_question)
            st.write("**Answer:**", response)

# Main
if __name__ == "__main__":
    run_streamlit_app()
