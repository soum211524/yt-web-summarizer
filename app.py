
import os
import validators
import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document

from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# --- Load API Key ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Streamlit UI ---
st.set_page_config(page_title="YT/Web Summarizer", page_icon="🧠")
st.title("🎥 Youtube and 🌐 Website Summarizer")
st.markdown("Paste a YouTube link or website URL to get a summarized version of its content.")

# --- Input ---
generic_url = st.text_input("🔗 Enter YouTube or Website URL")

# --- Prompt ---
prompt_template = """
Summarize the following content in approximately 300 words:

{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# YouTube transcript
def safe_youtube_loader(url: str) -> list[Document]:
    try:
        video_id = parse_qs(urlparse(url).query).get("v", [""])[0]
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-IN", "hi"]
        )
        full_text = " ".join([t["text"] for t in transcript])
        return [Document(page_content=full_text, metadata={"source": url})]
    except Exception as e:
        st.error(f"❌ Could not load YouTube transcript: {e}")
        return []

# lets summarize
if st.button("📄 Summarize"):
    if not groq_api_key or not groq_api_key.strip():
        st.error(" Please set your GROQ_API_KEY in the .env file.")
    elif not generic_url or not validators.url(generic_url):
        st.error("❌ Enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("🧠 Thinking..."):
                # Load 
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = safe_youtube_loader(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.warning("⚠️ No content found to summarize.")
                else:
                    llm = ChatGroq(
                        model_name="llama3-70b-8192",
                        api_key=groq_api_key
                    )
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)
                    st.success("✅ Summary generated!")
                    st.markdown(summary)

        except Exception as e:
            st.error("🚨 An unexpected error occurred.")
            st.exception(e)