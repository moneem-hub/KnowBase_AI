import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv

MODEL = os.getenv("GROQ MODEL", "openai/gpt-oss-120b")
SYSTEM_PROMPT = "You are KnowBase AI, a helful and concise assistant."
HISTORY_LIMIT = 8  # past messages sent to the model

def clean(text):
    # Remove <think>...</think> reasoning blocks some models print
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
        st.stop()
    return Groq(api_key=api_key)

st.set_page_config(page_title="KnowBase AI", page_icon="📚")
st.title("📚 KnowBase AI")
st.caption("Chat with an AI assistant. PDF question answering is coming next.")

client = get_client()

# Conversation memory: st.session_state survives Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL}")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle a new question
question = st.chat_input("Ask a question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + st.session_state.messages[-HISTORY_LIMIT:]
        + [{"role": "user", "content": question}]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, temperature=0.3, max_tokens=1000
                )
                answer = clean(response.choices[0].message.content)
            except Exception as e:
                answer = f"Something went wrong: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})