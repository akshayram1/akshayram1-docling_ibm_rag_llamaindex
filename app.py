import os
import gc
import tempfile
import uuid
import pandas as pd

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

import streamlit as st

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 10px;
    }
    .file-uploader {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .success-message {
        color: #4CAF50;
        font-weight: bold;
    }
    .chat-container {
        margin-top: 20px;
    }
    .clear-button {
        font-size: 1rem;
        background-color: #FF5733;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():
    llm = Ollama(model="llama3.2", request_timeout=120.0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_excel(file):
    st.markdown("### Excel Preview")
    df = pd.read_excel(file)
    st.dataframe(df)

# App Header
st.markdown("<div class='main-header'>RAG over Excel with Dockling & Llama-3.2</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sub-header'>Upload your documents</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose your `.xlsx` file", type=["xlsx", "xls"], key="file_uploader", help="Supported file types: .xlsx, .xls")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        reader = DoclingReader()
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            file_extractor={".xlsx": reader},
                        )
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # Setup LLM & embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

                    Settings.embed_model = embed_model
                    node_parser = MarkdownNodeParser()
                    index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

                    # Create the query engine
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # Customise prompt template
                    qa_prompt_tmpl_str = (
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.markdown("<div class='success-message'>Ready to Chat!</div>", unsafe_allow_html=True)
                display_excel(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown("<div class='sub-header'>Start Chatting with Your Data</div>", unsafe_allow_html=True)

with col2:
    st.button("Clear ↺", on_click=reset_chat, key="clear_button", help="Clear chat history and reset the context.", args=None, kwargs=None, type="primary", use_container_width=False, disabled=False)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = query_engine.query(prompt)

        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
