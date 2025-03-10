import os
import io
import base64
import json
import requests
from PIL import Image
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Tuple
from loguru import logger
from pydantic import BaseModel, Field
# pdf reader
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import tempfile

from prompts import SYSTEM_PROMPT, IMG_PROMPT, DOC_PROMPT

# Constants
DOC_SUPPORTED = ["pdf", "txt"]
IMG_SUPPORTED = ["png", "jpg", "jpeg"]


def load_pdf_page_as_image(pdf_path, page_number):
    try:
        pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        if not pages:
            raise ValueError(f"Page {page_number} not found in PDF.")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            pages[0].save(temp_image.name, "JPEG")
            temp_image_path = temp_image.name

        with open(temp_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        return base64_image

    except Exception as e:
        print(f"Error while processing PDF: {e}")
        return None


def get_llm_engine(
        model_id=None,
        api_key=None,
        max_new_tokens=400,
        temperature=0.0) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_id,
        max_tokens=max_new_tokens,
        temperature=temperature,
        model_kwargs={
        },
    )


def prepare_image(file):
    img = Image.open(file)
    st.image(img)
    img_base64 = base64.b64encode(file.read()).decode("utf-8")
    return img_base64


def prepare_image_path(image_path, image_format="JPEG"):
    try:
        image = Image.open(image_path)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image_format)
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Error while preparing image: {e}")
        return None


def send_request_with_image_to_openrouter(image_base64,
                                          user_prompt=IMG_PROMPT,
                                          system_prompt=SYSTEM_PROMPT,
                                          model="gpt-4o",
                                          temperature=0.0,
                                          api_key=""):
    """
    Sends a multimodal request (text + image) to OpenRouter's API.

    Parameters:
    - image_base64: str, Base64-encoded image data
    - user_prompt: str, Text prompt from the user
    - system_prompt: str, System instruction for the model
    - model: str, Model identifier (default: "gpt-4o")
    - temperature: float, Sampling temperature for response generation
    - api_key: str, OpenRouter API key

    Returns:
    - str: Response content from the model
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        logger.info(f"Response: {response.json()}")
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
    else:
        return f"Error: {response.status_code}, {response.text}"


def ask_llm(input, llm: ChatOpenAI, messages: list, **kwargs) -> str:
    messages = _convert_messages(messages)
    prompt = ChatPromptTemplate.from_messages(messages)
    openrouter_chain = prompt | llm
    params = {"input": input}
    kwargs.update(params)
    response = openrouter_chain.invoke(params)
    return response


def _convert_messages(messages: List[dict]) -> List[tuple]:
    roles_mapping = {"user": "human", "assistant": "assistant", "system": "system"}
    messages = [(roles_mapping[msg["role"]], msg["content"]) for msg in messages]
    messages.append(("human", "{input}"))
    return messages


def _convert_history(messages: List[dict]) -> List[dict]:
    exclude_roles = ["document"]
    return [{"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] not in exclude_roles]


def _format_message_as_html(content, role):
    color_mapping = {
        "user": "blue",
        "assistant": "green",
        "system": "red",
        "document": "purple"
    }
    return f'<p style="color:{color_mapping[role]}">{content}</p>'


def show_chat_messages(messages: List[dict]):
    for msg in messages:
        if msg["role"] == "document":
            with st.expander("Document"):
                st.markdown(msg["content"])
        else:
            avatar = "🩺" if msg["role"] == "assistant" else "😷"
            st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])


def process_image_with_llm(file, model="gpt-4o", api_key=""):
    img = Image.open(file)
    img = img.convert("RGB")
    img_path = file.name
    img_format = img.format
    img.save(img_path, format=img_format)
    st.image(img)
    img_base64 = prepare_image_path(img_path)
    if not img_base64:
        return
    response = send_request_with_image_to_openrouter(
        image_base64=img_base64, api_key=api_key, model=model
    )
    return response


def process_pdf_with_llm(file, model="gpt-4o"):
    pdf = PdfReader(file)
    text = ""
    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        text += page.extract_text()
    query = DOC_PROMPT.format(text=text)
    logger.info(f"🟢🟢🟢 Query: {query}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    response = ask_llm(query, get_llm_engine(model_id=model), messages)
    return response.content


def _update_messages_with_document(document):
    messages = st.session_state["messages_history"]
    user_input = "I uploaded a document, please analyze it."
    assistant_response = "I have analyzed the document. Here are the results:\n----\n{document}\n----\n"
    assistant_response = assistant_response.format(document=document)
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "system", "content": assistant_response})
    return messages


def process_uploaded_file(file, model_id):
    docs = [f"application/{doc}" for doc in DOC_SUPPORTED]
    imgs = [f"image/{img}" for img in IMG_SUPPORTED]
    if file.type in docs:
        text = ""
        if file.type == "application/pdf":
            if st.sidebar.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    pdf_content = st.session_state["pdf_reader"](file)
                    doc_length = len(pdf_content.pages)
                    st.write(f"Document has {doc_length} pages")
                    if doc_length > 100:
                        st.error("The document is too long to process. Please upload a shorter document.")
                    elif doc_length == 0:
                        st.error("The document is empty.")
                    # elif doc_length == 1:
                    #     # save pdf to temp file
                    #     with open(file.name, "wb") as f:
                    #         f.write(file.read())
                    #     image = load_pdf_page_as_image(file.name, 0)
                    #     st.image(image)
                    else:
                        text = process_pdf_with_llm(file, model=model_id)
                        st.session_state["document_content"][file.name] = text
                        st.session_state["messages_history"] = _update_messages_with_document(text)
                        st.markdown(text)

        elif file.type == "text/plain":
            if st.sidebar.button("Process text"):
                with st.spinner("Processing text..."):
                    text = file.read().decode("utf-8")
                    st.write(text)
        return text
    elif file.type in imgs:
        if st.sidebar.button("Process image"):
            with st.spinner("Processing image..."):
                response = process_image_with_llm(file, model=model_id, api_key=st.session_state["api_key"])
                if response:
                    st.session_state["document_content"][file.name] = response
                    st.session_state["messages_history"] = _update_messages_with_document(response)
                else:
                    st.error("Error while processing image.")


if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("OPENROUTER_API_KEY")
if not st.session_state["api_key"]:
    st.sidebar.warning("Please provide an OpenRouter API key to use the chat.")

# Initialize the chat
if "document_content" not in st.session_state:
    st.session_state["document_content"] = {}
if "pdf_reader" not in st.session_state:
    st.session_state["pdf_reader"] = PdfReader
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = [{"role": "system", "content": SYSTEM_PROMPT}]

with st.sidebar:
    if st.sidebar.button("Clear session"):
        st.session_state["messages_history"] = []
        st.session_state["document_content"] = {}
        st.session_state["pdf_reader"] = PdfReader

    with st.sidebar.expander("#### ⚙️ Settings"):
        api_key = st.text_input("OpenRouter API key", st.session_state["api_key"])
        st.session_state["api_key"] = api_key
        model_id = st.selectbox("Model", [
            "deepseek/deepseek-r1:free",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-pro-exp-02-05:free",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-nemo",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-r1-distill-llama-70b",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
            "qwen/qwq-32b",
            "x-ai/grok-2-1212"

        ])
        system_prompt = st.text_area("System prompt", SYSTEM_PROMPT)
        st.session_state["messages_history"][0]["content"] = system_prompt
        with st.sidebar.expander("Chat history"):
            show_chat_messages(st.session_state["messages_history"])
        with st.sidebar.expander("Uploaded documents"):
            for doc_name in st.session_state["document_content"].keys():
                st.write(doc_name)
        st.sidebar.markdown("------")

    documents = st.file_uploader("Upload documents", type=["pdf", "png", "txt", "jpg"], accept_multiple_files=False)
    if documents:
        process_uploaded_file(documents, model_id)

st.caption(f"🩺 Chat with the AI. Powered by {model_id}")
for msg in st.session_state.messages_history[1:]:
    st.chat_message(msg["role"]).write(msg["content"])

for doc_name, doc_content in st.session_state["document_content"].items():
    with st.sidebar.expander(f"📄 Document: {doc_name}"):
        st.write(doc_content)

if prompt := st.chat_input():
    logger.info(f"User input: {prompt}")
    st.chat_message("user").markdown(_format_message_as_html(prompt, "user"), unsafe_allow_html=True)
    client = get_llm_engine(model_id=model_id, api_key=st.session_state["api_key"])
    messages = _convert_history(st.session_state["messages_history"])
    response = ask_llm(prompt, client, messages)
    logger.info(f"AI response: {response.content}")
    msg = response.content

    st.session_state["messages_history"].append({"role": "user", "content": prompt})
    st.session_state["messages_history"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").markdown(_format_message_as_html(msg, "assistant"), unsafe_allow_html=True)
