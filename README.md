
# 🦙 Llama Chatbot with Streamlit & Ollama

A local, privacy-focused AI chatbot interface built with **Python**, **Streamlit**, and **Ollama**. This project allows you to chat with the **Llama** model (and others) running entirely locally on your machine—no API keys or internet required.

## 🚀 Features
* **100% Local:** Runs offline using Ollama. No data leaves your device.
* **Model Selection:** Easily switch between models (Llama 3, Mistral, Gemma) if installed.
* **Chat History:** Remembers previous messages in the session.
* **Clean UI:** Built with Streamlit for a simple, responsive interface.
* **Streaming Support:** Real-time text generation (typewriter effect).

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Backend:** [Ollama](https://ollama.com/)
* **Language:** Python 3.10+

---

## 📋 Prerequisites

### 1. Install Ollama
You need Ollama installed to run the LLM locally.
1.  Download it from [ollama.com](https://ollama.com).
2.  Pull the Llama model (run this in your terminal):
    ```bash
    ollama pull llama3
    ```

### 1. Install streamlit
Install Streamlit on your own machine.

Download it from [streamlit](https://docs.streamlit.io/).

### 2. Python Setup
Ensure you have Python installed. It is recommended to use a virtual environment.

```bash

# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## Start Ollama (Ensure the Ollama app is running in the background).

```

🏃‍♂️ How to Run streamlit 

```bash
streamlit run streamlit.py

```

```text
Open your browser at http://localhost:8501.

├── streamlit.py         # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Files to ignore in Git
├── README.md           # This file
└── data/               # Images/Screenshots/pdf file


📦 Requirements.txt Content

Make sure your requirements.txt contains these libraries:

```bash
ollama
chromadb
pdfplumber
langchain
langchain-core
langchain-ollama
langchain-community
langchain_text_splitters
unstructured
unstructured[all-docs]
fastembed
pdfplumber
sentence-transformers
elevenlabs
streamlit

```