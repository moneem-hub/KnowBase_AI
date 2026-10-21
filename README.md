# 📚 KnowBase AI

An AI chatbot that can chat normally or answer questions about your own PDF documents using **Retrieval-Augmented Generation (RAG)**, with page-level source references.

## Features

- **AI chat**: ask general questions and get AI-generated answers
- **Chat with PDF**: upload a PDF and ask questions about its contents
- **RAG pipeline**: relevant passages are retrieved before the answer is generated
- **Source references**: every answer shows the pages it came from
- **Conversation memory**: follow-up questions work within a session
- **Unknown-answer handling**: if the document doesn't contain the answer, the bot says so instead of inventing one

## How it works

```text
Document processing:  PDF -> text extraction -> chunking -> embeddings -> FAISS index
Question answering:   question -> retrieve top chunks -> Groq LLM -> answer + sources
```

The retrieval pipeline supplies information from the document, and the LLM writes the answer from it. If no chunk is similar enough to the question, the app skips the LLM call and returns a "not found" message.

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Chat interface |
| Groq API | LLM inference |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Text embeddings |
| FAISS | Similarity search over chunks |
| PyMuPDF | PDF text extraction |
| Jupyter Notebook | Experimentation (`chatbot.ipynb`) |

## Setup

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Create a `.env` file (see `.env.example`) with your key from the [Groq Console](https://console.groq.com/keys):
```text
   GROQ_API_KEY=your_key_here
```
5. Run the app:
```bash
   streamlit run app.py
```

Optional: set `GROQ_MODEL` in `.env` to use a different Groq model.

## Usage

1. Upload a text-based PDF in the sidebar and click **Process PDF**.
2. Ask questions in the chat box.
3. Open the **Sources** expander under an answer to see the pages it used.

## Project structure

```text
KnowBase_AI/
├── app.py              # Streamlit app and RAG pipeline
├── chatbot.ipynb       # Step-by-step experiments
├── requirements.txt
├── .env.example        # API key template
└── .gitignore
```

## Limitations

- Scanned PDFs (images without a text layer) are not supported, as there is no OCR.
- One document at a time; the index lives in memory and resets when the app restarts.
- Broad questions such as "summarize this document" can retrieve poorly, because search matches specific content.

## Possible improvements

- OCR for scanned PDFs
- Multiple documents and a saved index
- Smarter retrieval, such as reranking or query rewriting