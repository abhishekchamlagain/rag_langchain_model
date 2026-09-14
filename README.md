# LangChain RAG Model

This repository contains a simple Retrieval-Augmented Generation (RAG) project built with Python, LangChain, FAISS, and Hugging Face models.

The main workflow reads a text document from the data folder, splits it into chunks, creates vector embeddings, stores them in a FAISS vector database, and retrieves the most relevant context before generating an answer.

## Project Structure

- `rag_langchain.py` - Main Python script that builds the RAG pipeline.
- `RAG_with_langchain.ipynb` - Jupyter notebook version of the project.
- `data/sample_docs.txt` - Sample knowledge base used for retrieval.
- `requirements.txt` - Python dependencies for the project.

## Features

- Text loading from local files
- Chunking using `RecursiveCharacterTextSplitter`
- Embeddings using Hugging Face Sentence Transformers
- Vector search using FAISS
- Prompt construction and answer generation using LangChain
- Retrieval pipeline that answers questions using only the relevant chunks of context

## Installation

Create and activate a virtual environment:

```bash
python -m venv env

# Windows
env\Scripts\activate

# Linux / macOS
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Execute the main script:

```bash
python rag_langchain.py
```

You can also open the notebook file in Jupyter to experiment with the project interactively.

## Example Workflow

1. Load the sample text document.
2. Split the document into chunks.
3. Generate embeddings for each chunk.
4. Build a FAISS index from the chunks.
5. Retrieve the top relevant context using a retriever.
6. Send the prompt and retrieved context to the language model.
7. Return the generated answer.

## Requirements

The project uses the following major packages:

- `langchain-text-splitters`
- `langchain-huggingface`
- `langchain-community`
- `langchain-core`
- `sentence-transformers`
- `faiss-cpu`
- `transformers`
- `accelerate`

## License

This project is intended for educational and learning purposes.
