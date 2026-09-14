"""
rag_langchain.py
-----------------
RAG pipeline using LangChain's prebuilt components instead of writing 
FAISS/prompt-building code."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

DATA_PATH = "data/sample_docs.txt"
CHUNK_SIZE = 400       # characters
CHUNK_OVERLAP = 80
TOP_K = 3
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vectorstore(raw_text: str):
    print("Step 1: Chunking (RecursiveCharacterTextSplitter)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(raw_text)
    print(f"  -> {len(chunks)} chunks created")

    print("Step 2: Embedding + storing in FAISS (HuggingFaceEmbeddings + FAISS.from_texts)...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings_model)
    print("  -> Vector store built")
    return vectorstore


def build_chain(vectorstore):
    print("Step 3: Building retriever (.as_retriever())...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print("Step 4: Loading LLM (transformers pipeline wrapped in HuggingFacePipeline)...")
    hf_pipeline = pipeline("text-generation", model=LLM_MODEL, max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print("Step 5: Building prompt template (PromptTemplate)...")
    prompt_template = PromptTemplate(
        template="Answer using only this context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("Step 6: Assembling the chain (retriever | prompt | llm | parser)...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever, prompt_template, format_docs


def ask(rag_chain, retriever, prompt_template, format_docs, query: str) -> str:
    result = rag_chain.invoke(query)
    # This LLM isn't instruction-tuned for this pipeline path, so the raw
    # output echoes the prompt back — strip that off to get just the answer.
    echoed_prompt = prompt_template.format(
        context=format_docs(retriever.invoke(query)),
        question=query,
    )
    return result[len(echoed_prompt):].strip()


def main():
    raw_text = load_text(DATA_PATH)
    vectorstore = build_vectorstore(raw_text)
    rag_chain, retriever, prompt_template, format_docs = build_chain(vectorstore)

    query = input("\nAsk a question about the sample docs: ").strip()
    print("\nRunning the chain...")
    answer = ask(rag_chain, retriever, prompt_template, format_docs, query)

    print("\n=== ANSWER ===")
    print(answer)


if __name__ == "__main__":
    main()
