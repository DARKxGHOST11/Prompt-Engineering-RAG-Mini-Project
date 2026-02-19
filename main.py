import os
from dotenv import load_dotenv
import faiss
import numpy as np
import google.generativeai as genai
from typing import List
from sentence_transformers import SentenceTransformer


DATA_FOLDER = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 3
SIMILARITY_THRESHOLD = 2.0

GEMINI_MODEL = "models/gemini-2.5-flash"




load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(GEMINI_MODEL)



_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model



def load_documents(folder_path: str) -> List[str]:
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents



def chunk_text(text: str) -> List[str]:
    # If the document is short, return as a single chunk
    if len(text) < CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks



def build_faiss_index(chunks: List[str]):
    model = get_embedding_model()
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index



def retrieve(query: str, index, chunks: List[str]) -> List[str]:
    model = get_embedding_model()
    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), TOP_K)

    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks) and dist < SIMILARITY_THRESHOLD:
            retrieved.append(chunks[idx])

    return retrieved



def build_prompt_v1(context: str, question: str) -> str:
    return f"""
You are a policy assistant.

Use ONLY the provided context to answer the question.
If the answer is not found, say:
"The information is not available in the provided documents."

Context:
{context}

Question:
{question}
"""


def build_prompt_v2(context: str, question: str) -> str:
    return f"""
You are a strict company policy assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. If not found, respond exactly:
   "The information is not available in the provided documents."
4. If partially found, clearly explain what is known and what is missing.
5. Cite relevant clauses from the context.

Context:
----------------
{context}
----------------

Question:
{question}

Format:

Answer:
- ...

Source Evidence:
- ...
"""


def call_gemini(prompt: str) -> str:
    response = llm_model.generate_content(
        prompt,
        generation_config={"temperature": 0}
    )
    return response.text



def answer_question(question: str, index, chunks: List[str], prompt_version="v2"):
    retrieved_chunks = retrieve(question, index, chunks)

    if not retrieved_chunks:
        return "The information is not available in the provided documents."

    context = "\n\n".join(retrieved_chunks)

    if prompt_version == "v1":
        prompt = build_prompt_v1(context, question)
    else:
        prompt = build_prompt_v2(context, question)

    return call_gemini(prompt)


def build_system():
    print("Loading documents...")
    documents = load_documents(DATA_FOLDER)

    print("Chunking documents...")
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    print(f"Total chunks created: {len(all_chunks)}")

    print("Building FAISS index...")
    index = build_faiss_index(all_chunks)

    return index, all_chunks


evaluation_set = [
    "What is the refund timeline?",
    "Can I cancel after shipping?",
    "Are international returns free?",
    "Do you offer student discounts?",
    "Who is the CEO?"
]

def run_evaluation(index, chunks):
    print("\nRunning Evaluation...\n")

    for question in evaluation_set:
        print("Question:", question)
        answer = answer_question(question, index, chunks, "v2")
        print("Model Answer:\n", answer)
        print("Manual Score: \n")


if __name__ == "__main__":

    index, chunks = build_system()

    print("\n RAG System Ready!\n")

    while True:
        mode = input("Choose mode: (1) Ask Question  (2) Run Evaluation  (exit): ")

        if mode.lower() == "exit":
            break

        if mode == "1":
            question = input("\nEnter your question: ")
            answer = answer_question(question, index, chunks, "v2")
            print("\n=== ANSWER ===")
            print(answer)
            print("\n")

        elif mode == "2":
            run_evaluation(index, chunks)

        else:
            print("Invalid option.\n")
