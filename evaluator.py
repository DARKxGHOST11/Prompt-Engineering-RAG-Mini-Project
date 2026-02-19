from main import load_documents, chunk_text, build_faiss_index, answer_question

evaluation_set = [
    "What is the refund timeline?",
    "Can I cancel after shipping?",
    "Are international returns free?",
    "Do you offer student discounts?",
    "Who is the CEO?"
]

def run_evaluation():
    documents = load_documents("data")

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    index = build_faiss_index(all_chunks)

    for q in evaluation_set:
        print("\nQuestion:", q)
        answer = answer_question(q, index, all_chunks)
        print("Model Answer:\n", answer)
        print("Manual Score: \n")


if __name__ == "__main__":
    run_evaluation()
