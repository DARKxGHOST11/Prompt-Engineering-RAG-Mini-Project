# Prompt Engineering & RAG Mini Project

### Objective
Build a Retrieval-Augmented Generation (RAG) system to answer questions using company policy documents. Focus on prompt design, retrieval quality, and evaluation.

---

## 1. Data Preparation
- **Loading:** All policy documents in `data/` are loaded (TXT format).
- **Chunking:**
  - Chosen chunk size: 500 characters, 75 overlap.
  - **Reason:** 500 chars balances context (enough for a full policy clause) and retrieval precision. Overlap ensures no information is lost at chunk boundaries.

---

## 2. RAG Pipeline
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`).
- **Vector Store:** FAISS (in-memory, fast for small datasets).
- **Retrieval:** Top-3 semantic matches, threshold tuned for recall.
- **LLM:** Gemini (Google Generative AI).
- **Flow:**
  1. User question → embedding
  2. Retrieve top-k chunks
  3. Build prompt with context
  4. LLM generates answer

---

## 3. Prompt Engineering
### Initial Prompt (v1)
```
You are a policy assistant.
Use ONLY the provided context to answer the question.
If the answer is not found, say:
"The information is not available in the provided documents."
Context:
{context}
Question:
{question}
```

### Improved Prompt (v2)
```
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
```

#### **Prompt Iteration Explanation:**
- v2 adds stricter instructions, partial answer handling, and citation requirements for clarity and hallucination control.

---

## 4. Evaluation
### Evaluation Set
- What is the refund timeline? (✅)
- Can I cancel after shipping? (⚠️)
- Are international returns free? (❌)
- Do you offer student discounts? (❌)
- Who is the CEO? (❌)
- Are shipping charges refundable? (✅)
- What if my product is damaged? (✅)

### Scoring Rubric
- ✅ = Accurate, grounded, clear
- ⚠️ = Partially correct or incomplete
- ❌ = Hallucination or not found

### Results (Manual)
| Question                        | Model Answer Quality |
|----------------------------------|---------------------|
| What is the refund timeline?     | ✅                  |
| Can I cancel after shipping?     | ⚠️                  |
| Are international returns free?  | ❌                  |
| Do you offer student discounts?  | ❌                  |
| Who is the CEO?                  | ❌                  |
| Are shipping charges refundable? | ✅                  |
| What if my product is damaged?   | ✅                  |

---

## 5. Edge Case Handling
- If no relevant context is found, model responds: "The information is not available in the provided documents."
- For out-of-scope questions, same fallback response.

---

## Architecture Overview
- `main.py`: RAG pipeline, CLI
- `data/`: Policy documents
- `.env`: API key
- `requirements.txt`: Dependencies

---

## Setup Instructions
1. Clone repo
2. `pip install -r requirements.txt`
3. Add your Gemini API key to `.env`
4. Run: `python main.py`

---

## Trade-offs & Improvements
- **Proud of:** Prompt clarity, strict hallucination control, simple/robust pipeline.
- **Next improvement:** Add reranking, JSON output schema, or prompt templating (LangChain).

---

## Submission Note
- Most proud of: Clear, structured prompts and robust fallback for missing info.
- Next improvement: Add reranking or output schema validation for even more reliable answers.
