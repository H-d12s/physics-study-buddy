# Physics Study Buddy 

An Agentic AI assistant for B.Tech Physics students built using LangGraph, ChromaDB, and Groq.

## Project Info
- **Student:** Adrish
- **Roll Number:** 23051971
- **Batch:** 2023-2027
- **Course:** Agentic AI Hands-On Course 2026
- **Trainer:** Dr. Kanthi Kiran Sirra

## What it does
- Answers Physics questions from a 10-document ChromaDB knowledge base
- Performs numerical calculations using a calculator tool
- Remembers student name and conversation context using MemorySaver
- Self-evaluates answer faithfulness and retries if score < 0.7
- Admits clearly when a topic is outside the knowledge base

## Topics Covered
Newton's Laws, Kinematics, Work/Energy/Power, Gravitation, Thermodynamics,
Waves & Oscillations, Electrostatics, Current Electricity, Optics, Modern Physics

## Tech Stack
- LangGraph — agent framework
- ChromaDB — vector knowledge base
- Groq (LLaMA 3.3 70B) — LLM
- Sentence Transformers — embeddings
- Streamlit — web UI

## Files
- `day13_capstone.ipynb` — main notebook with all 8 parts
- `agent.py` — clean production agent code
- `capstone_streamlit.py` — Streamlit web UI

## How to Run
1. Install dependencies:pip install langgraph langchain langchain-groq chromadb sentence-transformers streamlit
2. Add your Groq API key in `agent.py`
3. Run the Streamlit app:streamlit run capstone_streamlit.py

## Evaluation
- Average Faithfulness Score: 0.98
- Manual LLM-based evaluation used as RAGAS fallback (OpenAI API key required for RAGAS)
