# FinSight AI

> Agentic Personal Finance Intelligence Platform — parses bank/credit card statements, 
> auto-categorizes transactions, detects anomalies, forecasts spending, and answers 
> natural-language financial questions via a tool-using LLM agent grounded in RAG.

## Status
🚧 In active development (Day 1 of 30) — building in public.

## Problem
Most people don't understand their own financial behavior. Statements are unstructured 
and time-consuming to analyze manually. This project builds an end-to-end AI system 
that automates categorization, surfaces anomalies, forecasts cash flow, and lets users 
converse with their own financial data.

## Architecture
(diagram coming soon — see docs/)

## Tech Stack
Python, pandas, scikit-learn, sentence-transformers, ChromaDB, LangChain, Prophet, 
FastAPI, Streamlit

## Roadmap
- [x] Day 1: Project setup
- [ ] Week 1: Data pipeline (synthetic data + statement parsing)
- [ ] Week 2: Categorization, anomaly detection, forecasting
- [ ] Week 3: RAG + agentic layer
- [ ] Week 4: API, frontend, deployment

## Setup
\`\`\`bash
git clone https://github.com/dhiraj1308/finsight-ai.git
cd finsight-ai
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`