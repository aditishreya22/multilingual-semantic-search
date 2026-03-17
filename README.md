# Multilingual Semantic Search System

This project implements a semantic search engine that retrieves text based on meaning rather than exact keyword matching across English and Indic languages.

## How it works
1. User enters a query
2. Text is converted into embeddings using a pretrained transformer
3. Cosine similarity is computed
4. Top-K most relevant results are returned

## Features
- Cross-lingual semantic search
- Transformer-based embeddings
- Cosine similarity ranking
- Streamlit UI for real-time search

## Example
Query: weather tomorrow  
Result: कल बारिश होगी

## Tech Stack
Python, Sentence Transformers, PyTorch, Streamlit

## My Contribution
- Built semantic search pipeline using embeddings
- Integrated pretrained multilingual transformer
- Developed Streamlit UI for real-time interaction

## Note
This project was developed as part of hands-on learning and experimentation with NLP systems.
