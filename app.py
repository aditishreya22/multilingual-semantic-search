import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("Multilingual Semantic Search (English ↔ Indic)")
st.caption("Embeddings + cosine similarity (MiniLM multilingual)")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

default_corpus = """Machine learning is amazing
मशीन लर्निंग बहुत शक्तिशाली है
I love pizza
मुझे क्रिकेट पसंद है
Artificial intelligence is the future
कल बारिश होगी"""

corpus_text = st.text_area("Corpus (one sentence per line)", value=default_corpus, height=180)
corpus = [x.strip() for x in corpus_text.splitlines() if x.strip()]

query = st.text_input("Query", value="weather tomorrow")
top_k = st.slider("Top-K results", 1, min(10, len(corpus)), 3)

if st.button("Search"):
    corpus_emb = model.encode(corpus, convert_to_tensor=True)
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, corpus_emb)[0]
    top = scores.topk(k=top_k)

    st.subheader("Results")
    for score, idx in zip(top.values, top.indices):
        st.write(f"**{score.item():.3f}**  →  {corpus[int(idx)]}")



