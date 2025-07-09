import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import drive_utils # Import our new utility file
from whitenoise import WhiteNoise # Import WhiteNoise

# --- Flask Setup ---
app = Flask(__name__)
# This tells your app how to serve static files like script.js and style.css in production.
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
CORS(app)


# --- Gemini Setup ---
# Load API Key from environment variable for security
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("FATAL ERROR: The GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based ONLY on the CONTEXT provided.
    If the answer is not found in the context, you MUST say: 'I cannot answer this question as the information is not in the provided documents.'
    When you cite a source from the context, you MUST use the exact link provided for that source.
    Format the citation as a clickable HTML link at the end of your answer, like this: <a href="THE_LINK_FROM_CONTEXT" target="_blank">[Source: File Name]</a>.
    Do not make up information. Be concise and professional.
    """
)

# --- Semantic Search ---
def get_relevant_chunks(question, chunks, top_k=5):
    """Finds the most relevant chunks using TF-IDF and cosine similarity."""
    if not chunks:
        return []
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words='english').fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()

    top_indices = [i for i in similarities.argsort()[-top_k:][::-1] if similarities[i] > 0.1]

    # De-duplicate chunks from the same source to avoid sending redundant context
    unique_chunks = []
    seen_links = set()
    for i in top_indices:
        chunk = chunks[i]
        if chunk['link'] not in seen_links:
            unique_chunks.append(chunk)
            seen_links.add(chunk['link'])

    return unique_chunks


# --- Routes ---
@app.route("/")
def index():
    # This will render the index.html file from the 'templates' folder
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    # Your Google Drive Folder ID
    folder_id = "1bS_LeR9Gcn0g22I7yWcQ-i5i1t5u1u3y"
    chunks = drive_utils.extract_all_chunks_with_links(folder_id)

    if not chunks:
        return jsonify({"answer": "I couldn’t read any usable files from Google Drive. Please double check that your folder contains readable Google Docs, Sheets, or PDFs."})

    relevant_chunks = get_relevant_chunks(question, chunks)
    if not relevant_chunks:
        return jsonify({"answer": "I cannot answer this question as the information is not in the provided documents."})

    context = "\n\n".join([f"FROM {chunk['source']} (Link: {chunk['link']}):\n{chunk['text']}" for chunk in relevant_chunks])
    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"

    try:
        gemini_response = model.generate_content(prompt)
        answer = getattr(gemini_response, "text", "Oops, no answer returned from the model.")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"answer": f"Server error when contacting the AI model: {str(e)}"}), 500

# --- Run Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
