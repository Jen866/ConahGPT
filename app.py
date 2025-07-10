import os
import io
import fitz  # PyMuPDF
import gspread
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Google Auth Setup ---
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly"
]

SERVICE_ACCOUNT_JSON = os.environ["SERVICE_ACCOUNT_JSON"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)
gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

# --- Gemini Setup ---
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based on the CONTEXT provided. Interpret the meaning — don’t require exact matches.
    If the answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    Always cite the source files used, with a clickable link to the exact paragraph in the source (if available).
    """
)

# --- File Readers ---
def list_all_files_in_shared_drive(shared_drive_id):
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/pdf'",
        corpora="drive",
        driveId=shared_drive_id,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        return "\n".join([str(row) for row in sheet.get_all_records()])
    except:
        return ""

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        content = doc.get("body", {}).get("content", [])
        paragraphs = []
        for idx, element in enumerate(content):
            if "paragraph" in element:
                text_run = ""
                for e in element["paragraph"].get("elements", []):
                    text_run += e.get("textRun", {}).get("content", "")
                if text_run.strip():
                    para_id = element.get("paragraph", {}).get("paragraphStyle", {}).get("namedStyleType", "")
                    heading = f"#heading={para_id}" if para_id.startswith("HEADING") else "#heading=NORMAL_TEXT"
                    link = f"https://docs.google.com/document/d/{file_id}/edit{heading}"
                    paragraphs.append({"text": text_run.strip(), "source": doc.get("title", "Google Doc"), "link": link})
        return paragraphs
    except:
        return []

def read_pdf(file_id):
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        text = ""
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for page in pdf_doc:
                text += page.get_text()
        if text.strip():
            return [{"text": text, "source": "PDF Document", "link": f"https://drive.google.com/file/d/{file_id}"}]
        else:
            return []
    except:
        return []

# --- Chunking & Embedding ---
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_all_chunks(shared_drive_id):
    files = list_all_files_in_shared_drive(shared_drive_id)
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
            if content.strip():
                for chunk in chunk_text(content):
                    chunks.append({"text": chunk, "source": name, "link": f"https://docs.google.com/spreadsheets/d/{file_id}"})
        elif mime_type == 'application/vnd.google-apps.document':
            paragraphs = read_google_doc(file_id)
            chunks.extend(paragraphs)
        elif mime_type == 'application/pdf':
            chunks.extend(read_pdf(file_id))
    return chunks

def get_relevant_chunks(question, chunks, top_k=5):
    if not chunks:
        return []
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.1]

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    if question.lower() in ["hi", "hello", "hey", "how are you", "what's up"]:
        return jsonify({"answer": "Hi there! 😊 How can I help you today with anything actuarial or business-related?"})

    shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
    chunks = extract_all_chunks(shared_drive_id)

    if not chunks:
        return jsonify({"answer": "I couldn’t read any usable files from Google Drive. Please double check that your folder contains readable Google Docs, Sheets, or PDFs."})

    relevant_chunks = get_relevant_chunks(question, chunks)
    if not relevant_chunks:
        return jsonify({"answer": "I cannot answer this question as the information is not in the provided documents."})

    context = "\n\n".join([f"FROM {chunk['source']} ({chunk['link']}):\n{chunk['text']}" for chunk in relevant_chunks])
    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"

    try:
        gemini_response = model.generate_content(prompt)
        answer = getattr(gemini_response, "text", "Oops, no answer returned.")

        # Add sources below the answer
        sources = "\n\n**Sources:**\n" + "\n".join(
            [f"- [{chunk['source']}]({chunk['link']})" for chunk in relevant_chunks]
        )
        return jsonify({"answer": answer + sources})
    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"}), 500

# --- Run ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
