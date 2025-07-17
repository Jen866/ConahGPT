import os
import io
import fitz  # PyMuPDF
import gspread
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

app = Flask(__name__)
CORS(app)

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

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# Gemini Setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, the expert AI assistant for Actuary Consulting.
    Your job is to answer the question fully using the provided CONTEXT.
    After your full answer, add citations in this format:
    (Source: Document Name, paragraph X — starts with: "first 5 words...") or
    (Source: PDF Name, page X — starts with: "first 5 words...")
    Do not list multiple copies of the same citation.
    Do not answer with just the reference. Always answer the question.
    """
)

# Utils

def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Readers

def list_all_files(shared_drive_id):
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' or mimeType='application/pdf'",
        corpora="drive",
        driveId=shared_drive_id,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        text = ""
        para_counter = 0
        chunks = []
        for content in doc.get("body", {}).get("content", []):
            if "paragraph" in content:
                para_text = ""
                for el in content["paragraph"].get("elements", []):
                    para_text += el.get("textRun", {}).get("content", "")
                if para_text.strip():
                    para_counter += 1
                    chunks.append({
                        "text": para_text.strip(),
                        "paragraph": para_counter
                    })
        return chunks
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
        chunks = []
        with fitz.open(stream=fh, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                for chunk in chunk_text(text):
                    chunks.append({
                        "text": clean_pdf_text(chunk),
                        "page": page_num
                    })
        return chunks
    except:
        return []

# Embedding + Search
def extract_chunks(shared_drive_id):
    files = list_all_files(shared_drive_id)
    all_chunks = []
    for file in files:
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        if mime == 'application/vnd.google-apps.document':
            doc_chunks = read_google_doc(file_id)
            for ch in doc_chunks:
                all_chunks.append({"text": ch['text'], "source": name, "link": link, "para": ch['paragraph']})
        elif mime == 'application/pdf':
            pdf_chunks = read_pdf(file_id)
            for ch in pdf_chunks:
                all_chunks.append({"text": ch['text'], "source": name, "link": link, "page": ch['page']})
    return all_chunks

def get_top_chunks(query, chunks, top_k=5):
    texts = [ch['text'] for ch in chunks]
    vectorizer = TfidfVectorizer().fit(texts + [query])
    doc_vecs = vectorizer.transform(texts)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    top = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top if sims[i] > 0.05]

# Format reference

def format_reference(chunk):
    prefix = f"(Source: <{chunk['link']}|{chunk['source']}"
    if 'para' in chunk:
        snippet = chunk['text'][:40].strip().replace("\n", " ")
        return f"{prefix}, paragraph {chunk['para']} — starts with: \"{snippet}...\")"
    if 'page' in chunk:
        snippet = chunk['text'][:40].strip().replace("\n", " ")
        return f"{prefix}, page {chunk['page']} — starts with: \"{snippet}...\")"
    return f"{prefix})"

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
        channel = event.get("channel")
        shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
        chunks = extract_chunks(shared_drive_id)
        rel_chunks = get_top_chunks(question, chunks)

        if not rel_chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([f"FROM {ch['source']}\n{ch['text']}" for ch in rel_chunks])
            prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
            gemini_response = model.generate_content(prompt)
            main_answer = getattr(gemini_response, "text", "[No answer generated]").strip()
            deduped = {}
            for ch in rel_chunks:
                deduped[(ch['source'], ch.get('para'), ch.get('page'))] = ch
            references = "\n".join([format_reference(ch) for ch in deduped.values()])
            reply = f"{main_answer}\n\n{references}"

        try:
            slack_client.chat_postMessage(channel=channel, text=reply)
        except SlackApiError as e:
            print("Slack Error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT Slack bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
