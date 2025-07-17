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

# Gemini setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based on the CONTEXT provided. Interpret meaning — don’t require exact matches.
    Always cite the source files with clickable links. For PDFs, include page number. For Docs, include paragraph number + snippet.
    If answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Util ---
def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def list_all_files(shared_drive_id):
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/pdf'",
        corpora="drive",
        driveId=shared_drive_id,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

# Readers
def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        return "\n".join([str(row) for row in sheet.get_all_records()])
    except:
        return ""

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        text_chunks = []
        para_counter = 0
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                paragraph_text = ""
                for element in content_item["paragraph"].get("elements", []):
                    paragraph_text += element.get("textRun", {}).get("content", "")
                if paragraph_text.strip():
                    para_counter += 1
                    text_chunks.append({
                        "text": paragraph_text.strip(),
                        "meta": f"paragraph {para_counter} — starts with: \"{paragraph_text.strip()[:60]}...\""
                    })
        return text_chunks
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
        with fitz.open(stream=fh, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text()
                for chunk in chunk_text(text):
                    chunks.append({
                        "text": clean_pdf_text(chunk),
                        "meta": f"page {i + 1}"
                    })
        return chunks
    except:
        return []

def extract_all_chunks(shared_drive_id):
    chunks = []
    for file in list_all_files(shared_drive_id):
        file_id, name, mime = file['id'], file['name'], file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        if mime == 'application/pdf':
            for chunk in read_pdf(file_id):
                chunks.append({"text": chunk['text'], "meta": f"{name}, {chunk['meta']}", "link": link})
        elif mime == 'application/vnd.google-apps.document':
            for chunk in read_google_doc(file_id):
                chunks.append({"text": chunk['text'], "meta": f"{name}, {chunk['meta']}", "link": link})
        elif mime == 'application/vnd.google-apps.spreadsheet':
            sheet_text = read_google_sheet(file_id)
            for chunk in chunk_text(sheet_text):
                chunks.append({"text": chunk, "meta": name, "link": link})
    return chunks

def get_relevant_chunks(question, chunks, top_k=5):
    docs = [c['text'] for c in chunks]
    vectorizer = TfidfVectorizer().fit(docs + [question])
    doc_vectors = vectorizer.transform(docs)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if sims[i] > 0.05]

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
        channel_id = event.get("channel")
        shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
        chunks = extract_all_chunks(shared_drive_id)

        if not chunks:
            reply = "I couldn’t read any usable files from Google Drive."
        else:
            relevant = get_relevant_chunks(question, chunks)
            if not relevant:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([f"FROM {c['meta']}\n({c['link']}):\n{c['text']}" for c in relevant])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"
                try:
                    response = model.generate_content(prompt)
                    reply = getattr(response, "text", "Oops, no answer returned.")
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print("Slack API Error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT Slack bot with PDF+Doc refs is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
