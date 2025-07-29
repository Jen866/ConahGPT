import os
import io
import fitz  # PyMuPDF
import gspread
import json
import re
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime
import google.generativeai as genai

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Google Auth ---
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly"
]
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)
gspread_client = gspread.authorize(creds)

# --- Gemini Setup ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, a business assistant for Actuary Consulting.
    Answer user questions fully using only the provided CONTEXT.
    Provide only one answer per question. Include citations in this format:
    (Source: [Document Name], paragraph 5 — starts with: "Preview...")
    or
    (Source: [Document Name], page 4 — starts with: "Preview...")
    If answer not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    """
)

# --- Slack Client ---
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Shared Drive ---
SHARED_DRIVE_ID = "0AL5LG1aWrCL2Uk9PVA"

# --- Helper Functions ---
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def read_google_doc(file_id):
    chunks = []
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        para_count = 0
        for content in doc.get("body", {}).get("content", []):
            if "paragraph" in content:
                text = "".join([el.get("textRun", {}).get("content", "") for el in content["paragraph"].get("elements", [])])
                text = text.strip()
                if text:
                    para_count += 1
                    chunks.append({"text": clean_text(text), "meta": f"paragraph {para_count}"})
    except:
        pass
    return chunks

def read_pdf(file_id):
    chunks = []
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with fitz.open(stream=fh, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                if text:
                    for chunk in chunk_text(clean_text(text)):
                        chunks.append({"text": chunk, "meta": f"page {page_num}"})
    except:
        pass
    return chunks

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        content = "\n".join([str(row) for row in sheet.get_all_records()])
        return [{"text": chunk, "meta": f"row {i+1}"} for i, chunk in enumerate(chunk_text(content))]
    except:
        return []

def list_files_recursive(folder_id):
    files = []
    try:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            corpora="drive",
            driveId=SHARED_DRIVE_ID,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id, name, mimeType)"
        ).execute()
        for file in response.get("files", []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                files.extend(list_files_recursive(file['id']))
            else:
                files.append(file)
    except Exception as e:
        print("Drive traversal error:", e)
    return files

def extract_chunks():
    files = list_files_recursive(SHARED_DRIVE_ID)
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        doc_chunks = []
        if mime_type == 'application/vnd.google-apps.document':
            doc_chunks = read_google_doc(file_id)
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            doc_chunks = read_google_sheet(file_id)
        elif mime_type == 'application/pdf':
            doc_chunks = read_pdf(file_id)
        for chunk in doc_chunks:
            chunk['source'] = name
            chunk['link'] = link
            chunks.append(chunk)
    return chunks

def get_relevant_chunks(question, chunks, top_k=3):
    docs = [c['text'] for c in chunks]
    vec = TfidfVectorizer(stop_words='english').fit(docs + [question])
    doc_vecs = vec.transform(docs)
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    top = sims.argsort()[-top_k*2:][::-1]
    seen = set()
    results = []
    for i in top:
        key = chunks[i]['link'] + chunks[i]['meta']
        if sims[i] > 0.1 and key not in seen:
            seen.add(key)
            results.append(chunks[i])
        if len(results) >= top_k:
            break
    return results

def format_citations(chunks):
    out = []
    seen = set()
    for chunk in chunks:
        key = chunk['link'] + chunk['meta']
        if key in seen: continue
        seen.add(key)
        snippet = chunk['text'][:60].split('. ')[0].strip() + "..."
        out.append(f"(Source: <{chunk['link']}|{chunk['source']}>, {chunk['meta']} — starts with: \"{snippet}\")")
    return "\n\n" + "\n".join(out)

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        text = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
        channel = event.get("channel")

        try:
            chunks = extract_chunks()
            if not chunks:
                reply = "I couldn’t find any usable content in the Drive."
            else:
                relevant = get_relevant_chunks(text, chunks)
                if not relevant:
                    reply = "I cannot answer this question as the information is not in the provided documents."
                else:
                    context = "\n\n".join([c['text'] for c in relevant])
                    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{text}"
                    answer = model.generate_content(prompt).text
                    reply = answer + format_citations(relevant)
            slack_client.chat_postMessage(channel=channel, text=reply)
        except Exception as e:
            slack_client.chat_postMessage(channel=channel, text=f"⚠️ Error: {e}")

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
