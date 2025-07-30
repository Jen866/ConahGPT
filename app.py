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
    You are Conah GPT, a precise business assistant for Actuary Consulting.
    Fully answer each question clearly based on the provided CONTEXT.
    Only respond once per question. Include source citations at the end:
    (Source: [Document Name], paragraph 5 — starts with: "Preview snippet...")
    If source is a PDF, show page number instead of paragraph.
    If answer is not available, say:
    'I cannot answer this question as the information is not in the provided documents.'
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# Helpers for reading

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def list_all_files_recursive(drive_id, parent_id=None):
    query = f"'{parent_id or drive_id}' in parents and trashed = false"
    page_token = None
    files = []
    while True:
        response = drive_service.files().list(
            q=query,
            corpora="drive",
            driveId=drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            pageToken=page_token
        ).execute()
        for f in response.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                files.extend(list_all_files_recursive(drive_id, f["id"]))
            else:
                files.append(f)
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        chunks = []
        para_count = 0
        for content in doc.get("body", {}).get("content", []):
            if "paragraph" in content:
                text = "".join([elem.get("textRun", {}).get("content", "") for elem in content["paragraph"].get("elements", [])]).strip()
                if text:
                    para_count += 1
                    chunks.append((para_count, clean_text(text)))
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
        with fitz.open(stream=fh, filetype="pdf") as pdf:
            for page_num, page in enumerate(pdf, 1):
                text = page.get_text().strip()
                if text:
                    for chunk in chunk_text(text):
                        chunks.append((page_num, clean_text(chunk)))
        return chunks
    except:
        return []

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        records = sheet.get_all_records()
        lines = [str(row) for row in records]
        return chunk_text(" ".join(lines))
    except:
        return []

def extract_relevant_chunks(question, files):
    chunks = []
    for file in files:
        file_id, name, mime = file["id"], file["name"], file["mimeType"]
        link = f"https://drive.google.com/file/d/{file_id}"
        if mime == "application/vnd.google-apps.document":
            for para_num, para in read_google_doc(file_id):
                chunks.append({"text": para, "meta": f"paragraph {para_num}", "source": name, "link": link})
        elif mime == "application/pdf":
            for page_num, text in read_pdf(file_id):
                chunks.append({"text": text, "meta": f"page {page_num}", "source": name, "link": link})
        elif mime == "application/vnd.google-apps.spreadsheet":
            for i, chunk in enumerate(read_google_sheet(file_id)):
                chunks.append({"text": chunk, "meta": f"row {i+1}", "source": name, "link": link})
    return get_relevant_chunks(question, chunks)

def get_relevant_chunks(question, chunks, top_k=5):
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    seen = set()
    results = []
    for i in top_indices:
        key = chunks[i]['link'] + chunks[i]['meta']
        if sims[i] > 0.05 and key not in seen:
            seen.add(key)
            results.append(chunks[i])
    return results

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if data.get("event", {}).get("type") == "app_mention":
        text = re.sub(r"<@[^>]+>", "", data["event"].get("text", "")).strip()
        channel = data["event"].get("channel")
        try:
            drive_id = "0AL5LG1aWrCL2Uk9PVA"
            all_files = list_all_files_recursive(drive_id)
            top_chunks = extract_relevant_chunks(text, all_files)

            if not top_chunks:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([chunk["text"] for chunk in top_chunks])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{text}"
                gemini_response = model.generate_content(prompt)
                answer = getattr(gemini_response, "text", "No answer returned.")
                citations = []
                seen = set()
                for c in top_chunks:
                    key = c['link'] + c['meta']
                    if key not in seen:
                        seen.add(key)
                        snippet = c['text'][:60].split(". ")[0].strip() + "..."
                        citations.append(f"(Source: <{c['link']}|{c['source']}>, {c['meta']} — starts with: \"{snippet}\")")
                reply = answer + "\n\n" + "\n".join(citations)
            slack_client.chat_postMessage(channel=channel, text=reply)
        except Exception as e:
            slack_client.chat_postMessage(channel=channel, text=f"⚠️ Error: {str(e)}")
    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT with lazy loading is live."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
