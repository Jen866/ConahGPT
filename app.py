import os
import io
import fitz  # PyMuPDF
import gspread
import json
import re
import hashlib
import google.generativeai as genai
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import numpy as np

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

shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based on the CONTEXT provided. Interpret meaning — don’t require exact matches.
    If the answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    Always cite the source files used, with a clickable link, paragraph or page number, and a short snippet.
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
recent_event_hashes = set()

def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def list_all_files():
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
        text = ""
        para_count = 0
        chunks = []
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                para_text = ""
                for element in content_item["paragraph"].get("elements", []):
                    para_text += element.get("textRun", {}).get("content", "")
                if para_text.strip():
                    chunks.append({
                        "text": para_text.strip(),
                        "source": doc.get("title", "Google Doc"),
                        "link": f"https://docs.google.com/document/d/{file_id}",
                        "paragraph": para_count + 1,
                        "snippet": para_text.strip()[:60]
                    })
                    para_count += 1
        return chunks
    except:
        return []

def read_pdf(file_id):
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        while not downloader.next_chunk()[1]:
            pass
        fh.seek(0)
        chunks = []
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                text = page.get_text().strip()
                if text:
                    chunks.append({
                        "text": clean_pdf_text(text),
                        "source": pdf_doc.name or "PDF",
                        "link": f"https://drive.google.com/file/d/{file_id}",
                        "page": page_num,
                        "snippet": text.split("\n")[0][:60]
                    })
        return chunks
    except:
        return []

def extract_all_chunks():
    files = list_all_files()
    all_chunks = []
    for file in files:
        fid, name, mime = file['id'], file['name'], file['mimeType']
        if mime == 'application/vnd.google-apps.document':
            all_chunks.extend(read_google_doc(fid))
        elif mime == 'application/pdf':
            chunks = read_pdf(fid)
            for chunk in chunks:
                chunk['source'] = name
            all_chunks.extend(chunks)
        elif mime == 'application/vnd.google-apps.spreadsheet':
            text = read_google_sheet(fid)
            if text:
                for c in chunk_text(text):
                    all_chunks.append({
                        "text": c,
                        "source": name,
                        "link": f"https://drive.google.com/file/d/{fid}",
                        "snippet": c[:60]
                    })
    return all_chunks

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_relevant_chunks(question, chunks, top_k=5):
    docs = [chunk["text"] for chunk in chunks]
    vec = TfidfVectorizer().fit(docs + [question])
    doc_vecs = vec.transform(docs)
    q_vec = vec.transform([question])
    sim = cosine_similarity(q_vec, doc_vecs).flatten()
    indices = sim.argsort()[-top_k:][::-1]
    return [chunks[i] for i in indices if sim[i] > 0.05]

def build_source_note(chunk):
    base = f"(Source: <{chunk['link']}|{chunk['source']}"  # clickable link in Slack
    if 'page' in chunk:
        base += f", page {chunk['page']}"
    elif 'paragraph' in chunk:
        base += f", paragraph {chunk['paragraph']}"
    snippet = chunk.get('snippet', '').strip()
    if snippet:
        base += f" — starts with: \"{snippet[:50]}...\")"
    else:
        base += ")"
    return base

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        text = event.get("text", "")
        channel = event.get("channel")
        user = event.get("user")
        event_hash = hashlib.sha256(f"{user}:{text}".encode()).hexdigest()
        if event_hash in recent_event_hashes:
            return Response(), 200
        recent_event_hashes.add(event_hash)

        clean_text = re.sub(r"<@[^>]+>", "", text).strip()
        chunks = extract_all_chunks()
        if not chunks:
            reply = "❌ No readable documents found."
        else:
            relevant = get_relevant_chunks(clean_text, chunks)
            if not relevant:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([f"FROM {c['source']}\n{c['text']}" for c in relevant])
                prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{clean_text}"
                try:
                    answer = model.generate_content(prompt).text.strip()
                    final_note = build_source_note(relevant[0])
                    reply = f"{answer}\n\n{final_note}"
                except Exception as e:
                    reply = f"⚠️ Gemini Error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel, text=reply)
        except SlackApiError as e:
            print("Slack error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
