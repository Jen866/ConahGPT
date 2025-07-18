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

# Setup Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based on the CONTEXT provided. Interpret the meaning — don’t require exact matches.
    If the answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    Always cite the source files used, with a clickable link.
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

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
        text = ""
        for i, content_item in enumerate(doc.get("body", {}).get("content", [])):
            if "paragraph" in content_item:
                for element in content_item["paragraph"].get("elements", []):
                    text += element.get("textRun", {}).get("content", "")
        return text.strip()
    except:
        return ""

def read_pdf(file_id):
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            return " ".join([clean_pdf_text(page.get_text()) for page in pdf_doc])
    except:
        return ""

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_all_chunks(shared_drive_id):
    files = list_all_files_in_shared_drive(shared_drive_id)
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        doc_link = f"https://drive.google.com/file/d/{file_id}"
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
        elif mime_type == 'application/vnd.google-apps.document':
            content = read_google_doc(file_id)
        elif mime_type == 'application/pdf':
            content = read_pdf(file_id)
        else:
            continue

        for i, chunk in enumerate(chunk_text(content)):
            chunks.append({
                "text": chunk,
                "source": name,
                "link": doc_link,
                "meta": f"paragraph {i + 1}" if 'document' in mime_type else f"page {i + 1}"
            })
    return chunks

def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        text_hash = hash(chunk['text'])
        if text_hash not in seen:
            unique_chunks.append(chunk)
            seen.add(text_hash)
    return unique_chunks

def get_relevant_chunks(question, chunks, top_k=3):
    if not chunks:
        return []
    chunks = deduplicate_chunks(chunks)
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.05]

def format_citations(chunks):
    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk['source'], chunk['meta'])
        if key not in seen:
            seen.add(key)
            preview = ' '.join(chunk['text'].split()[:10])
            citation = f"(Source: [{chunk['source']}]({chunk['link']}), {chunk['meta']} — starts with: \"{preview}...\")"
            citations.append(citation)
    return citations

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if data.get("event", {}).get("type") == "app_mention":
        user_text = data["event"].get("text", "")
        channel_id = data["event"].get("channel", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
        chunks = extract_all_chunks(shared_drive_id)

        if not chunks:
            reply = "I couldn’t read any usable files from Google Drive. Please check the folder."
        else:
            relevant_chunks = get_relevant_chunks(clean_text, chunks)
            if not relevant_chunks:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"
                try:
                    gemini_response = model.generate_content(prompt)

                    # Remove duplicate paragraphs from Gemini output
                    raw_answer = getattr(gemini_response, "text", "").strip()
                    paragraphs = [p.strip() for p in raw_answer.split("\n\n") if p.strip()]
                    seen_paragraphs = set()
                    deduped_paragraphs = []
                    for para in paragraphs:
                        if para not in seen_paragraphs:
                            deduped_paragraphs.append(para)
                            seen_paragraphs.add(para)
                    raw_answer = "\n\n".join(deduped_paragraphs) or "Oops, no answer returned."

                    citations = format_citations(relevant_chunks)
                    reply = f"{raw_answer}\n\n" + "\n".join(citations)
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print("Slack API Error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT Slack bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
