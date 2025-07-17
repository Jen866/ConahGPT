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
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from collections import defaultdict
import numpy as np

# --- Flask Setup ---
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
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, an expert business assistant for Actuary Consulting.
    You answer questions based on the CONTEXT provided. Interpret the meaning — don’t require exact matches.
    If the answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
    Always cite the source files used, with a clickable link, including page number for PDFs or paragraph with preview for Docs.
    """
)

# --- Slack Setup ---
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Utility Functions ---
def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

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
        paragraphs = []
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                paragraph_text = ""
                for element in content_item["paragraph"].get("elements", []):
                    paragraph_text += element.get("textRun", {}).get("content", "")
                if paragraph_text.strip():
                    paragraphs.append(paragraph_text.strip())
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
        pages = []
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for page_number, page in enumerate(pdf_doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages.append((page_number, clean_pdf_text(text)))
        return pages
    except:
        return []

# --- Chunking & Search ---
def extract_all_chunks(shared_drive_id):
    files = list_all_files_in_shared_drive(shared_drive_id)
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        doc_link = f"https://drive.google.com/file/d/{file_id}"
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
            for i, chunk in enumerate(chunk_text(content)):
                chunks.append({"text": chunk, "source": name, "link": doc_link})
        elif mime_type == 'application/vnd.google-apps.document':
            paragraphs = read_google_doc(file_id)
            for idx, paragraph in enumerate(paragraphs):
                chunks.append({"text": paragraph, "source": name, "link": doc_link, "paragraph": idx + 1})
        elif mime_type == 'application/pdf':
            pages = read_pdf(file_id)
            for page_num, page_text in pages:
                for chunk in chunk_text(page_text):
                    chunks.append({"text": chunk, "source": name, "link": doc_link, "page": page_num})
    return chunks

def get_relevant_chunks(question, chunks, top_k=5):
    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.05]

# --- Slack Events Route ---
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if data.get("event", {}).get("type") == "app_mention" and "bot_id" not in data["event"]:
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
                grouped_by_source = defaultdict(list)
                for chunk in relevant_chunks:
                    grouped_by_source[(chunk["source"], chunk["link"])].append(chunk)

                citations = []
                for (source, link), chs in grouped_by_source.items():
                    if "paragraph" in chs[0]:
                        para = chs[0]["paragraph"]
                        preview = chs[0]["text"][:40].strip()
                        citations.append(f"(Source: <{link}|{source}>, paragraph {para} — starts with: \"{preview}...\")")
                    elif "page" in chs[0]:
                        page = chs[0]["page"]
                        citations.append(f"(Source: <{link}|{source}>, page {page})")
                    else:
                        citations.append(f"(Source: <{link}|{source}>)")

                context = "\n\n".join([f"FROM {chunk['source']}:\n{chunk['text']}" for chunk in relevant_chunks])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"

                try:
                    gemini_response = model.generate_content(prompt)
                    answer = getattr(gemini_response, "text", "Oops, no answer returned.")
                    reply = f"{answer}\n\n" + "\n".join(citations)
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
