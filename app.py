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
import hashlib

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
    Always cite the source files used in this format (Source: Doc Name, paragraph X — starts with: "Snippet...").
    Only provide ONE answer. Never repeat the same answer twice.
    """
)

# --- Slack Setup ---
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Message Tracker ---
recent_messages = set()

# --- Utility ---
def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def generate_hash(event):
    return hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()

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

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        paragraphs = []
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                para_text = "".join(
                    element.get("textRun", {}).get("content", "")
                    for element in content_item["paragraph"].get("elements", [])
                )
                paragraphs.append(para_text.strip())
        chunks = []
        for i, para in enumerate(paragraphs):
            if para:
                chunks.append({
                    "text": para,
                    "source": f"paragraph {i+1} — starts with: \"{para[:40].strip()}...\"",
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
        pdf_chunks = []
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for i, page in enumerate(pdf_doc):
                text = clean_pdf_text(page.get_text())
                if text:
                    pdf_chunks.append({
                        "text": text,
                        "source": f"page {i+1} — starts with: \"{text[:40].strip()}...\""
                    })
        return pdf_chunks
    except:
        return []

# --- Chunking & Retrieval ---
def extract_all_chunks(shared_drive_id):
    files = list_all_files_in_shared_drive(shared_drive_id)
    all_chunks = []
    for file in files:
        name, file_id, mime_type = file['name'], file['id'], file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        if mime_type == 'application/vnd.google-apps.document':
            chunks = read_google_doc(file_id)
        elif mime_type == 'application/pdf':
            chunks = read_pdf(file_id)
        else:
            continue
        for chunk in chunks:
            all_chunks.append({"text": chunk['text'], "citation": f"(Source: <{link}|{name}>, {chunk['source']})"})
    return all_chunks

def get_relevant_chunks(question, chunks, top_k=3):
    docs = [c['text'] for c in chunks]
    if not docs:
        return []
    vectorizer = TfidfVectorizer().fit(docs + [question])
    doc_vecs = vectorizer.transform(docs)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, doc_vecs).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices if sims[i] > 0.05]

# --- Slack Event Handler ---
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    event_id = generate_hash(event)
    if event_id in recent_messages:
        return Response(), 200
    recent_messages.add(event_id)

    if event.get("type") == "app_mention":
        user_input = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
        channel = event.get("channel")

        chunks = extract_all_chunks("0AL5LG1aWrCL2Uk9PVA")
        top_chunks = get_relevant_chunks(user_input, chunks)

        if not top_chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([f"{c['text']}\n{c['citation']}" for c in top_chunks])
            prompt = f"Answer the question based only on the CONTEXT below. Use natural language and include the source reference at the end.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{user_input}"
            try:
                gemini_reply = model.generate_content(prompt)
                reply = getattr(gemini_reply, "text", "Sorry, no response generated.")
            except Exception as e:
                reply = f"⚠️ Gemini error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel, text=reply)
        except SlackApiError as e:
            print("Slack error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT Slack bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
