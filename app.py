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
    You answer user questions based on the provided CONTEXT.
    Your response must:
    - Provide a full answer to the user's question.
    - After the answer, add: (Source: [Document Name], paragraph X — starts with: "Snippet...")
    - Only show each source once. Do not repeat answers or cite the same source twice.
    If you do not find the answer, say: 
    "I cannot answer this question as the information is not in the provided documents."
    """
)

# --- Slack Setup ---
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# --- Utility Functions ---
def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

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
        text = ""
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                for element in content_item["paragraph"].get("elements", []):
                    text += element.get("textRun", {}).get("content", "")
        return text
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
        text = ""
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for i, page in enumerate(pdf_doc):
                page_text = page.get_text()
                text += f"\n[PAGE {i+1}]\n" + page_text
        return text
    except:
        return ""

# --- Chunking ---
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_all_chunks(shared_drive_id):
    files = list_all_files_in_shared_drive(shared_drive_id)
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        content = ""
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
        elif mime_type == 'application/vnd.google-apps.document':
            content = read_google_doc(file_id)
        elif mime_type == 'application/pdf':
            content = read_pdf(file_id)
        if not content.strip():
            continue

        paragraphs = content.split("\n") if 'application/vnd.google-apps.document' in mime_type else None
        for i, chunk in enumerate(chunk_text(content)):
            if paragraphs:
                paragraph_num = i + 1
                snippet = paragraphs[i].strip()[:50] if i < len(paragraphs) else ""
                source_info = f"paragraph {paragraph_num} — starts with: \"{snippet}...\""
            else:
                match = re.search(r"\\[PAGE (\\d+)\\]", chunk)
                page_number = match.group(1) if match else "?"
                snippet = chunk.strip()[:50]
                source_info = f"page {page_number} — starts with: \"{snippet}...\""

            chunks.append({
                "text": chunk,
                "source": name,
                "link": link,
                "ref": source_info
            })
    return chunks

def get_relevant_chunks(question, chunks, top_k=5):
    documents = [chunk['text'] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    seen = set()
    results = []
    for i in top_indices:
        key = (chunks[i]['source'], chunks[i]['ref'])
        if key not in seen and similarities[i] > 0.05:
            results.append(chunks[i])
            seen.add(key)
    return results

# --- Slack Endpoint ---
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention" and not event.get("bot_id"):
        user_text = event.get("text", "")
        channel_id = event.get("channel", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
        chunks = extract_all_chunks(shared_drive_id)
        relevant_chunks = get_relevant_chunks(clean_text, chunks)

        if not relevant_chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([f"FROM {c['source']} ({c['link']}):\n{c['text']}" for c in relevant_chunks])
            prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"
            try:
                gemini_response = model.generate_content(prompt)
                raw_answer = getattr(gemini_response, "text", "Oops, no answer returned.").strip()
                sources_used = set()
                for c in relevant_chunks:
                    key = (c['source'], c['ref'])
                    if key not in sources_used:
                        raw_answer += f"\n\n(Source: <{c['link']}|{c['source']}>, {c['ref']})"
                        sources_used.add(key)
                reply = raw_answer
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
