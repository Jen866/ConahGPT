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
from threading import Lock
from datetime import datetime, timedelta

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Global Configurations & Clients ---
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

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
     You are Conah GPT, an expert business assistant for Actuary Consulting.
     You answer questions based on the CONTEXT provided. Interpret the meaning — don’t require exact matches.
     If the answer is not found, say: 'I cannot answer this question as the information is not in the provided documents.'
     Your answer should be concise and directly address the user's question.
     Do not repeat information. Cite each source document only once.
     """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
SHARED_DRIVE_ID = "0AL5LG1aWrCL2Uk9PVA" # Your Shared Drive ID

# --- Caching Mechanism (Improvement #4) ---
drive_chunks_cache = []
cache_last_updated = None
CACHE_DURATION = timedelta(minutes=10)
cache_lock = Lock()

def clean_text(text):
    """Removes extra whitespace and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

# --- Google Drive Reading Functions (with improved chunking metadata) ---

def read_google_sheet(file_id):
    """Reads a Google Sheet and returns content as a single string."""
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        return "\n".join([str(row) for row in sheet.get_all_records()])
    except Exception:
        return ""

def read_google_doc(file_id):
    """Reads a Google Doc and returns a list of chunks, one per paragraph."""
    chunks = []
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        doc_content = doc.get("body", {}).get("content", [])
        paragraph_index = 0
        for content_item in doc_content:
            if "paragraph" in content_item:
                paragraph_index += 1
                elements = content_item.get("paragraph", {}).get("elements", [])
                current_paragraph = "".join(
                    [elem.get("textRun", {}).get("content", "") for elem in elements]
                )
                if current_paragraph.strip():
                    chunks.append({
                        "text": clean_text(current_paragraph),
                        "meta": {"paragraph": paragraph_index}
                    })
    except Exception:
        return []
    return chunks

def read_pdf(file_id):
    """Reads a PDF and returns a list of chunks, one per page."""
    chunks = []
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for i, page in enumerate(pdf_doc):
                text = page.get_text()
                if text.strip():
                    chunks.append({
                        "text": clean_text(text),
                        "meta": {"page": i + 1}
                    })
    except Exception:
        return []
    return chunks

# --- Content Processing and Caching ---

def extract_all_chunks(shared_drive_id):
    """
    Scans a Shared Drive for files, extracts text content, and creates structured chunks.
    This is the core of the caching logic.
    """
    print("Refreshing Drive cache...")
    all_chunks = []
    try:
        files = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/pdf'",
            corpora="drive",
            driveId=shared_drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id, name, mimeType, webViewLink)"
        ).execute().get('files', [])
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

    for file in files:
        file_id, name = file['id'], file['name']
        doc_link = f"https://drive.google.com/file/d/{file_id}"

        content_chunks = []
        if file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
            if content:
                words = content.split()
                for i in range(0, len(words), 300):
                    text_chunk = " ".join(words[i:i+300])
                    content_chunks.append({
                        "text": text_chunk,
                        "meta": {"block": (i // 300) + 1}
                    })
        elif file['mimeType'] == 'application/vnd.google-apps.document':
            content_chunks = read_google_doc(file_id)
        elif file['mimeType'] == 'application/pdf':
            content_chunks = read_pdf(file_id)

        for chunk in content_chunks:
            chunk['source'] = name
            chunk['link'] = doc_link
            all_chunks.append(chunk)

    print(f"Cache refreshed. Total chunks: {len(all_chunks)}")
    return all_chunks


def refresh_cache_if_needed():
    """Checks cache freshness and refreshes it if it's stale."""
    global drive_chunks_cache, cache_last_updated
    with cache_lock:
        if not cache_last_updated or (datetime.utcnow() - cache_last_updated > CACHE_DURATION):
            drive_chunks_cache = extract_all_chunks(SHARED_DRIVE_ID)
            cache_last_updated = datetime.utcnow()

# --- RAG Core Logic ---

def get_relevant_chunks(question, chunks, top_k=3):
    """
    Finds the most relevant chunks using TF-IDF and cosine similarity.
    (Improvement #1 & #4: top_k=3, deduplication handled post-retrieval)
    """
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words='english').fit(documents + [question])
        doc_vectors = vectorizer.transform(documents)
        question_vector = vectorizer.transform([question])
    except ValueError:
        return []

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_k*2:][::-1]
    
    relevant_chunks = []
    for i in top_indices:
        if similarities[i] > 0.1: # Threshold to ensure relevance
            chunk_with_score = chunks[i].copy()
            chunk_with_score['similarity'] = similarities[i]
            relevant_chunks.append(chunk_with_score)

    relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)

    # --- Deduplication Logic (Improvement #1) ---
    # Keep only the *most relevant* chunk for each source document.
    unique_sources = {}
    deduplicated_chunks = []
    for chunk in relevant_chunks:
        if chunk['source'] not in unique_sources:
            unique_sources[chunk['source']] = True
            deduplicated_chunks.append(chunk)

    return deduplicated_chunks[:top_k]

# --- Slack Integration & Flask Routes ---

def format_citations(chunks):
    """
    Formats citations according to the specified format.
    (Improvement #2 & #3: Single citation per source, with link and preview)
    """
    if not chunks:
        return ""

    citations = []
    for chunk in chunks:
        source_name = chunk['source']
        link = chunk['link']
        meta_info = chunk.get('meta', {})
        
        if 'page' in meta_info:
            location = f"page {meta_info['page']}"
        elif 'paragraph' in meta_info:
            location = f"paragraph {meta_info['paragraph']}"
        elif 'block' in meta_info:
             location = f"data block {meta_info['block']}"
        else:
            location = "start of document"
            
        preview = " ".join(chunk['text'].split()[:10]) + "..."
        
        citation = f'(Source: [{source_name}]({link}), {location} — starts with: "{preview}")'
        citations.append(citation)
        
    return "\n\n**Sources:**\n" + "\n".join(citations)

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        user_text = event.get("text", "")
        channel_id = event.get("channel", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        refresh_cache_if_needed()
        
        if not drive_chunks_cache:
            reply = "I couldn’t read any usable files from Google Drive. Please check folder permissions or content."
        else:
            relevant_chunks = get_relevant_chunks(clean_text, drive_chunks_cache, top_k=3)

            if not relevant_chunks:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in relevant_chunks])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"

                try:
                    gemini_response = model.generate_content(prompt)
                    raw_answer = getattr(gemini_response, 'text', "I'm sorry, I couldn't generate a response.")
                    citations = format_citations(relevant_chunks)
                    reply = raw_answer + citations
                except Exception as e:
                    reply = f"⚠️ An error occurred while generating the answer: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print(f"Slack API Error: {e.response['error']}")

    return Response(), 200

@app.route("/")
def index():
    if not cache_last_updated:
        refresh_cache_if_needed()
    status = "Cache is populated." if drive_chunks_cache else "Cache is empty, check Drive connection."
    return f"✅ ConahGPT Slack bot is running. {status}"

if __name__ == "__main__":
    refresh_cache_if_needed()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
