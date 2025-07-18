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

# Load credentials from environment variable
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    raise ValueError("The SERVICE_ACCOUNT_JSON environment variable is not set.")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)

# Build API clients
gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

# Configure the Generative AI model with a more forceful system instruction
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are an expert assistant. You MUST answer the user's question using ONLY the information from the provided CONTEXT.
    Do not use any prior knowledge. The CONTEXT is the absolute source of truth.
    If the answer is not available in the CONTEXT, you must say: 'I cannot answer this question as the information is not in the provided documents.'
    Your answer should be concise. Cite each source document only once.
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
SHARED_DRIVE_ID = "0AL5LG1aWrCL2Uk9PVA"  # Your Shared Drive ID

# --- Caching Mechanism for Performance ---
drive_chunks_cache = []
cache_last_updated = None
CACHE_DURATION = timedelta(minutes=10)
cache_lock = Lock()

def clean_text(text):
    """Removes extra whitespace and newlines for cleaner processing."""
    return re.sub(r'\s+', ' ', text).strip()

# --- Google Drive Reading Functions with Improved Metadata ---

def read_google_sheet(file_id):
    """Reads a Google Sheet and returns its content as a single string."""
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        return "\n".join([str(row) for row in sheet.get_all_records()])
    except Exception as e:
        print(f"Error reading Google Sheet {file_id}: {e}")
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
    except Exception as e:
        print(f"Error reading Google Doc {file_id}: {e}")
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
    except Exception as e:
        print(f"Error reading PDF {file_id}: {e}")
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
            fields="files(id, name, mimeType)"
        ).execute().get('files', [])
    except Exception as e:
        print(f"Error listing files from Drive: {e}")
        return []

    for file in files:
        file_id, name = file['id'], file['name']
        doc_link = f"https://drive.google.com/file/d/{file_id}"

        content_chunks = []
        if file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
            if content:
                # Chunk large sheets into blocks
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

    print(f"Cache refreshed. Total chunks found: {len(all_chunks)}")
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
    Finds the most relevant and unique chunks using TF-IDF and cosine similarity.
    """
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]
    try:
        # Fit vectorizer on all docs + question to handle vocab
        vectorizer = TfidfVectorizer(stop_words='english').fit(documents + [question])
        doc_vectors = vectorizer.transform(documents)
        question_vector = vectorizer.transform([question])
    except ValueError: # Handles case where vocab is empty
        return []

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    
    # Get more than top_k initially to allow for deduplication
    top_indices = similarities.argsort()[-top_k * 2:][::-1]
    
    relevant_chunks = []
    for i in top_indices:
        if similarities[i] > 0.1:  # Relevance threshold
            chunk_with_score = chunks[i].copy()
            chunk_with_score['similarity'] = similarities[i]
            relevant_chunks.append(chunk_with_score)

    # Sort by similarity to prioritize the best chunk for each source
    relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)

    # Deduplicate: Keep only the *most relevant* chunk for each source document.
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
    Formats citations according to the required format:
    (Source: [Name](link), location — starts with: "preview...")
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
            
        # Create a preview of the first 8-12 words
        preview = " ".join(chunk['text'].split()[:10]) + "..."
        
        citation = f'(Source: [{source_name}]({link}), {location} — starts with: "{preview}")'
        citations.append(citation)
        
    return "\n\n**Sources:**\n" + "\n".join(citations)

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    # Respond to the Slack challenge
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    # Handle app mentions
    if event.get("type") == "app_mention":
        user_text = event.get("text", "")
        channel_id = event.get("channel", "")
        # Remove the bot's @mention from the text
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        # Ensure the cache is fresh
        refresh_cache_if_needed()
        
        if not drive_chunks_cache:
            reply = "I couldn’t read any usable files from Google Drive. Please check folder permissions or content."
        else:
            # Step 1: Find relevant documents
            relevant_chunks = get_relevant_chunks(clean_text, drive_chunks_cache, top_k=3)

            if not relevant_chunks:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                # Step 2: Construct a direct and forceful prompt for the model
                context = "\n\n".join([f"Source Document: {chunk['source']}\nContent: {chunk['text']}" for chunk in relevant_chunks])
                prompt = f"Based on the following CONTEXT, please provide a direct answer to the USER QUESTION.\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}\n\nANSWER:"

                try:
                    # Step 3: Generate the answer
                    gemini_response = model.generate_content(prompt)
                    raw_answer = getattr(gemini_response, 'text', "I'm sorry, I couldn't generate a response.")

                    # Step 4: Add citations ONLY if the model provides a real answer
                    if "I cannot answer" in raw_answer:
                        reply = raw_answer
                    else:
                        citations = format_citations(relevant_chunks)
                        reply = raw_answer + citations

                except Exception as e:
                    print(f"Error generating content from Gemini: {e}")
                    reply = f"⚠️ An error occurred while generating the answer: {str(e)}"

        # Step 5: Post the reply to Slack
        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print(f"Slack API Error: {e.response['error']}")

    return Response(), 200

@app.route("/")
def index():
    """A simple health check endpoint."""
    if not cache_last_updated:
        refresh_cache_if_needed()
    status = f"Cache is populated with {len(drive_chunks_cache)} chunks." if drive_chunks_cache else "Cache is empty, check Drive connection."
    return f"✅ ConahGPT Slack bot is running. {status}"

if __name__ == "__main__":
    # Pre-warm the cache on startup
    refresh_cache_if_needed()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
