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
from datetime import datetime, timedelta
from threading import Lock
import google.generativeai as genai

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

SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    raise ValueError("The SERVICE_ACCOUNT_JSON environment variable is not set.")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)

gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are an expert assistant. You MUST answer the user's question using ONLY the information from the provided CONTEXT.
    Do not use any prior knowledge. The CONTEXT is the absolute source of truth.
    If the answer is not available in the CONTEXT, you must say: 'I cannot answer this question as the information is not in the provided documents.'
    Your answer should be concise and in a single paragraph. Cite each source document only once.
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
SHARED_DRIVE_ID = "0AL5LG1aWrCL2Uk9PVA"

# --- Caching Mechanism ---
drive_chunks_cache = []
cache_last_updated = None
CACHE_DURATION = timedelta(minutes=10)
cache_lock = Lock()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# --- Google Drive Reading Functions ---

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        return "\n".join([str(row) for row in sheet.get_all_records()])
    except Exception as e:
        print(f"Error reading Google Sheet {file_id}: {e}")
        return ""

def read_google_doc(file_id):
    """
    ✅ DEFINITIVE FIX: This logic combines heading detection with Q&A pairing.
    It identifies headings, then groups questions and their subsequent answers
    into a single chunk, associated with the correct heading for robust context
    and accurate citations.
    """
    chunks = []
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        doc_content = doc.get("body", {}).get("content", [])
        
        # First, create a clean list of paragraphs with their styles.
        # This avoids the complexity of the raw API response structure.
        all_paragraphs_with_style = []
        for content_item in doc_content:
            if "paragraph" in content_item:
                elements = content_item.get("paragraph", {}).get("elements", [])
                p_text = "".join([elem.get("textRun", {}).get("content", "") for elem in elements]).strip()
                
                if not p_text:
                    continue

                # Determine if the paragraph is a heading (bold and underlined)
                is_heading = False
                if elements:
                    is_heading = all(
                        elem.get("textRun", {}).get("textStyle", {}).get("bold") and
                        elem.get("textRun", {}).get("textStyle", {}).get("underline")
                        for elem in elements if elem.get("textRun", {}).get("content", "").strip()
                    )
                all_paragraphs_with_style.append({"text": p_text, "is_heading": is_heading})

        # Now, iterate through the clean list to build chunks.
        current_heading = "General"
        i = 0
        while i < len(all_paragraphs_with_style):
            para_info = all_paragraphs_with_style[i]
            p_text = para_info["text"]
            
            if para_info["is_heading"]:
                current_heading = p_text
                i += 1
                continue

            # It's a regular paragraph. Check if it's a question.
            if p_text.endswith('?'):
                question_text = p_text
                answer_text = ""
                
                # Look at the next paragraph to see if it's the answer.
                if (i + 1) < len(all_paragraphs_with_style):
                    next_para_info = all_paragraphs_with_style[i+1]
                    # The answer is the next line if it's not another heading.
                    if not next_para_info["is_heading"]:
                        answer_text = next_para_info["text"]
                        i += 1 # Consume the answer paragraph.
                
                # Create the combined chunk for full context.
                full_text = f"Question: {question_text} Answer: {answer_text}"
                chunks.append({
                    "text": clean_text(full_text),
                    "meta": {"heading": current_heading}
                })
            else:
                # It's a standalone paragraph (not a question, not a heading).
                chunks.append({
                    "text": clean_text(p_text),
                    "meta": {"heading": current_heading}
                })
            
            i += 1
            
    except Exception as e:
        print(f"Error in read_google_doc (Heading & Q&A Logic): {e}")
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
                words = content.split()
                for i in range(0, len(words), 300):
                    text_chunk = " ".join(words[i:i+300])
                    content_chunks.append({"text": text_chunk, "meta": {"block": (i // 300) + 1}})
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
    global drive_chunks_cache, cache_last_updated
    with cache_lock:
        if not cache_last_updated or (datetime.utcnow() - cache_last_updated > CACHE_DURATION):
            drive_chunks_cache = extract_all_chunks(SHARED_DRIVE_ID)
            cache_last_updated = datetime.utcnow()

# --- RAG Core Logic ---

def get_relevant_chunks(question, chunks, top_k=3):
    if not chunks: return []
    documents = [chunk["text"] for chunk in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words='english').fit(documents + [question])
        doc_vectors = vectorizer.transform(documents)
        question_vector = vectorizer.transform([question])
    except ValueError:
        return []

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k * 2:][::-1]
    
    results = []
    for i in top_indices:
        if similarities[i] > 0.1:
            results.append(chunks[i])

    unique_sources = {}
    deduplicated_chunks = []
    for chunk in results:
        if chunk['source'] not in unique_sources:
            unique_sources[chunk['source']] = True
            deduplicated_chunks.append(chunk)
            
    return deduplicated_chunks[:top_k]

# --- Slack Integration & Flask Routes ---

def format_citations(chunks):
    if not chunks: return ""
    citations = []
    for chunk in chunks:
        source_name = chunk['source']
        link = chunk['link']
        meta_info = chunk.get('meta', {})
        
        if 'heading' in meta_info:
            location = f'in section "{meta_info["heading"]}"'
        elif 'page' in meta_info:
            location = f"on page {meta_info['page']}"
        elif 'block' in meta_info:
            location = f"in data block {meta_info['block']}"
        else:
            location = "near the start of the document"
        
        preview = " ".join(chunk['text'].split()[:10]) + "..."
        citations.append(f'(Source: [{source_name}]({link}), {location} — starts with: "{preview}")')

    return "\n\n**Sources:**\n" + "\n".join(citations)

def process_slack_event(channel_id, clean_text):
    refresh_cache_if_needed()
    if not drive_chunks_cache:
        reply = "I couldn’t read any usable files from Google Drive. Please check folder permissions or content."
    else:
        relevant_chunks = get_relevant_chunks(clean_text, drive_chunks_cache, top_k=3)
        
        if not relevant_chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([f"Source Document: {chunk['source']}\nContent: {chunk['text']}" for chunk in relevant_chunks])
            prompt = f"Based on the following CONTEXT, please provide a direct answer to the USER QUESTION.\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}\n\nANSWER:"
            
            try:
                gemini_response = model.generate_content(prompt)
                raw_answer = getattr(gemini_response, 'text', "I'm sorry, I couldn't generate a response.")
                if "I cannot answer" in raw_answer:
                    reply = raw_answer
                else:
                    citations = format_citations(relevant_chunks)
                    reply = raw_answer + citations
            except Exception as e:
                print(f"Error generating content from Gemini: {e}")
                reply = f"⚠️ An error occurred while generating the answer: {str(e)}"
    try:
        slack_client.chat_postMessage(channel=channel_id, text=reply)
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']}")

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
    
    if request.headers.get('X-Slack-Retry-Num'):
        return Response(status=200)

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event.get("channel", "")
        user_text = event.get("text", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()
        
        thread = threading.Thread(target=process_slack_event, args=(channel_id, clean_text))
        thread.start()

    return Response(status=200)

@app.route("/")
def index():
    if not cache_last_updated:
        refresh_cache_if_needed()
    status = f"Cache is populated with {len(drive_chunks_cache)} chunks." if drive_chunks_cache else "Cache is empty, check Drive connection."
    return f"✅ ConahGPT Slack bot is running. {status}"

if __name__ == "__main__":
    refresh_cache_if_needed()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
