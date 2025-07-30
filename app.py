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
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)
gspread_client = gspread.authorize(creds)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
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

SHARED_DRIVE_ID = "0AL5LG1aWrCL2Uk9PVA"

# ---------------------- File Readers ----------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        rows = sheet.get_all_records()
        return [str(row) for row in rows]
    except:
        return []

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        paragraphs = []
        count = 0
        for content in doc.get("body", {}).get("content", []):
            if "paragraph" in content:
                p_text = "".join([e.get("textRun", {}).get("content", "") for e in content["paragraph"].get("elements", [])]).strip()
                if p_text:
                    count += 1
                    paragraphs.append((count, clean_text(p_text)))
        return paragraphs
    except:
        return []

def read_pdf(file_id):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        chunks = []
        with fitz.open(stream=fh, filetype="pdf") as doc:
            for i, page in enumerate(doc, 1):
                text = page.get_text().strip()
                if text:
                    chunks.append((i, clean_text(text)))
        return chunks
    except:
        return []

# ---------------------- Lazy Chunk Search ----------------------
def list_all_files_recursively():
    query = "mimeType='application/vnd.google-apps.document' or " \
            "mimeType='application/vnd.google-apps.spreadsheet' or " \
            "mimeType='application/pdf'"
    files = []
    page_token = None
    while True:
        response = drive_service.files().list(
            q=query,
            corpora="drive",
            driveId=SHARED_DRIVE_ID,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            pageSize=100,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break
    return files

# ---------------------- Chunking & Relevance ----------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_relevant_chunks(question, files, top_k=5):
    vectorizer = TfidfVectorizer()
    text_chunks = []
    meta_data = []

    for file in files:
        file_id, name, mime = file['id'], file['name'], file['mimeType']
        link = f"https://drive.google.com/file/d/{file_id}"
        doc_texts = []

        if mime == 'application/vnd.google-apps.spreadsheet':
            doc_texts = read_google_sheet(file_id)
            meta_format = lambda i: f"row {i+1}"
        elif mime == 'application/vnd.google-apps.document':
            doc_texts = [para for _, para in read_google_doc(file_id)]
            meta_format = lambda i: f"paragraph {i+1}"
        elif mime == 'application/pdf':
            pdf_chunks = read_pdf(file_id)
            doc_texts = [text for _, text in pdf_chunks]
            meta_format = lambda i: f"page {pdf_chunks[i][0]}"
        else:
            continue

        for i, text in enumerate(doc_texts):
            text_chunks.append(text)
            meta_data.append({"source": name, "link": link, "meta": meta_format(i)})

    if not text_chunks:
        return []

    vectors = vectorizer.fit_transform(text_chunks + [question])
    scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]

    return [{"text": text_chunks[i], **meta_data[i]} for i in top_indices if scores[i] > 0.05]

# ---------------------- Slack Handler ----------------------
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event.get("channel", "")
        user_text = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()

        files = list_all_files_recursively()
        chunks = get_relevant_chunks(user_text, files)

        if not chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([c["text"] for c in chunks])
            prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_text}"
            try:
                gemini_response = model.generate_content(prompt)
                answer = getattr(gemini_response, 'text', 'No answer generated.')
                citations = []
                seen = set()
                for chunk in chunks:
                    key = chunk['link'] + chunk['meta']
                    if key not in seen:
                        seen.add(key)
                        snippet = chunk['text'][:60].split(". ")[0].strip() + "..."
                        citations.append(f"(Source: <{chunk['link']}|{chunk['source']}>, {chunk['meta']} — starts with: \"{snippet}\")")
                reply = f"{answer}\n\n" + "\n".join(citations)
            except Exception as e:
                reply = f"⚠️ Gemini Error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print("Slack error:", e.response["error"])

    return Response(), 200

@app.route("/")
def index():
    return "✅ ConahGPT is live and using lazy loading."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
