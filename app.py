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
gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
    You are Conah GPT, a precise business assistant for Actuary Consulting.
    Fully answer each question clearly based on the provided CONTEXT.
    Only respond once per question. Include source citations at the end:
    (Source: [Document Name], paragraph 5 — starts with: \"Preview snippet...\")
    If source is a PDF, show page number instead of paragraph.
    If answer is not available, say:
    'I cannot answer this question as the information is not in the provided documents.'
    """
)

slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))


def clean_pdf_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def get_all_file_ids_recursive(folder_id):
    files = []
    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            corpora="drive",
            driveId="0AL5LG1aWrCL2Uk9PVA",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        for file in response.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                files += get_all_file_ids_recursive(file['id'])
            else:
                files.append(file)

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return files

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
        para_count = 0
        for content_item in doc.get("body", {}).get("content", []):
            if "paragraph" in content_item:
                para_text = ""
                for element in content_item["paragraph"].get("elements", []):
                    para_text += element.get("textRun", {}).get("content", "")
                if para_text.strip():
                    para_count += 1
                    paragraphs.append((para_count, para_text.strip()))
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
        chunks = []
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                text = page.get_text().strip()
                if text:
                    chunks.append((page_num, clean_pdf_text(text)))
        return chunks
    except:
        return []

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_relevant_chunks(question, files, top_k=5):
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        doc_link = f"https://drive.google.com/file/d/{file_id}"
        content_chunks = []
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            content = read_google_sheet(file_id)
            for i, chunk in enumerate(chunk_text(content)):
                content_chunks.append({"text": chunk, "source": name, "link": doc_link, "meta": f"row {i+1}"})
        elif mime_type == 'application/vnd.google-apps.document':
            for para_num, para in read_google_doc(file_id):
                content_chunks.append({"text": para, "source": name, "link": doc_link, "meta": f"paragraph {para_num}"})
        elif mime_type == 'application/pdf':
            for page_num, text in read_pdf(file_id):
                for i, chunk in enumerate(chunk_text(text)):
                    content_chunks.append({"text": chunk, "source": name, "link": doc_link, "meta": f"page {page_num}"})

        chunks.extend(content_chunks)

    documents = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(documents + [question])
    doc_vectors = vectorizer.transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    seen_keys = set()
    relevant = []
    for i in top_indices:
        key = chunks[i]['link'] + chunks[i]['meta']
        if similarities[i] > 0.05 and key not in seen_keys:
            seen_keys.add(key)
            relevant.append(chunks[i])
    return relevant

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if data.get("event", {}).get("type") == "app_mention":
        user_text = data["event"].get("text", "")
        channel_id = data["event"].get("channel", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        all_files = get_all_file_ids_recursive("0AL5LG1aWrCL2Uk9PVA")
        relevant_chunks = get_relevant_chunks(clean_text, all_files)

        if not relevant_chunks:
            reply = "I cannot answer this question as the information is not in the provided documents."
        else:
            context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
            prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"
            try:
                gemini_response = model.generate_content(prompt)
                raw_answer = getattr(gemini_response, "text", "Oops, no answer returned.")
                citations = []
                seen = set()
                for chunk in relevant_chunks:
                    key = chunk['link'] + chunk['meta']
                    if key not in seen:
                        seen.add(key)
                        snippet = chunk['text'][:60].split(". ")[0].strip() + "..."
                        citations.append(f"(Source: <{chunk['link']}|{chunk['source']}>, {chunk['meta']} — starts with: \"{snippet}\")")
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
    return "✅ ConahGPT is live."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
