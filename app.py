import os
import re
import json
import time
import logging
import fitz  # PyMuPDF
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from vertexai.generative_models import GenerativeModel, Part
import vertexai

# === CONFIG ===
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CLIENT = WebClient(token=SLACK_BOT_TOKEN)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
DRIVE_FOLDER_ID = "0AL5LG1aWrCL2Uk9PVA"

# === INIT ===
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
vertexai.init(project="your-project-id", location="us-central1")
gemini = GenerativeModel("gemini-1.5-pro-preview-0409")

# === AUTH ===
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=creds)
docs_service = build("docs", "v1", credentials=creds)

# === HELPERS ===
def list_drive_files(folder_id):
    files = []
    page_token = None
    while True:
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()
        files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if not page_token:
            break
    return files

def extract_paragraphs(doc_id):
    doc = docs_service.documents().get(documentId=doc_id).execute()
    text_elements = doc.get("body", {}).get("content", [])
    paragraphs = []
    for elem in text_elements:
        if "paragraph" in elem:
            texts = elem["paragraph"].get("elements", [])
            full_text = "".join(t.get("textRun", {}).get("content", "") for t in texts)
            full_text = full_text.strip()
            if full_text:
                paragraphs.append(full_text)
    return paragraphs

def extract_pdf_chunks(file_id):
    request_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(request_url, headers=headers)
    if response.status_code != 200:
        return []
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        for para in text.split("\n\n"):
            snippet = para.strip()
            if snippet:
                chunks.append({"text": snippet, "page": page_num})
    return chunks

def build_context():
    files = list_drive_files(DRIVE_FOLDER_ID)
    all_chunks = []
    for f in files:
        if f["mimeType"] == "application/vnd.google-apps.document":
            paras = extract_paragraphs(f["id"])
            for i, p in enumerate(paras):
                if len(p) > 20:
                    all_chunks.append({
                        "text": p,
                        "source": f["name"],
                        "url": f"https://docs.google.com/document/d/{f['id']}",
                        "meta": f"paragraph {i+1}"
                    })
        elif f["mimeType"] == "application/pdf":
            chunks = extract_pdf_chunks(f["id"])
            for c in chunks:
                all_chunks.append({
                    "text": c["text"],
                    "source": f["name"],
                    "url": f"https://drive.google.com/file/d/{f['id']}",
                    "meta": f"page {c['page']}"
                })
    return all_chunks

def answer_question_with_context(question, chunks):
    prompt = f"""
You are ConahGPT, a helpful assistant that answers user questions using the provided company documents.

Answer the question **in full**, directly, and clearly.
At the end of your answer, include the **source reference** like this:
(Source: [Document Name](URL), paragraph 4 — starts with: "First words...") or
(Source: [PDF Name](URL), page 3 — starts with: "First words...")

Question: {question}

Relevant context:
"""
    selected_chunks = []
    seen = set()
    for c in chunks:
        key = c['text'][:100]
        if key not in seen:
            selected_chunks.append(f"From {c['source']} ({c['meta']}): {c['text']}")
            seen.add(key)
        if len(selected_chunks) >= 10:
            break

    prompt += "\n\n".join(selected_chunks)
    response = gemini.generate_content(prompt)
    return response.text

# === SLACK ===
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return data["challenge"]

    if "event" in data:
        event = data["event"]
        if event.get("type") == "app_mention":
            text = event.get("text", "")
            user = event.get("user")
            channel = event.get("channel")
            question = re.sub(r"<@[^>]+>", "", text).strip()
            if question:
                try:
                    chunks = build_context()
                    answer = answer_question_with_context(question, chunks)
                    SLACK_CLIENT.chat_postMessage(channel=channel, text=answer)
                except SlackApiError as e:
                    logging.error(f"Slack error: {e.response['error']}")
    return "", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
