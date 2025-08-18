# app.py
import os
import io
import re
import json
import heapq
import string
import threading
from datetime import datetime
from typing import Dict, Generator, List, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import fitz  # PyMuPDF
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import google.generativeai as genai


# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Config / Clients
# -----------------------------
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly"
]

SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    raise ValueError("SERVICE_ACCOUNT_JSON not set.")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)

drive_service = build("drive", "v3", credentials=creds)
docs_service = build("docs", "v1", credentials=creds)
gspread_client = gspread.authorize(creds)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
slack_client = WebClient(token=SLACK_BOT_TOKEN)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")  # REQUIRED (single folder, searched recursively)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction=(
        "You are ConahGPT. Answer the user's question ONLY using the small CONTEXT provided. "
        "If the answer is not in CONTEXT, reply exactly: "
        "'I cannot answer this question as the information is not in the provided documents.' "
        "Answer in one short paragraph. Do not include citations in the text; they are added by the app."
    )
)

# -----------------------------
# Helpers
# -----------------------------
STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the their to was were will with you your
""".split())

def normalize_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(s: str) -> List[str]:
    s = s.lower().translate(str.maketrans({c: " " for c in string.punctuation}))
    return [w for w in s.split() if w and w not in STOPWORDS]

def overlap_score(query_tokens: List[str], text: str) -> int:
    # very fast bag-of-words overlap; avoids TF-IDF allocations (prevents OOM)
    qt = set(query_tokens)
    tt = set(tokenize(text))
    return len(qt & tt)

def drive_query(q: str, fields: str = "files(id,name,mimeType,parents)", page_token: str = None):
    return drive_service.files().list(
        q=q,
        corpora="drive",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields=fields + ",nextPageToken",
        pageToken=page_token,
        pageSize=200
    ).execute()

def list_files_recursive(folder_id: str) -> List[Dict]:
    """List Docs/Sheets/PDFs recursively under a single folder. Returns lightweight metadata only."""
    if not folder_id:
        return []
    # BFS over folders to avoid deep recursion
    queue = [folder_id]
    files: List[Dict] = []
    seen_folders = set()

    mime_doc = "application/vnd.google-apps.document"
    mime_sheet = "application/vnd.google-apps.spreadsheet"
    mime_folder = "application/vnd.google-apps.folder"
    mime_pdf = "application/pdf"

    while queue:
        fid = queue.pop(0)
        if fid in seen_folders:
            continue
        seen_folders.add(fid)

        page = None
        while True:
            res = drive_query(
                q=f"('{fid}' in parents) and trashed=false",
                fields="files(id,name,mimeType),nextPageToken",
                page_token=page
            )
            for f in res.get("files", []):
                mt = f["mimeType"]
                if mt == mime_folder:
                    queue.append(f["id"])
                elif mt in (mime_doc, mime_sheet, mime_pdf):
                    files.append(f)
            page = res.get("nextPageToken")
            if not page:
                break

    return files

# -----------------------------
# Chunk generators (memory-light)
# -----------------------------
def iter_google_doc_chunks(file_id: str, doc_name: str) -> Generator[Dict, None, None]:
    """
    Yield small chunks with heading metadata.
    We use Google Docs named style types (HEADING_1..HEADING_6).
    """
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        content = doc.get("body", {}).get("content", [])

        current_heading = "General"
        for c in content:
            if "paragraph" not in c:
                continue
            para = c["paragraph"]
            elements = para.get("elements", [])
            text = "".join([e.get("textRun", {}).get("content", "") for e in elements]).strip()
            if not text:
                continue

            style = para.get("paragraphStyle", {})
            named = style.get("namedStyleType", "")
            if named and named.startswith("HEADING_"):
                current_heading = text
                continue

            # Pair simple Q&A (question line followed by the next non-heading paragraph)
            if text.endswith("?"):
                # Try to peek next paragraph's text (best-effort)
                answer = ""
                nxt = content[content.index(c) + 1] if (content.index(c) + 1) < len(content) else None
                if nxt and "paragraph" in nxt:
                    nxt_style = nxt["paragraph"].get("paragraphStyle", {})
                    nxt_named = nxt_style.get("namedStyleType", "")
                    if not (nxt_named and nxt_named.startswith("HEADING_")):
                        answer = "".join([e.get("textRun", {}).get("content", "") for e in nxt["paragraph"].get("elements", [])]).strip()
                text = f"Question: {text} Answer: {answer}"

            yield {
                "file_id": file_id,
                "file_name": doc_name,
                "mime": "gdoc",
                "link": f"https://docs.google.com/document/d/{file_id}/edit",
                "meta": {"section": current_heading},
                "text": normalize_text(text)
            }
    except Exception as e:
        print(f"[Docs] {doc_name} ({file_id}) read error: {e}")

def iter_pdf_chunks(file_id: str, pdf_name: str) -> Generator[Dict, None, None]:
    try:
        # Stream download to memory once; we only keep top K chunks overall later
        req = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)

        with fitz.open(stream=fh, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text() or ""
                text = normalize_text(text)
                if not text:
                    continue
                yield {
                    "file_id": file_id,
                    "file_name": pdf_name,
                    "mime": "pdf",
                    "link": f"https://drive.google.com/file/d/{file_id}/preview#page={i}",
                    "meta": {"page": i},
                    "text": text
                }
    except Exception as e:
        print(f"[PDF] {pdf_name} ({file_id}) read error: {e}")

def iter_sheet_chunks(file_id: str, sheet_name: str) -> Generator[Dict, None, None]:
    # Keep sheets light: flatten rows in small blocks
    try:
        sh = gspread_client.open_by_key(file_id).sheet1
        rows = sh.get_all_values()
        block = []
        block_idx = 1
        for r in rows:
            line = normalize_text(" | ".join(r))
            if line:
                block.append(line)
            if len(block) >= 20:  # ~ small block
                yield {
                    "file_id": file_id,
                    "file_name": sheet_name,
                    "mime": "gsheet",
                    "link": f"https://docs.google.com/spreadsheets/d/{file_id}/edit",
                    "meta": {"block": block_idx},
                    "text": normalize_text(" ".join(block))
                }
                block = []
                block_idx += 1
        if block:
            yield {
                "file_id": file_id,
                "file_name": sheet_name,
                "mime": "gsheet",
                "link": f"https://docs.google.com/spreadsheets/d/{file_id}/edit",
                "meta": {"block": block_idx},
                "text": normalize_text(" ".join(block))
            }
    except Exception as e:
        print(f"[Sheet] {sheet_name} ({file_id}) read error: {e}")

# -----------------------------
# Retrieval (top-K without OOM)
# -----------------------------
def retrieve_top_chunks(question: str, max_files: int = 40, top_k: int = 3) -> Tuple[str, List[Dict]]:
    """
    Streams through files & chunks, keeps only top_k chunks (heap) by simple overlap score.
    This avoids building any large vector indices (prevents OOM).
    """
    files = list_files_recursive(DRIVE_FOLDER_ID)
    if not files:
        return "No files available in Drive folder.", []

    # Light file pre-filter by name keywords to go faster
    q_tokens = tokenize(question)
    scored_files: List[Tuple[int, Dict]] = []
    for f in files:
        name_score = overlap_score(q_tokens, f["name"])
        heapq.heappush(scored_files, (-name_score, f))
    # Take top N files by name match (still includes zero-score files)
    chosen_files = [heapq.heappop(scored_files)[1] for _ in range(min(max_files, len(scored_files)))]

    # Stream chunks and keep a min-heap of size top_k
    heap: List[Tuple[int, int, Dict]] = []  # (score, tiebreak, chunk)
    tiebreak = 0

    for f in chosen_files:
        mt = f["mimeType"]
        fid = f["id"]
        fname = f["name"]

        if mt == "application/vnd.google-apps.document":
            gen = iter_google_doc_chunks(fid, fname)
        elif mt == "application/pdf":
            gen = iter_pdf_chunks(fid, fname)
        elif mt == "application/vnd.google-apps.spreadsheet":
            gen = iter_sheet_chunks(fid, fname)
        else:
            continue

        for ch in gen:
            sc = overlap_score(q_tokens, ch["text"])
            if sc <= 0:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (sc, tiebreak, ch))
            else:
                if sc > heap[0][0]:
                    heapq.heapreplace(heap, (sc, tiebreak, ch))
            tiebreak += 1

    if not heap:
        return "", []

    # Highest scores first
    top_chunks = [h[2] for h in sorted(heap, key=lambda x: (-x[0], x[1]))]
    # Build tiny context (cap to ~8k chars)
    ctx_parts = []
    total = 0
    for ch in top_chunks:
        piece = f"Source: {ch['file_name']}\nContent: {ch['text']}\n"
        if total + len(piece) > 8000:
            break
        ctx_parts.append(piece)
        total += len(piece)
    context = "\n".join(ctx_parts)
    return context, top_chunks

# -----------------------------
# Answering + Single Citation
# -----------------------------
def format_single_citation(chunk: Dict) -> str:
    name = chunk["file_name"]
    link = chunk["link"]
    meta = chunk.get("meta", {})
    if chunk["mime"] == "pdf":
        return f'(Source: [{name}]({link}), on page {meta.get("page")})'
    if chunk["mime"] == "gdoc":
        section = meta.get("section", "General")
        return f'(Source: [{name}]({link}), in section "{section}")'
    if chunk["mime"] == "gsheet":
        return f'(Source: [{name}]({link}), in data block {meta.get("block")})'
    return f'(Source: [{name}]({link}))'

def answer_question(user_q: str) -> str:
    context, chunks = retrieve_top_chunks(user_q, max_files=40, top_k=3)

    if not chunks or not context:
        return "I cannot answer this question as the information is not in the provided documents."

    prompt = (
        "CONTEXT (use this only):\n"
        f"{context}\n\n"
        f"QUESTION: {user_q}\n\n"
        "ANSWER:"
    )
    try:
        resp = gemini.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        text = normalize_text(text)
        if not text or text.lower().startswith("i cannot answer"):
            return "I cannot answer this question as the information is not in the provided documents."
    except Exception as e:
        print(f"[Gemini] error: {e}")
        return "I cannot answer this question as the information is not in the provided documents."

    # append exactly one citation (best chunk)
    best = chunks[0]
    citation = format_single_citation(best)
    return f"{text} {citation}"

# -----------------------------
# Slack Events
# -----------------------------
def handle_slack_mention(channel_id: str, raw_text: str):
    # Strip the @mention token
    clean = re.sub(r"<@[^>]+>", "", raw_text).strip()
    if not clean:
        return
    reply = answer_question(clean)
    try:
        slack_client.chat_postMessage(channel=channel_id, text=reply)
    except SlackApiError as e:
        print(f"[Slack] post error: {e.response.get('error')}")

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.get_json(force=True, silent=True) or {}
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Ensure Slack doesn't get duplicates from retries
    if request.headers.get("X-Slack-Retry-Num"):
        return Response(status=200)

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event.get("channel")
        user_text = event.get("text", "")
        # Offload work so we ACK immediately (keeps Slack happy + snappy UX)
        threading.Thread(target=handle_slack_mention, args=(channel_id, user_text), daemon=True).start()

    return Response(status=200)

@app.route("/")
def index():
    return "✅ ConahGPT is running. Ready to answer."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
