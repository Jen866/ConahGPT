# app.py
import os
import io
import re
import json
import heapq
import string
import threading
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
    "https://www.googleapis.com/auth/documents.readonly",
]
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    raise ValueError("SERVICE_ACCOUNT_JSON not set.")
creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)

drive = build("drive", "v3", credentials=creds)
docs = build("docs", "v1", credentials=creds)
gspread_client = gspread.authorize(creds)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    raise ValueError("SLACK_BOT_TOKEN not set.")
slack = WebClient(token=SLACK_BOT_TOKEN)

# IMPORTANT: This may be a Shared Drive ID (like 0AL5...) OR a Folder ID.
DRIVE_CONTAINER_ID = os.environ.get("DRIVE_CONTAINER_ID", "").strip()
if not DRIVE_CONTAINER_ID:
    raise ValueError("DRIVE_CONTAINER_ID not set (use your shared drive ID 0AL5LG1aWrCL2Uk9PVA).")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction=(
        "You are ConahGPT. Answer the user's question ONLY using the small CONTEXT provided. "
        "If the answer is not in CONTEXT, reply exactly: "
        "'I cannot answer this question as the information is not in the provided documents.' "
        "Answer in one short paragraph. Do not include citations in the text; they are added by the app."
    ),
)

# -----------------------------
# Helpers
# -----------------------------
STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the their to was were will with you your
""".split())

def norm(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def toks(s: str) -> List[str]:
    s = s.lower().translate(str.maketrans({c: " " for c in string.punctuation}))
    return [w for w in s.split() if w and w not in STOPWORDS]

def overlap(query_tokens: List[str], text: str) -> int:
    return len(set(query_tokens) & set(toks(text)))


# -----------------------------
# File listing (handles Shared Drive OR Folder)
# -----------------------------
MIME_DOC = "application/vnd.google-apps.document"
MIME_SHEET = "application/vnd.google-apps.spreadsheet"
MIME_PDF = "application/pdf"
MIME_FOLDER = "application/vnd.google-apps.folder"

def list_in_shared_drive(drive_id: str) -> List[Dict]:
    files: List[Dict] = []
    page = None
    q = (
        "trashed=false and ("
        f"mimeType='{MIME_DOC}' or mimeType='{MIME_SHEET}' or mimeType='{MIME_PDF}'"
        ")"
    )
    while True:
        res = drive.files().list(
            corpora="drive",
            driveId=drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            q=q,
            fields="files(id,name,mimeType),nextPageToken",
            pageToken=page,
            pageSize=200,
        ).execute()
        files.extend(res.get("files", []))
        page = res.get("nextPageToken")
        if not page:
            break
    return files

def list_in_folder_recursive(folder_id: str) -> List[Dict]:
    queue = [folder_id]
    files: List[Dict] = []
    seen = set()
    while queue:
        fid = queue.pop(0)
        if fid in seen:
            continue
        seen.add(fid)
        page = None
        while True:
            res = drive.files().list(
                q=f"'{fid}' in parents and trashed=false",
                corpora="allDrives",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="files(id,name,mimeType),nextPageToken",
                pageToken=page,
                pageSize=200,
            ).execute()
            for f in res.get("files", []):
                mt = f["mimeType"]
                if mt == MIME_FOLDER:
                    queue.append(f["id"])
                elif mt in (MIME_DOC, MIME_SHEET, MIME_PDF):
                    files.append(f)
            page = res.get("nextPageToken")
            if not page:
                break
    return files

def list_files(container_id: str) -> List[Dict]:
    """
    Try Shared Drive listing first (for IDs like 0AL5...), else fall back to folder recursion.
    """
    try:
        files = list_in_shared_drive(container_id)
        if files:
            return files
    except Exception as e:
        print(f"[List] Shared drive listing failed: {e}")
    try:
        return list_in_folder_recursive(container_id)
    except Exception as e:
        print(f"[List] Folder listing failed: {e}")
        return []


# -----------------------------
# Chunk generators
# -----------------------------
def iter_gdoc_chunks(file_id: str, name: str) -> Generator[Dict, None, None]:
    """
    Parse Google Docs. Use named heading styles (HEADING_1..6) to set 'section'.
    Treat any line ending with '?' as a question and pair it with the next non-heading paragraph as the answer.
    """
    try:
        doc = docs.documents().get(documentId=file_id).execute()
        content = doc.get("body", {}).get("content", [])
        current_section = "General"

        # iterate with index to safely peek next paragraph
        for i in range(len(content)):
            c = content[i]
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
                current_section = text
                continue

            # Q&A pairing
            if text.endswith("?"):
                ans = ""
                if i + 1 < len(content) and "paragraph" in content[i + 1]:
                    nxt = content[i + 1]["paragraph"]
                    nstyle = nxt.get("paragraphStyle", {})
                    nnamed = nstyle.get("namedStyleType", "")
                    if not (nnamed and nnamed.startswith("HEADING_")):
                        ans = "".join([e.get("textRun", {}).get("content", "") for e in nxt.get("elements", [])]).strip()
                text = f"Question: {text} Answer: {ans}"

            yield {
                "file_id": file_id,
                "file_name": name,
                "mime": "gdoc",
                "link": f"https://docs.google.com/document/d/{file_id}/edit",
                "meta": {"section": current_section},
                "text": norm(text),
            }
    except Exception as e:
        print(f"[Docs] {name} ({file_id}) read error: {e}")

def iter_pdf_chunks(file_id: str, name: str) -> Generator[Dict, None, None]:
    try:
        req = drive.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with fitz.open(stream=fh, filetype="pdf") as pdf:
            for page_num, page in enumerate(pdf, start=1):
                txt = norm(page.get_text() or "")
                if not txt:
                    continue
                yield {
                    "file_id": file_id,
                    "file_name": name,
                    "mime": "pdf",
                    "link": f"https://drive.google.com/file/d/{file_id}/preview#page={page_num}",
                    "meta": {"page": page_num},
                    "text": txt,
                }
    except Exception as e:
        print(f"[PDF] {name} ({file_id}) read error: {e}")

def iter_sheet_chunks(file_id: str, name: str) -> Generator[Dict, None, None]:
    try:
        sh = gspread_client.open_by_key(file_id).sheet1
        rows = sh.get_all_values()
        block, idx = [], 1
        for r in rows:
            line = norm(" | ".join(r))
            if line:
                block.append(line)
            if len(block) >= 20:
                yield {
                    "file_id": file_id,
                    "file_name": name,
                    "mime": "gsheet",
                    "link": f"https://docs.google.com/spreadsheets/d/{file_id}/edit",
                    "meta": {"block": idx},
                    "text": norm(" ".join(block)),
                }
                block, idx = [], idx + 1
        if block:
            yield {
                "file_id": file_id,
                "file_name": name,
                "mime": "gsheet",
                "link": f"https://docs.google.com/spreadsheets/d/{file_id}/edit",
                "meta": {"block": idx},
                "text": norm(" ".join(block)),
            }
    except Exception as e:
        print(f"[Sheet] {name} ({file_id}) read error: {e}")


# -----------------------------
# Retrieval (heap keeps top-k)
# -----------------------------
def retrieve_top_chunks(question: str, max_files: int = 200, top_k: int = 3) -> Tuple[str, List[Dict]]:
    files = list_files(DRIVE_CONTAINER_ID)
    if not files:
        print("[Retrieve] No files found under container ID.")
        return "", []

    # quick prefilter by filename overlap
    qtok = toks(question)
    scored = [(-overlap(qtok, f["name"]), f) for f in files]
    heapq.heapify(scored)
    chosen = [heapq.heappop(scored)[1] for _ in range(min(max_files, len(scored)))]

    heap: List[Tuple[int, int, Dict]] = []
    tiebreak = 0

    def push(ch: Dict):
        nonlocal heap, tiebreak
        sc = overlap(qtok, ch["text"])
        if sc < 0:  # defensive
            sc = 0
        if len(heap) < top_k:
            heapq.heappush(heap, (sc, tiebreak, ch))
        else:
            if sc > heap[0][0]:
                heapq.heapreplace(heap, (sc, tiebreak, ch))
        tiebreak += 1

    for f in chosen:
        mt, fid, name = f["mimeType"], f["id"], f["name"]
        gen = None
        if mt == MIME_DOC:
            gen = iter_gdoc_chunks(fid, name)
        elif mt == MIME_PDF:
            gen = iter_pdf_chunks(fid, name)
        elif mt == MIME_SHEET:
            gen = iter_sheet_chunks(fid, name)
        if not gen:
            continue
        for ch in gen:
            push(ch)

    # Fallback: if no overlap, still use first chunk(s) so Gemini has context
    if not heap:
        for f in chosen[:5]:
            mt, fid, name = f["mimeType"], f["id"], f["name"]
            gen = (
                iter_gdoc_chunks(fid, name) if mt == MIME_DOC
                else iter_pdf_chunks(fid, name) if mt == MIME_PDF
                else iter_sheet_chunks(fid, name) if mt == MIME_SHEET
                else None
            )
            if not gen:
                continue
            try:
                ch = next(gen)
                push(ch)
            except StopIteration:
                continue

    if not heap:
        return "", []

    top = [h[2] for h in sorted(heap, key=lambda x: (-x[0], x[1]))]
    # compact context (~8k cap)
    ctx, total = [], 0
    for ch in top:
        part = f"Source: {ch['file_name']}\nContent: {ch['text']}\n"
        if total + len(part) > 8000:
            break
        ctx.append(part); total += len(part)
    return "\n".join(ctx), top


# -----------------------------
# Answer + single citation
# -----------------------------
def citation_for(ch: Dict) -> str:
    name, link, meta, mime = ch["file_name"], ch["link"], ch.get("meta", {}), ch["mime"]
    if mime == "pdf":
        return f'(Source: [{name}]({link}), on page {meta.get("page")})'
    if mime == "gdoc":
        sec = meta.get("section", "General")
        return f'(Source: [{name}]({link}), in section "{sec}")'
    if mime == "gsheet":
        return f'(Source: [{name}]({link}), in data block {meta.get("block")})'
    return f'(Source: [{name}]({link}))'

def answer(user_q: str) -> str:
    context, chunks = retrieve_top_chunks(user_q, max_files=200, top_k=3)
    if not context or not chunks:
        return "I cannot answer this question as the information is not in the provided documents."

    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {user_q}\n\nANSWER:"
    try:
        resp = gemini.generate_content(prompt)
        text = norm(getattr(resp, "text", "") or "")
        if not text or text.lower().startswith("i cannot answer"):
            return "I cannot answer this question as the information is not in the provided documents."
    except Exception as e:
        print(f"[Gemini] error: {e}")
        return "I cannot answer this question as the information is not in the provided documents."

    best = chunks[0]  # highest scoring chunk
    return f"{text} {citation_for(best)}"


# -----------------------------
# Slack Events
# -----------------------------
def handle_mention(channel_id: str, raw_text: str):
    q = re.sub(r"<@[^>]+>", "", raw_text).strip()
    if not q:
        return
    reply = answer(q)
    try:
        slack.chat_postMessage(channel=channel_id, text=reply)
    except SlackApiError as e:
        print(f("[Slack] post error: {e.response.get('error')}"))

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.get_json(force=True, silent=True) or {}
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
    # avoid duplicate replies on Slack retries
    if request.headers.get("X-Slack-Retry-Num"):
        return Response(status=200)

    event = data.get("event", {})
    if event.get("type") == "app_mention":
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        threading.Thread(target=handle_mention, args=(channel_id, text), daemon=True).start()

    return Response(status=200)

@app.route("/")
def index():
    return "✅ ConahGPT is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
