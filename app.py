import os
import io
import time
import json
import re
import fitz  # PyMuPDF
import gspread
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

# -------------------- Flask --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Tunables / Memory Caps --------------------
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))      # 10 minutes
MAX_FILES = int(os.getenv("MAX_FILES", "25"))                       # cap total files read
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "8"))                # first N pages per PDF
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "60000"))  # truncate long docs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "240"))                    # words per chunk
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "15000"))  # vectorizer cap
TOP_K = int(os.getenv("TOP_K", "3"))                                # relevant chunks

CHUNK_CACHE = {"data": None, "ts": 0}

# -------------------- Google Auth --------------------
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly",
]

SERVICE_ACCOUNT_JSON = os.environ["SERVICE_ACCOUNT_JSON"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    json.loads(SERVICE_ACCOUNT_JSON), SCOPES
)
gspread_client = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)
docs_service = build("docs", "v1", credentials=creds)

# -------------------- Gemini --------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-pro",
    system_instruction="""
You are Conah GPT, an expert business assistant for Actuary Consulting.
Answer the question using ONLY the provided CONTEXT. Interpret meaning; do not require exact matches.
If the answer is not found, reply: "I cannot answer this question as the information is not in the provided documents."
Return the answer ONCE, in a single concise paragraph (no bullets, no repetition). Do not restate the answer.
"""
)

# -------------------- Slack --------------------
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# -------------------- Helpers --------------------
def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ---- Readers that preserve page/paragraph positions ----
def read_google_doc_paragraphs(file_id):
    """
    Return list of (paragraph_text, paragraph_index starting at 1).
    We STITCH short/heading/question paragraphs with the next one(s),
    so 'Do I pay for Affidavits?' + 'No' becomes a single block for retrieval.
    """
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()

        # 1) Collect raw paragraphs
        raw_paras = []
        for content in doc.get("body", {}).get("content", []):
            para = content.get("paragraph")
            if not para:
                continue
            parts = []
            for el in para.get("elements", []):
                tr = el.get("textRun", {})
                parts.append(tr.get("content", ""))
            txt = clean_ws("".join(parts))
            if txt:
                raw_paras.append(txt[:MAX_CHARS_PER_FILE])

        # 2) Stitch short/question/headers with following answers
        stitched = []
        i = 0
        p_idx = 0
        while i < len(raw_paras):
            cur = raw_paras[i]
            looks_short = len(cur.split()) < 8
            looks_q = cur.endswith("?")
            looks_header = bool(re.match(r"^([A-Z][A-Z \-:0-9/]{2,}|[•\-\u2022])", cur))

            block = cur
            first_index = i + 1  # 1-based index for the citation
            j = i + 1

            if looks_short or looks_q or looks_header:
                pulls = 0
                while j < len(raw_paras) and pulls < 2:
                    nxt = raw_paras[j]
                    # stop if next is screaming header
                    if len(nxt.split()) >= 30 and nxt.isupper():
                        break
                    block = (block + " " + nxt).strip()
                    pulls += 1
                    j += 1
                i = j
            else:
                i += 1

            p_idx += 1
            stitched.append((block, first_index))

        return stitched

    except Exception:
        return []

def read_google_sheet_text(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        rows = sheet.get_all_records()
        text = "\n".join([str(row) for row in rows])[:MAX_CHARS_PER_FILE]
        return clean_ws(text)
    except Exception:
        return ""

def read_pdf_pages(file_id):
    """Return list of (page_text, page_number starting at 1)."""
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        pages = []
        with fitz.open(stream=fh, filetype="pdf") as pdf:
            n = min(MAX_PDF_PAGES, pdf.page_count)
            for i in range(n):
                t = clean_ws(pdf[i].get_text())
                if t:
                    pages.append((t[:MAX_CHARS_PER_FILE], i + 1))
        return pages
    except Exception:
        return []

# ---- Ingest to chunks with metadata ----
def list_files(shared_drive_id):
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' "
          "or mimeType='application/vnd.google-apps.spreadsheet' "
          "or mimeType='application/pdf'",
        corpora="drive",
        driveId=shared_drive_id,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get("files", [])[:MAX_FILES]

def extract_all_chunks_uncached(shared_drive_id):
    files = list_files(shared_drive_id)
    chunks = []
    for f in files:
        file_id, name, mime = f["id"], f["name"], f["mimeType"]
        link = f"https://drive.google.com/file/d/{file_id}"

        if mime == "application/vnd.google-apps.document":
            for p_text, p_no in read_google_doc_paragraphs(file_id):
                for piece in chunk_text(p_text):
                    chunks.append({
                        "text": piece,
                        "source": name,
                        "link": link,
                        "meta": f"paragraph {p_no}",
                        "kind": "doc",
                    })

        elif mime == "application/vnd.google-apps.spreadsheet":
            text = read_google_sheet_text(file_id)
            if text:
                for idx, piece in enumerate(chunk_text(text)):
                    chunks.append({
                        "text": piece,
                        "source": name,
                        "link": link,
                        "meta": f"paragraph {idx + 1}",
                        "kind": "sheet",
                    })

        elif mime == "application/pdf":
            for page_text, page_no in read_pdf_pages(file_id):
                for piece in chunk_text(page_text):
                    chunks.append({
                        "text": piece,
                        "source": name,
                        "link": link,
                        "meta": f"page {page_no}",
                        "kind": "pdf",
                    })

    return chunks

def get_chunks(shared_drive_id):
    now = time.time()
    if CHUNK_CACHE["data"] is None or now - CHUNK_CACHE["ts"] > CACHE_TTL_SECONDS:
        CHUNK_CACHE["data"] = extract_all_chunks_uncached(shared_drive_id)
        CHUNK_CACHE["ts"] = now
    return CHUNK_CACHE["data"]

def deduplicate_chunks(chunks):
    seen = set()
    unique = []
    for c in chunks:
        key = hash(c["text"])
        if key not in seen:
            unique.append(c)
            seen.add(key)
    return unique

def get_relevant_chunks(question, chunks, top_k=TOP_K):
    if not chunks:
        return []
    chunks = deduplicate_chunks(chunks)
    docs = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english").fit(docs + [question])
    doc_vectors = vectorizer.transform(docs)
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx if sims[i] > 0.05]

def keep_single_paragraph(answer_text: str) -> str:
    """Keep only the first meaningful paragraph (>=5 words)."""
    if not answer_text:
        return ""
    paras = re.split(r"\n\s*\n+", answer_text.strip())
    for p in paras:
        clean = p.strip()
        if len(clean.split()) >= 5:
            return clean
    for line in answer_text.splitlines():
        if len(line.split()) >= 5:
            return line.strip()
    return answer_text.strip()

def format_citations(relevant_chunks):
    """One citation per (source, meta) with preview ~10 words."""
    seen = set()
    lines = []
    for c in relevant_chunks:
        key = (c["source"], c["meta"])
        if key in seen:
            continue
        seen.add(key)
        preview = " ".join(c["text"].split()[:10])
        lines.append(
            f"(Source: [{c['source']}]({c['link']}), {c['meta']} — starts with: \"{preview}...\")"
        )
    if not lines:
        return ""
    return "**Sources:**\n" + "\n".join(lines)

# -------------------- Slack Events --------------------
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

        shared_drive_id = "0AL5LG1aWrCL2Uk9PVA"
        all_chunks = get_chunks(shared_drive_id)

        if not all_chunks:
            reply = "I couldn’t read any usable files from Google Drive. Please check the folder."
        else:
            relevant = get_relevant_chunks(clean_text, all_chunks, top_k=TOP_K)
            if not relevant:
                reply = "I cannot answer this question as the information is not in the provided documents."
            else:
                context = "\n\n".join([c["text"] for c in relevant])
                prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{clean_text}"

                try:
                    response = model.generate_content(prompt)
                    answer = getattr(response, "text", "").strip()
                    single = keep_single_paragraph(answer)
                    citations_md = format_citations(relevant)
                    reply = f"{single}\n\n{citations_md}" if citations_md else single
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"

        try:
            slack_client.chat_postMessage(channel=channel_id, text=reply)
        except SlackApiError as e:
            print("Slack API Error:", e.response.get("error"))

    return Response(), 200

# -------------------- Health --------------------
@app.route("/")
def index():
    return "✅ ConahGPT Slack bot is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
