import os
import json
import markdown
from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai

app = Flask(__name__)

# Set up Gemini API key from environment
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Load service account credentials from environment
SERVICE_ACCOUNT_INFO = json.loads(os.environ.get("SERVICE_ACCOUNT_JSON"))
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=SCOPES
)

# Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Google Drive folder to pull context from
FOLDER_ID = "10ZMJ54LvgXBZDe5PdFmcqUzt06KrMJyl"

# Utility: Load Google Drive files
def get_files_from_folder(folder_id):
    drive_service = build("drive", "v3", credentials=creds)
    query = f"'{folder_id}' in parents and trashed = false"
    response = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return response.get("files", [])

# Utility: Extract content from Google Docs
def extract_text_from_google_doc(file_id):
    docs_service = build("docs", "v1", credentials=creds)
    doc = docs_service.documents().get(documentId=file_id).execute()
    body = doc.get("body", {}).get("content", [])
    text = ""
    for content in body:
        if "paragraph" in content:
            elements = content["paragraph"].get("elements", [])
            for elem in elements:
                text += elem.get("textRun", {}).get("content", "")
    return text.strip()

# Utility: Extract content from Google Sheets
def extract_text_from_google_sheet(file_id):
    sheets_service = build("sheets", "v4", credentials=creds)
    sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=file_id).execute()
    sheet_names = [s["properties"]["title"] for s in sheet_metadata["sheets"]]

    all_text = ""
    for name in sheet_names:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=file_id, range=name
        ).execute()
        values = result.get("values", [])
        for row in values:
            all_text += " | ".join(row) + "\n"
    return all_text.strip()

# Utility: Extract content from PDFs
def extract_text_from_pdf(file_id):
    from io import BytesIO
    from PyPDF2 import PdfReader

    drive_service = build("drive", "v3", credentials=creds)
    request = drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = request.execute()
    fh.write(downloader)
    fh.seek(0)
    reader = PdfReader(fh)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# Route: Web interface
@app.route("/")
def index():
    return render_template("index.html")

# Route: Ask Conah GPT
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"answer": "Please provide a question."})

    files = get_files_from_folder(FOLDER_ID)

    context_blocks = []
    sources = set()  # Avoid duplicates

    for file in files:
        file_id = file["id"]
        name = file["name"]
        mime_type = file["mimeType"]

        try:
            if "document" in mime_type:
                content = extract_text_from_google_doc(file_id)
            elif "spreadsheet" in mime_type:
                content = extract_text_from_google_sheet(file_id)
            elif "pdf" in mime_type or name.lower().endswith(".pdf"):
                content = extract_text_from_pdf(file_id)
            else:
                continue

            doc_link = f"https://drive.google.com/file/d/{file_id}/view"
            context_blocks.append(f"From {name}:\n{content}")
            sources.add(f"[{name}]({doc_link})")

        except Exception as e:
            print(f"Failed to read {name}: {e}")

    prompt = (
        f"You are Conah GPT, a helpful assistant for actuaries.\n\n"
        f"User Question: {user_question}\n\n"
        f"Context:\n{chr(10).join(context_blocks)}"
    )

    try:
        response = model.generate_content(prompt)
        answer = response.text
        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
        return jsonify({"answer": markdown.markdown(answer)})
    except Exception as e:
        return jsonify({"answer": f"Error generating response: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
