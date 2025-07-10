from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from googleapiclient.discovery import build
from PyPDF2 import PdfReader
import os
import io
import docx
import google.generativeai as genai

app = Flask(__name__)

# === CONFIGURATION ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'conah-gpt-service-key.json'  # <-- change this if your file has a different name
FOLDER_ID = '10ZMJ54LvgXBZDe5PdFmcqUzt06KrMJyl'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# === HELPERS ===
def get_file_list():
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed = false",
        pageSize=1000,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def extract_text_from_google_doc(file_id):
    doc = docs_service.documents().get(documentId=file_id).execute()
    text = ""
    for content in doc.get("body", {}).get("content", []):
        paragraph = content.get("paragraph")
        if paragraph:
            for element in paragraph.get("elements", []):
                text += element.get("textRun", {}).get("content", "")
    return text.strip()

def extract_text_from_google_sheet(file_id):
    result = sheets_service.spreadsheets().values().get(
        spreadsheetId=file_id, range="A1:Z1000").execute()
    rows = result.get('values', [])
    return "\n".join([", ".join(row) for row in rows])

def extract_text_from_pdf(file_id):
    file = drive_service.files().get_media(fileId=file_id).execute()
    reader = PdfReader(io.BytesIO(file))
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_id):
    file = drive_service.files().get_media(fileId=file_id).execute()
    doc = docx.Document(io.BytesIO(file))
    return "\n".join([para.text for para in doc.paragraphs])

def get_gdrive_file_link(file_id):
    return f"https://drive.google.com/file/d/{file_id}/view"

# === MAIN LOGIC ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_question = data.get('question', '')

    files = get_file_list()
    contexts = []
    used_sources = set()

    for f in files:
        try:
            if f['mimeType'] == 'application/vnd.google-apps.document':
                content = extract_text_from_google_doc(f['id'])
            elif f['mimeType'] == 'application/vnd.google-apps.spreadsheet':
                content = extract_text_from_google_sheet(f['id'])
            elif f['mimeType'] == 'application/pdf':
                content = extract_text_from_pdf(f['id'])
            elif f['mimeType'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                content = extract_text_from_docx(f['id'])
            else:
                continue

            file_link = get_gdrive_file_link(f['id'])
            if file_link in used_sources:
                continue  # avoid repeating the same source
            used_sources.add(file_link)

            contexts.append(f"Source: {f['name']} ({file_link})\n{content}")
        except Exception as e:
            print(f"Error reading {f['name']}: {e}")

    prompt = f"""You are Conah GPT, an expert assistant trained on actuarial, legal, and consulting documents. 
Answer the question below using only the context provided from internal company files. Be concise and factual. 
If unsure, say "I'm not sure based on the available files."

Question: {user_question}

---CONTEXT---
{chr(10).join(contexts[:5])}
"""

    response = model.generate_content(prompt)
    answer = response.text

    # Extract source links from prompt (top 5 only) and include just once
    source_links = [get_gdrive_file_link(f['id']) for f in files[:5]]
    unique_links = list(dict.fromkeys(source_links))  # remove duplicates while preserving order
    sources_md = "\n".join([f"- [{f['name']}]({get_gdrive_file_link(f['id'])})" for f in files[:5]])

    return jsonify({"answer": f"{answer}\n\n**Sources:**\n{sources_md}"})


if __name__ == '__main__':
    app.run(debug=True)
