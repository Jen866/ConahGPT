import os
import json
import google.auth
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__)

# Load Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load Google credentials
SERVICE_ACCOUNT_INFO = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)
drive_service = build("drive", "v3", credentials=creds)
docs_service = build("docs", "v1", credentials=creds)

# Folder ID where your files are stored
FOLDER_ID = "10ZMJ54LvgXBZDe5PdFmcqUzt06KrMJyl"

def extract_google_doc_content(file_id):
    doc = docs_service.documents().get(documentId=file_id).execute()
    body = doc.get("body", {}).get("content", [])
    text = ""
    for value in body:
        if "paragraph" in value:
            elements = value["paragraph"].get("elements", [])
            for elem in elements:
                if "textRun" in elem:
                    text += elem["textRun"].get("content", "")
    return text

def extract_pdf_content(file_id):
    return "[PDF content not supported in this version]"

def get_files_from_drive():
    query = f"'{FOLDER_ID}' in parents and trashed = false"
    response = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return response.get("files", [])

def build_prompt(question, file_data):
    context = "\n\n".join(
        f"---\nFilename: {f['name']}\n\n{f['content']}" for f in file_data if f["content"]
    )
    return f"""You are Conah GPT, a helpful assistant answering questions using ACTUARY CONSULTING documents. 
Answer the question below using only the context provided. 
If the answer is not in the documents, say so.

Context:
{context}

Question: {question}
Answer:"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    files = get_files_from_drive()

    file_data = []
    sources = []

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        link = f"https://docs.google.com/document/d/{file_id}/edit"

        if file["mimeType"] == "application/vnd.google-apps.document":
            content = extract_google_doc_content(file_id)
            file_data.append({"name": file_name, "content": content, "link": link})
        elif file["mimeType"] == "application/pdf":
            content = extract_pdf_content(file_id)
            file_data.append({"name": file_name, "content": content, "link": link})

    prompt = build_prompt(question, file_data)

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    answer_text = response.text

    # Add single link to first matching document as source
    for f in file_data:
        if f["name"].lower() in answer_text.lower() or any(word in answer_text.lower() for word in f["content"].lower().split()[:50]):
            answer_text += f"\n\n[Source]({f['link']})"
            break

    return jsonify({"answer": answer_text})

if __name__ == "__main__":
    app.run(debug=True)
