from flask import Flask, request, jsonify
import google.auth
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os
import re
from google.generativeai import configure, GenerativeModel

app = Flask(__name__)

# Set your Gemini API key
configure(api_key="YOUR_GEMINI_API_KEY")

# Setup credentials for Google Drive API
SERVICE_ACCOUNT_FILE = 'service_account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

# Folder ID of shared folder with documents
FOLDER_ID = '10ZMJ54LvgXBZDe5PdFmcqUzt06KrMJyl'

def list_documents_in_folder(folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    results = drive_service.files().list(
        q=query, pageSize=100, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def extract_text_from_google_doc(file_id):
    doc = docs_service.documents().get(documentId=file_id).execute()
    text = ''
    for element in doc.get('body').get('content'):
        if 'paragraph' in element:
            for run in element['paragraph'].get('elements', []):
                text += run.get('textRun', {}).get('content', '')
    return text

def create_bookmark_links(file_id, content):
    # Extract headings/bookmarks and map them
    links = []
    for match in re.finditer(r'(#+\s*)(.+)', content):
        heading = match.group(2).strip()
        # Turn heading into a pseudo-bookmark ID for the link
        fragment = heading.lower().replace(' ', '-').replace('?', '')
        url = f'https://docs.google.com/document/d/{file_id}/edit#heading={fragment}'
        links.append((heading, url))
    return links

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get("question")

    # Get all docs in the folder
    files = list_documents_in_folder(FOLDER_ID)
    context_blocks = []

    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.document':
            doc_text = extract_text_from_google_doc(file['id'])
            doc_link = f"https://docs.google.com/document/d/{file['id']}/edit"
            context_blocks.append(f"{file['name']}:\n{doc_text}\n[Link]({doc_link})")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are Conah GPT, an expert actuarial assistant. Use only the context below to answer the user's question. Include clickable links in [text](url) markdown format to the document where the answer was found (only once in the paragraph, not repeatedly). Do not add a separate 'Sources' section at the end. Avoid repeating the same source.

Context:
{context}

User Question: {user_question}
Answer:
"""

    model = GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return jsonify({"answer": response.text})

if __name__ == '__main__':
    app.run(debug=True)
