import os
import io
import gspread
import fitz  # PyMuPDF
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Service account credentials file
SERVICE_ACCOUNT_FILE = 'conah-gpt-creds.json'

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly"
]

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=SCOPES
)

gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

def list_all_files_in_folder(folder_id):
    """Return all files (of any type) in a folder."""
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def read_google_sheet(file_id):
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        records = sheet.get_all_records()
        return "\n".join([str(row) for row in records])
    except Exception as e:
        return f"Error reading sheet: {e}"

def read_google_doc(file_id):
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        content = doc.get("body", {}).get("content", [])
        text = ""
        for c in content:
            p = c.get("paragraph")
            if p:
                for el in p.get("elements", []):
                    text += el.get("textRun", {}).get("content", "")
        return text
    except Exception as e:
        return f"Error reading doc: {e}"

def read_pdf(file_id):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        text = ""
        with fitz.open(stream=fh, filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_all_text_from_folder(folder_id):
    files = list_all_files_in_folder(folder_id)
    results = []

    for file in files:
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']
        text = ""

        if mime == 'application/vnd.google-apps.spreadsheet':
            text = read_google_sheet(file_id)
        elif mime == 'application/vnd.google-apps.document':
            text = read_google_doc(file_id)
        elif mime == 'application/pdf':
            text = read_pdf(file_id)
        else:
            text = f"Unsupported file type: {mime}"

        results.append({
            "file_name": name,
            "mime_type": mime,
            "content": text
        })

    return results

