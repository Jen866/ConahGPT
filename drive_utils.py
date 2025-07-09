import os
import io
import gspread
import fitz  # PyMuPDF
import json
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- Google Auth Setup ---
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly"
]

SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
if not SERVICE_ACCOUNT_JSON:
    try:
        with open('conah-gpt-creds.json') as f:
            SERVICE_ACCOUNT_JSON = f.read()
    except FileNotFoundError:
        raise ValueError("FATAL ERROR: The SERVICE_ACCOUNT_JSON environment variable is not set and the local 'conah-gpt-creds.json' file was not found.")

creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(SERVICE_ACCOUNT_JSON), SCOPES)
gspread_client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)
docs_service = build('docs', 'v1', credentials=creds)

def list_all_files_in_shared_drive(shared_drive_id):
    """Lists all supported files in a given shared drive."""
    # THIS IS THE CORRECTED QUERY LOGIC FOR A SHARED DRIVE
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/pdf'",
        corpora="drive",
        driveId=shared_drive_id,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def read_google_sheet_by_row(file_id):
    """Reads a Google Sheet and yields each row as a string."""
    try:
        sheet = gspread_client.open_by_key(file_id).sheet1
        records = sheet.get_all_records()
        for record in records:
            yield ", ".join([f"{k}: {v}" for k, v in record.items() if v])
    except Exception as e:
        print(f"Error reading Google Sheet {file_id}: {e}")
        return

def read_google_doc_by_paragraph(file_id):
    """Reads a Google Doc and yields each paragraph's text along with a direct link to its bookmark."""
    try:
        doc = docs_service.documents().get(documentId=file_id).execute()
        doc_content = doc.get('body', {}).get('content', [])
        for content_item in doc_content:
            if 'paragraph' in content_item:
                paragraph = content_item['paragraph']
                paragraph_text = ""
                bookmark_id = None
                if paragraph.get('elements'):
                    for element in paragraph.get('elements'):
                        if element.get('bookmarkId'):
                            bookmark_id = element.get('bookmarkId')
                            break
                for element in paragraph.get('elements', []):
                    if element.get('textRun'):
                        paragraph_text += element['textRun'].get('content', '')
                if paragraph_text.strip():
                    link = f"https://docs.google.com/document/d/{file_id}/edit#bookmark={bookmark_id}" if bookmark_id else f"https://docs.google.com/document/d/{file_id}"
                    yield {"text": paragraph_text.strip(), "link": link}
    except Exception as e:
        print(f"Error reading Google Doc {file_id}: {e}")
        return

def read_pdf_by_page(file_id):
    """Reads a PDF and yields each page's text along with a link to that specific page."""
    try:
        request_file = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_file)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        with fitz.open(stream=fh, filetype="pdf") as pdf_doc:
            for page_num, page in enumerate(pdf_doc, start=1):
                text = page.get_text()
                if text.strip():
                    link = f"https://drive.google.com/file/d/{file_id}#page={page_num}"
                    yield {"text": text.strip(), "link": link}
    except Exception as e:
        print(f"Error reading PDF {file_id}: {e}")
        return

def extract_all_chunks_with_links(shared_drive_id):
    """Extracts content from all files, creating chunks with specific links to their source."""
    files = list_all_files_in_shared_drive(shared_drive_id) # Using the corrected function name
    chunks = []
    for file in files:
        file_id, name, mime_type = file['id'], file['name'], file['mimeType']
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            doc_link = f"https://docs.google.com/spreadsheets/d/{file_id}"
            for row_text in read_google_sheet_by_row(file_id):
                if row_text:
                    chunks.append({"text": row_text, "source": name, "link": doc_link})
        elif mime_type == 'application/vnd.google-apps.document':
            for para_info in read_google_doc_by_paragraph(file_id):
                chunks.append({"text": para_info["text"], "source": name, "link": para_info["link"]})
        elif mime_type == 'application/pdf':
            for page_info in read_pdf_by_page(file_id):
                page_text_chunks = [page_info["text"][i:i+1500] for i in range(0, len(page_info["text"]), 1500)]
                for text_chunk in page_text_chunks:
                    chunks.append({"text": text_chunk, "source": name, "link": page_info["link"]})
    return chunks
