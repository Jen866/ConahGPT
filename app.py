import os
import io
import re
from flask import Flask, request, jsonify, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PyPDF2 import PdfReader
import google.generativeai as genai

# === Flask setup ===
app = Flask(__name__)

# === Environment Variables ===
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SERVICE_ACCOUNT_JSON_PATH = os.environ.get("SERVICE_ACCOUNT_JSON_PATH") or "service_account.json"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SHARED_DRIVE_ID = "10ZMJ54LvgXBZDe5PdFmcqUzt06KrMJyl"

# === Slack client ===
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# === Google API clients ===
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON_PATH, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)
docs_service = build("docs", "v1", credentials=credentials)
sheets_service = build("sheets", "v4", credentials=credentials)

# === Gemini setup ===
genai.configure(api_key=GEMINI_API_KEY)

def get_file_content(file_id, mime_type):
    if mime_type == "application/pdf":
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        reader = PdfReader(fh)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    elif mime_type == "application/vnd.google-apps.document":
        doc = docs_service.documents().get(documentId=file_id).execute()
        content = doc.get("body", {}).get("content", [])
        text = ""
        for item in content:
            if "paragraph" in item:
                for el in item["paragraph"].get("elements", []):
                    text += el.get("textRun", {}).get("content", "")
        return text
    
    elif mime_type == "application/vnd.google-apps.spreadsheet":
        sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=file_id).execute()
        sheet_title = sheet_metadata["sheets"][0]["properties"]["title"]
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=file_id, range=f"{sheet_title}!A1:Z100"
        ).execute()
        rows = result.get("values", [])
        return "\n".join([", ".join(row) for row in rows])
    
    return ""

def get_gemini_response(question):
    files = drive_service.files().list(
        corpora="drive",
        driveId=SHARED_DRIVE_ID,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        q="mimeType contains 'application/pdf' or mimeType contains 'document' or mimeType contains 'spreadsheet'",
        fields="files(id, name, mimeType, webViewLink)"
    ).execute().get("files", [])

    prompt = f"You are ConahGPT, a helpful assistant. Use the following documents to answer:\n\n"
    for file in files:
        try:
            content = get_file_content(file["id"], file["mimeType"])
            if content:
                prompt += f"{content[:4000]}\n(Source: [{file['name']}]({file['webViewLink']}))\n\n"
        except Exception as e:
            print(f"❌ Error reading {file['name']}: {e}")

    prompt += f"\nAnswer the question: {question}"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini Error:", e)
        return "⚠️ Gemini failed to generate a response."

@app.route("/")
def index():
    return "✅ ConahGPT Slack Bot is running!"

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    print("📩 Slack Event:", data)

    # Slack verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # Handle mentions
    if data.get("event", {}).get("type") == "app_mention":
        event = data["event"]
        user_text = event.get("text", "")
        channel_id = event.get("channel", "")
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        print("🧠 User asked:", clean_text)

        response_text = get_gemini_response(clean_text)

        try:
            slack_client.chat_postMessage(channel=channel_id, text=response_text)
            print("✅ Replied to Slack.")
        except SlackApiError as e:
            print("❌ Slack API Error:", e.response["error"])

    return Response(), 200

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "")
        answer = get_gemini_response(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
