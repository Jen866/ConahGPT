import os
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai

# Load environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SERVICE_ACCOUNT_JSON = json.loads(os.environ.get("SERVICE_ACCOUNT_JSON"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Slack App setup
app = App(token=SLACK_BOT_TOKEN)

# Google Docs API setup
SCOPES = ["https://www.googleapis.com/auth/documents.readonly", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(SERVICE_ACCOUNT_JSON, SCOPES)
docs_service = build("docs", "v1", credentials=creds)
drive_service = build("drive", "v3", credentials=creds)

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# Utility: Load content from Google Docs
def get_all_docs_content():
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        corpora="drive",
        driveId="0AL5LG1aWrCL2Uk9PVA",  # Replace with your shared drive ID
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id, name)"
    ).execute()

    docs = results.get("files", [])
    context = ""
    for doc in docs:
        doc_id = doc["id"]
        name = doc["name"]
        doc_data = docs_service.documents().get(documentId=doc_id).execute()
        text = ""
        for item in doc_data.get("body", {}).get("content", []):
            for element in item.get("paragraph", {}).get("elements", []):
                text += element.get("textRun", {}).get("content", "")
        if text.strip():
            context += f"\n\nFROM {name}:\n{text}"
    return context or "No document content available."

# Respond to app mentions
@app.event("app_mention")
def handle_mention(body, say):
    event = body.get("event", {})
    user_question = event.get("text", "").split(">")[-1].strip()  # Removes "@ConahGPT"

    context = get_all_docs_content()
    prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_question}"

    try:
        response = model.generate_content(prompt)
        answer = getattr(response, "text", "No answer returned.")
        say(answer)
    except Exception as e:
        say(f"⚠️ Error generating response: {str(e)}")

# Start the Socket Mode listener
if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
