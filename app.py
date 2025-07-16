import os
import re
from flask import Flask, request, jsonify, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import google.generativeai as genai

# === Flask app setup ===
app = Flask(__name__)

# === Slack client setup ===
slack_token = os.environ.get("SLACK_BOT_TOKEN")
slack_client = WebClient(token=slack_token)

# === Gemini setup ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return "⚠️ Gemini is temporarily unavailable."

@app.route("/")
def index():
    return "✅ ConahGPT is running!"

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    print("📩 Received Slack event:", data)

    # 1. Respond to Slack verification challenge
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # 2. Handle app_mention event
    if data.get("event", {}).get("type") == "app_mention":
        event = data.get("event", {})
        user_text = event.get("text", "")
        channel_id = event.get("channel", "")

        # Clean the text (remove bot mention)
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()
        print("🧠 User asked:", clean_text)

        # === FAST HARD-CODED REPLY for testing/demo ===
        response_text = f"✅ ConahGPT is alive! You said: *{clean_text}*"

        # === OPTIONAL: Use Gemini instead
        # response_text = get_gemini_response(clean_text)

        try:
            slack_client.chat_postMessage(channel=channel_id, text=response_text)
            print("✅ Replied to Slack successfully.")
        except SlackApiError as e:
            print("❌ Slack API Error:", e.response["error"])

    return Response(), 200

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "")
        response = get_gemini_response(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
