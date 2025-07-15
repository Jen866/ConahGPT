import os
import re
from flask import Flask, request, jsonify, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import google.generativeai as genai

# === Flask setup ===
app = Flask(__name__)

# === Gemini setup ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return "⚠️ Gemini could not generate a response right now."

# === Slack setup ===
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

@app.route("/")
def index():
    return "✅ ConahGPT is running!"

# === Slack Events Endpoint ===
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    print("📩 Received Slack event:", data)

    # 1. Respond to Slack's URL verification challenge
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    # 2. Handle bot mention
    if data.get("event", {}).get("type") == "app_mention":
        user_text = data["event"].get("text", "")
        channel_id = data["event"].get("channel", "")
        print("🔔 Bot was mentioned:", user_text)

        # Remove bot mention (e.g., <@U01ABCDE>)
        clean_text = re.sub(r"<@[^>]+>", "", user_text).strip()

        # === TEST REPLY ONLY ===
        # response_text = f"✅ ConahGPT received: *{clean_text}*"

        # === REAL REPLY VIA GEMINI ===
        response_text = get_gemini_response(clean_text)

        # Post message back to Slack
        try:
            slack_client.chat_postMessage(channel=channel_id, text=response_text)
        except SlackApiError as e:
            print("❌ Slack API error:", e.response["error"])

    return Response(), 200

# === Gemini HTTP route (optional) ===
@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "")
        print("🧠 Received web question:", question)
        response = get_gemini_response(question)
        return jsonify({"answer": response})
    except Exception as e:
        print("❌ /ask error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
