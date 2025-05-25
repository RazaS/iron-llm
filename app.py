import os
import uuid
import json
import base64
from datetime import datetime
from flask import Flask, request, render_template_string, session
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# === Required API Keys ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_CREDS_B64 = os.getenv("GOOGLE_CREDS_B64")

# === Other Constants ===
INDEX_NAME = "multi-pdf-rag-docling-e5"
SPREADSHEET_NAME = "Iron-LLM logs"

# === Initialize Services ===
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer("intfloat/e5-base")

# === Flask App ===
app = Flask(__name__)
app.secret_key = os.urandom(24)  # no env var used, resets on server restart

# === HTML Template ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Guideline Chatbot</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background-color: #fafafa; }
    h2 { text-align: center; }
    form { display: flex; justify-content: center; gap: 10px; margin-bottom: 30px; }
    input[name="query"] { width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }
    input[type="submit"] { padding: 10px 20px; font-size: 16px; background-color: #0074d9; color: white; border: none; border-radius: 6px; cursor: pointer; }
    .chat-box { background: #fff; border-left: 4px solid #0074d9; padding: 16px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .user { font-weight: bold; margin-bottom: 4px; color: #0074d9; }
    .bot { white-space: pre-wrap; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <h2>Iron Guidelines Chatbot</h2>
  <form method="POST">
    <input name="query" placeholder="Ask a question..." required autofocus>
    <input type="submit" value="Ask">
  </form>
  <div id="chat-history">
    {% for pair in history %}
      <div class="chat-box">
        <div class="user">You:</div>
        <div>{{ pair.query }}</div>
        <br>
        <div class="user">Bot:</div>
        <div class="bot" id="bot-{{ loop.index }}"></div>
        <script>
          document.getElementById("bot-{{ loop.index }}").innerHTML = marked.parse(`{{ pair.response | tojson | safe }}`);
        </script>
      </div>
    {% endfor %}
  </div>
</body>
</html>
"""

# === Load Google credentials from base64
def get_google_credentials():
    if not GOOGLE_CREDS_B64:
        raise ValueError("Missing GOOGLE_CREDS_B64 environment variable.")
    creds_json = base64.b64decode(GOOGLE_CREDS_B64).decode("utf-8")
    info = json.loads(creds_json)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    return Credentials.from_service_account_info(info, scopes=scopes)

# === Log a row to the Google Sheet
def log_to_google_sheet(session_id, user_input, bot_response):
    try:
        creds = get_google_credentials()
        client = gspread.authorize(creds)
        sheet = client.open(SPREADSHEET_NAME).sheet1
        sheet.append_row([
            session_id,
            datetime.utcnow().isoformat(),
            user_input,
            bot_response
        ])
    except Exception as e:
        print(f"❌ Failed to log to Google Sheet: {e}")

# === Retrieve context from Pinecone
def retrieve_context(query, top_k=20):
    formatted_query = f"query: {query}"
    q_emb = embedder.encode([formatted_query])[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    context_pieces = [
        f"{m['metadata']['text']} (source: {m['metadata'].get('source', 'unknown')})"
        for m in results["matches"]
    ]
    return "\n\n".join(context_pieces)

# === Flask route
@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []
        session["session_id"] = str(uuid.uuid4())

    response = None
    if request.method == "POST":
        query = request.form["query"]
        context = retrieve_context(query)

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            result = model.generate_content(f"Use this context to answer the question:\n\n{context}\n\nQuestion: {query}")
            response = result.text
        except Exception as e:
            response = f"⚠️ Error generating response: {str(e)}"

        session["history"].append({"query": query, "response": response})
        session.modified = True

        log_to_google_sheet(session["session_id"], query, response)

    return render_template_string(HTML_TEMPLATE, response=response, history=reversed(session["history"]))

# === Run locally or in Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)