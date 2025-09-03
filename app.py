from pymongo import MongoClient
import secrets
import subprocess
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import sys

app = Flask(__name__, template_folder="templates")
app.secret_key = secrets.token_hex(16)

# MongoDB Configuration
client = MongoClient("mongodb+srv://abhilashvisakan2004:lCz2zRCnMNxZBDTV@cluster0.onbhxed.mongodb.net/")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Access database and collection
db = client["UserData"]
collection = db["Users"]

@app.route("/")
def index():
    return render_template('hi.html')

@app.route("/ai")
def ai():
    return render_template('ai.html')

@app.route("/submit", methods=["POST"])
def submit():
    fullname = request.form["fullname"]
    email = request.form["email"]
    code = request.form["code"]
    phone = request.form["phone"]

    user_data = {
        "fullname": fullname,
        "email": email,
        "phone": f"{code} {phone}"
    }

    collection.insert_one(user_data)

    session["toast"] = f"Welcome {fullname}, You have successfully logged in."
    return redirect(url_for("HomePage"))

@app.route("/HomePage")
def HomePage():
    toast_message = session.pop("toast", None)
    return render_template("index.html", toast_message=toast_message)

# Define paths to model scripts
MODEL_SCRIPTS = {
    "Anemia": os.path.join(BASE_DIR, "DP", "anemia_app.py"),
    #"Heart Disease": "Heart_app.py",
    # "Brain Tumor": "Brain_app.py",
    # "Pneumonia": "Lung_app.py"
}

def start_anemia_app():
    """Starts the Anemia Flask app if not already running."""
    # try:
    #     # Check if port 5002 (where Anemia Flask app runs) is active
    #     response = requests.get("http://localhost:5002/", timeout=2)
    #     if response.status_code == 200:
    #         print("✅ Anemia app is already running.")
    #         return
    # except requests.exceptions.RequestException:
    #     pass  # The app is not running

    try:
        script_path = MODEL_SCRIPTS["Anemia"]
        python_executable = sys.executable
        subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("🚀 Launched Anemia app at http://127.0.0.1:5003/")
    except Exception as e:
        print(f"❌ Failed to launch Anemia app: {e}")

@app.route('/r_model', methods=['POST'])
def handle_ai():
    data = request.get_json()
    disease = data.get('disease')

    if disease == "Anemia":
        try:
            start_anemia_app()  # Ensures Anemia app is running
            anemia_link = "http://127.0.0.1:5003/"
            return jsonify({
                "ai_response": "Click here to start: <a href='http://127.0.0.1:5003/' target='_blank' style='color:blue; text-decoration: underline;'>Anemia Detection</a>"
            })
        except Exception as e:
            return jsonify({
                "ai_response": f"Failed to run Anemia model: {str(e)}"
            }), 500
    elif disease == "Skin Disease":
        return jsonify({
                "ai_response": "Click here to start: <a href='http://skin.test.woza.work/' target='_blank' style='color:blue; text-decoration: underline;'>Skin Disease Detection</a>"
            })
    else:
        return jsonify({
            "ai_response": "The system can currently recognize symptoms related to:  Anemia and Skin Disease."
        })

if __name__ == "__main__":
    # Start main Flask app
    app.run(host='127.0.0.1', port=5000, debug=False)
