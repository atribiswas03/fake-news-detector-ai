from flask import Flask, render_template, render_template_string, request, redirect, url_for, session
from pymongo import MongoClient
from flask_mail import Mail, Message
from datetime import datetime
import random
import string
import pandas as pd
from fake_news_pipeline import predict_news, load_model_and_vectorizer, train_model, load_and_prepare_data
from bson import ObjectId
import os
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = "a8d3c7b4e9d4a1e2b5d6f8a9c3e7b2f4"

load_dotenv()
# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["FAKENEWS"]
collection = db["predictions"]

# Load model and vectorizer or train if not available
model, vectorizer = load_model_and_vectorizer()
if model is None or vectorizer is None:
    print("Training model as no saved model found...")
    X, y = load_and_prepare_data()
    model, vectorizer = train_model(X, y)

# Flask-Mail setup
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)


otp_code = None

@app.route("/real-news")
def real_news():
    return render_template("R_News.html")  # or any valid response


@app.route("/fake-news")
def fake_news():
    return render_template("F_News.html")  # or any valid response



@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    news_text = ""

    if request.method == "POST":
        if "feedback" in request.form:
            text = request.form.get("text")
            prediction = request.form.get("prediction")
            confidence = float(request.form.get("confidence", 0))
            feedback_value = request.form.get("feedback")

            collection.insert_one({
                "text": text,
                "result": prediction,
                "confidence": confidence,
                "user_feedback": feedback_value,
                "timestamp": datetime.now()
            })
            return redirect("/")

        news_text = request.form["message"]
        result, confidence = predict_news(news_text, model, vectorizer)

    return render_template("index.html", result=result, news_text=news_text, confidence=confidence)




@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    global otp_code
    if request.method == "POST":
        email = request.form["email"]

        # Check if email exists in MongoDB collection (case-insensitive match for safety)
        admin_entry = db.admin_emails.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})

        if not admin_entry:
            return render_template_string("""
                <h3 style='color:red; text-align:center;'>❌ Unauthorized Email!</h3>
                <script>setTimeout(() => { window.location.href = "/admin-login"; }, 3000);</script>
            """)

        # Generate OTP
        otp_code = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px;">
            <div style="text-align: center;">
                <img src="https://lh3.googleusercontent.com/d/1axKWatF3vfJIKIWk2aed6sDxkMY-dG4M" alt="Fake News Detector Logo" style="max-width: 120px; margin-bottom: 20px;">
                <h2 style="color: #2c3e50;">Admin OTP Verification</h2>
            </div>
            <p>Hello Admin,</p>
            <p>You have requested to log in to the <b>Fake News Detector Admin Panel</b>. Please use the OTP below to complete your login:</p>
            <p style="font-size: 20px; font-weight: bold; color: #e74c3c; text-align: center; padding: 10px 0;">{otp_code}</p>
            <p>If you did not request this, please ignore this email.</p>
            <br>
            <p style="font-size: 14px; color: #7f8c8d;">Best Regards,<br>Fake News Detector Team</p>
        </div>
        """

        msg = Message("Admin OTP Verification - Fake News Detector", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.html = html_body
        msg.body = f"Your OTP is: {otp_code}"  # fallback plain text
        mail.send(msg)

        session["email"] = email
        return redirect(url_for("verify_otp"))

    return render_template("admin_login.html")






@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if "otp_attempts" not in session:
        session["otp_attempts"] = 0

    if request.method == "POST":
        entered_otp = request.form.get("otp", "")
        if entered_otp == otp_code:
            session["authenticated"] = True
            session.pop("otp_attempts", None)
            return render_template_string("""
                <script>window.open("/admin-dashboard", "_blank"); window.close();</script>
            """)

        session["otp_attempts"] += 1

        if session["otp_attempts"] == 1:
            return render_template("verify_otp.html", error="❌ Invalid OTP. Please try again.")

        session.pop("otp_attempts", None)
        return render_template_string("""
            <script>
                window.parent.location.href = "/";
            </script>
        """)

    return render_template("verify_otp.html")


@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("authenticated"):
        return render_template_string("""
            <h3 style="color:red; text-align:center;">⚠️ Unauthorized Access Denied</h3>
            <script>setTimeout(() => { window.location.href = "/"; }, 3000);</script>
        """)

    data = [{**doc, "_id": str(doc["_id"])} for doc in collection.find()]
    admin_emails = list(db["admin_emails"].find())
    for admin in admin_emails:
        admin["_id"] = str(admin["_id"])
    return render_template("admin_dashboard.html", data=data, admin_emails=admin_emails)


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("home"))


@app.route('/delete/<id>', methods=['POST'])
def delete_entry(id):
    if not session.get("authenticated"):
        return "Unauthorized", 401
    collection.delete_one({"_id": ObjectId(id)})
    return redirect(url_for('admin_dashboard'))


@app.route("/delete-all", methods=["POST"])
def delete_all():
    collection.delete_many({})
    return redirect(url_for("admin_dashboard"))



@app.route("/approve/<id>", methods=["POST"])
def approve(id):
    entry = collection.find_one({"_id": ObjectId(id)})
    if not entry or not entry.get("text"):
        return {"success": False, "error": "Invalid entry"}

    feedback = str(entry.get("user_feedback", "")).strip().lower()
    result = str(entry.get("result", "")).strip().lower()

    label_map = {
        ("yes", "real"): 1,
        ("yes", "fake"): 0,
        ("no", "real"): 0,
        ("no", "fake"): 1
    }
    label = label_map.get((feedback, result))
    if label is None:
        return {"success": False, "error": "Invalid mapping"}

    new_data = pd.DataFrame({"text": [entry["text"]], "label": [label]})
    path = "news_dataset.csv"
    try:
        existing = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        updated = pd.concat([existing, new_data], ignore_index=True)
    except pd.errors.EmptyDataError:
        updated = new_data

    updated.to_csv(path, index=False)

    if len(updated["label"].unique()) >= 2:
        try:
            train_model(updated["text"], updated["label"])
        except Exception as e:
            print("Training error:", e)

    collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": "Approved"}})
    return {"success": True, "status": "Approved"}



@app.route("/reject/<id>", methods=["POST"])
def reject(id):
    if not session.get("authenticated"):
        return {"success": False, "error": "Unauthorized"}, 401

    entry = collection.find_one({"_id": ObjectId(id)})

    if not entry or not entry.get("text"):
        return {"success": False, "error": "Missing entry or text"}

    # 🧠 Determine correct label based on feedback and result
    feedback = str(entry.get("user_feedback", "")).strip().lower()
    result = str(entry.get("result", "")).strip().lower()

    label_map = {
        ("yes", "real"): 0,
        ("yes", "fake"): 1,
        ("no",  "real"): 1,
        ("no",  "fake"): 0
    }

    label = label_map.get((feedback, result))
    if label is None:
        return {"success": False, "error": f"Invalid mapping: ({feedback}, {result})"}

    new_data = pd.DataFrame({
        "text": [entry["text"]],
        "label": [label]
    })

    dataset_path = "news_dataset.csv"
    if os.path.exists(dataset_path):
        try:
            full_data = pd.read_csv(dataset_path)
            updated_data = pd.concat([full_data, new_data], ignore_index=True)
        except pd.errors.EmptyDataError:
            updated_data = new_data
    else:
        updated_data = new_data

    updated_data.to_csv(dataset_path, index=False)

    if len(updated_data["label"].unique()) >= 2:
        X = updated_data["text"]
        y = updated_data["label"]
        train_model(X, y)
    else:
        print("⚠️ Skipping training: only one class present.")

    # ✅ Update DB and return JSON
    collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": "Rejected"}})

    return {"success": True, "status": "Rejected"}


@app.route("/add-admin-email", methods=["POST"])
def add_admin_email():
    if not session.get("authenticated"):
        return "Unauthorized", 401
    email = request.form["email"].strip().lower()
    if db["admin_emails"].find_one({"email": email}):
        return "<script>alert('⚠️ Email already exists!'); window.location.href='/admin-dashboard';</script>"
    db["admin_emails"].insert_one({"email": email})
    return redirect(url_for("admin_dashboard"))


@app.route("/delete-admin-email/<id>", methods=["POST"])
def delete_admin_email(id):
    if not session.get("authenticated"):
        return "Unauthorized", 401
    db["admin_emails"].delete_one({"_id": ObjectId(id)})
    return redirect(url_for("admin_dashboard"))




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
