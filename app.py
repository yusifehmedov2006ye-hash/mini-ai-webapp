import os
import sqlite3
import random
import re
from collections import defaultdict
from flask import Flask, request, redirect, session, render_template_string, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mini-ai-gizli-anahtar-2026")

DB_FILE = "app.db"

# ======================
# VERİTABANI
# ======================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()

    # Varsayılan admin
    if not conn.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )
        conn.commit()
        print(">>> Admin: admin / admin123  (hemen değiştir!)")
    conn.close()

init_db()

# ======================
# MARKOV + Q&A SİSTEMİ
# ======================
def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", "", text)
    return text

def load_markov():
    """Bütün öğretilen cümlelerden Markov zinciri kurar"""
    conn = get_db()
    rows = conn.execute("SELECT text FROM sentences").fetchall()
    conn.close()

    markov = defaultdict(list)
    for row in rows:
        words = clean(row["text"]).split()
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            key = words[i]
            markov[key].append(words[i + 1])
        # 2-gram da ekle
        for i in range(len(words) - 2):
            key = words[i] + " " + words[i + 1]
            markov[key].append(words[i + 2])
    return markov

def generate_markov(seed="", max_words=18):
    markov = load_markov()
    if not markov:
        return None

    words = clean(seed).split()
    if not words:
        current = random.choice(list(markov.keys()))
    else:
        # En uzun eşleşen key'i bul
        current = None
        for length in [2, 1]:
            if len(words) >= length:
                key = " ".join(words[-length:])
                if key in markov:
                    current = key
                    break
        if current is None:
            current = random.choice(list(markov.keys()))

    result = current.split()
    for _ in range(max_words):
        key = " ".join(result[-2:]) if len(result) >= 2 else result[-1]
        if key not in markov:
            key = result[-1]
        if key not in markov:
            break
        next_word = random.choice(markov[key])
        result.append(next_word)
        if len(result) > 25:
            break

    return " ".join(result)

def find_best_qa(question):
    """Öğretilmiş soru-cevap çiftlerinden en iyi eşleşeni bul"""
    q = clean(question)
    if not q:
        return None

    conn = get_db()
    rows = conn.execute("SELECT question, answer FROM qa").fetchall()
    conn.close()

    best_score = 0
    best_answers = []

    q_words = set(q.split())

    for row in rows:
        known = clean(row["question"])
        known_words = set(known.split())
        if not known_words:
            continue

        common = len(q_words & known_words)
        score = common / max(len(q_words), 1)

        # Tam eşleşme öncelikli
        if known == q:
            return row["answer"]

        if score > best_score and score >= 0.35:
            best_score = score
            best_answers = [row["answer"]]
        elif score == best_score and score >= 0.35:
            best_answers.append(row["answer"])

    if best_answers:
        return random.choice(best_answers)
    return None

FALLBACKS = [
    "Anlamadım, biraz daha açık yazar mısın?",
    "Bu konuda henüz bir şey öğrenmedim. Bana öğretebilirsin.",
    "İlginç bir soru. Şimdilik bilmiyorum.",
    "Daha basit bir şekilde sorabilir misin?",
    "Hâlâ öğreniyorum. Bana birkaç örnek cümle öğret.",
]

def get_answer(user_text):
    # 1. Q&A yaddaşı
    qa = find_best_qa(user_text)
    if qa:
        return qa

    # 2. Markov üretimi
    markov = generate_markov(user_text)
    if markov and len(markov.split()) >= 3:
        return markov

    # 3. Fallback
    return random.choice(FALLBACKS)

# ======================
# HTML
# ======================
LAYOUT = """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini AI</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 520px; margin: 20px auto; padding: 0 16px; background:#0f0f0f; color:#eee; }
input, textarea, button { width: 100%; padding: 12px; margin: 8px 0; box-sizing: border-box; border-radius: 10px; border: 1px solid #333; background:#1a1a1a; color:#eee; font-size: 16px; }
button { background: #22c55e; color: white; border: none; cursor: pointer; font-weight: 600; }
button.secondary { background: #3b82f6; }
a { color: #60a5fa; }
.msg-model { background:#1e293b; padding:14px; border-radius:12px; margin:14px 0; line-height:1.5; white-space: pre-wrap; }
.topbar { display:flex; justify-content: space-between; align-items:center; margin-bottom: 20px; }
.flash { background:#7c2d12; padding:10px; border-radius:8px; margin:10px 0; }
</style>
</head>
<body>
<div class="topbar">
  <h2>Mini AI</h2>
  <div>
  {% if session.get('username') %}
    {{ session['username'] }} ({{ session['role'] }}) | <a href="{{ url_for('logout') }}">Çıkış</a>
  {% else %}
    <a href="{{ url_for('login') }}">Giriş</a>
  {% endif %}
  </div>
</div>
{% with messages = get_flashed_messages() %}
  {% for m in messages %}<div class="flash">{{ m }}</div>{% endfor %}
{% endwith %}
{{ body|safe }}
</body>
</html>
"""

def render(body_html, **kwargs):
    return render_template_string(LAYOUT, body=render_template_string(body_html, **kwargs))

# ======================
# AUTH
# ======================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or not password:
            flash("Kullanıcı adı ve şifre boş olamaz.")
            return redirect(url_for("register"))
        conn = get_db()
        if conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            flash("Bu kullanıcı adı zaten alınmış.")
            conn.close()
            return redirect(url_for("register"))
        conn.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
                     (username, generate_password_hash(password)))
        conn.commit()
        conn.close()
        flash("Kayıt başarılı.")
        return redirect(url_for("login"))
    return render("""
    <h3>Kayıt Ol</h3>
    <form method="post">
      <input name="username" placeholder="Kullanıcı adı" required>
      <input name="password" type="password" placeholder="Şifre" required>
      <button type="submit">Kayıt Ol</button>
    </form>
    <p><a href="{{ url_for('login') }}">Giriş yap</a></p>
    """)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("index"))
        flash("Hatalı kullanıcı adı veya şifre.")
        return redirect(url_for("login"))
    return render("""
    <h3>Giriş Yap</h3>
    <form method="post">
      <input name="username" placeholder="Kullanıcı adı" required>
      <input name="password" type="password" placeholder="Şifre" required>
      <button type="submit">Giriş Yap</button>
    </form>
    <p><a href="{{ url_for('register') }}">Kayıt ol</a></p>
    """)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def login_required(role=None):
    if "username" not in session:
        return False
    if role and session.get("role") != role:
        return False
    return True

# ======================
# ANA SAYFA
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    if not login_required():
        return redirect(url_for("login"))

    model_reply = None
    if request.method == "POST":
        action = request.form.get("action")
        user_text = request.form.get("text", "").strip()

        if action == "ask" and user_text:
            model_reply = get_answer(user_text)

        elif action == "teach" and user_text:
            # Basit öğretme: tüm cümleyi kaydet + Markov'a ekle
            conn = get_db()
            conn.execute("INSERT INTO sentences (username, text) VALUES (?, ?)",
                         (session["username"], user_text.lower()))
            conn.commit()
            conn.close()
            flash("Öğrendim! Bu cümle artık modelde.")

    return render("""
    <h3>Modelle konuş</h3>
    <form method="post">
      <textarea name="text" rows="3" placeholder="Bir kelime veya cümle yaz..." required></textarea>
      <button type="submit" name="action" value="ask">Sor / Devamını Üret</button>
      <button type="submit" name="action" value="teach" class="secondary">Modele Öğret</button>
    </form>
    {% if model_reply %}
    <div class="msg-model"><b>Model:</b><br>{{ model_reply }}</div>
    {% endif %}
    {% if session['role'] == 'admin' %}
    <p style="margin-top:28px"><a href="{{ url_for('admin_panel') }}">Admin paneline git →</a></p>
    {% endif %}
    """, model_reply=model_reply)

# ======================
# ADMIN + Q&A ÖĞRETME
# ======================
@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if not login_required(role="admin"):
        flash("Sadece admin.")
        return redirect(url_for("index"))

    if request.method == "POST":
        action = request.form.get("action")

        if action == "add_qa":
            q = request.form.get("question", "").strip()
            a = request.form.get("answer", "").strip()
            if q and a:
                conn = get_db()
                conn.execute("INSERT INTO qa (question, answer) VALUES (?, ?)", (q, a))
                conn.commit()
                conn.close()
                flash("Soru-cevap eklendi.")

        elif action == "clear_sentences":
            conn = get_db()
            conn.execute("DELETE FROM sentences")
            conn.commit()
            conn.close()
            flash("Tüm öğretilen cümleler silindi.")

    conn = get_db()
    sentences = conn.execute("SELECT * FROM sentences ORDER BY id DESC LIMIT 40").fetchall()
    qas = conn.execute("SELECT * FROM qa ORDER BY id DESC LIMIT 30").fetchall()
    users = conn.execute("SELECT username, role FROM users").fetchall()
    conn.close()

    rows = "".join(f"<div style='background:#1e293b;padding:8px;border-radius:8px;margin:5px 0'><b>{s['username']}</b>: {s['text']}</div>" for s in sentences)
    qa_rows = "".join(f"<div style='background:#1e293b;padding:8px;border-radius:8px;margin:5px 0'><b>S:</b> {q['question']}<br><b>C:</b> {q['answer']}</div>" for q in qas)
    users_html = "".join(f"<div>{u['username']} ({u['role']})</div>" for u in users)

    return render("""
    <h3>Admin Paneli</h3>

    <h4>Soru-Cevap Ekle (en iyi yöntem)</h4>
    <form method="post">
      <input name="question" placeholder="Soru (örnek: merhaba)" required>
      <input name="answer" placeholder="Cevap (örnek: Merhaba! Nasılsın?)" required>
      <button type="submit" name="action" value="add_qa">Ekle</button>
    </form>

    <h4 style="margin-top:24px">Kayıtlı Soru-Cevaplar</h4>
    {{ qa_rows|safe }}

    <h4 style="margin-top:24px">Öğretilen Cümleler</h4>
    {{ rows|safe }}

    <form method="post" style="margin-top:16px">
      <button type="submit" name="action" value="clear_sentences" style="background:#b91c1c">Tüm cümleleri sil</button>
    </form>

    <h4 style="margin-top:24px">Kullanıcılar</h4>
    {{ users_html|safe }}

    <p style="margin-top:28px"><a href="{{ url_for('index') }}">← Sohbete dön</a></p>
    """, rows=rows, qa_rows=qa_rows, users_html=users_html)

# App Inventor API
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"answer": "Boş mesaj"})
    return jsonify({"answer": get_answer(text)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
