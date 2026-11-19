import os
import sqlite3
import numpy as np
from flask import Flask, request, redirect, session, render_template_string, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "degistir-bu-cok-onemli-gizli-bir-anahtar"  # CANLIYA ALIRKEN MUTLAKA DEGISTIR

DB_FILE = "app.db"
MODEL_DIR = "model_data"
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# VERITABANI KURULUMU
# ----------------------------------------------------------------------

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
            text TEXT NOT NULL,
            used_in_training INTEGER DEFAULT 0
        )
    """)
    conn.commit()

    # Hic admin yoksa, varsayilan bir admin olustur (ilk calistirmada)
    existing_admin = conn.execute("SELECT * FROM users WHERE role='admin'").fetchone()
    if not existing_admin:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )
        conn.commit()
        print("Varsayilan admin olusturuldu -> kullanici: admin  sifre: admin123")
        print("GUVENLIK ICIN BU SIFREYI HEMEN DEGISTIR!")
    conn.close()


init_db()

# ----------------------------------------------------------------------
# MINI RNN MODEL (sende Pydroid'de calisan ayni mantik)
# ----------------------------------------------------------------------

BASE_TEXT = """merhaba dunya bu benim ilk yapay zeka modelim ben bir yapay sinir agi ile calisiyorum her gun biraz daha ogreniyorum veri cok onemlidir cunku model veriden ogrenir ne kadar cok veri o kadar iyi sonuc kodlama ogrenmek zaman alir ama sabirla her sey ogrenilir yapay zeka gelecegin teknolojisidir bilgisayarlar insanlar gibi dusunmeye calisir ancak hala cok sinirlidir makine ogrenmesi istatistik ve matematik temeline dayanir python en cok kullanilan dildir cunku kolay ve guclu kutuphaneler sunar ben bu modeli sifirdan yaziyorum torch kullanmadan sadece numpy ile boylece arka planda ne oldugunu daha iyi anliyorum
yapay zeka modelleri veri ile egitilir daha fazla veri daha iyi performans demektir sinir aglari katmanlardan olusur her katman onceki katmandan gelen bilgiyi isler aktivasyon fonksiyonlari onemlidir tanh ve relu sik kullanilir gradient descent ile agirliklar guncellenir loss dusmeye basladiginda model ogreniyor demektir
dil modelleri metin icinde kaliplari ogrenir bir kelimeden sonra hangi kelimenin gelecegini tahmin etmeye calisir ne kadar cok ornek gorurse o kadar iyi tahmin eder bu yuzden buyuk veri setleri onemlidir"""

HIDDEN_SIZE = 64
SEQ_LEN = 15
LR = 0.005

model_state = {}  # Wxh, Whh, Why, bh, by, char_to_idx, idx_to_char, vocab_size, m/v degerleri
training_in_progress = {"flag": False}


def get_all_training_text():
    conn = get_db()
    rows = conn.execute("SELECT text FROM sentences").fetchall()
    conn.close()
    extra = "\n".join(r["text"] for r in rows)
    return (BASE_TEXT + "\n" + extra).lower()


def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


def init_weights(vocab_size):
    np.random.seed(42)
    return {
        "Wxh": np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01,
        "Whh": np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01,
        "Why": np.random.randn(vocab_size, HIDDEN_SIZE) * 0.01,
        "bh": np.zeros((HIDDEN_SIZE, 1)),
        "by": np.zeros((vocab_size, 1)),
    }


def load_or_init_model():
    text = get_all_training_text()
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)

    weight_files = ["Wxh", "Whh", "Why", "bh", "by"]
    paths = {w: os.path.join(MODEL_DIR, w + ".npy") for w in weight_files}
    all_exist = all(os.path.exists(p) for p in paths.values())

    if all_exist:
        loaded = {w: np.load(paths[w]) for w in weight_files}
        if loaded["Wxh"].shape[1] != vocab_size:
            # vocab degisti (yeni karakterler geldi), yeniden baslat
            loaded = init_weights(vocab_size)
    else:
        loaded = init_weights(vocab_size)

    model_state["weights"] = loaded
    model_state["char_to_idx"] = char_to_idx
    model_state["idx_to_char"] = idx_to_char
    model_state["vocab_size"] = vocab_size
    model_state["text"] = text


load_or_init_model()


def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def generate_text(start_text, length=80, temperature=0.7):
    w = model_state["weights"]
    c2i = model_state["char_to_idx"]
    i2c = model_state["idx_to_char"]
    vocab_size = model_state["vocab_size"]

    clean = "".join([c for c in start_text.lower() if c in c2i])
    if len(clean) == 0:
        return "(bu karakterleri taniyamadim)"

    h = np.zeros((HIDDEN_SIZE, 1))
    result = clean
    idx = c2i[clean[-1]]

    for ch in clean[:-1]:
        x = one_hot(c2i[ch], vocab_size)
        h = np.tanh(w["Wxh"] @ x + w["Whh"] @ h + w["bh"])

    for _ in range(length):
        x = one_hot(idx, vocab_size)
        h = np.tanh(w["Wxh"] @ x + w["Whh"] @ h + w["bh"])
        y = (w["Why"] @ h + w["by"]) / temperature
        p = softmax(y)
        idx = np.random.choice(vocab_size, p=p.ravel())
        result += i2c[idx]
    return result


def train_model(epochs=200):
    """Admin tarafindan tetiklenir. Tum ogretilen cumleleri dahil ederek yeniden egitir."""
    training_in_progress["flag"] = True
    try:
        load_or_init_model()  # guncel veriyle vocab/agirliklari tazele
        w = model_state["weights"]
        c2i = model_state["char_to_idx"]
        vocab_size = model_state["vocab_size"]
        text = model_state["text"]
        data = np.array([c2i[ch] for ch in text])

        Wxh, Whh, Why, bh, by = w["Wxh"], w["Whh"], w["Why"], w["bh"], w["by"]
        m_Wxh = np.zeros_like(Wxh); v_Wxh = np.zeros_like(Wxh)
        m_Whh = np.zeros_like(Whh); v_Whh = np.zeros_like(Whh)
        m_Why = np.zeros_like(Why); v_Why = np.zeros_like(Why)
        m_bh = np.zeros_like(bh); v_bh = np.zeros_like(bh)
        m_by = np.zeros_like(by); v_by = np.zeros_like(by)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        def adam(param, dparam, m, v, t):
            m = beta1 * m + (1 - beta1) * dparam
            v = beta2 * v + (1 - beta2) * (dparam ** 2)
            mc = m / (1 - beta1 ** t)
            vc = v / (1 - beta2 ** t)
            param -= LR * mc / (np.sqrt(vc) + eps)
            return param, m, v

        for epoch in range(epochs):
            h = np.zeros((HIDDEN_SIZE, 1))
            for i in range(0, len(data) - SEQ_LEN - 1, SEQ_LEN):
                step += 1
                inputs = data[i:i+SEQ_LEN]
                targets = data[i+1:i+SEQ_LEN+1]

                xs, hs, ps = {}, {-1: h}, {}
                for t in range(len(inputs)):
                    xs[t] = one_hot(inputs[t], vocab_size)
                    hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t-1] + bh)
                    ys = Why @ hs[t] + by
                    ps[t] = softmax(ys)

                dWxh = np.zeros_like(Wxh); dWhh = np.zeros_like(Whh); dWhy = np.zeros_like(Why)
                dbh = np.zeros_like(bh); dby = np.zeros_like(by)
                dh_next = np.zeros_like(hs[0])

                for t in reversed(range(len(xs))):
                    dy = ps[t].copy()
                    dy[targets[t]] -= 1
                    dWhy += dy @ hs[t].T
                    dby += dy
                    dh = Why.T @ dy + dh_next
                    dh_raw = (1 - hs[t] ** 2) * dh
                    dbh += dh_raw
                    dWxh += dh_raw @ xs[t].T
                    dWhh += dh_raw @ hs[t-1].T
                    dh_next = Whh.T @ dh_raw

                for d in [dWxh, dWhh, dWhy, dbh, dby]:
                    np.clip(d, -5, 5, out=d)

                Wxh, m_Wxh, v_Wxh = adam(Wxh, dWxh, m_Wxh, v_Wxh, step)
                Whh, m_Whh, v_Whh = adam(Whh, dWhh, m_Whh, v_Whh, step)
                Why, m_Why, v_Why = adam(Why, dWhy, m_Why, v_Why, step)
                bh, m_bh, v_bh = adam(bh, dbh, m_bh, v_bh, step)
                by, m_by, v_by = adam(by, dby, m_by, v_by, step)

                h = hs[len(inputs) - 1]

        np.save(os.path.join(MODEL_DIR, "Wxh.npy"), Wxh)
        np.save(os.path.join(MODEL_DIR, "Whh.npy"), Whh)
        np.save(os.path.join(MODEL_DIR, "Why.npy"), Why)
        np.save(os.path.join(MODEL_DIR, "bh.npy"), bh)
        np.save(os.path.join(MODEL_DIR, "by.npy"), by)

        model_state["weights"] = {"Wxh": Wxh, "Whh": Whh, "Why": Why, "bh": bh, "by": by}

        conn = get_db()
        conn.execute("UPDATE sentences SET used_in_training = 1")
        conn.commit()
        conn.close()
    finally:
        training_in_progress["flag"] = False


# ----------------------------------------------------------------------
# HTML SABLONLARI (basit, tek dosyada)
# ----------------------------------------------------------------------

LAYOUT = """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini AI</title>
<style>
body { font-family: sans-serif; max-width: 500px; margin: 20px auto; padding: 0 15px; background:#111; color:#eee; }
input, textarea, button { width: 100%; padding: 10px; margin: 6px 0; box-sizing: border-box; border-radius: 6px; border: 1px solid #444; background:#222; color:#eee; }
button { background: #2a6; color: white; border: none; cursor: pointer; }
a { color: #6cf; }
.msg-user { background:#234; padding:8px; border-radius:8px; margin:6px 0; }
.msg-model { background:#333; padding:8px; border-radius:8px; margin:6px 0; }
.topbar { display:flex; justify-content: space-between; align-items:center; }
.flash { background:#642; padding:8px; border-radius:6px; margin:6px 0;}
</style>
</head>
<body>
<div class="topbar">
<h2>Mini AI</h2>
<div>
{% if session.get('username') %}
  {{ session['username'] }} ({{ session['role'] }}) | <a href="{{ url_for('logout') }}">Cikis</a>
{% else %}
  <a href="{{ url_for('login') }}">Giris</a>
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


# ----------------------------------------------------------------------
# ROTALAR: KAYIT / GIRIS / CIKIS
# ----------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or not password:
            flash("Kullanici adi ve sifre bos olamaz.")
            return redirect(url_for("register"))
        conn = get_db()
        existing = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            flash("Bu kullanici adi zaten alinmis.")
            conn.close()
            return redirect(url_for("register"))
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
            (username, generate_password_hash(password))
        )
        conn.commit()
        conn.close()
        flash("Kayit basarili, simdi giris yapabilirsin.")
        return redirect(url_for("login"))

    return render("""
    <h3>Kayit Ol</h3>
    <form method="post">
      <input name="username" placeholder="Kullanici adi" required>
      <input name="password" type="password" placeholder="Sifre" required>
      <button type="submit">Kayit Ol</button>
    </form>
    <p>Zaten hesabin var mi? <a href="{{ url_for('login') }}">Giris yap</a></p>
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
        flash("Kullanici adi veya sifre hatali.")
        return redirect(url_for("login"))

    return render("""
    <h3>Giris Yap</h3>
    <form method="post">
      <input name="username" placeholder="Kullanici adi" required>
      <input name="password" type="password" placeholder="Sifre" required>
      <button type="submit">Giris Yap</button>
    </form>
    <p>Hesabin yok mu? <a href="{{ url_for('register') }}">Kayit ol</a></p>
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


# ----------------------------------------------------------------------
# ANA SAYFA: SOHBET + OGRETME
# ----------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if not login_required():
        return redirect(url_for("login"))

    model_reply = None
    if request.method == "POST":
        action = request.form.get("action")
        user_text = request.form.get("text", "").strip()

        if action == "ask" and user_text:
            model_reply = generate_text(user_text)

        elif action == "teach" and user_text:
            conn = get_db()
            conn.execute(
                "INSERT INTO sentences (username, text) VALUES (?, ?)",
                (session["username"], user_text.lower())
            )
            conn.commit()
            conn.close()
            flash("Tesekkurler! Cumlen kaydedildi, admin egitimi baslattiginda modele dahil olacak.")

    return render("""
    <h3>Modelle konus</h3>
    <form method="post">
      <textarea name="text" placeholder="Bir kelime veya cumle yaz..." required></textarea>
      <button type="submit" name="action" value="ask">Sor / Devamini Uret</button>
      <button type="submit" name="action" value="teach">Modele Ogret</button>
    </form>
    {% if model_reply %}
    <div class="msg-model"><b>Model:</b> {{ model_reply }}</div>
    {% endif %}
    {% if session['role'] == 'admin' %}
    <p><a href="{{ url_for('admin_panel') }}">Admin paneline git</a></p>
    {% endif %}
    """, model_reply=model_reply)


# ----------------------------------------------------------------------
# ADMIN PANELI
# ----------------------------------------------------------------------

@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if not login_required(role="admin"):
        flash("Bu sayfaya sadece admin erisebilir.")
        return redirect(url_for("index"))

    if request.method == "POST" and request.form.get("action") == "retrain":
        if training_in_progress["flag"]:
            flash("Zaten bir egitim devam ediyor, bekle.")
        else:
            train_model(epochs=200)
            flash("Egitim tamamlandi, model guncellendi.")

    conn = get_db()
    sentences = conn.execute("SELECT * FROM sentences ORDER BY id DESC").fetchall()
    users = conn.execute("SELECT username, role FROM users").fetchall()
    conn.close()

    rows_html = "".join(
        f"<div class='msg-user'>[{'egitildi' if s['used_in_training'] else 'bekliyor'}] "
        f"<b>{s['username']}</b>: {s['text']}</div>"
        for s in sentences
    )
    users_html = "".join(f"<div>{u['username']} ({u['role']})</div>" for u in users)

    return render("""
    <h3>Admin Paneli</h3>
    <form method="post">
      <button type="submit" name="action" value="retrain">Modeli Simdi Yeniden Egit</button>
    </form>
    <h4>Kayitli Kullanicilar</h4>
    {{ users_html|safe }}
    <h4>Ogretilen Cumleler</h4>
    {{ rows_html|safe }}
    <p><a href="{{ url_for('index') }}">Sohbete don</a></p>
    """, rows_html=rows_html, users_html=users_html)


if __name__ == "__main__":
    import os
app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
