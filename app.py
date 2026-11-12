"""Rain Predictor - aplikasi Streamlit prediksi hujan besok (dataset weatherAUS)."""
import base64
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Hujan", page_icon="🌦️", layout="centered")

# ---------------------------------------------------------------- konfigurasi
FEATURE_META = {
    # key: (label, satuan, min, max, step, tipe, keterangan)
    "Humidity3pm":   ("Kelembapan jam 15.00", "%", 0, 100, 1, int, "Fitur paling berpengaruh"),
    "Pressure3pm":   ("Tekanan udara jam 15.00", "hPa", 980.0, 1040.0, 0.1, float, "Tekanan rendah → peluang hujan naik"),
    "Sunshine":      ("Lama sinar matahari hari ini", "jam", 0.0, 14.5, 0.1, float, "Makin sedikit matahari, makin mendung"),
    "WindGustSpeed": ("Kecepatan hembusan angin terkuat", "km/jam", 0, 135, 1, int, "Angin kencang sering mendahului hujan"),
    "Cloud3pm":      ("Tutupan awan jam 15.00", "oktas (0-8)", 0, 8, 1, int, "0 = cerah, 8 = tertutup penuh"),
    "Rainfall":      ("Curah hujan hari ini", "mm", 0.0, 150.0, 0.2, float, "Hujan hari ini memengaruhi hari esok"),
}


@st.cache_resource(show_spinner="Memuat model prediksi...")
def load_model():
    return joblib.load("model.joblib"))


bundle = load_model()
FEATURES = bundle["features"]
LOCATIONS = bundle["locations"]

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("dark", False)
ss.setdefault("history", [])
ss.setdefault("last", None)
ss.setdefault("saved_ids", set())


def apply_location_defaults():
    d = bundle["location_defaults"][ss["location"]]
    for k in FEATURES:
        typ = FEATURE_META[k][5]
        lo, hi = FEATURE_META[k][2], FEATURE_META[k][3]
        v = typ(min(max(d[k], lo), hi))
        ss[k], ss[f"{k}_s"], ss[f"{k}_n"] = v, v, v


if "location" not in ss:
    ss["location"] = "Sydney" if "Sydney" in LOCATIONS else LOCATIONS[0]
    apply_location_defaults()


def sync(key, src):
    v = ss[f"{key}_{src}"]
    ss[key] = ss[f"{key}_s"] = ss[f"{key}_n"] = v


# ---------------------------------------------------------------- tema
LIGHT = dict(bg="#F5F8FC", card="#FFFFFF", text="#12263A", muted="#5B6B7C",
             border="#DCE5EF", accent="#0B6E8A", input="#FFFFFF")
DARK = dict(bg="#0E1621", card="#16212F", text="#E8EEF5", muted="#9FB0C3",
            border="#26364A", accent="#4FC3E0", input="#1C2A3B")
C = DARK if ss["dark"] else LIGHT

st.markdown(f"""
<style>
.stApp {{ background:{C['bg']}; color:{C['text']}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color:{C['text']}; }}
[data-testid="stCaptionContainer"], .muted {{ color:{C['muted']} !important; }}
.block-container {{ max-width:860px; padding-top:2.2rem; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"],
div[data-baseweb="input"] input, .stNumberInput input {{
    background:{C['input']} !important; color:{C['text']} !important; border-color:{C['border']} !important; }}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] ul {{
    background:{C['card']} !important; color:{C['text']} !important; }}
.stNumberInput button {{ background:{C['input']} !important; color:{C['text']} !important; }}
div[data-testid="stSlider"] [role="slider"] {{ background:{C['accent']} !important; }}
.stButton > button, .stDownloadButton > button {{
    border-radius:12px; border:1px solid {C['border']};
    background:{C['card']}; color:{C['text']}; font-weight:600; }}
.stButton > button[kind="primary"] {{
    background:{C['accent']}; color:#fff; border:none; padding:.7rem 1rem; font-size:1.05rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{C['accent']}; color:{C['accent']}; }}
.stButton > button[kind="primary"]:hover {{ color:#fff; filter:brightness(1.08); }}
.hero h1 {{ font-size:2.1rem; margin:0 0 .15rem 0; letter-spacing:-.02em; }}
.hero p {{ margin:0; color:{C['muted']}; }}
.card {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:18px;
         padding:1.2rem 1.4rem; margin:.4rem 0 1rem 0; }}
.result {{ text-align:center; }}
.result .title {{ font-size:1.6rem; font-weight:700; margin:.2rem 0; }}
.result .prob {{ font-size:3rem; font-weight:800; color:{C['accent']}; line-height:1.1; }}
.bar {{ height:10px; border-radius:99px; background:{C['border']}; overflow:hidden; margin:.7rem auto; max-width:420px; }}
.bar > div {{ height:100%; border-radius:99px; background:linear-gradient(90deg,#4FC3E0,#0B6E8A,#1E3A8A); }}
.stDataFrame {{ border-radius:12px; overflow:hidden; }}
details, [data-testid="stExpander"] {{ background:{C['card']}; border-color:{C['border']} !important; border-radius:14px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- ilustrasi cuaca (SVG animasi)
def weather_svg(level, dark):
    cloud_c = "#7C8DA3" if dark else "#B8C6D6"
    cloud_d = "#56677D" if dark else "#7F92A8"
    sun = '''<g><circle cx="150" cy="95" r="38" fill="#FFC53D"/>
      <g stroke="#FFC53D" stroke-width="6" stroke-linecap="round">
      <animateTransform attributeName="transform" type="rotate" from="0 150 95" to="360 150 95" dur="24s" repeatCount="indefinite"/>
      <line x1="150" y1="35" x2="150" y2="22"/><line x1="150" y1="155" x2="150" y2="168"/>
      <line x1="90" y1="95" x2="77" y2="95"/><line x1="210" y1="95" x2="223" y2="95"/>
      <line x1="108" y1="53" x2="99" y2="44"/><line x1="192" y1="137" x2="201" y2="146"/>
      <line x1="108" y1="137" x2="99" y2="146"/><line x1="192" y1="53" x2="201" y2="44"/></g></g>'''

    def cloud(x, y, s, col):
        return (f'<g transform="translate({x} {y}) scale({s})" fill="{col}">'
                '<circle cx="40" cy="50" r="28"/><circle cx="78" cy="38" r="34"/>'
                '<circle cx="118" cy="52" r="26"/><rect x="40" y="50" width="78" height="28" rx="2"/>'
                '<rect x="12" y="50" width="130" height="28" rx="14"/></g>')

    def drops(n, col, speed):
        out = ""
        for i in range(n):
            x = 95 + i * (150 / max(n - 1, 1))
            out += (f'<line x1="{x}" y1="140" x2="{x-6}" y2="158" stroke="{col}" stroke-width="4" '
                    f'stroke-linecap="round" opacity="0"><animate attributeName="opacity" values="0;1;0" '
                    f'dur="{speed}s" begin="{i*0.13:.2f}s" repeatCount="indefinite"/>'
                    f'<animateTransform attributeName="transform" type="translate" values="0 0;-6 42" '
                    f'dur="{speed}s" begin="{i*0.13:.2f}s" repeatCount="indefinite"/></line>')
        return out

    bolt = ('<polygon points="150,128 128,162 146,162 136,192 172,150 152,150 164,128" fill="#FFD43B">'
            '<animate attributeName="opacity" values="0;0;1;0;1;0;0" dur="3s" repeatCount="indefinite"/></polygon>')

    if level == 0:      # cerah
        body = sun
    elif level == 1:    # berawan
        body = sun + cloud(70, 85, 1.0, cloud_c)
    elif level == 2:    # mungkin hujan
        body = cloud(60, 60, 1.2, cloud_c) + drops(5, "#4FC3E0", 1.4)
    else:               # hujan
        body = cloud(60, 50, 1.25, cloud_d) + bolt + drops(9, "#3AA6D0", 0.9)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200" width="260" height="173">'
            f'{body}</svg>')


def svg_img(svg):
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:260px;width:100%;"/>'


VERDICTS = [
    (0.0, 0, "Cerah", "Hampir pasti tidak hujan besok. Aman untuk beraktivitas di luar."),
    (0.0, 1, "Cerah Berawan", "Peluang hujan rendah. Langit mungkin berawan, tapi umumnya kering."),
    (0.0, 2, "Berpotensi Hujan", "Peluang hujan cukup besar. Sebaiknya bawa payung."),
    (0.0, 3, "Hujan", "Hujan sangat mungkin turun besok. Siapkan jas hujan atau ubah rencana."),
]


def classify(prob, thr):
    # level ditentukan relatif terhadap threshold keputusan model
    if prob < thr * 0.5:
        lvl = 0
    elif prob < thr:
        lvl = 1
    elif prob < max(thr * 1.9, 0.6):
        lvl = 2
    else:
        lvl = 3
    return lvl, VERDICTS[lvl][2], VERDICTS[lvl][3]


# ---------------------------------------------------------------- header
head_l, head_r = st.columns([5, 1.4], vertical_alignment="center")
with head_l:
    st.markdown('<div class="hero"><h1>🌦️ Rain Predictor</h1>'
                '<p>Prediksi peluang hujan besok berdasarkan kondisi cuaca hari ini.</p></div>',
                unsafe_allow_html=True)
with head_r:
    st.toggle("🌙 Mode gelap", key="dark")

st.write("")

# ---------------------------------------------------------------- form input
st.markdown("#### Data cuaca")
st.selectbox("📍 Lokasi", LOCATIONS, key="location", on_change=apply_location_defaults,
             help="Nilai awal slider menyesuaikan kondisi umum di lokasi ini.")
st.caption("Geser slider atau ketik angka secara manual. Keduanya saling tersinkron.")

cols = st.columns(2, gap="large")
for i, key in enumerate(FEATURES):
    label, unit, lo, hi, step, typ, hint = FEATURE_META[key]
    with cols[i % 2]:
        st.markdown(f"**{label}**  <span class='muted'>({unit})</span>", unsafe_allow_html=True)
        s_col, n_col = st.columns([3, 1.6], vertical_alignment="center")
        with s_col:
            st.slider(label, typ(lo), typ(hi), step=typ(step), key=f"{key}_s",
                      on_change=sync, args=(key, "s"), label_visibility="collapsed")
        with n_col:
            st.number_input(label, typ(lo), typ(hi), step=typ(step), key=f"{key}_n",
                            on_change=sync, args=(key, "n"), label_visibility="collapsed")
        st.caption(hint)

st.write("")
go = st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True)

# ---------------------------------------------------------------- prediksi
if go:
    row = {"Location": ss["location"], **{k: ss[k] for k in FEATURES}}
    prob = float(bundle["pipeline"].predict_proba(pd.DataFrame([row]))[0, 1])
    thr = bundle["threshold"]
    lvl, title, desc = classify(prob, thr)
    ss["last"] = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lokasi": ss["location"], **{k: ss[k] for k in FEATURES},
        "peluang_hujan_%": round(prob * 100, 1),
        "prediksi": "Hujan" if prob >= thr else "Tidak hujan",
        "kondisi": title, "desc": desc,
        "level": lvl,
    }

res = ss["last"]
if res:
    st.markdown("#### Hasil prediksi")
    st.markdown(
        f"""<div class="card result">
        {svg_img(weather_svg(res['level'], ss['dark']))}
        <div class="title">{res['kondisi']}</div>
        <div class="muted">Prediksi besok di <b>{res['lokasi']}</b></div>
        <div class="prob">{res['peluang_hujan_%']:.0f}%</div>
        <div class="muted">peluang hujan</div>
        <div class="bar"><div style="width:{res['peluang_hujan_%']}%"></div></div>
        <p>{res['desc']}</p></div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Keputusan model", res["prediksi"])
    m2.metric("Peluang hujan", f"{res['peluang_hujan_%']:.1f}%")
    base = bundle["location_rain_rate"].get(res["lokasi"], 0) * 100
    m3.metric(f"Rata-rata hujan {res['lokasi']}", f"{base:.0f}%",
              help="Persentase hari dengan hujan keesokan harinya pada data historis.")

    if res["id"] in ss["saved_ids"]:
        st.success("✅ Hasil ini sudah tersimpan di riwayat.")
    elif st.button("💾 Simpan hasil ini", use_container_width=True):
        ss["history"].append({k: v for k, v in res.items() if k not in ("id", "desc", "level")} | {"id": res["id"]})
        ss["saved_ids"].add(res["id"])
        st.rerun()

# ---------------------------------------------------------------- riwayat
if ss["history"]:
    st.markdown("#### Riwayat tersimpan")
    hist = pd.DataFrame(ss["history"]).drop(columns=["id"])
    hist = hist.rename(columns={k: f"{FEATURE_META[k][0]} ({FEATURE_META[k][1]})" for k in FEATURES})
    st.dataframe(hist, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Unduh CSV", hist.to_csv(index=False).encode("utf-8-sig"),
                       file_name="riwayat_prediksi_hujan.csv", mime="text/csv",
                       use_container_width=True)
    if d2.button("🗑️ Hapus riwayat", use_container_width=True):
        ss["history"], ss["saved_ids"] = [], set()
        st.rerun()

# ---------------------------------------------------------------- info model
st.write("")
with st.expander("ℹ️ Tentang model"):
    mt = bundle["metrics"]
    st.write(
        "Model **XGBoost** dilatih pada data **weatherAUS** (~142 ribu baris) dengan alur yang sama "
        "seperti notebook: buang baris tanpa target, hapus duplikat, split 80/20 stratified. "
        "Aplikasi hanya memakai **6 fitur paling berpengaruh** ditambah lokasi."
    )
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{mt['roc_auc']:.3f}")
    b.metric("Akurasi", f"{mt['accuracy']:.3f}")
    c.metric("Recall", f"{mt['recall']:.3f}")
    d.metric("F1", f"{mt['f1']:.3f}")
    imp = pd.Series(bundle["importance"]).sort_values(ascending=False)
    st.caption("Pentingnya fitur (penurunan ROC-AUC saat fitur diacak)")
    st.bar_chart(imp, horizontal=True, height=240)
    st.caption("Prediksi bersifat statistik dan tidak menggantikan prakiraan resmi BMKG/BoM.")
