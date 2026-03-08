import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Amazon Sales Analysis",
    page_icon="📊🔍🛒",
    layout="wide"
)

st.markdown("""
<style>
/* Importăm fonturi noi: Plus Jakarta Sans pt titluri, Inter pt text, JetBrains Mono pt cod */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0B1120; /* Albastru foarte închis (Midnight) */
    --surface:  #1E293B; /* Slate închis pt carduri */
    --border:   #334155;
    --accent:   #38BDF8; /* Sky Blue */
    --accent2:  #A78BFA; /* Violet deschis */
    --text:     #F8FAFC;
    --muted:    #94A3B8;
    --success:  #34D399; /* Verde mentă */
    --mono:     'JetBrains Mono', monospace;
    --display:  'Plus Jakarta Sans', sans-serif;
    --body:     'Inter', sans-serif;
    --radius:   12px; /* Margini mult mai rotunjite */
}

/* Setări generale */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}

h1, h2, h3, h4 {
    font-family: var(--display) !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

/* Bara laterală (Sidebar) */
[data-testid="stSidebar"] {
    background-color: #0F172A !important; /* Un pic diferit de fundalul principal */
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Streamlit defaults */
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; line-height: 1.6; }
.stAlert p { color: #000 !important; }

/* Secțiunea Hero (Antet principal) */
.hero {
    background: linear-gradient(145deg, var(--surface) 0%, #0F172A 100%);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    padding: 48px 40px;
    border-radius: var(--radius);
    margin-bottom: 40px;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
}
.hero-label {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    font-weight: 500;
}
.hero-title {
    font-family: var(--display);
    font-size: 56px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    margin: 0 0 16px 0;
}
.hero-sub {
    font-size: 16px;
    color: var(--muted);
    max-width: 650px;
    line-height: 1.6;
}

/* Titluri de secțiune */
.sec-header {
    font-family: var(--display);
    font-size: 20px;
    font-weight: 700;
    color: var(--accent2);
    margin: 40px 0 20px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Carduri pentru pagini (Navigare Home -> Dashboard) */
.page-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-top: 10px;
}
.page-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    transition: all 0.3s ease;
    cursor: pointer;
}
.page-card:hover { 
    border-color: var(--accent); 
    transform: translateY(-3px);
    box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.1);
}
.page-num {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    padding: 4px 8px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
    margin-bottom: 12px;
}
.page-name {
    font-family: var(--display);
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
}
.page-desc {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.5;
}

/* Cutii cu sfaturi (Tip box) */
.tip {
    background: rgba(167, 139, 250, 0.05); /* Fundal mov foarte transparent */
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: var(--radius);
    padding: 20px;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
}
.tip strong { color: var(--accent2); }
.tip code {
    background: #0F172A;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
}

/* Zona de upload (Upload prompt) */
.upload-prompt {
    border: 2px dashed var(--border);
    background: rgba(30, 41, 59, 0.5);
    border-radius: var(--radius);
    padding: 40px 20px;
    text-align: center;
    color: var(--muted);
    font-size: 15px;
    margin: 20px 0;
    transition: border-color 0.3s ease;
}
.upload-prompt:hover { border-color: var(--accent); }
.upload-prompt strong { color: var(--accent); font-weight: 600; }

/* Box-uri pentru statistici (Stat pills) - perfecte pentru Dashboard.py */
.stat-row {
    display: flex;
    gap: 16px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.stat-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; /* Mai rotunjit pentru statistici */
    padding: 16px 24px;
    flex: 1;
    min-width: 140px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.stat-pill .sv {
    font-family: var(--display);
    font-size: 24px; /* am micșorat de la 32px */
    font-weight: 800;
    color: var(--text);
    line-height: 1.2;
    white-space: nowrap; /* forțează numărul să stea pe un singur rând */
}
.stat-pill .sl {
    font-family: var(--body);
    font-size: 13px;
    color: var(--accent);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.markdown("### 📦📈 Amazon Dashboard")
st.sidebar.markdown("**Analiza Vânzărilor & Performanță**")
st.sidebar.markdown("---")

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">Analiză E-Commerce &nbsp;/&nbsp; Python &amp; Streamlit</div>
    <div class="hero-title">Amazon Sales<br>Dashboard</div>
    <div class="hero-sub">
        Un instrument interactiv pentru analiza performanței pe Amazon — de la procesarea 
        seturilor de date brute până la vizualizarea tendințelor, a profitabilității 
        și a logisticii. Construit cu Pandas, Plotly și Streamlit.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-header">Structura proiectului</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    <div class="struct-box">
        amazon_dashboard/<br>
        <span class="dim">│</span><br>
        <span class="dim">├──</span> <span class="hl">Home.py</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">← pagina principală (încărcare CSV)</span><br>
        <span class="dim">├──</span> amazon_sales.csv &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">← setul de date brut</span><br>
        <span class="dim">├──</span> requirements.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">← dependințele pt. deployment</span><br>
        <span class="dim">└──</span> pages/<br>
        &nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">└──</span> 1_Dashboard.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="dim">← analiza și vizualizările</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tip">
        <strong>Reguli de Arhitectură & Deployment</strong><br><br>
        Salvăm CSV-ul în <code>st.session_state</code> în <code>Home.py</code> pentru a fi accesat în <code>1_Dashboard.py</code>, evitând reîncărcarea.<br><br>
        Pentru deployment pe <b>Streamlit Cloud / GitHub</b>, fișierul <code>requirements.txt</code> este obligatoriu (trebuie să conțină <i>streamlit, pandas, matplotlib, plotly</i>).<br><br>
        <code>st.set_page_config()</code> trebuie să fie prima comandă în <code>Home.py</code>.<br><br>
        Se rulează local cu: <code>streamlit run Home.py</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="sec-header">Încarcă dataset-ul</div>', unsafe_allow_html=True)

st.markdown("""
Încarcă fișierul **amazon_sales.csv** o singură dată, aici.
Datele sunt salvate în `st.session_state` și rămân disponibile în toate paginile aplicației.
""")

if "df" not in st.session_state:
    st.markdown(
        '<div class="upload-prompt">Încarcă fișierul <strong>amazon_sales.csv</strong> pentru a activa dashboard-ul</div>',
        unsafe_allow_html=True)
    fisier = st.file_uploader("Alege fișierul CSV", type=["csv"], label_visibility="collapsed")

    if fisier is not None:
        df = pd.read_csv(fisier)
        st.session_state["df"] = df
        st.rerun()
else:
    df = st.session_state["df"]

    # Calculăm metricile folosind exact coloanele din CSV-ul tău
    nr_comenzi = len(df)
    nr_categorii = df['product_category'].nunique()
    venit_total = f"${df['total_revenue'].sum():,.2f}"
    profit_total = f"${df['profit'].sum():,.2f}"
    rating_mediu = f"{df['rating'].mean():.2f} ⭐"

    st.markdown(f"""
    <div class="ok-banner">
        <strong>Date încărcate cu succes.</strong> Dataset-ul conține
        <strong>{nr_comenzi} comenzi</strong> din
        <strong>{nr_categorii} categorii</strong> de produse.
        Poți naviga acum către pagina de <strong>Dashboard</strong> din meniul lateral.
    </div>
    <div class="stat-row">
        <div class="stat-pill"><div class="sv">{nr_comenzi}</div><div class="sl">Comenzi</div></div>
        <div class="stat-pill"><div class="sv">{nr_categorii}</div><div class="sl">Categorii</div></div>
        <div class="stat-pill"><div class="sv">{venit_total}</div><div class="sl">Venit Total</div></div>
        <div class="stat-pill"><div class="sv">{profit_total}</div><div class="sl">Profit Total</div></div>
        <div class="stat-pill"><div class="sv">{rating_mediu}</div><div class="sl">Rating Mediu</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Afișăm primele 5 rânduri pentru validare vizuală
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Încarcă un alt set de date"):
        del st.session_state["df"]
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#94A3B8; font-size:13px; font-family:\'JetBrains Mono\', monospace;">Amazon Sales Dashboard · Analiză E-Commerce · Python & Streamlit</p>',
    unsafe_allow_html=True)