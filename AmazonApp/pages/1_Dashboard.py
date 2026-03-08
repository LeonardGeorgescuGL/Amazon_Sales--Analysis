import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# ── Configurare pagină și CSS ─────────────────────────────────────
st.set_page_config(page_title="Sales & Profit Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:      #0f0f0f;
    --surface: #1a1a1a;
    --border:  #2e2e2e;
    --accent:  #ff5c00;
    --text:    #e8e8e8;
    --mono:    'IBM Plex Mono', monospace;
    --display: 'Syne', sans-serif;
    --body:    'DM Sans', sans-serif;
}

html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--body) !important; }
h1, h2, h3 { font-family: var(--display) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background-color: #141414 !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.page-header {
    border-left: 4px solid var(--accent);
    padding: 28px 36px;
    background: var(--surface);
    border-radius: 4px;
    margin-bottom: 36px;
}
.page-header h1 { font-size: 40px !important; font-weight: 800; margin: 0 !important; line-height: 1.1; }
.sec-header {
    font-family: var(--mono); font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); margin: 40px 0 16px 0; padding-bottom: 10px; border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>📊 Global Sales & Profit Dashboard</h1>
    <p style="color:#888; margin-top:10px;">Analiza performanței financiare pe regiuni și categorii de produse.</p>
</div>
""", unsafe_allow_html=True)

# ── Încărcare Date ────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.info("Încarcă dataset-ul pentru a continua.")
    fisier = st.file_uploader("Încarcă fișierul CSV", type=["csv"])
    if fisier is None:
        st.stop()
    df = pd.read_csv(fisier)

# ── FILTRE ÎN SIDEBAR ─────────────────────────────────────────────
st.sidebar.markdown("### 🎯 Filtrează Datele")

# Filtru 1: Regiune (Multiselect)
regiuni = df["customer_region"].unique().tolist()
selectie_regiuni = st.sidebar.multiselect("Regiune Clienți", regiuni, default=regiuni)

# Filtru 2: Categorie (Multiselect)
categorii = df["product_category"].unique().tolist()
selectie_categorii = st.sidebar.multiselect("Categorii Produse", categorii, default=categorii)

# Filtru 3: Discount (Slider)
max_disc = int(df["discount_percent"].max())
selectie_discount = st.sidebar.slider("Discount Maxim (%)", 0, max_disc, max_disc)

# 🚀 EXTRA 1: Toggle pentru Produse de Top (Rating >= 4.5)
st.sidebar.markdown("---")
doar_top = st.sidebar.toggle("🌟 Arată doar produsele de top (Rating >= 4.5)")

# ── APLICARE FILTRE ───────────────────────────────────────────────
df_filtrat = df[
    (df["customer_region"].isin(selectie_regiuni)) &
    (df["product_category"].isin(selectie_categorii)) &
    (df["discount_percent"] <= selectie_discount)
    ]

if doar_top:
    df_filtrat = df_filtrat[df_filtrat["rating"] >= 4.5]

# ── STATISTICI (Metrics) ──────────────────────────────────────────
st.markdown('<div class="sec-header">KPIs Financiari</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Comenzi Totale", len(df_filtrat))
col2.metric("Venit Total (Revenue)", f"${df_filtrat['total_revenue'].sum():,.2f}")
col3.metric("Profit Total", f"${df_filtrat['profit'].sum():,.2f}")
col4.metric("Rating Mediu", f"{df_filtrat['rating'].mean():.2f} ⭐" if not df_filtrat.empty else "N/A")

# ── DATAFRAME & DOWNLOAD ──────────────────────────────────────────
st.markdown('<div class="sec-header">Tabel Date Filtrate</div>', unsafe_allow_html=True)
st.dataframe(
    df_filtrat[["order_id", "product_category", "customer_region", "total_revenue", "profit", "rating"]].head(100),
    use_container_width=True,
    hide_index=True
)

# 🚀 EXTRA 2: Buton de Export
st.download_button(
    label="📥 Descarcă Raportul Filtrat (CSV)",
    data=df_filtrat.to_csv(index=False).encode('utf-8'),
    file_name='raport_vanzari_filtrat.csv',
    mime='text/csv'
)

# ── GRAFICE (Plotly & Matplotlib) ─────────────────────────────────
st.markdown('<div class="sec-header">Analiză Vizuală</div>', unsafe_allow_html=True)

grafic_col1, grafic_col2 = st.columns(2)

with grafic_col1:
    # Grafic 1: Plotly (Interactiv) - Profit pe Regiuni
    st.markdown("**Profit Total pe Regiuni (Plotly)**")
    if not df_filtrat.empty:
        df_profit_regiune = df_filtrat.groupby("customer_region")["profit"].sum().reset_index()
        fig_plotly = px.pie(
            df_profit_regiune,
            names="customer_region",
            values="profit",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        fig_plotly.update_layout(margin=dict(t=20, b=20, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e8e8")
        st.plotly_chart(fig_plotly, use_container_width=True)

with grafic_col2:
    # Grafic 2: Matplotlib - Corelația dintre Discount și Profit
    st.markdown("**Distribuția Metodelor de Plată (Matplotlib)**")
    if not df_filtrat.empty:
        fig_mat, ax = plt.subplots(figsize=(6, 4))
        fig_mat.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.spines['bottom'].set_color('#e8e8e8')
        ax.spines['left'].set_color('#e8e8e8')
        ax.tick_params(axis='x', colors='#e8e8e8')
        ax.tick_params(axis='y', colors='#e8e8e8')

        plati = df_filtrat['payment_method'].value_counts()
        ax.bar(plati.index, plati.values, color="#ff5c00", edgecolor="#1a1a1a")
        ax.set_ylabel("Număr de comenzi", color="#e8e8e8")

        st.pyplot(fig_mat)
        plt.close(fig_mat)