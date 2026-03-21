import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Machine Learning", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:      #0B1120;
    --surface: #1E293B;
    --border:  #334155;
    --accent:  #38BDF8;
    --accent2: #A78BFA;
    --accent3: #34D399;
    --accent4: #FB923C;
    --text:    #F8FAFC;
    --muted:   #94A3B8;
    --mono:    'JetBrains Mono', monospace;
    --display: 'Plus Jakarta Sans', sans-serif;
    --body:    'Inter', sans-serif;
    --radius:  12px;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--body) !important; }
h1,h2,h3,h4 { font-family: var(--display) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
.page-header {
    border-left: 4px solid var(--accent2);
    padding: 28px 36px;
    background: var(--surface);
    border-radius: var(--radius);
    margin-bottom: 36px;
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.4);
}
.page-header h1 { font-size: 38px !important; font-weight: 800; margin: 0 !important; }
.sec-header {
    font-family: var(--mono); font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); margin: 40px 0 16px 0; padding-bottom: 10px; border-bottom: 1px solid var(--border);
}
.tip {
    background: rgba(167,139,250,0.05);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: var(--radius);
    padding: 20px; font-size: 14px; color: #cbd5e1; line-height: 1.6;
}
.tip strong { color: var(--accent2); }
.tip code { background: #0F172A; padding: 2px 6px; border-radius: 4px;
    font-family: var(--mono); font-size: 12px; color: var(--accent); }
.interpret-box {
    background: rgba(52,211,153,0.05);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent3);
    border-radius: var(--radius);
    padding: 20px; font-size: 14px; color: #cbd5e1; line-height: 1.6; margin-top: 16px;
}
.interpret-box strong { color: var(--accent3); }
.cluster-box {
    background: rgba(251,146,60,0.05);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent4);
    border-radius: var(--radius);
    padding: 20px; font-size: 14px; color: #cbd5e1; line-height: 1.6; margin-top: 16px;
}
.cluster-box strong { color: var(--accent4); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="page-header">
    <h1> Machine Learning — Amazon Sales</h1>
    <p style="color:#94A3B8; margin-top:10px;">
        Regresie · Clasificare · Clusterizare Ierarhică cu PCA.
        Alegerile de preprocesare din pagina anterioară influențează direct performanța fiecărui model.
    </p>
</div>
""", unsafe_allow_html=True)

# Verificare date
if "df" not in st.session_state:
    st.warning(" Mergi la **Home** și încarcă dataset-ul mai întâi.")
    st.stop()

df_original = st.session_state["df"].copy()

# Preluăm datele preprocesate dacă există, altfel folosim default
if "df_procesat" in st.session_state and "alegeri_preprocesare_amazon" in st.session_state:
    df_proc = st.session_state["df_procesat"].copy()
    cfg = st.session_state["alegeri_preprocesare_amazon"]
    st.info(f" Folosind datele preprocesate din Pagina 2: **{cfg['n_randuri']:,} rânduri**, **{cfg['n_coloane']} coloane**, scalare: **{cfg['metoda_scal']}**")
else:
    # Preprocesare default simplă
    df_proc = df_original.copy()
    id_cols = [c for c in ["order_id", "product_id"] if c in df_proc.columns]
    df_proc = df_proc.drop(columns=id_cols)
    for cat_col in ["product_category", "customer_region", "payment_method"]:
        if cat_col in df_proc.columns:
            le = LabelEncoder()
            df_proc[cat_col] = le.fit_transform(df_proc[cat_col].astype(str))
    cfg = {
        "metoda_rating": "Median", "metoda_disc": "Median",
        "metoda_outlieri": "Păstrează toți outlierii",
        "enc_categ": "Label Encoding", "enc_regiune": "Label Encoding",
        "enc_plata": "Label Encoding", "metoda_scal": "Fără scalare",
        "n_randuri": len(df_proc), "n_coloane": df_proc.shape[1]
    }
    st.warning("⚠ Nu ai trecut prin Pagina 2 — se folosește preprocesarea implicită (Label Encoding, fără scalare).")

st.markdown("---")
st.markdown("""
### Ce modele antrenăm?

| Model | Task | Variabilă target | Metrică principală |
|---|---|---|---|
| **Regresia Liniară** | Predicție numerică | `profit` | R², MAE, RMSE |
| **Random Forest** | Predicție/clasificare | `profit` / segment | R² sau Accuracy |
| **Regresia Logistică** | Clasificare | Segment profit (Low/Med/High) | Accuracy, F1 |
| **Clusterizare Ierarhică + PCA** | Nesupervizat | — (grupăm produse) | Dendrogramă, silhouette |
""")

# Helpers
def get_numeric_features(df, exclude=None):
    """Returnează coloanele numerice, excluzând target-urile."""
    if exclude is None:
        exclude = []
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]

def creeaza_segmente_profit(df):
    """Creează segmente Low/Medium/High bazate pe cuantile."""
    df = df.copy()
    q33 = df["profit"].quantile(0.33)
    q66 = df["profit"].quantile(0.66)
    df["segment_profit"] = pd.cut(
        df["profit"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Low Profit", "Medium Profit", "High Profit"]
    )
    return df, q33, q66


# SECȚIUNEA 1 — Regresia Liniară
st.markdown("---")
st.header("1. Regresia Liniară — Predicția profitului")

st.markdown("""
**Scopul:** Pornind de la `price`, `discount_percent`, `quantity_sold`, `rating` etc.,
modelul încearcă să prezică exact **valoarea profitului** al unei comenzi.

**Regresia Liniară** presupune o relație liniară între features și target:
```
profit = β₀ + β₁·price + β₂·discount + β₃·quantity + ... + ε
```
E cel mai interpretabil model — coeficienții `β` spun direct
**cât crește profitul la o unitate de creștere a fiecărei variabile**.

**Metrica R²:** proporția din varianța profitului explicată de model.
R²=1 → predicție perfectă. R²=0 → modelul nu e mai bun decât media.
""")

with st.form("form_lm"):
    col1, col2, col3 = st.columns(3)
    with col1:
        target_lm = st.selectbox(
            "Target:", ["profit", "total_revenue", "discounted_price"],
            key="tgt_lm"
        )
    with col2:
        test_size_lm = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_lm")
    with col3:
        random_state_lm = st.number_input("Random state", 0, 100, 42, key="rs_lm")
    submit_lm = st.form_submit_button("▶ Antrenează Regresia Liniară")

if submit_lm:
    exclude_cols = [target_lm, "profit", "total_revenue", "discounted_price"]
    feat_cols = get_numeric_features(df_proc, exclude=exclude_cols)
    df_model = df_proc[feat_cols + [target_lm]].dropna()

    X = df_model[feat_cols]
    y = df_model[target_lm]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size_lm, random_state=int(random_state_lm))

    with st.spinner("Antrenez modelul de regresie liniară..."):
        model_lm = LinearRegression()
        model_lm.fit(X_tr, y_tr)
        y_pred_lm = model_lm.predict(X_te)
        r2   = r2_score(y_te, y_pred_lm)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred_lm))
        mae  = mean_absolute_error(y_te, y_pred_lm)

    st.success(" Antrenare completă!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.4f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("MAE", f"{mae:,.2f}")
    col4.metric("Date test", f"{len(X_te):,}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_lm1 = px.scatter(
            x=y_te, y=y_pred_lm,
            labels={"x": f"{target_lm} real", "y": f"{target_lm} prezis"},
            title="Real vs. Prezis",
            color_discrete_sequence=["#38BDF8"]
        )
        max_v = max(float(y_te.max()), float(y_pred_lm.max()))
        min_v = min(float(y_te.min()), float(y_pred_lm.min()))
        fig_lm1.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                          line=dict(color="#A78BFA", dash="dash", width=2))
        fig_lm1.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_lm1, use_container_width=True)
        st.caption("Punctele pe linia violet = predicții perfecte. Abaterile = erori ale modelului.")

    with col_g2:
        erori_lm = y_pred_lm - y_te
        fig_lm2 = px.histogram(
            x=erori_lm, nbins=40,
            labels={"x": "Eroare (prezis − real)"},
            title="Distribuția erorilor",
            color_discrete_sequence=["#34D399"]
        )
        fig_lm2.add_vline(x=0, line_dash="dash", line_color="#A78BFA")
        fig_lm2.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_lm2, use_container_width=True)
        st.caption("Distribuție centrată în 0 și îngustă = model bine calibrat, fără bias sistematic.")

    # Coeficienți
    df_coef = pd.DataFrame({
        "Feature": feat_cols,
        "Coeficient": model_lm.coef_
    }).sort_values("Coeficient", key=abs, ascending=True).tail(12)

    fig_coef = px.bar(
        df_coef, x="Coeficient", y="Feature", orientation="h",
        title=f"Coeficienți Regresie Liniară — impact asupra `{target_lm}`",
        color="Coeficient",
        color_continuous_scale=[[0, "#A78BFA"], [0.5, "#1E293B"], [1, "#38BDF8"]]
    )
    fig_coef.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Cum interpretezi coeficienții?</strong><br><br>
    Un coeficient <strong>pozitiv</strong> înseamnă că o creștere cu 1 unitate a acelei variabile
    <em>crește</em> profitul cu valoarea coeficientului (celelalte variabile rămânând constante).<br>
    Un coeficient <strong>negativ</strong> înseamnă că variabila <em>reduce</em> profitul.<br><br>
    Exemplu: dacă `discount_percent` are coeficient negativ mare → reducerile mai mari
    scad profitul per comandă. Dacă `quantity_sold` are coeficient pozitiv mare →
    volumul mai mare crește profitul.
    </div>
    """, unsafe_allow_html=True)

    st.session_state["rezultat_lm"] = {"r2": r2, "rmse": rmse, "mae": mae, "target": target_lm}


# SECȚIUNEA 2 — Random Forest
st.markdown("---")
st.header("2. Random Forest — Regresie & Importanța Variabilelor")

st.markdown("""
**Random Forest** construiește sute de arbori de decizie pe subseturi aleatoare
din date și features, apoi mediazează predicțiile lor.

**Avantaje față de Regresia Liniară:**
- Nu presupune relații liniare între features și target
- Robust la outlieri (arborii individuali nu sunt afectați la fel de mult)
- Oferă **feature importance** — care variabilă contribuie cel mai mult la predicție

**Feature importance** în RF = cât de mult reduce fiecare feature eroarea
medie de predicție (impurity reduction) pe toți arborii.
""")

with st.form("form_rf"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_rf = st.selectbox("Target:", ["profit", "total_revenue"], key="tgt_rf")
    with col2:
        n_trees = st.slider("Nr. arbori", 50, 500, 100, 50, key="trees_rf")
    with col3:
        max_depth = st.select_slider("Adâncime maximă", options=[3, 5, 10, 20, None], value=10, key="depth_rf")
    with col4:
        test_size_rf = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_rf")
    submit_rf = st.form_submit_button("▶ Antrenează Random Forest")

if submit_rf:
    exclude_cols = [target_rf, "profit", "total_revenue", "discounted_price"]
    feat_cols_rf = get_numeric_features(df_proc, exclude=exclude_cols)
    df_rf = df_proc[feat_cols_rf + [target_rf]].dropna()

    X_rf = df_rf[feat_cols_rf]
    y_rf = df_rf[target_rf]
    X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=test_size_rf, random_state=42)

    with st.spinner(f"Antrenez Random Forest cu {n_trees} arbori..."):
        model_rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42, n_jobs=-1)
        model_rf.fit(X_tr, y_tr)
        y_pred_rf = model_rf.predict(X_te)
        r2_rf   = r2_score(y_te, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_te, y_pred_rf))
        mae_rf  = mean_absolute_error(y_te, y_pred_rf)

    st.success(" Antrenare completă!")

    # Comparare cu LM dacă există
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2_rf:.4f}",
        delta=f"{r2_rf - st.session_state.get('rezultat_lm', {}).get('r2', r2_rf):.4f} vs LM"
        if "rezultat_lm" in st.session_state else None)
    col2.metric("RMSE", f"{rmse_rf:,.2f}",
        delta=f"{rmse_rf - st.session_state.get('rezultat_lm', {}).get('rmse', rmse_rf):,.2f} vs LM"
        if "rezultat_lm" in st.session_state else None, delta_color="inverse")
    col3.metric("MAE", f"{mae_rf:,.2f}")
    col4.metric("Nr. arbori", n_trees)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_rf1 = px.scatter(
            x=y_te, y=y_pred_rf,
            labels={"x": f"{target_rf} real", "y": f"{target_rf} prezis"},
            title="Real vs. Prezis — Random Forest",
            color_discrete_sequence=["#FB923C"]
        )
        max_v = max(float(y_te.max()), float(y_pred_rf.max()))
        min_v = min(float(y_te.min()), float(y_pred_rf.min()))
        fig_rf1.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                          line=dict(color="#A78BFA", dash="dash", width=2))
        fig_rf1.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_rf1, use_container_width=True)

    with col_g2:
        imp_df = pd.DataFrame({
            "Feature": feat_cols_rf,
            "Importanță (%)": model_rf.feature_importances_ * 100
        }).sort_values("Importanță (%)", ascending=True).tail(12)

        fig_imp = px.bar(
            imp_df, x="Importanță (%)", y="Feature", orientation="h",
            title="Top Features după importanță (Random Forest)",
            color="Importanță (%)",
            color_continuous_scale=[[0, "#1E293B"], [1, "#FB923C"]]
        )
        fig_imp.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Cum interpretam Feature Importance?</strong><br><br>
    Barele arată ce % din reducerea erorii totale vine de la fiecare variabilă.
    O valoare mare înseamnă că variabila respectivă este un predictor puternic al profitului.<br><br>
    Dacă <code>price</code> domină importanța → profitul este determinat în principal de prețul produsului.<br>
    Dacă <code>discount_percent</code> are importanță mare → reducerile sunt un driver major al profitabilității.<br><br>
    <strong>Atenție:</strong> spre deosebire de coeficienții din Regresia Liniară,
    importanța RF nu spune <em>direcția</em> efectului (pozitiv/negativ) — doar magnitudinea.
    </div>
    """, unsafe_allow_html=True)

    st.session_state["rezultat_rf"] = {"r2": r2_rf, "rmse": rmse_rf, "target": target_rf}


# SECȚIUNEA 3 — Regresia Logistică (Clasificare)
st.markdown("---")
st.header("3. Regresia Logistică — Clasificarea segmentului de profit")

st.markdown("""
**Regresia Logistică** este un model de **clasificare** — nu prezice o valoare numerică,
ci **probabilitatea că o comandă aparține unui segment**.

Împărțim comenzile în 3 segmente pe baza distribuției profitului:

| Segment | Definiție |
|---|---|
| 🔴 **Low Profit** | Sub percentila 33% a profitului |
| 🟡 **Medium Profit** | Între percentilele 33% și 66% |
| 🟢 **High Profit** | Peste percentila 66% a profitului |

**Avantaj:** Putem întreba modelul „care e probabilitatea că această comandă
este High Profit?" și putem acționa în consecință (promovare, stoc prioritar).
""")

with st.form("form_lr"):
    col1, col2, col3 = st.columns(3)
    with col1:
        solver_lr = st.selectbox("Solver:", ["lbfgs", "saga", "liblinear"], key="solver_lr")
    with col2:
        max_iter_lr = st.slider("Max iterații", 200, 2000, 1000, 100, key="iter_lr")
    with col3:
        test_size_lr = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_lr")
    submit_lr = st.form_submit_button("▶ Antrenează Regresia Logistică")

if submit_lr:
    df_cls, q33, q66 = creeaza_segmente_profit(df_proc)
    etichete_cls = ["Low Profit", "Medium Profit", "High Profit"]

    exclude_cls = ["profit", "total_revenue", "discounted_price", "segment_profit"]
    feat_cls = get_numeric_features(df_cls, exclude=exclude_cls)
    df_cls_clean = df_cls[feat_cls + ["segment_profit"]].dropna()

    X_cls = df_cls_clean[feat_cls]
    y_cls = df_cls_clean["segment_profit"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cls, y_cls, test_size=test_size_lr, random_state=42, stratify=y_cls
    )

    with st.spinner("Antrenez Regresia Logistică..."):
        # Scalăm dacă nu s-a scalat deja pe pagina de preprocesare
        needs_scale = cfg.get("metoda_scal", "Fără scalare") == "Fără scalare"
        if needs_scale:
            scaler_lr = StandardScaler()
            X_tr_s = scaler_lr.fit_transform(X_tr)
            X_te_s = scaler_lr.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr.values, X_te.values

        model_lr_cls = LogisticRegression(
            max_iter=max_iter_lr, solver=solver_lr,
            random_state=42, multi_class="auto"
        )
        model_lr_cls.fit(X_tr_s, y_tr)
        y_pred_lr = model_lr_cls.predict(X_te_s)
        acc_lr = accuracy_score(y_te, y_pred_lr)

    st.success(" Antrenare completă!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc_lr:.4f}")
    col2.metric("Date antrenare", f"{len(X_tr):,}")
    col3.metric("Prag Low (q33)", f"{q33:.2f}")
    col4.metric("Prag High (q66)", f"{q66:.2f}")

    if needs_scale:
        st.caption(" Scalare StandardScaler aplicată automat (necesară pentru Regresia Logistică).")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        cm_lr = confusion_matrix(y_te, y_pred_lr, labels=etichete_cls)
        fig_cm = px.imshow(
            cm_lr, text_auto=True,
            x=etichete_cls, y=etichete_cls,
            labels=dict(x="Prezis", y="Real"),
            title="Matrice de Confuzie",
            color_continuous_scale=[[0, "#0F172A"], [0.5, "#1E4D3A"], [1, "#34D399"]]
        )
        fig_cm.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonala = predicții corecte. Valorile în afara diagonalei = erori de clasificare.")

    with col_g2:
        dist_df = pd.DataFrame({
            "Segment": etichete_cls,
            "Real": [sum(y_te == s) for s in etichete_cls],
            "Prezis": [sum(y_pred_lr == s) for s in etichete_cls]
        }).melt(id_vars="Segment", var_name="Tip", value_name="Count")

        fig_dist = px.bar(
            dist_df, x="Segment", y="Count", color="Tip", barmode="group",
            title="Distribuție segmente — Real vs. Prezis",
            color_discrete_sequence=["#38BDF8", "#A78BFA"]
        )
        fig_dist.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_dist, use_container_width=True)

    with st.expander(" Raport detaliat per segment"):
        raport_lr = classification_report(y_te, y_pred_lr, labels=etichete_cls, output_dict=True)
        df_raport = pd.DataFrame(raport_lr).transpose().round(3)
        st.dataframe(df_raport, use_container_width=True)
        st.markdown("""
        - **precision** — din predicțiile pentru un segment, câte sunt corecte
        - **recall** — din comenzile reale dintr-un segment, câte le-a identificat corect modelul
        - **f1-score** — media armonică precision/recall (echilibrul între cele două)
        - **support** — numărul de comenzi reale din segment
        """)

    st.markdown("""
    <div class="interpret-box">
    <strong>Ce înseamnă un recall mic pentru „High Profit"?</strong><br><br>
    Dacă recall-ul pentru <strong>High Profit</strong> este mic, modelul ratează
    multe comenzi cu profit mare — le clasifică greșit ca Medium sau Low.<br>
    Asta înseamnă că nu putem folosi modelul cu încredere pentru a identifica
    comenzile/produsele cele mai profitabile.<br><br>
    <strong>Cauze posibile:</strong> prea puțini outlieri de tip High Profit în datele de antrenare
    (mai ales dacă ai eliminat outlierii pe Pagina 2), sau features insuficiente pentru
    a distinge comenzile cu profit mare.
    </div>
    """, unsafe_allow_html=True)

    st.session_state["rezultat_lr"] = {"acc": acc_lr}


# SECȚIUNEA 4 — PCA + Clusterizare Ierarhică
st.markdown("---")
st.header("4. PCA + Clusterizare Ierarhică — Gruparea Produselor")

st.markdown("""
**Clusterizarea ierarhică** este un algoritm **nesupervizat** — nu avem labels/clase/etichete predefinite.
Grupăm produsele/comenzile în funcție de similaritate, fără să le spunem dinainte
câte grupuri există.

**PCA (Principal Component Analysis)** reduce dimensionalitatea datelor:
din `n` coloane numerice, creăm 2–3 **componente principale** care captează
cea mai mare variabilitate. Acest lucru ne permite să **vizualizăm clusterele în 2D**.

**Componentele principale pentru Amazon Sales:**
""")

col_pca1, col_pca2 = st.columns(2)
with col_pca1:
    st.markdown("""
    <div class="tip">
    <strong>PC1 — Performanță Comercială</strong><br><br>
    Prima componentă captează variabilitatea legată de <strong>valoarea financiară</strong>
    a unei comenzi. Are loadings mari pozitivi pe:<br>
    → <code>price</code>, <code>total_revenue</code>, <code>profit</code>, <code>discounted_price</code><br><br>
    O comandă cu PC1 mare = produs scump, profit ridicat, venit mare.<br>
    O comandă cu PC1 mic = produs ieftin, marjă scăzută.<br><br>
    <strong>Interpretare:</strong> PC1 separă segmentul <em>premium</em> de cel <em>budget</em>.
    </div>
    """, unsafe_allow_html=True)

with col_pca2:
    st.markdown("""
    <div class="tip">
    <strong>PC2 — Engagement & Sensibilitate la Reduceri</strong><br><br>
    A doua componentă captează variabilitatea legată de <strong>popularitatea produsului
    și comportamentul de cumpărare</strong>. Are loadings mari pe:<br>
    → <code>rating</code>, <code>review_count</code>, <code>quantity_sold</code>, <code>discount_percent</code><br><br>
    O comandă cu PC2 mare = produs popular, multe recenzii, discount ridicat.<br>
    O comandă cu PC2 mic = nișă, puțin recenzat, fără reduceri semnificative.<br><br>
    <strong>Interpretare:</strong> PC2 separă produsele <em>viral/populare</em> de cele <em>de nișă</em>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.form("form_cluster"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_clusters = st.slider("Număr de clustere", 2, 6, 3, key="n_clust")
    with col2:
        linkage_method = st.selectbox(
            "Metodă de linkage:",
            ["ward", "complete", "average", "single"],
            key="link_method"
        )
    with col3:
        n_comp_pca = st.slider("Componente PCA", 2, 4, 2, key="n_pca")
    with col4:
        n_sample = st.slider("Sample pt. dendrogramă", 50, 300, 100, 50, key="n_samp")

    feat_options = [c for c in ["price", "discount_percent", "quantity_sold",
                                "rating", "review_count", "total_revenue", "profit",
                                "discounted_price"]
                    if c in df_proc.columns]
    feat_cluster = st.multiselect(
        "Features pentru clustering:",
        options=feat_options,
        default=[c for c in ["price", "discount_percent", "quantity_sold", "rating",
                              "review_count", "profit"] if c in feat_options],
        key="feat_clust"
    )
    submit_clust = st.form_submit_button("▶ Rulează PCA + Clusterizare Ierarhică")

if submit_clust:
    if len(feat_cluster) < 2:
        st.error("Selectează cel puțin 2 features pentru clustering.")
    else:
        df_clust = df_proc[feat_cluster].dropna().copy()
        df_clust = df_clust.sample(min(2000, len(df_clust)), random_state=42).reset_index(drop=True)

        # Scalare obligatorie pentru clustering
        scaler_cl = StandardScaler()
        X_scaled = scaler_cl.fit_transform(df_clust)

        with st.spinner("Calculez PCA și clusterizarea ierarhică..."):
            # PCA
            pca = PCA(n_components=n_comp_pca, random_state=42)
            coords_pca = pca.fit_transform(X_scaled)
            var_expl = pca.explained_variance_ratio_ * 100

            # Linkage & clustere
            Z = linkage(coords_pca, method=linkage_method)
            labels_clust = fcluster(Z, n_clusters, criterion="maxclust")

        st.success(" PCA și clusterizare completă!")

        # Varianță explicată
        st.subheader("4a. Varianță explicată de fiecare componentă PCA")

        fig_var = px.bar(
            x=[f"PC{i+1}" for i in range(n_comp_pca)],
            y=var_expl,
            labels={"x": "Componentă", "y": "Varianță explicată (%)"},
            title="Varianță explicată de fiecare PC",
            color=var_expl,
            color_continuous_scale=[[0, "#1E293B"], [1, "#A78BFA"]],
            text=[f"{v:.1f}%" for v in var_expl]
        )
        fig_var.update_traces(textposition="outside")
        fig_var.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
            showlegend=False
        )
        st.plotly_chart(fig_var, use_container_width=True)

        st.metric("Varianță totală explicată (PC1+PC2)",
                  f"{sum(var_expl[:2]):.1f}%",
                  help="Cât din informația totală e capturată de primele 2 componente")

        # Loadings
        st.subheader("4b. Loadings — contribuția fiecărei variabile la PC1 și PC2")

        loadings_df = pd.DataFrame(
            pca.components_[:2].T,
            columns=["PC1 — Performanță Comercială", "PC2 — Engagement & Reduceri"],
            index=feat_cluster
        )

        fig_load = go.Figure()
        fig_load.add_trace(go.Bar(
            name="PC1 — Performanță Comercială",
            x=feat_cluster,
            y=loadings_df["PC1 — Performanță Comercială"],
            marker_color="#38BDF8"
        ))
        fig_load.add_trace(go.Bar(
            name="PC2 — Engagement & Reduceri",
            x=feat_cluster,
            y=loadings_df["PC2 — Engagement & Reduceri"],
            marker_color="#A78BFA"
        ))
        fig_load.update_layout(
            barmode="group",
            title="Loadings PCA — cât contribuie fiecare variabilă la fiecare componentă",
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC"
        )
        st.plotly_chart(fig_load, use_container_width=True)

        st.markdown("""
        <div class="interpret-box">
        <strong>Cum citim loadings-urile(variantele standardizate)?</strong><br><br>
        Un loading <strong>pozitiv mare</strong> pe PC1 înseamnă că variabila respectivă
        crește odată cu Performanța Comercială.<br>
        Un loading <strong>negativ</strong> pe PC2 înseamnă că variabila scade când
        Engagement-ul crește (sau invers).<br><br>
        Variabilele cu loading aproape de 0 contribuie puțin la acea componentă
        și pot fi mai puțin relevante pentru clusterizare.
        </div>
        """, unsafe_allow_html=True)

        # Dendrogramă
        st.subheader("4c. Dendrogramă — structura ierarhică a clusterelor")

        idx_sample = np.random.choice(len(coords_pca), size=min(n_sample, len(coords_pca)), replace=False)
        sample_coords = coords_pca[idx_sample]
        Z_sample = linkage(sample_coords, method=linkage_method)

        fig_dendo, ax_dendo = plt.subplots(figsize=(14, 5))
        fig_dendo.patch.set_facecolor("#0B1120")
        ax_dendo.set_facecolor("#0B1120")
        for spine in ax_dendo.spines.values():
            spine.set_color("#334155")
        ax_dendo.tick_params(colors="#94A3B8")
        ax_dendo.set_xlabel("Index comandă (sample)", color="#94A3B8")
        ax_dendo.set_ylabel("Distanță", color="#94A3B8")
        ax_dendo.set_title(
            f"Dendrogramă — linkage={linkage_method} | {n_sample} comenzi (sample)",
            color="#F8FAFC", fontsize=13
        )

        color_thresh = Z_sample[-(n_clusters - 1), 2]
        dendrogram(Z_sample, ax=ax_dendo, color_threshold=color_thresh,
                   leaf_rotation=90, leaf_font_size=7,
                   above_threshold_color="#334155")

        ax_dendo.axhline(y=color_thresh, color="#FB923C",
                         linestyle="--", linewidth=1.5, label=f"Tăietură la {n_clusters} clustere")
        ax_dendo.legend(facecolor="#1E293B", edgecolor="#334155",
                        labelcolor="#F8FAFC", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig_dendo)
        plt.close(fig_dendo)

        st.markdown("""
        <div class="cluster-box">
        <strong>Cum citim dendrograma?</strong><br><br>
        Fiecare frunză = o comandă. Ramurile care se unesc jos = comenzi foarte similare.
        Ramurile care se unesc sus = grupuri mai diferite între ele.<br><br>
        <strong>Linia portocalie</strong> arată unde „tăiem" ierarhia pentru a obține numărul
        de clustere ales. Tot ce este la stânga liniei formează un cluster.<br><br>
        <strong>Metoda Ward</strong> (recomandată) minimizează variația intra-cluster —
        produce clustere compacte și echilibrate ca dimensiune.
        </div>
        """, unsafe_allow_html=True)

        # Scatter PCA colorat
        st.subheader("4d. Proiecție PCA — clusterele vizualizate în 2D")

        df_viz = pd.DataFrame({
            "PC1 — Performanță Comercială": coords_pca[:, 0],
            "PC2 — Engagement & Reduceri":  coords_pca[:, 1],
            "Cluster": [f"Cluster {c}" for c in labels_clust]
        })

        # Adăugăm coloana originală de categorie pentru tooltip
        if len(df_clust) == len(df_viz):
            if "product_category" in df_original.columns:
                df_orig_sample = df_original.dropna(subset=feat_cluster).sample(
                    min(2000, len(df_original.dropna(subset=feat_cluster))), random_state=42
                ).reset_index(drop=True)
                if "product_category" in df_orig_sample.columns:
                    df_viz["Categorie"] = df_orig_sample["product_category"].values[:len(df_viz)]

        cluster_colors = ["#38BDF8", "#A78BFA", "#34D399", "#FB923C", "#F472B6", "#FBBF24"]
        fig_pca = px.scatter(
            df_viz,
            x="PC1 — Performanță Comercială",
            y="PC2 — Engagement & Reduceri",
            color="Cluster",
            hover_data=["Categorie"] if "Categorie" in df_viz.columns else None,
            title=f"Clustere în spațiul PCA — {n_clusters} grupuri ({linkage_method})",
            color_discrete_sequence=cluster_colors[:n_clusters],
            opacity=0.7
        )
        fig_pca.update_layout(
            height=520, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC"
        )
        fig_pca.update_traces(marker=dict(size=5))
        # Axe cu linii de referință la 0
        fig_pca.add_hline(y=0, line_color="#334155", line_dash="dot", line_width=1)
        fig_pca.add_vline(x=0, line_color="#334155", line_dash="dot", line_width=1)
        st.plotly_chart(fig_pca, use_container_width=True)

        # Profil clustere
        st.subheader("4e. Profilul fiecărui cluster")

        df_profil = df_clust.copy()
        df_profil["Cluster"] = [f"Cluster {c}" for c in labels_clust]
        df_profil_group = df_profil.groupby("Cluster")[feat_cluster].mean().round(2)

        # Adăugăm dimensiunea
        dim_clust = df_profil["Cluster"].value_counts().sort_index()
        df_profil_group["Comenzi (n)"] = dim_clust.values

        st.dataframe(df_profil_group, use_container_width=True)

        st.markdown("""
        <div class="cluster-box">
        <strong>Cum interpretam profilul clusterelor?</strong><br><br>
        Fiecare rând = valorile medii ale comenzilor din acel cluster.<br><br>
        <strong>Exemple de pattern-uri tipice:</strong><br>
        → Un cluster cu <code>price</code> mare + <code>profit</code> mare + <code>discount_percent</code> mic
        = <em>produse premium, marjă mare, vândute la preț întreg</em>.<br>
        → Un cluster cu <code>quantity_sold</code> mare + <code>rating</code> mare + <code>discount_percent</code> mare
        = <em>produse populare în promoție, volum ridicat, marjă mai mică</em>.<br>
        → Un cluster cu toate valorile mici = <em>produse nișă, vânzări slabe</em>.<br><br>
        Aceste tipare pot ghida strategia de pricing, promovare și gestionarea stocului.
        </div>
        """, unsafe_allow_html=True)

        # Distribuție categorii per cluster
        if "product_category" in df_original.columns and "Categorie" in df_viz.columns:
            st.subheader("4f. Distribuția categoriilor de produse per cluster")
            df_categ_clust = df_viz[["Cluster", "Categorie"]].copy()
            df_categ_cnt = df_categ_clust.groupby(["Cluster", "Categorie"]).size().reset_index(name="Count")
            fig_categ = px.bar(
                df_categ_cnt, x="Cluster", y="Count", color="Categorie", barmode="stack",
                title="Ce categorii de produse sunt în fiecare cluster?",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_categ.update_layout(
                height=400, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC"
            )
            st.plotly_chart(fig_categ, use_container_width=True)

        st.session_state["rezultat_cluster"] = {
            "n_clusters": n_clusters,
            "linkage": linkage_method,
            "var_expl_pc1": var_expl[0],
            "var_expl_pc2": var_expl[1],
        }

# SECȚIUNEA 5 — Comparație directă modele
st.markdown("---")
st.header("5. Comparație directă — toate modelele")

st.markdown("Antrenează toate cele 3 modele supervizate și compară-le simultan.")

with st.form("form_comparatie"):
    col1, col2 = st.columns(2)
    with col1:
        n_trees_comp = st.slider("Arbori RF", 50, 300, 100, 50, key="trees_comp")
        test_comp = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_comp")
    with col2:
        st.markdown("""
        <div class="tip">
        <strong>Ce comparăm?</strong><br><br>
        Regresie: R² și RMSE (predict <code>profit</code>)<br>
        Clasificare: Accuracy (segment Low/Med/High)<br><br>
        Același split train/test — comparație corectă.
        </div>
        """, unsafe_allow_html=True)
    submit_comp = st.form_submit_button("▶ Compară toate modelele")

if submit_comp:
    rezultate_comp = []
    progress_bar = st.progress(0, text="Model 1/3 — Regresie Liniară...")

    exclude_base = ["profit", "total_revenue", "discounted_price"]
    feat_base = get_numeric_features(df_proc, exclude=exclude_base)
    df_base = df_proc[feat_base + ["profit"]].dropna()
    X_b = df_base[feat_base]
    y_b = df_base["profit"]
    X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=test_comp, random_state=42)

    with st.spinner("Antrenez modelele..."):
        # 1. Regresie Liniară
        lm = LinearRegression()
        lm.fit(X_tr_b, y_tr_b)
        p_lm = lm.predict(X_te_b)
        progress_bar.progress(0.33, text="Model 2/3 — Random Forest...")
        rezultate_comp.append({
            "Model": "Regresie Liniară", "Task": "Regresie",
            "R²": round(r2_score(y_te_b, p_lm), 4),
            "RMSE": int(np.sqrt(mean_squared_error(y_te_b, p_lm))),
            "Accuracy": "—"
        })

        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=n_trees_comp, random_state=42, n_jobs=-1)
        rf.fit(X_tr_b, y_tr_b)
        p_rf = rf.predict(X_te_b)
        progress_bar.progress(0.66, text="Model 3/3 — Regresie Logistică...")
        rezultate_comp.append({
            "Model": "Random Forest", "Task": "Regresie",
            "R²": round(r2_score(y_te_b, p_rf), 4),
            "RMSE": int(np.sqrt(mean_squared_error(y_te_b, p_rf))),
            "Accuracy": "—"
        })

        # 3. Regresia Logistică
        df_cls_comp, _, _ = creeaza_segmente_profit(df_proc)
        X_cls_c = df_cls_comp[feat_base].dropna()
        y_cls_c = df_cls_comp.loc[X_cls_c.index, "segment_profit"]
        X_trc, X_tec, y_trc, y_tec = train_test_split(
            X_cls_c, y_cls_c, test_size=test_comp, random_state=42, stratify=y_cls_c
        )
        sc_comp = StandardScaler()
        lr_comp = LogisticRegression(max_iter=1000, random_state=42)
        lr_comp.fit(sc_comp.fit_transform(X_trc), y_trc)
        p_lr_comp = lr_comp.predict(sc_comp.transform(X_tec))
        progress_bar.progress(1.0, text="✅ Toate modelele antrenate!")
        rezultate_comp.append({
            "Model": "Regresie Logistică", "Task": "Clasificare",
            "R²": "—", "RMSE": "—",
            "Accuracy": round(accuracy_score(y_tec, p_lr_comp), 4)
        })

    df_comp = pd.DataFrame(rezultate_comp)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_reg = df_comp[df_comp["Task"] == "Regresie"].copy()
        df_reg["R²_num"] = df_reg["R²"].astype(float)
        fig_r2 = px.bar(
            df_reg, x="Model", y="R²_num",
            title="R² — modele de regresie (mai mare = mai bun)",
            color="Model",
            color_discrete_sequence=["#38BDF8", "#FB923C"],
            text="R²_num"
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                             showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_g2:
        df_cls_c2 = df_comp[df_comp["Task"] == "Clasificare"].copy()
        df_cls_c2["Acc_num"] = df_cls_c2["Accuracy"].astype(float)
        fig_acc = px.bar(
            df_cls_c2, x="Model", y="Acc_num",
            title="Accuracy — model de clasificare",
            color="Model",
            color_discrete_sequence=["#A78BFA"],
            text="Acc_num"
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
                             showlegend=False, yaxis_range=[0, 1.1])
        st.plotly_chart(fig_acc, use_container_width=True)

    best_reg = df_comp[df_comp["Task"] == "Regresie"].copy()
    best_reg["R²_num"] = best_reg["R²"].astype(float)
    best_model = best_reg.loc[best_reg["R²_num"].idxmax(), "Model"]
    best_r2    = best_reg["R²_num"].max()
    st.info(f"🏆 **Cel mai bun model de regresie:** {best_model} cu R² = {best_r2:.4f}")

# REZUMAT
st.markdown("---")
st.header("📌 Rezumat — ce am demonstrat!")

st.markdown("""
**Modele antrenate și ce ne-au arătat:**

- **Regresia Liniară** — relația directă dintre features și profit, interpretabilă
  prin coeficienți. Bună pentru a înțelege *direcția* efectelor.

- **Random Forest** — model mai puternic, captează relații non-liniare.
  Feature importance ne arată ce variabile *contează cel mai mult* pentru profit.

- **Regresia Logistică** — clasificarea comenzilor în segmente de profitabilitate.
  Utilă operațional: putem prioritiza comenzile/produsele cu potențial High Profit.

- **PCA + Clusterizare Ierarhică** — segmentarea nesupervizată a produselor în:
  - **PC1 (Performanță Comercială):** separă produsele scumpe/profitabile de cele ieftine
  - **PC2 (Engagement & Reduceri):** separă produsele populare/cu discount de cele de nișă

**Concluzie:** Preprocesarea (Pagina 2) influențează direct toate aceste modele.
Același algoritm, date diferite = rezultate diferite. Decizia de a elimina outlieri,
de a folosi OHE vs Label Encoding sau de a scala datele nu este neutră.
""")

st.success(" Felicitări! Ai parcurs întregul pipeline de la date brute la modele ML interpretabile.")
