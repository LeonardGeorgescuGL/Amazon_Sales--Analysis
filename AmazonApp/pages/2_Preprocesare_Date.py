import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Configurare pagină și CSS
st.set_page_config(page_title="Preprocesare Date", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:      #0B1120;
    --surface: #1E293B;
    --border:  #334155;
    --accent:  #38BDF8;
    --accent2: #A78BFA;
    --text:    #F8FAFC;
    --muted:   #94A3B8;
    --success: #34D399;
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
    border-left: 4px solid var(--accent);
    padding: 28px 36px;
    background: var(--surface);
    border-radius: var(--radius);
    margin-bottom: 36px;
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.4);
}
.page-header h1 { font-size: 38px !important; font-weight: 800; margin: 0 !important; line-height: 1.1; }
.sec-header {
    font-family: var(--mono); font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); margin: 40px 0 16px 0; padding-bottom: 10px; border-bottom: 1px solid var(--border);
}
.tip {
    background: rgba(167, 139, 250, 0.05);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: var(--radius);
    padding: 20px;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
}
.tip strong { color: var(--accent2); }
.tip code { background: #0F172A; padding: 2px 6px; border-radius: 4px;
    font-family: var(--mono); font-size: 12px; color: var(--accent); }
.stat-row { display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap; }
.stat-pill {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    padding: 16px 24px; flex: 1; min-width: 140px;
}
.stat-pill .sv { font-family: var(--display); font-size: 22px; font-weight: 800;
    color: var(--text); line-height: 1.2; white-space: nowrap; }
.stat-pill .sl { font-family: var(--body); font-size: 13px; color: var(--accent);
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="page-header">
    <h1> Preprocesare Date Amazon</h1>
    <p style="color:#94A3B8; margin-top:10px;">
        Curățăm, transformăm și pregătim datele pentru modelele ML.
        Alegerile de aici influențează direct performanța din pagina următoare.
    </p>
</div>
""", unsafe_allow_html=True)

# Încărcare date
if "df" not in st.session_state:
    st.warning("⚠️ Dataset-ul nu este încărcat. Mergi la **Home** și încarcă fișierul CSV mai întâi.")
    st.stop()

df_original = st.session_state["df"].copy()

st.markdown("""
### De ce contează preprocesarea?

Datele reale de e-commerce conțin **valori lipsă**, **outlieri** (comenzi cu prețuri extreme),
**variabile categoriale** (categorie produs, regiune, metodă de plată) și **scale diferite**
(prețul în USD vs. discount-ul în procente 0–100).

Fiecare decizie de preprocesare — cum tratam outlierii, cum codifica, categoriile,
dacă scalam sau nu — modifică ce „vede" modelul ML. Pe această pagină
iei aceste decizii conștient și le salvezi pentru pagina de Machine Learning.
""")

st.info("""
**Flux:** Pasul 1 → Pasul 2 → Pasul 3 → Pasul 4.
Fiecare pas primește datele ieșite din pasul anterior.
Urmărim metricile după fiecare transformare.
""")

# PASUL 1 — Valori lipsă
st.markdown("---")
st.header("Pasul 1 — Verificarea și tratarea valorilor lipsă")

st.markdown("""
`st.progress` afișează o bară de progres între `0.0` și `1.0` — util pentru a
vizualiza **completitudinea** fiecărei coloane dintr-o singură privire.

```python
st.progress(0.95, text="95% valori prezente")
```
""")

col_stanga, col_dreapta = st.columns(2)

with col_stanga:
    st.subheader("Situația valorilor lipsă")
    lipsa = df_original.isnull().sum().reset_index()
    lipsa.columns = ["Coloană", "Valori lipsă"]
    lipsa["Procent (%)"] = (lipsa["Valori lipsă"] / len(df_original) * 100).round(2)
    lipsa_reala = lipsa[lipsa["Valori lipsă"] > 0]

    if lipsa_reala.empty:
        st.success(" Dataset-ul nu conține valori lipsă! Totuși, putem simula tratarea lor pentru a înțelege metodele.")
        # Simulăm 5% valori lipsă în 'rating' și 'discount_percent' pentru exercițiu
        df_sim = df_original.copy()
        np.random.seed(42)
        idx_rating = np.random.choice(df_sim.index, size=int(len(df_sim) * 0.05), replace=False)
        idx_disc   = np.random.choice(df_sim.index, size=int(len(df_sim) * 0.03), replace=False)
        df_sim.loc[idx_rating, "rating"] = np.nan
        df_sim.loc[idx_disc, "discount_percent"] = np.nan
        df_lucru = df_sim
        lipsa_afisaj = df_sim.isnull().sum().reset_index()
        lipsa_afisaj.columns = ["Coloană", "Valori lipsă"]
        lipsa_afisaj["Procent (%)"] = (lipsa_afisaj["Valori lipsă"] / len(df_sim) * 100).round(2)
        lipsa_afisaj = lipsa_afisaj[lipsa_afisaj["Valori lipsă"] > 0]
        st.info("Am simulat valori lipsă în `rating` (5%) și `discount_percent` (3%) pentru demonstrație.")
    else:
        df_lucru = df_original.copy()
        lipsa_afisaj = lipsa_reala
        st.dataframe(lipsa_afisaj, use_container_width=True, hide_index=True)

with col_dreapta:
    st.subheader("Completitudine per coloană")
    for col in df_original.columns:
        valori_prezente = df_original[col].notnull().sum() / len(df_original)
        st.markdown(f"**{col}**")
        st.progress(float(valori_prezente), text=f"{valori_prezente*100:.1f}% completă")

if "df_lucru" not in dir():
    df_lucru = df_original.copy()

# Afișăm valorile lipsă simulate dacă există
if "lipsa_afisaj" in dir() and not lipsa_afisaj.empty:
    st.subheader("Alege metoda de imputare")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**`rating`** — evaluare produs (0–5)")
        metoda_rating = st.selectbox(
            "Metodă pentru rating:",
            ["Mean", "Median", "KNN Imputer (bazat pe price, discount_percent)"],
            key="met_rating"
        )
    with col2:
        st.markdown("**`discount_percent`** — procentul de reducere aplicat")
        metoda_disc = st.selectbox(
            "Metodă pentru discount_percent:",
            ["Median", "Mean", "Completează cu 0 (fără reducere)", "KNN Imputer"],
            key="met_disc"
        )

    with st.expander(" Când folosești fiecare metodă?"):
        st.markdown("""
        | Metodă | Când e potrivită | Risc |
        |---|---|---|
        | **Mean** | Distribuție simetrică, fără outlieri | Sensibilă la valori extreme |
        | **Median** | Distribuție skewed (asimetrică) | Poate subreprezenta valorile rare |
        | **Completează cu 0** | Absența valorii are sens logic | Distorsionează media coloanei |
        | **KNN Imputer** | Când există corelații între coloane | Lent pe dataset-uri mari |

        **Context Amazon:** `rating`-ul mediu al unui produs tinde să fie corelat
        cu prețul și categoria — KNN poate fi mai precis decât o simplă mediană globală.
        """)

    # Aplicare imputare
    df_pas1 = df_lucru.copy()

    if metoda_rating == "Mean":
        df_pas1["rating"] = df_pas1["rating"].fillna(df_pas1["rating"].mean())
    elif metoda_rating == "Median":
        df_pas1["rating"] = df_pas1["rating"].fillna(df_pas1["rating"].median())
    elif "KNN" in metoda_rating:
        knn = KNNImputer(n_neighbors=5)
        num_cols = ["rating", "price", "discount_percent", "quantity_sold"]
        num_cols = [c for c in num_cols if c in df_pas1.columns]
        df_pas1[num_cols] = knn.fit_transform(df_pas1[num_cols])

    if metoda_disc == "Mean":
        df_pas1["discount_percent"] = df_pas1["discount_percent"].fillna(df_pas1["discount_percent"].mean())
    elif metoda_disc == "Median":
        df_pas1["discount_percent"] = df_pas1["discount_percent"].fillna(df_pas1["discount_percent"].median())
    elif metoda_disc == "Completează cu 0 (fără reducere)":
        df_pas1["discount_percent"] = df_pas1["discount_percent"].fillna(0)
    elif "KNN" in metoda_disc:
        knn2 = KNNImputer(n_neighbors=5)
        cols2 = ["discount_percent", "price", "quantity_sold"]
        cols2 = [c for c in cols2 if c in df_pas1.columns]
        df_pas1[cols2] = knn2.fit_transform(df_pas1[cols2])

    lipsa_dupa = df_pas1.isnull().sum().sum()
    if lipsa_dupa == 0:
        st.success(f" Toate valorile lipsă au fost tratate. Dataset complet: **{len(df_pas1)} rânduri**.")
    else:
        st.warning(f"Au rămas **{lipsa_dupa}** valori lipsă.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rating mediu — înainte", f"{df_lucru['rating'].mean():.3f}")
    col2.metric("Rating mediu — după", f"{df_pas1['rating'].mean():.3f}")
    col3.metric("Diferență", f"{df_pas1['rating'].mean() - df_lucru['rating'].mean():.4f}", delta_color="off")
else:
    df_pas1 = df_original.copy()
    metoda_rating = "Median"
    metoda_disc = "Median"


# PASUL 2 — Outlieri
st.markdown("---")
st.header("Pasul 2 — Detectarea și tratarea outlierilor")

st.markdown("""
În datele Amazon, outlierii pot apărea în mai multe forme:
- **Comenzi cu prețuri extreme** (produse de lux sau erori de înregistrare)
- **Discount-uri de 0% sau 100%** (promoții speciale sau produse gratuite)
- **Profit negativ** (returnări sau costuri neașteptate)

Folosim **metoda IQR** (Interquartile Range) — cea mai robustă pentru date skewed:
```python
Q1  = df["col"].quantile(0.25)
Q3  = df["col"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```
""")

col_viz, col_stats = st.columns([2, 1])

with col_viz:
    col_outlier = st.selectbox(
        "Selectează coloana pentru analiză outlieri:",
        ["price", "discount_percent", "total_revenue", "profit", "quantity_sold"],
        key="col_out_select"
    )
    fig_box = px.box(
        df_pas1, x="product_category", y=col_outlier,
        title=f"Boxplot {col_outlier} pe categorii de produse",
        color="product_category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_box.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC",
        showlegend=False
    )
    fig_box.update_xaxes(tickangle=30)
    st.plotly_chart(fig_box, use_container_width=True)

with col_stats:
    Q1_  = df_pas1[col_outlier].quantile(0.25)
    Q3_  = df_pas1[col_outlier].quantile(0.75)
    IQR_ = Q3_ - Q1_
    n_out = len(df_pas1[
        (df_pas1[col_outlier] < Q1_ - 1.5 * IQR_) |
        (df_pas1[col_outlier] > Q3_ + 1.5 * IQR_)
    ])
    pct_out = n_out / len(df_pas1) * 100
    st.metric("Q1 (25%)", f"{Q1_:,.2f}")
    st.metric("Q3 (75%)", f"{Q3_:,.2f}")
    st.metric("IQR", f"{IQR_:,.2f}")
    st.metric("Outlieri detectați", f"{n_out} ({pct_out:.1f}%)")

st.subheader("Alege metoda de tratare")
col1, col2 = st.columns(2)

with col1:
    cols_de_tratat = st.multiselect(
        "Coloane pe care aplici tratarea:",
        ["price", "discount_percent", "total_revenue", "profit", "quantity_sold"],
        default=["price", "profit"],
        key="cols_outlier"
    )
    metoda_out = st.radio(
        "Metodă:", ["Păstrează toți outlierii", "Elimină rândurile outlieri", "Capping la percentile"],
        key="met_out"
    )

with col2:
    if metoda_out == "Capping la percentile":
        perc_jos = st.slider("Percentilă inferioară (%)", 0, 10, 1, key="pjos")
        perc_sus = st.slider("Percentilă superioară (%)", 90, 100, 99, key="psus")
    else:
        st.markdown("""
        <div class="tip">
        <strong>Sfat practic:</strong><br>
        Pe datele Amazon, <code>Eliminarea</code> e riscantă —
        poți pierde comenzile cu profit mare (luxury items).<br><br>
        <code>Capping</code> păstrează toate rândurile dar limitează
        valorile extreme, menținând structura distribuției.
        </div>
        """, unsafe_allow_html=True)

df_pas2 = df_pas1.copy()

for col_t in cols_de_tratat:
    if col_t not in df_pas2.columns:
        continue
    if metoda_out == "Elimină rândurile outlieri":
        Q1t = df_pas2[col_t].quantile(0.25)
        Q3t = df_pas2[col_t].quantile(0.75)
        IQRt = Q3t - Q1t
        df_pas2 = df_pas2[
            (df_pas2[col_t] >= Q1t - 1.5 * IQRt) &
            (df_pas2[col_t] <= Q3t + 1.5 * IQRt)
        ]
    elif metoda_out == "Capping la percentile":
        low  = df_pas2[col_t].quantile(perc_jos / 100)
        high = df_pas2[col_t].quantile(perc_sus / 100)
        df_pas2[col_t] = df_pas2[col_t].clip(low, high)

df_pas2 = df_pas2.reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rânduri înainte", f"{len(df_pas1):,}")
col2.metric("Rânduri după", f"{len(df_pas2):,}")
col3.metric("Rânduri eliminate", f"{len(df_pas1) - len(df_pas2):,}")


# PASUL 3 — Encoding variabile categoriale
st.markdown("---")
st.header("Pasul 3 — Encoding variabile categoriale")

st.markdown("""
Dataset-ul Amazon conține **3 variabile categoriale** principale:

| Coloană | Tip | Recomandare |
|---|---|---|
| `product_category` | Nominală (fără ordine) | One-Hot Encoding |
| `customer_region` | Nominală | One-Hot Encoding sau Label |
| `payment_method` | Nominală | One-Hot Encoding sau Label |

**One-Hot Encoding** creează câte o coloană binară per valoare unică —
modelul nu presupune nicio relație ordinală între categorii.

**Label Encoding** atribuie un număr întreg fiecărei categorii.
E mai compact, dar introduce o ordine artificială (ex: 0=Credit Card < 1=Debit Card < 2=PayPal).
""")

col1, col2, col3 = st.columns(3)
with col1:
    enc_categ = st.selectbox(
        "Encoding `product_category`:",
        ["Label Encoding", "One-Hot Encoding"],
        key="enc_categ"
    )
    n_categ = df_pas2["product_category"].nunique()
    st.caption(f"Valori unice: {n_categ} → OHE adaugă {n_categ} coloane")

with col2:
    enc_regiune = st.selectbox(
        "Encoding `customer_region`:",
        ["Label Encoding", "One-Hot Encoding"],
        key="enc_reg"
    )
    n_reg = df_pas2["customer_region"].nunique()
    st.caption(f"Valori unice: {n_reg} → OHE adaugă {n_reg} coloane")

with col3:
    enc_plata = st.selectbox(
        "Encoding `payment_method`:",
        ["Label Encoding", "One-Hot Encoding"],
        key="enc_plata"
    )
    n_plata = df_pas2["payment_method"].nunique()
    st.caption(f"Valori unice: {n_plata} → OHE adaugă {n_plata} coloane")

include_id = st.checkbox(
    "Exclude coloanele ID (`order_id`, `product_id`)?",
    value=True,
    key="excl_id"
)

df_pas3 = df_pas2.copy()
if include_id:
    id_cols = [c for c in ["order_id", "product_id"] if c in df_pas3.columns]
    df_pas3 = df_pas3.drop(columns=id_cols)

le = LabelEncoder()

if enc_categ == "Label Encoding":
    df_pas3["product_category"] = le.fit_transform(df_pas3["product_category"].astype(str))
else:
    df_pas3 = pd.get_dummies(df_pas3, columns=["product_category"], prefix="cat")

if enc_regiune == "Label Encoding":
    le2 = LabelEncoder()
    df_pas3["customer_region"] = le2.fit_transform(df_pas3["customer_region"].astype(str))
else:
    df_pas3 = pd.get_dummies(df_pas3, columns=["customer_region"], prefix="reg")

if enc_plata == "Label Encoding":
    le3 = LabelEncoder()
    df_pas3["payment_method"] = le3.fit_transform(df_pas3["payment_method"].astype(str))
else:
    df_pas3 = pd.get_dummies(df_pas3, columns=["payment_method"], prefix="pay")

# Convertim bool -> int (din get_dummies)
bool_cols = df_pas3.select_dtypes(include="bool").columns
df_pas3[bool_cols] = df_pas3[bool_cols].astype(int)

col1, col2 = st.columns(2)
col1.metric("Coloane înainte de encoding", df_pas2.shape[1])
col2.metric("Coloane după encoding", df_pas3.shape[1])

st.markdown("**Primele 5 rânduri după encoding:**")
st.dataframe(df_pas3.head(5), use_container_width=True)


# PASUL 4 — Scalare
st.markdown("---")
st.header("Pasul 4 — Scalarea caracteristicilor")

st.markdown("""
**De ce scalăm?** Coloanele din dataset-ul Amazon au scale foarte diferite:
- `price`: 0 – 500+ USD
- `discount_percent`: 0 – 100
- `rating`: 1 – 5
- `quantity_sold`: 1 – 1000+

Algoritmii bazați pe **distanță** (KNN, clusterizare ierarhică, PCA) sunt extrem
de sensibili la scale diferite — o coloană cu valori 0–500 va domina una cu valori 1–5.

| Metodă | Formula | Când o folosești |
|---|---|---|
| **StandardScaler** | `(x - μ) / σ` | Distribuții aproximativ normale, SVM, PCA |
| **MinMaxScaler** | `(x - min) / (max - min)` | Rețele neuronale, K-Means, date cu limite clare |
| **Fără scalare** | — | Random Forest, Decision Tree (nu sunt afectate de scale) |
""")

col1, col2 = st.columns([1, 2])
with col1:
    metoda_scal = st.radio(
        "Metodă de scalare:",
        ["Fără scalare", "StandardScaler", "MinMaxScaler"],
        key="met_scal"
    )

    cols_numerice = df_pas3.select_dtypes(include="number").columns.tolist()
    target_exclude = ["total_revenue", "profit"]
    cols_de_scalat = [c for c in cols_numerice if c not in target_exclude]

    st.caption(f"Se vor scala {len(cols_de_scalat)} coloane numerice (excluzând target-urile).")

with col2:
    # Grafic distribuție înainte de scalare
    col_preview = st.selectbox(
        "Previzualizează distribuția:",
        [c for c in ["price", "discount_percent", "rating", "quantity_sold"] if c in df_pas3.columns],
        key="col_prev_scal"
    )
    fig_hist = px.histogram(
        df_pas3, x=col_preview, nbins=40,
        title=f"Distribuția `{col_preview}` — înainte de scalare",
        color_discrete_sequence=["#38BDF8"]
    )
    fig_hist.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

df_pas4 = df_pas3.copy()
scaler_obj = None

if metoda_scal == "StandardScaler":
    scaler_obj = StandardScaler()
    df_pas4[cols_de_scalat] = scaler_obj.fit_transform(df_pas4[cols_de_scalat])
elif metoda_scal == "MinMaxScaler":
    scaler_obj = MinMaxScaler()
    df_pas4[cols_de_scalat] = scaler_obj.fit_transform(df_pas4[cols_de_scalat])

if metoda_scal != "Fără scalare":
    col1, col2, col3 = st.columns(3)
    col_check = cols_de_scalat[0] if cols_de_scalat else None
    if col_check:
        col1.metric(f"`{col_check}` — medie după scalare",
                    f"{df_pas4[col_check].mean():.4f}")
        col2.metric(f"`{col_check}` — std după scalare",
                    f"{df_pas4[col_check].std():.4f}")
        col3.metric("Scalare aplicată", metoda_scal)

    if metoda_scal == "StandardScaler":
        st.success(" StandardScaler: medie ≈ 0, deviație standard ≈ 1 pentru fiecare coloană scalată.")
    else:
        st.success(" MinMaxScaler: toate valorile scalate sunt în intervalul [0, 1].")
else:
    st.info(" Nicio scalare aplicată. Random Forest și arborii de decizie nu au nevoie de scalare.")

# SUMAR + SALVARE
st.markdown("---")
st.header(" Sumar — alegerile tale de preprocesare")

st.markdown("Acestea sunt deciziile luate. Pagina de Machine Learning le va prelua automat.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Pasul 1 — Valori lipsă**")
    st.markdown(f"- `rating`: **{metoda_rating}**")
    st.markdown(f"- `discount_percent`: **{metoda_disc}**")
with col2:
    st.markdown("**Pasul 2 — Outlieri**")
    st.markdown(f"- Metodă: **{metoda_out}**")
    st.markdown(f"- Rânduri rămase: **{len(df_pas2):,}** / {len(df_original):,}")
with col3:
    st.markdown("**Pasul 3 — Encoding**")
    st.markdown(f"- `product_category`: **{enc_categ}**")
    st.markdown(f"- `customer_region`: **{enc_regiune}**")
    st.markdown(f"- `payment_method`: **{enc_plata}**")
    st.markdown(f"- Coloane finale: **{df_pas3.shape[1]}**")
with col4:
    st.markdown("**Pasul 4 — Scalare**")
    st.markdown(f"- Metodă: **{metoda_scal}**")
    st.markdown(f"- Rânduri finale: **{len(df_pas4):,}**")
    st.markdown(f"- Coloane finale: **{df_pas4.shape[1]}**")

# Salvăm în session_state
st.session_state["df_procesat"]          = df_pas4
st.session_state["df_procesat_nescalat"] = df_pas3  # pentru interpretare
st.session_state["df_pas2_amazon"]       = df_pas2  # cu outlieri tratați, fără encoding
st.session_state["alegeri_preprocesare_amazon"] = {
    "metoda_rating":   metoda_rating,
    "metoda_disc":     metoda_disc,
    "metoda_outlieri": metoda_out,
    "enc_categ":       enc_categ,
    "enc_regiune":     enc_regiune,
    "enc_plata":       enc_plata,
    "metoda_scal":     metoda_scal,
    "n_randuri":       len(df_pas4),
    "n_coloane":       df_pas4.shape[1],
}

st.success(" Datele preprocesate au fost salvate. Navighează la **Machine Learning** pentru a antrena modelele.")

st.markdown("**Preview date finale:**")
st.dataframe(df_pas4.head(5), use_container_width=True)
st.download_button(
    " Descarcă datele preprocesate (CSV)",
    data=df_pas4.to_csv(index=False).encode("utf-8"),
    file_name="amazon_procesat.csv",
    mime="text/csv"
)
