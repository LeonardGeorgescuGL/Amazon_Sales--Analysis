# 🛒 Amazon Sales Dashboard — Analysis & Machine Learning

An interactive, multi-page data application built with **Python** and **Streamlit** for analyzing Amazon e-commerce sales performance, from raw data exploration to supervised and unsupervised Machine Learning models.

🎓 **Academic Context:** Developed as a seminar assignment for the course **"Software Packages: Development of Applications for Data Analysis and Machine Learning in Python and SAS"**.

🌐 **Live Demo:** https://amazonsales--analysis-dpwxx4quar2atcxuxkrtkb.streamlit.app/

---

## 📋 Project Structure

```
amazon_app/
│
├── Home.py                    ← Main page (CSV upload & data overview)
├── amazon_sales.csv           ← Raw dataset
├── requirements.txt           ← Deployment dependencies
└── pages/
    ├── 1_Dashboard.py         ← Visual analysis & financial KPIs
    ├── 2_Preprocesare.py      ← Interactive data preprocessing pipeline
    └── 3_Machine_Learning.py  ← Supervised & unsupervised ML models
```

---

## 🚀 Features

### 🏠 Home
- CSV file upload with persistence via `st.session_state`
- Data preview and summary statistics (orders, categories, total revenue, total profit, average rating)

### 📊 Dashboard (Page 1)
- **Dynamic filtering** in real time: customer region, product category, maximum discount, top-rated products toggle (rating ≥ 4.5)
- **Financial KPIs:** total orders, total revenue, total profit, average rating
- **Interactive charts:** profit by region (Plotly — donut chart), payment method distribution (Matplotlib)
- **Filtered data export** as a downloadable CSV report

### 🔧 Data Preprocessing (Page 2)
An interactive 4-step pipeline — choices made here directly affect model performance on Page 3:

| Step | Operation | Widgets |
|------|-----------|---------|
| 1 | Missing value treatment (Mean / Median / KNN Imputer) | `st.progress`, `st.selectbox` |
| 2 | Outlier detection & handling (IQR, Capping, Removal) | `st.radio`, `st.slider`, `px.box` |
| 3 | Categorical encoding (Label Encoding / One-Hot Encoding) | `st.selectbox`, `st.checkbox`, `st.data_editor` |
| 4 | Feature scaling (StandardScaler / MinMaxScaler / None) | `st.radio`, `px.histogram` |

### 🤖 Machine Learning (Page 3)
Four models trained on the preprocessed data, with guided interpretation at each step:

| Model | Type | Target | Metrics |
|-------|------|--------|---------|
| **Linear Regression** | Supervised — Regression | `profit` | R², RMSE, MAE |
| **Random Forest** | Supervised — Regression | `profit` | R², RMSE, Feature Importance |
| **Logistic Regression** | Supervised — Classification | Profit Segment (Low/Med/High) | Accuracy, F1, Confusion Matrix |
| **PCA + Hierarchical Clustering** | Unsupervised | — | Dendrogram, Cluster Profiles |

#### 🔍 PCA Components — interpreted for Amazon Sales data:
- **PC1 — Commercial Performance:** captures variance related to the financial value of orders (`price`, `profit`, `total_revenue`, `discounted_price`) — separates premium from budget products
- **PC2 — Engagement & Discount Sensitivity:** captures product popularity and promotional behavior (`rating`, `review_count`, `quantity_sold`, `discount_percent`) — separates viral/popular products from niche ones

---

## 🛠️ Technologies

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.12 | Backend |
| **Streamlit** | ≥ 1.32 | Interactive web interface |
| **Pandas** | ≥ 2.0 | Data manipulation & cleaning |
| **Plotly** | ≥ 5.18 | Interactive charts |
| **Matplotlib** | ≥ 3.8 | Static charts (dendrogram, histograms) |
| **Scikit-learn** | ≥ 1.4 | ML models, scaling, encoding, PCA |
| **SciPy** | ≥ 1.12 | Hierarchical clustering (linkage, dendrogram) |
| **NumPy** | ≥ 1.26 | Numerical operations |

---

## 📦 Local Installation

```bash
# Clone the repository
git clone https://github.com/<username>/amazon_app.git
cd amazon_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```

Then open your browser at `http://localhost:8501` and upload the `amazon_sales.csv` file.

---

## 📁 Dataset Format (`amazon_sales.csv`)

The application expects a CSV file with **at least** the following columns:

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Unique order identifier |
| `product_id` | string | Unique product identifier |
| `product_category` | string | Product category |
| `price` | float | Original price (USD) |
| `discount_percent` | float | Discount percentage (0–100) |
| `quantity_sold` | int | Units sold |
| `customer_region` | string | Customer region |
| `payment_method` | string | Payment method |
| `rating` | float | Product rating (1–5) |
| `review_count` | int | Number of reviews |
| `discounted_price` | float | Price after discount |
| `total_revenue` | float | Total revenue for the order |
| `profit` | float | Profit for the order |

---

## 👤 Author

Academic project — Faculty of Cybernetics, Statistics and Economic Informatics
