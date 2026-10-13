import os, io, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Credit Risk AI | Underwriting Workbench", page_icon="🏦", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.25rem;padding-bottom:2rem}
[data-testid="stMetric"]{background:#f6f8fc;border:1px solid #e2e8f0;padding:14px;border-radius:12px;color:#111827}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricValue"],[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:#111827 !important}
[data-testid="stMetric"] [data-testid="stMetricLabel"]{color:#374151 !important}
section[data-testid="stSidebar"]{background:#10284b}
section[data-testid="stSidebar"] *{color:#f8fafc}
.small-muted{color:#64748b;font-size:.86rem}
div[data-testid="stExpander"]{border:1px solid #e2e8f0;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("🏦 AI Credit Risk Assessment Dashboard")
st.caption("Model training · Application scoring · Explainability · Monitoring · AI assistant")

DATA_PATH = Path(__file__).parent / "data" / "credit_applications.csv"
TARGET_CANDIDATES = ["default", "loan_default", "default_payment_next_month", "target", "bad_loan", "risk_label"]
ID_COLS = {"id", "customer_id", "application_id", "name", "full_name", "email", "phone", "address", "customer_name"}

def make_sample_data(n=4500, seed=42):
    """Generate explicitly synthetic teaching data; not suitable for real lending decisions."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 71, n)
    income = np.clip(rng.lognormal(np.log(65000), .55, n), 12000, 450000).round(0)
    loan = np.clip(income * rng.uniform(2, 16, n), 25000, 2500000).round(0)
    tenure = rng.choice([6, 12, 18, 24, 36, 48, 60], n, p=[.04,.10,.06,.22,.30,.12,.16])
    emi = loan / tenure * rng.uniform(1.03, 1.18, n)
    existing_emi = np.clip(income * rng.beta(1.6, 5.0, n), 0, income*.8)
    utilization = np.clip(rng.beta(2.1, 3.8, n)*100, 0, 100)
    late_payments = np.clip(rng.poisson(.55, n), 0, 8)
    employment = rng.choice(["Salaried", "Self-employed", "Contract", "Other"], n, p=[.58,.25,.10,.07])
    years_employed = np.clip(rng.gamma(2.1, 2.2, n), 0, 35).round(1)
    credit_history = np.clip(rng.normal(7.5, 4.1, n), .2, 30).round(1)
    dti = (existing_emi + emi) / np.maximum(income, 1)
    logit = -3.15 + 3.4*dti + 0.022*np.maximum(utilization-35,0) + .43*late_payments \
            - .075*np.minimum(credit_history,15) - .035*np.minimum(years_employed,10) \
            + .25*(employment=="Contract") + .20*(employment=="Other") \
            + .00000035*loan - .000004*np.minimum(income,150000)
    p = 1/(1+np.exp(-logit))
    default = rng.binomial(1, p)
    df = pd.DataFrame({
        "age":age, "monthly_income":income, "loan_amount":loan, "tenure_months":tenure,
        "existing_monthly_emi":existing_emi.round(0), "credit_utilization_pct":utilization.round(1),
        "late_payments_12m":late_payments, "employment_type":employment,
        "years_employed":years_employed, "credit_history_years":credit_history,
        "default":default
    })
    return df

@st.cache_data(show_spinner="Loading the prepared UCI credit-card default dataset...")
def load_uci_credit_data():
    """Load the validated local CSV derived from the official UCI dataset."""
    csv_path = Path(__file__).parent / "data" / "credit_card_default_uci.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put credit_card_default_uci.csv in the project's data folder."
        )
    df = pd.read_csv(csv_path)
    if "default" not in df.columns:
        raise ValueError("Expected target column 'default' is missing from the UCI CSV.")
    df["default"] = pd.to_numeric(df["default"], errors="coerce")
    df = df.dropna(subset=["default"]).copy()
    if not set(df["default"].unique()).issubset({0, 1}) or df["default"].nunique() != 2:
        raise ValueError("Target column 'default' must contain both binary values 0 and 1.")
    df["default"] = df["default"].astype(int)
    if len(df) != 30000 or len(df.columns) != 25:
        st.warning(f"Loaded UCI dataset has {len(df):,} rows and {len(df.columns)} columns; expected 30,000 rows and 25 columns.")
    return df

def find_target(df, chosen=None):
    if chosen and chosen in df.columns: return chosen
    for c in TARGET_CANDIDATES:
        if c in df.columns: return c
    return None

def normalize_target(s):
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(pd.Series(s.dropna().unique()).tolist())
        if len(vals) == 2:
            # Convention: 1 means default/bad; 0 means non-default/good.
            return (s == max(vals)).astype(int), "Numeric binary target; largest value treated as default/bad (verify this convention)."
    mapping = {
        "yes":1, "y":1, "true":1, "default":1, "bad":1, "late":1, "1":1, "high risk":1, "high-risk":1,
        "no":0, "n":0, "false":0, "non-default":0, "non default":0, "good":0, "paid":0, "0":0, "low risk":0, "low-risk":0
    }
    vals = s.astype(str).str.strip().str.lower()
    mapped = vals.map(mapping)
    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int), "Text target mapped to 0=non-default and 1=default."
    unique = list(vals.dropna().unique())
    if len(unique) == 2:
        st.warning("Target is binary text but labels are not standard. Select a positive/default label.")
        pos = st.selectbox("Which target label represents default / bad outcome?", unique, key="positive_label")
        return (vals == pos).astype(int), f"User selected {pos!r} as default/bad outcome."
    raise ValueError("Target must have exactly two classes (default vs non-default).")

def make_model(name, numeric, categorical):
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    if name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    elif name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced",
                                           n_jobs=-1, random_state=42, max_features="sqrt")
    else:
        estimator = HistGradientBoostingClassifier(max_iter=180, learning_rate=.07, l2_regularization=1.0,
                                                    random_state=42)
        # HGB does not accept sparse matrices; use dense preprocessing for this model.
        prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                    ("scale", StandardScaler())]), numeric),
                                  ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)],
                                 sparse_threshold=0)
    return Pipeline([("preprocess", prep), ("model", estimator)])

def train_models(df, target, model_name, test_size, seed):
    y, target_note = normalize_target(df[target])
    X = df.drop(columns=[target]).copy()
    # Drop identifiers / free text fields which should not be model features.
    drop = [c for c in X.columns if c.lower().strip() in ID_COLS or X[c].dtype == "object" and X[c].nunique() > 80]
    X = X.drop(columns=drop, errors="ignore")
    # Convert numeric-looking text columns where possible.
    for c in X.columns:
        if X[c].dtype == "object":
            converted = pd.to_numeric(X[c], errors="coerce")
            if converted.notna().mean() > .85: X[c] = converted
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    if len(X.columns) == 0: raise ValueError("No usable predictor columns after removing identifiers.")
    if y.nunique() != 2: raise ValueError("Target must have two classes.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    names = [model_name] if model_name != "Compare all three" else ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    fitted, rows = {}, []
    for name in names:
        model = make_model(name, numeric, categorical)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:,1]
        pred = (proba >= .5).astype(int)
        metrics = {
            "Model":name, "ROC-AUC":roc_auc_score(yte, proba) if yte.nunique()==2 else np.nan,
            "Accuracy":accuracy_score(yte,pred), "Precision":precision_score(yte,pred,zero_division=0),
            "Recall":recall_score(yte,pred,zero_division=0), "F1":f1_score(yte,pred,zero_division=0),
            "Brier score":brier_score_loss(yte,proba), "Test rows":len(yte)
        }
        fitted[name] = {"pipeline":model, "features":X.columns.tolist(), "numeric":numeric,
                        "categorical":categorical, "X_test":Xte, "y_test":yte, "proba":proba, "pred":pred}
        rows.append(metrics)
    return fitted, pd.DataFrame(rows), target_note, drop, (Xtr, Xte, ytr, yte)

if "dataset" not in st.session_state:
    st.session_state.dataset = make_sample_data()
if "trained" not in st.session_state:
    st.session_state.trained = None

with st.sidebar:
    st.header("Workbench")
    page = st.radio("Navigate", ["Overview", "Train & Evaluate", "Score an Applicant", "AI Assistant", "Data & Governance"])
    st.divider()
    st.caption("DATA SOURCE")
    data_source = st.radio(
        "Choose dataset",
        ["Official UCI real dataset", "Upload my CSV", "Synthetic demo (optional)"],
        index=0,
        help="Default source is the public UCI Default of Credit Card Clients dataset. Synthetic data is optional."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) if data_source == "Upload my CSV" else None

    if data_source != st.session_state.get("last_source"):
        try:
            if data_source == "Official UCI real dataset":
                st.session_state.dataset = load_uci_credit_data()
                st.session_state.dataset_source_label = "UCI Machine Learning Repository — Default of Credit Card Clients (ID 350)"
                st.session_state.trained = None
            elif data_source == "Synthetic demo (optional)":
                st.session_state.dataset = make_sample_data()
                st.session_state.dataset_source_label = "Synthetic demonstration data — not real customer records"
                st.session_state.trained = None
            st.session_state.last_source = data_source
        except Exception as e:
            st.error(f"Could not load the selected dataset: {e}")
            if data_source == "Official UCI real dataset":
                st.info("Check the app's internet connection and try again. Synthetic data will not be silently presented as real.")

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            if not uploaded_df.empty:
                st.session_state.dataset = uploaded_df
                st.session_state.dataset_source_label = f"User-uploaded CSV ({len(uploaded_df):,} rows)"
                st.session_state.trained = None
                st.session_state.last_source = data_source
                st.success(f"Loaded {len(uploaded_df):,} rows")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.caption("The UCI benchmark contains historical credit-card data from Taiwan, not current Indian bank data. Use properly licensed and representative data for real-world research.")

df = st.session_state.dataset.copy()
st.session_state.dataset = df
if df.empty:
    st.error("Dataset is empty. Upload a valid CSV.")
    st.stop()

if page == "Overview":
    st.subheader("Portfolio overview")
    st.info(f"**Current data source:** {st.session_state.get('dataset_source_label', 'Dataset source not specified')}. This is an educational research prototype; model outputs are not suitable for real lending decisions.")
    target_guess = find_target(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Input features", max(0,len(df.columns)-(1 if target_guess else 0)))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Candidate target", target_guess or "Select in Train")
    left,right = st.columns([1.3,1])
    with left:
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Data quality snapshot")
        quality = pd.DataFrame({"Column":df.columns, "Missing %":(df.isna().mean()*100).round(1).values,
                                "Unique values":[df[c].nunique(dropna=True) for c in df.columns],
                                "Type":[str(df[c].dtype) for c in df.columns]})
        st.dataframe(quality, use_container_width=True, hide_index=True, height=420)
    if target_guess:
        st.markdown("#### Outcome balance")
        vc = df[target_guess].astype(str).value_counts().rename_axis("Outcome").reset_index(name="Rows")
        st.plotly_chart(px.bar(vc, x="Outcome", y="Rows", color="Outcome", title="Target label counts"), use_container_width=True)

elif page == "Train & Evaluate":
    st.subheader("Train and evaluate a model")
    st.write("Select the target column representing whether a borrower defaulted. The app uses a stratified holdout split and reports metrics on unseen test rows.")
    default_target = find_target(df)
    target = st.selectbox("Target column (default / bad outcome)", df.columns.tolist(),
                          index=df.columns.get_loc(default_target) if default_target in df.columns else len(df.columns)-1)
    model_choice = st.selectbox("Model", ["Compare all three", "Logistic Regression", "Random Forest", "Gradient Boosting"])
    a,b,c = st.columns(3)
    test_size = a.slider("Test-set fraction", .15, .40, .25, .05)
    seed = b.number_input("Random seed", 1, 99999, 42)
    threshold = c.slider("Default-risk threshold", .10, .90, .50, .05)
    if st.button("Train model(s)", type="primary", use_container_width=True):
        with st.spinner("Training pipelines and evaluating holdout predictions..."):
            try:
                fitted, metrics, target_note, dropped, split = train_models(df, target, model_choice, test_size, int(seed))
                st.session_state.trained = {"fitted":fitted,"metrics":metrics,"target":target,"target_note":target_note,
                                            "dropped":dropped,"threshold":threshold,"model_choice":model_choice}
                st.success("Training complete. Metrics below are calculated on the holdout test split.")
            except Exception as e:
                st.error(f"Training failed: {e}")
    trained = st.session_state.trained
    if trained:
        st.markdown("#### Holdout metrics")
        st.caption(trained["target_note"])
        st.dataframe(trained["metrics"].style.format({c:"{:.3f}" for c in ["ROC-AUC","Accuracy","Precision","Recall","F1","Brier score"]}),
                     use_container_width=True, hide_index=True)
        best_name = trained["metrics"].sort_values("ROC-AUC",ascending=False).iloc[0]["Model"]
        st.info(f"Highest holdout ROC-AUC in this run: **{best_name}**. This alone does not establish production suitability.")
        model_name = st.selectbox("Inspect model", list(trained["fitted"].keys()), index=list(trained["fitted"].keys()).index(best_name))
        item = trained["fitted"][model_name]
        preds = (item["proba"] >= trained["threshold"]).astype(int)
        cm = confusion_matrix(item["y_test"], preds, labels=[0,1])
        cc1,cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted non-default","Predicted default"],
                                      y=["Actual non-default","Actual default"], title="Confusion matrix", color_continuous_scale="Blues"), use_container_width=True)
        with cc2:
            st.plotly_chart(px.histogram(pd.DataFrame({"Predicted default probability":item["proba"],
                                                       "Actual outcome":item["y_test"].map({0:"Non-default",1:"Default"}).values}),
                                         x="Predicted default probability", color="Actual outcome", nbins=25,
                                         barmode="overlay", opacity=.7, title="Holdout probability distribution"), use_container_width=True)
        st.markdown("#### Global feature importance (permutation, test set)")
        with st.spinner("Calculating permutation importance..."):
            try:
                pi = permutation_importance(item["pipeline"], item["X_test"], item["y_test"], n_repeats=3,
                                            random_state=42, scoring="roc_auc", n_jobs=-1)
                imp = pd.DataFrame({"Feature":item["X_test"].columns,"Importance":pi.importances_mean}).sort_values("Importance",ascending=True).tail(15)
                st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Permutation importance (ROC-AUC decrease)"), use_container_width=True)
            except Exception as e: st.warning(f"Feature importance unavailable for this model: {e}")
        st.download_button("Download model metrics (CSV)", trained["metrics"].to_csv(index=False), "model_metrics.csv", "text/csv")
        st.download_button("Download fitted model (pickle)", pickle.dumps(item["pipeline"]), "credit_risk_model.pkl", "application/octet-stream")
        st.warning("Model pickle files should only be loaded from trusted sources. Validate calibration, stability, fairness, and data leakage before any production use.")

elif page == "Score an Applicant":
    st.subheader("Score an individual application")
    trained = st.session_state.trained
    if not trained:
        st.warning("Train a model first in **Train & Evaluate**. Then return here to score an applicant.")
    else:
        model_name = st.selectbox("Trained model", list(trained["fitted"].keys()))
        item = trained["fitted"][model_name]
        feat = item["features"]
        st.caption("Enter values matching the columns and units used in the training dataset. Unknown categories are handled by the pipeline; missing values are imputed.")
        input_values = {}
        cols = st.columns(3)
        for i, f in enumerate(feat):
            series = df[f]
            with cols[i%3]:
                if pd.api.types.is_numeric_dtype(series):
                    med = float(series.median()) if series.notna().any() else 0.0
                    input_values[f] = st.number_input(f, value=med, key=f"score_{f}")
                else:
                    opts = sorted(series.dropna().astype(str).unique().tolist())
                    input_values[f] = st.selectbox(f, opts or ["Unknown"], key=f"score_{f}")
        threshold = st.slider("Decision-support threshold for flagging default risk", .10, .90, float(trained["threshold"]), .05)
        if st.button("Score application", type="primary"):
            xnew = pd.DataFrame([input_values], columns=feat)
            probability = float(item["pipeline"].predict_proba(xnew)[0,1])
            flagged = probability >= threshold
            m1,m2,m3 = st.columns(3)
            m1.metric("Estimated default probability", f"{probability:.1%}")
            m2.metric("Risk band", "Higher" if probability >= .45 else ("Medium" if probability >= .20 else "Lower"))
            m3.metric("Threshold flag", "Review required" if flagged else "Below threshold")
            st.progress(probability)
            st.write("**Interpretation:**", "This application crosses the selected review threshold. Escalate for human review; do not auto-decline." if flagged else "This application is below the selected flagging threshold. This is not an approval decision.")
            st.json({"model":model_name,"estimated_default_probability":round(probability,5),"threshold":threshold,
                     "flagged_for_review":bool(flagged),"decision":"human_review_required","disclaimer":"Model score is decision support only."})

elif page == "AI Assistant":
    st.subheader("Credit risk assistant")
    st.write("On the deployed website, the assistant uses Groq when you configure a key in Streamlit Secrets. Locally, without a Groq key, it can use your Ollama model.")
    prompt = st.text_area("Ask a credit risk question", placeholder="Explain ROC-AUC vs recall, or how to validate a default model.")
    if st.button("Ask AI", type="primary") and prompt.strip():
        system_message = (
            "You are a responsible credit-risk analytics assistant for an educational prototype. "
            "Explain methods and limitations. Never claim a score guarantees default or recommend automated denial. "
            "Emphasize human review, data protection, fairness, validation, and applicable law. "
            "State that the UCI dataset is historical credit-card data from Taiwan, not current Indian bank data."
        )
        groq_key = os.environ.get("GROQ_API_KEY")
        try:
            if not groq_key:
                try:
                    groq_key = st.secrets.get("GROQ_API_KEY", None)
                except Exception:
                    groq_key = None
            if groq_key:
                from groq import Groq
                with st.spinner("Generating a response with Groq..."):
                    client = Groq(api_key=groq_key)
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt.strip()}],
                        temperature=0.3,
                        max_tokens=900
                    )
                st.markdown(completion.choices[0].message.content or "No response text returned.")
            else:
                import urllib.request
                payload = {"model": "llama3.2:3b", "stream": False, "messages": [
                    {"role": "system", "content": system_message}, {"role": "user", "content": prompt.strip()}]}
                req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode("utf-8"),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with st.spinner("Generating a response with local Ollama..."):
                    with urllib.request.urlopen(req, timeout=300) as response:
                        result = json.loads(response.read().decode("utf-8"))
                st.markdown(result.get("message", {}).get("content", "No response text returned."))
        except Exception as e:
            st.error(f"AI request failed: {e}")
            st.info("For cloud deployment, add GROQ_API_KEY in Streamlit app Secrets. For local Ollama, make sure Ollama is running and llama3.2:3b is installed.")

    q = st.selectbox("Quick question", ["What does ROC-AUC mean?", "Why is recall important in default detection?", "What should be checked before deployment?", "How should a flagged applicant be handled?"])
    if st.button("Show answer"):
        answers = {
          "What does ROC-AUC mean?": "ROC-AUC measures how well a model ranks default cases above non-default cases across thresholds. It does not measure calibration, fairness, or business value by itself.",
          "Why is recall important in default detection?": "Recall is the share of actual defaults detected by the model. Low recall means more defaults are missed. Increasing recall can also increase false positives, so thresholds require cost and fairness analysis.",
          "What should be checked before deployment?": "Validate on representative and preferably time-based holdouts; check leakage, probability calibration, subgroup performance, drift, privacy, explainability, security, adverse-action reasons, monitoring, and human oversight.",
          "How should a flagged applicant be handled?": "Flag for review, verify source data, assess affordability and policy requirements, provide meaningful explanations, and have an authorized human make and document the decision. A model score should not be the sole basis for denial."
        }
        st.markdown(answers[q])

elif page == "Data & Governance":
    st.subheader("Data quality & responsible model governance")
    st.markdown("#### Missingness by column")
    miss = (df.isna().mean()*100).sort_values(ascending=False).rename("Missing %").reset_index().rename(columns={"index": "Column"})
    miss["Missing cells"] = df.isna().sum().reindex(miss["Column"]).to_numpy()
    if int(miss["Missing cells"].sum()) == 0:
        st.success(f"No missing values found in this dataset ({len(df):,} rows × {len(df.columns):,} columns).")
        st.caption("The chart is empty because every column has 0% missingness. This is expected for the selected UCI dataset.")
    else:
        miss_nonzero = miss[miss["Missing %"] > 0].sort_values("Missing %", ascending=False).head(25)
        st.plotly_chart(px.bar(miss_nonzero, x="Column", y="Missing %", title="Columns with missing values (%)", text_auto=".1f"), use_container_width=True)
    with st.expander("View missingness table for all columns"):
        st.dataframe(miss[["Column", "Missing cells", "Missing %"]].style.format({"Missing %": "{:.2f}%"}), use_container_width=True, hide_index=True)
    st.markdown("#### Duplicate rows")
    duplicate_count = int(df.duplicated().sum())
    st.metric("Duplicate rows", f"{duplicate_count:,}")
    if duplicate_count == 0:
        st.success("No duplicate rows detected.")
    st.markdown("#### Governance checklist")
    checks = [
      "Confirm lawful basis, consent/permissions, minimization and retention for all personal data.",
      "Define the default outcome, observation window and label-maturity period before training.",
      "Remove identifiers and audit leakage-prone variables; split data by time where appropriate.",
      "Evaluate discrimination, calibration, false-positive/false-negative costs and subgroup performance.",
      "Document model version, dataset lineage, thresholds, approvals, limitations and audit trail.",
      "Set drift monitoring, periodic revalidation, incident handling and human-review procedures."
    ]
    for c in checks: st.checkbox(c, key="gov_"+str(checks.index(c)))
    st.warning("A software demo is not a regulatory approval. Real lending deployment requires appropriate legal, risk, security, validation, fairness and governance reviews.")

st.divider()
st.caption(f"Educational prototype · Data source: {st.session_state.get('dataset_source_label', 'not specified')} · Model scores are not suitable for real lending decisions.")
