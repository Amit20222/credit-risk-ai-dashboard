# AI Credit Risk Assessment Dashboard

A runnable Streamlit + scikit-learn application with:
- CSV upload and dataset inspection
- Training and comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Holdout metrics: ROC-AUC, accuracy, precision, recall, F1 and Brier score
- Confusion matrix, probability distribution and permutation feature importance
- Individual applicant scoring using the trained model
- Rule-based credit-risk Q&A; optional generative assistant via OpenAI API
- Data-quality and responsible-AI checklist
- CSV metrics and model export

## Run locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. To access from another device on the same network, run:

```bash
streamlit run app.py --server.address 0.0.0.0
```

The computer hosting the app must remain running. For public access, deploy to Streamlit Community Cloud, Render, or another hosting platform. Do not expose real personal or financial data on a public deployment without proper access controls.

## Optional generative AI

The app works without an API key, with a rule-based assistant. To enable generative answers, set an OpenAI API key in your environment:

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

API usage may incur charges. Never put API keys in source code or commit them to a public repository.

## Train on your own CSV

1. Choose **Upload my CSV** in the sidebar.
2. Upload a CSV containing one binary target column and one or more predictor columns.
3. Go to **Train & Evaluate**, select the target column representing default/bad outcome, choose model(s), and click **Train model(s)**.
4. Review the holdout metrics and model diagnostics.
5. Go to **Score an Applicant** and enter feature values using the same names, meanings and units as the training data.

### Target label convention
For numeric binary targets, the app treats the larger value as default/bad outcome (e.g. 1=default, 0=non-default). Verify that convention for your dataset. Common text labels are mapped automatically; unusual labels require you to choose the positive/default class.

## Important limitations

- The built-in dataset is **synthetic**, generated only so the app can run immediately without external data access. Results are not evidence of real-world credit performance.
- This app is a working ML workflow, not a validated production lending system. The default estimate is a model score, not a guaranteed probability.
- Before real use, obtain representative and lawfully collected data, establish the target and observation window, perform leakage checks, probability calibration, time-based validation, fairness/subgroup testing, security/privacy review, model risk approval, monitoring, and human decision controls.
- Do not use the built-in synthetic data or unvalidated model outputs to approve/decline real applicants.


## API key setup (Windows, local only)

The app reads `OPENAI_API_KEY` from the environment first, then `.streamlit/secrets.toml`.

1. Create `.streamlit/secrets.toml` inside this project folder.
2. Add this line, replacing the placeholder with your new key:
   `OPENAI_API_KEY = "YOUR_NEW_API_KEY"`
3. Save the file and restart Streamlit.

Never share the key or commit `secrets.toml`. API use may incur charges. If a key has appeared in a screenshot or chat, revoke it and create a new one.


## Free local AI assistant (Ollama)

The AI Assistant uses a local Ollama service and the `llama3.2:3b` model; it does not need an OpenAI API key or paid API credits. Install Ollama from https://ollama.com/download/windows, then in a terminal run `ollama run llama3.2:3b`. Keep Ollama running while using the dashboard. The first model download requires internet and disk space; after downloading, inference runs locally and has no provider API-credit quota. Performance depends on your hardware.
