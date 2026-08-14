# SKUNIVO — Public MVP

## 1. Product overview

SKUNIVO is an explainable e-commerce Decision Copilot. It helps merchandising
teams decide which product listings to protect, review, test, maintain, or
deprioritize. The application consumes precomputed outputs; it does not retrain a
model during normal use.

Primary promise: **Turn marketplace signals into explainable product decisions.**

Live app: https://skunivo.streamlit.app/  
Source repository: https://github.com/mhiunguyen/SKUNIVO

Important boundary: this is a decision-support prototype. It does not forecast
transactional demand, estimate causal promotion lift, optimize profit or inventory,
calculate ROAS, automate pricing, or guarantee outcomes.

## 2. Main features

- Premium landing experience with a real recommendation preview
- Executive portfolio overview with normalized market views
- Indonesia-first product prioritization with a simple market selector and collapsed
  advanced filters for shop, category, decision, confidence, score, promotion,
  official-shop, engagement, and local price
- Auditable product explanations with peer percentiles and business-language reasons
- A visible AI-assisted contextual benchmark using shop-grouped out-of-fold predictions
- A human Decision Log for accepting, overriding, or deferring recommendations
- Transparent six-component what-if score simulator
- Methodology, model diagnostics, robustness, limitations, and data roadmap
- Structured usability feedback with local append or public-session download
- English and Vietnamese interface switching with stable internal decision keys
- Eight-page in-app navigation and a three-step, three-minute judge Demo Guide

## 3. Project structure

```text
app.py                         Streamlit entry point and eight page views
app_components/
  data_loader.py               Cached loading, aliases, schema validation
  filters.py                   Filter, sort, active-review, price-label logic
  charts.py                    Chart helpers and normalized summaries
  recommendation_ui.py         Score formula, presets, tiers, guidance
  feedback.py                  Validation and append/session export behavior
  decision_log.py              Human decision record validation and export
  persistence.py               Google Apps Script delivery without secret exposure
  i18n.py                      English/Vietnamese display translations
  styles.py                    Brand and responsive application styling
src/                           Reproducible feature, model, score, and recommendation code
notebooks/                     Discovery, EDA, and decision-intelligence entry notebooks
outputs/                       Precomputed recommendation and evaluation files
outputs/charts/                High-resolution chart assets
proposal/                      Final competition PPTX and PDF
.streamlit/config.toml         Public-safe Streamlit configuration
.streamlit/secrets.toml.example Optional secret template; contains no credentials
tests/                         Formula, data, page-flow, persistence, and UI smoke tests
```

The app resolves every data path relative to `app.py`; no local absolute path is
required at runtime.

## 4. Local installation

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## 5. Local run instructions

From the project root:

```powershell
streamlit run app.py
```

The local app opens at the URL printed by Streamlit, normally
`http://localhost:8501`.

To validate the complete application:

```powershell
python -m unittest discover -s tests -v
```

## 6. Public deployment instructions

The repository is ready for Streamlit Community Cloud.

1. Create a Git repository and include `app.py`, `app_components/`,
   `requirements.txt`, `.streamlit/config.toml`, the required `outputs/*.csv`
   files, and `outputs/charts/*.png`.
2. Push the repository to GitHub.
3. In Streamlit Community Cloud, choose **Create app**.
4. Select the repository and branch, then set **Main file path** to `app.py`.
5. Add the Google Sheets secrets described below if persistent feedback and
   Decision Log records are required.
6. Deploy and confirm the three-step judge demo, all eight pages, charts,
   Decision Log, and language switching.

Exact first-push commands (replace the placeholder; do not commit secrets):

```powershell
git init
git add .
git commit -m "Build SKUNIVO public MVP"
git branch -M main
git remote add origin <REPOSITORY_URL>
git push -u origin main
```

## 7. Data limitations

- 1,157 latest listings across 20 shops and two markets
- Three snapshot dates only: 2026-07-01 to 2026-07-03
- Engagement measures are cumulative
- No orders, customers, cost, inventory history, ad spend, realized revenue,
  experimental treatment/control, or currency metadata
- Shop-category coverage is 66.1%; missing mappings display as unavailable
- Indonesia and Vietnam prices remain in local units and are never directly compared
- The actionable Indonesia ML benchmark is useful; Vietnam ranking quality is limited
- The descriptive ML experiment includes historical sold value and is leakage-prone

## 8. Feedback persistence note

Local development appends valid rows to `outputs/mvp_user_feedback.csv`; existing
rows are never overwritten. The file is ignored by Git.

Streamlit Community Cloud local files are not durable. The production path uses the
Google Apps Script endpoint in `deployment/google_apps_script.gs`. It appends feedback
to a `Feedback` sheet, decisions to a `Decisions` sheet, and can email Team YOUNGHTT
when a new row arrives. If the endpoint is not configured, the app offers a CSV fallback.
Personal details remain optional.

### Google Sheets and email setup

1. Create one Google Sheet for the prototype evaluation.
2. Open **Extensions → Apps Script** and paste `deployment/google_apps_script.gs`.
3. In **Project Settings → Script Properties**, add:
   - `SPREADSHEET_ID`: the ID from the Google Sheet URL.
   - `MERCHPILOT_TOKEN`: a long random secret.
   - `NOTIFICATION_EMAIL`: the Team YOUNGHTT Gmail address that receives alerts.
4. Deploy the script as a **Web app**, executing as the owner.
5. Add these values to Streamlit Community Cloud secrets:

```toml
[google_sheets]
webhook_url = "https://script.google.com/macros/s/your-deployment-id/exec"
webhook_token = "the-same-long-random-secret"
```

Do not send or commit a Google password, verification code, or the real webhook token.

## 9. Troubleshooting

- **Missing output message:** confirm the required CSVs and eight PNG charts are
  present under `outputs/`.
- **Schema validation message:** regenerate the precomputed decision outputs or
  compare column names with `app_components/data_loader.py`. Reasonable aliases are
  supported centrally.
- **Charts do not render:** confirm Git LFS did not replace PNGs with pointer files.
- **No price filter:** expand **Advanced filters** after selecting Indonesia or
  Vietnam. The app intentionally exposes only one local-currency market at a time.
- **Feedback is temporary:** configure the Google Apps Script webhook and confirm the
  app shows “connected to Team YOUNGHTT's Google Sheet.”
- **Port already in use:** run `streamlit run app.py --server.port 8502`.

## 10. Screenshot placeholders

Capture these after deployment and replace the placeholders in project materials:

- `[Screenshot: premium Home hero and real recommendation preview]`
- `[Screenshot: Executive Overview with normalized market selector]`
- `[Screenshot: Product Prioritization filters and ranked table]`
- `[Screenshot: one Product Explanation decision record]`
- `[Screenshot: AI-assisted contextual benchmark and Decision Log]`
- `[Screenshot: What-if Score Explorer with contribution chart]`
- `[Screenshot: Methodology model results and limitations]`
- `[Screenshot: User Feedback form and persistence notice]`

## 11. Demo testing script

1. Open **Home** and select **Start 3-minute judge demo**.
2. Confirm Indonesia is selected and advanced filters are collapsed. Open a
   high-priority product from the queue.
3. Read **Decision at a glance** first; expand supporting evidence only when needed.
4. Open **Decision Log**, accept or override the recommendation, select an action,
   define a success metric, and save.
5. Confirm the Google Sheet receives a row under `Decisions`.
6. Open **What-if Score Explorer**, choose each preset, and move one component from
   0 to 100.
7. Confirm all-zero components score 0, all-100 score 100, and all-50 score 50.
8. Review Indonesia and Vietnam model notes under **Methodology and Transparency**.
9. Submit the feedback form and confirm the `Feedback` sheet and email notification.
