# SKUNIVO Project Handoff

This document is the durable handoff for Team YOUNGHTT. It is designed for a
new maintainer or a different Codex account to continue the project without
depending on the original chat history.

## 1. Start here

- Repository: https://github.com/mhiunguyen/SKUNIVO
- Live application: https://skunivo.streamlit.app/
- Default branch: `main`
- Application entry point: `app.py`
- Deployment platform: Streamlit Community Cloud
- Team: YOUNGHTT
- Competition: AREA 303, Track 1, Topic 05 — E-commerce Decision Intelligence

The project is an explainable decision-support prototype for e-commerce
merchandising. It ranks product listings, explains peer-relative signals, and
lets a human accept, override, or defer a recommendation. It does not
automatically change prices, promotions, inventory, or advertising.

## 2. Current product decision

The proposal asks for a four-week, Indonesia-first pilot of SKUNIVO.

Why Indonesia first:

- 475 current listings in the app output
- 83 of 98 active-review candidates are in Indonesia
- deeper displayed-discount environment than Vietnam
- stronger shop-held-out actionable-model ranking quality

Vietnam remains visible, but the AI benchmark is deliberately shown with lower
confidence and stronger human-review language.

## 3. Current user journey

The primary judge flow is intentionally only three steps:

1. **Product Prioritization** — Indonesia is selected by default; advanced
   filters are collapsed.
2. **Product Explanation** — recommendation, reasons, and suggested action are
   shown first; technical evidence and model details are optional.
3. **Decision Log** — the reviewer accepts, overrides, or requests more
   evidence and records the action, rationale, metric, and review date.

Home contains a prominent **Start 3-minute judge demo** button. Do not make the
default experience more complex unless usability evidence supports the change.

## 4. Repository map

```text
app.py                         Streamlit shell and eight application pages
app_components/
  data_loader.py               Load/validate precomputed app outputs
  filters.py                   Market-aware filtering and sorting
  charts.py                    Portfolio/chart helpers
  recommendation_ui.py         Transparent score logic and guidance
  ai_model.py                  Load and apply the precomputed model artifact
  ai_benchmark.py              Business-language AI benchmark explanation
  decision_log.py              Decision records and validation
  feedback.py                  Usability feedback records and validation
  persistence.py               Google Apps Script delivery
  i18n.py                      English/Vietnamese display translations
  styles.py                    Responsive SKUNIVO visual system
src/                           Discovery-to-recommendation pipeline code
notebooks/                     Executable notebook entry points
outputs/                       Deployable derived data/model/chart artifacts
proposal/                      Final competition PPTX and PDF
deployment/google_apps_script.gs
tests/                         Logic and Streamlit page-flow tests
```

## 5. Data and model boundary

The GitHub repository intentionally does **not** contain the raw competition
dataset. Raw files remain local under `Data/` and are ignored by Git.

The deployable repository contains only derived artifacts required by the app:

- `outputs/product_recommendations.csv`
- `outputs/processed_latest_products.csv`
- `outputs/ai_model_artifact.json`
- model metrics, feature importance, score sensitivity, country top lists
- eight precomputed chart images

Known evidence boundary:

- 1,157 latest listings, 20 shops, Indonesia and Vietnam
- three snapshot dates: 2026-07-01 through 2026-07-03
- no orders, customers, costs, inventory history, ad spend, realized revenue,
  or treatment/control flags
- sold value is a marketplace proxy, not audited revenue
- no defensible demand forecast, causal promotion lift, profit optimization, or
  ROAS claim is currently possible

The AI layer is a contextual benchmark, not an automated decision maker. The
transparent score remains visible and the merchant owns the action.

## 6. Run and verify locally

Recommended Python: 3.11 or 3.12.

```powershell
git clone https://github.com/mhiunguyen/SKUNIVO.git
cd SKUNIVO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

Expected test result at handoff: 20 tests pass. Streamlit normally opens at
`http://localhost:8501`.

Minimum manual QA:

1. Home loads without missing assets.
2. English/Vietnamese switching preserves the current page.
3. The three-minute demo starts in the Indonesia queue.
4. Indonesia and Vietnam Executive Overview views render without chart errors.
5. Open Product Explanation from the ranked queue.
6. Confirm reasons/actions appear before supporting technical evidence.
7. Open Decision Log and verify required-field validation.
8. Submit a test record only in an authorized test Sheet.

## 7. Deployment ownership

Streamlit Community Cloud deploys `app.py` from the `main` branch. A push to
`main` normally triggers redeployment.

The new maintainer needs:

1. GitHub access to `mhiunguyen/SKUNIVO`.
2. Streamlit Community Cloud access to the deployed app, if they must manage
   secrets, logs, reboot, or deployment settings.
3. Google Sheet and Apps Script access only if they must administer feedback
   and Decision Log persistence.

Do not move credentials through chat or commit them to Git. Configure these in
Streamlit Secrets:

```toml
[google_sheets]
webhook_url = "https://script.google.com/macros/s/.../exec"
webhook_token = "a-rotated-random-secret"
```

Configure matching Apps Script Properties:

- `SPREADSHEET_ID`
- `MERCHPILOT_TOKEN`
- `NOTIFICATION_EMAIL` (optional)

Security action: a webhook token was previously pasted into a chat. Treat that
token as exposed and rotate it in both Apps Script Properties and Streamlit
Secrets. Never place the old or new value in this repository.

## 8. Competition deliverables

The final editable deck and submission PDF are:

- `proposal/YOUNGHTT_Track 01_05.pptx`
- `proposal/YOUNGHTT_Track 01_05.pdf`

The deck has 18 slides and follows a proposal narrative:

- explicit four-week Indonesia pilot ask
- decision problem and data boundary
- governed workflow and explainable scoring
- product-level example
- validation and robustness evidence
- human accountability
- business value without invented revenue claims
- feasibility, pilot success gates, and roadmap

## 9. Known ambiguities and next priorities

Unresolved business questions:

- whether `monthly_sold_value` is units, value, or another platform proxy
- how promoted status was assigned and whether campaign timing exists
- whether repeated snapshots are available beyond the supplied three days
- who owns each commercial action and which success metric is accepted
- whether Indonesia and Vietnam category mappings are fully comparable

Recommended next work, in order:

1. Run five user tests using the three-minute judge flow and capture
   time-to-triage, clarity, trust, and navigation scores.
2. Verify Google Sheet delivery and rotate the shared webhook secret.
3. Add transaction outcomes and treatment/control flags before claiming lift.
4. Add inventory and cost data before stock or profit decisions.
5. Add scheduled ingestion, access control, monitoring, and model governance
   before production use.

Do not build a more complex ML model until outcome data and a measurable
decision target exist.

## 10. Copy this prompt into the new Codex account

```text
You are taking over the SKUNIVO project for Team YOUNGHTT.

Repository: https://github.com/mhiunguyen/SKUNIVO
Local workspace, if available: D:\YOUNGHTT
Live app: https://skunivo.streamlit.app/

First read HANDOFF.md, README.md, and README_MVP.md completely. Then inspect
git status, the latest commits, app.py, app_components/, tests/, outputs/model_metrics.csv,
outputs/score_sensitivity.csv, and the final proposal deck/PDF.

Preserve these product decisions:
- SKUNIVO is decision support, not automatic execution.
- The primary proposal is a four-week Indonesia-first pilot.
- The judge demo must remain a simple three-step flow.
- Put recommendation, reasons, and action before technical model detail.
- Do not claim demand forecasting, causal promotion lift, revenue, profit,
  inventory optimization, or ROAS with the current data.
- Never commit raw Data/ files or any secret/token.

Before changing code, run:
python -m unittest discover -s tests -v

After changing code, rerun all tests and manually verify the deployed user
journey. Preserve unrelated local and untracked files. Explain any assumption
that changes the proposal or evidence boundary.
```

## 11. Handoff completion checklist

- [ ] New Codex account can access the repository or existing local workspace.
- [ ] GitHub authentication is configured under the new account/session.
- [ ] The new maintainer has read this file and run the tests.
- [ ] Streamlit ownership/access is confirmed.
- [ ] Google Sheet and Apps Script ownership/access are confirmed if needed.
- [ ] The exposed webhook token has been rotated.
- [ ] Raw data remains outside GitHub.
- [ ] The live app and final proposal open successfully.
