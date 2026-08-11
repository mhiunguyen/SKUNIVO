from __future__ import annotations


APP_CSS = """
<style>
:root {
  --mp-navy: #07111f;
  --mp-ink: #132033;
  --mp-paper: #f7f5ef;
  --mp-lime: #c9ff4a;
  --mp-violet: #8b7cff;
  --mp-cyan: #48dcff;
  --mp-teal: #11b8a5;
  --mp-orange: #f28b30;
  --mp-muted: #667085;
}

.stApp {
  background:
    radial-gradient(circle at 92% 8%, rgba(139,124,255,.08), transparent 27rem),
    linear-gradient(180deg, #fbfaf6 0%, #f4f2ec 100%);
  color: var(--mp-ink);
}

[data-testid="stSidebar"] {
  background: var(--mp-navy);
  border-right: 1px solid rgba(255,255,255,.09);
}
[data-testid="stSidebar"] * { color: #f8fafc; }
[data-testid="stSidebar"] label { font-weight: 650; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: rgba(255,255,255,.07);
  border-color: rgba(255,255,255,.16);
}

.block-container { max-width: 1240px; padding-top: 1.7rem; padding-bottom: 4rem; }

.mp-brand-lockup {
  display: flex; align-items: center; margin: .2rem 0 .55rem;
}
.mp-brand-wordmark { width: 10.75rem; max-width: 100%; height: auto; display: block; }
.mp-brand-sub {
  color: #aeb8c8; font-size: .7rem; line-height: 1.4;
  letter-spacing: .015em; margin: 0 0 1rem;
}

.mp-hero {
  position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,.12);
  border-radius: 24px; color: white; padding: clamp(1.6rem, 3.3vw, 2.9rem);
  background:
    radial-gradient(circle at 88% 20%, rgba(72,220,255,.23), transparent 19rem),
    radial-gradient(circle at 70% 88%, rgba(139,124,255,.25), transparent 24rem),
    linear-gradient(125deg, #06101d 0%, #0b1628 65%, #111c31 100%);
  box-shadow: 0 22px 70px rgba(7,17,31,.2);
}
.mp-hero:after {
  content:""; position:absolute; inset:0; opacity:.2; pointer-events:none;
  background-image: radial-gradient(rgba(255,255,255,.45) .7px, transparent .7px);
  background-size: 22px 22px;
  mask-image: linear-gradient(120deg, black, transparent 65%);
}
.mp-hero-mark {
  position: absolute; z-index: 0; right: clamp(1.1rem, 4vw, 3.5rem); top: 1.25rem;
  width: clamp(4.5rem, 10vw, 8.5rem); height: auto; opacity: .16;
}
.mp-hero > *:not(.mp-hero-mark) { position: relative; z-index: 1; }
.mp-eyebrow {
  color: var(--mp-lime); font-size: .78rem; font-weight: 800; letter-spacing: .14em;
  text-transform: uppercase; margin-bottom: 1rem;
}
.mp-hero h1 {
  max-width: 850px; font-size: clamp(2.2rem, 4vw, 3.75rem); line-height: 1;
  letter-spacing: -.055em; margin: 0 0 .9rem; color: white;
}
.mp-hero p {
  max-width: 750px; color: #c9d2df; font-size: clamp(.92rem, 1.35vw, 1.05rem);
  line-height: 1.48; margin: 0;
}
.mp-accent { color: var(--mp-lime); }

.mp-page-head { margin: .5rem 0 1.8rem; }
.mp-page-head h1 { letter-spacing: -.045em; font-size: clamp(2.15rem, 4vw, 3.8rem); margin-bottom: .4rem; }
.mp-page-head p { color: var(--mp-muted); font-size: 1.08rem; max-width: 820px; }

.mp-section-title { font-size: 1.75rem; letter-spacing: -.035em; margin: 2.8rem 0 .35rem; }
.mp-section-copy { color: var(--mp-muted); max-width: 780px; margin-bottom: 1.2rem; }

.mp-card, .mp-preview, .mp-insight, .mp-boundary {
  border: 1px solid #dce1e7; border-radius: 16px; background: rgba(255,255,255,.82);
  padding: 1.25rem; box-shadow: 0 8px 26px rgba(12,24,42,.05); height: 100%;
}
.mp-card h3, .mp-preview h3 { margin: 0 0 .45rem; letter-spacing: -.025em; }
.mp-card p, .mp-preview p { color: var(--mp-muted); margin: 0; line-height: 1.55; }
.mp-dark-card { background: var(--mp-navy); border-color: #172438; color: white; }
.mp-dark-card p { color: #b9c4d2; }

.mp-kicker {
  display: inline-block; color: #455266; font-weight: 750; font-size: .72rem;
  letter-spacing: .1em; text-transform: uppercase; margin-bottom: .55rem;
}
.mp-score {
  font-size: 3.1rem; line-height: 1; font-weight: 850; letter-spacing: -.06em;
}
.mp-score small { font-size: 1rem; color: var(--mp-muted); letter-spacing: 0; }
.mp-badge {
  display: inline-flex; align-items:center; border-radius: 999px; padding: .32rem .7rem;
  background: #edf0f5; color: #273447; font-size: .78rem; font-weight: 720; margin: .15rem .25rem .15rem 0;
}
.mp-badge-lime { background: #eaffb9; color: #263d00; }
.mp-badge-violet { background: #ebe8ff; color: #3d318a; }
.mp-badge-teal { background: #dcfaf6; color: #075f54; }
.mp-badge-orange { background: #fff0e3; color: #8a4205; }

.mp-metric-grid {
  display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: .75rem; margin: 1rem 0;
}
.mp-metric {
  border: 1px solid #dfe3e8; border-radius: 14px; padding: 1rem;
  background: rgba(255,255,255,.75);
}
.mp-metric strong { display:block; font-size:1.65rem; letter-spacing:-.04em; }
.mp-metric span { color:var(--mp-muted); font-size:.82rem; }

.mp-workflow {
  display:grid; grid-template-columns: repeat(5,1fr); gap:.5rem; margin:1.25rem 0 2rem;
}
.mp-step {
  border:1px solid #dce1e7; border-radius:12px; padding:.8rem; text-align:center;
  font-weight:720; background:white; position:relative;
}

.mp-boundary {
  border-left: 4px solid var(--mp-violet); background: #f2f0ff; color:#342c69;
  margin: 1.5rem 0;
}
.mp-warning {
  border-left:4px solid #f6b73c; background:#fff8e8; padding:1rem 1.1rem;
  border-radius:10px; color:#62480f; margin:1rem 0;
}
.mp-success {
  border-left:4px solid var(--mp-teal); background:#eafbf8; padding:1rem 1.1rem;
  border-radius:10px; color:#075f54;
}
.mp-reason { border-left:3px solid var(--mp-cyan); padding:.65rem .9rem; margin:.55rem 0; background:#f4fbfd; }

.mp-market-id { color: var(--mp-teal); font-weight: 760; }
.mp-market-vn { color: var(--mp-orange); font-weight: 760; }

.mp-footer {
  border-top:1px solid #d9dee4; margin-top:4rem; padding:1.4rem 0 0;
  color:var(--mp-muted); font-size:.82rem;
}

div[data-testid="stMetric"] {
  background:rgba(255,255,255,.8); border:1px solid #dde2e8; padding:1rem;
  border-radius:14px; box-shadow:0 5px 18px rgba(12,24,42,.04);
}
div[data-testid="stMetricLabel"] { color:#5b6778; }
div[data-testid="stMetricValue"] { letter-spacing:-.035em; }
.stButton > button, .stDownloadButton > button {
  border-radius:999px; font-weight:760; min-height:2.75rem; border:1px solid #bfc7d1;
}
.stButton > button[kind="primary"] {
  background:var(--mp-navy); color:white; border-color:var(--mp-navy);
}
.stButton > button[kind="primary"]:hover { background:#152239; border-color:#152239; color:var(--mp-lime); }

@media (max-width: 760px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .mp-hero { border-radius:18px; padding:2rem 1.25rem; }
  .mp-metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .mp-workflow { grid-template-columns:1fr; }
}
</style>
"""
