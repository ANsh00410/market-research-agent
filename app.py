"""
app.py — Streamlit UI for Indian Market Research Agent + Portfolio Tracker
With Candlestick Charts + AI Prediction Engine (Bulkowski-based)
Run with: python -m streamlit run app.py
"""

import os
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── Page config ───────────────────────────────────

st.set_page_config(
    page_title="Indian Market Research Agent",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── Custom CSS ────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
  h1,h2,h3{font-family:'Syne',sans-serif!important;}

  .hero-header{
    background:linear-gradient(135deg,#FF6B35 0%,#FF9A3C 40%,#FFB347 100%);
    border-radius:16px;padding:36px 40px;margin-bottom:28px;
    color:white;text-align:center;box-shadow:0 8px 32px rgba(255,107,53,.3);
  }
  .hero-header h1{font-family:'Syne',sans-serif!important;font-size:2.4rem;
    font-weight:800;margin:0 0 8px 0;letter-spacing:-0.5px;}
  .hero-header p{font-size:1.05rem;opacity:.92;margin:0;}

  .stButton>button{
    border-radius:24px!important;font-family:'DM Sans',sans-serif!important;
    font-weight:500!important;transition:all .2s ease!important;
    border:1.5px solid #FF6B35!important;color:#FF6B35!important;
    background:transparent!important;
  }
  .stButton>button:hover{
    background:#FF6B35!important;color:white!important;
    transform:translateY(-1px)!important;
    box-shadow:0 4px 12px rgba(255,107,53,.25)!important;
  }
  .status-log{
    background:#0e1117;border:1px solid #2d2d2d;border-radius:10px;
    padding:16px 20px;font-family:monospace;font-size:.82rem;
    color:#00ff88;max-height:200px;overflow-y:auto;line-height:1.8;
  }

  /* ── Prediction cards ── */
  .pred-bull{
    background:linear-gradient(135deg,#0d5e2a 0%,#1a8c42 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #00ff88;
    box-shadow:0 4px 20px rgba(0,255,136,.2);margin-bottom:16px;
  }
  .pred-bear{
    background:linear-gradient(135deg,#5e1a1a 0%,#a02020 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #ff4444;
    box-shadow:0 4px 20px rgba(255,68,68,.2);margin-bottom:16px;
  }
  .pred-neutral{
    background:linear-gradient(135deg,#2a2a0d 0%,#6b6b00 100%);
    border-radius:14px;padding:20px 24px;color:white;
    border-left:5px solid #ffd700;
    box-shadow:0 4px 20px rgba(255,215,0,.15);margin-bottom:16px;
  }
  .pred-direction{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:1px;}
  .pred-pattern{font-size:.9rem;opacity:.85;margin-top:4px;}
  .pred-target{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;margin-top:10px;}
  .pred-conf{font-size:.82rem;opacity:.8;margin-top:4px;}
  .pred-reasoning{
    font-size:.85rem;opacity:.9;margin-top:12px;
    border-top:1px solid rgba(255,255,255,.2);padding-top:10px;line-height:1.5;
  }
  .pred-metrics{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;}
  .pred-metric{
    background:rgba(255,255,255,.12);border-radius:8px;
    padding:7px 12px;font-size:.8rem;
  }
  .pred-metric-label{opacity:.7;font-size:.72rem;display:block;}
  .conf-bar-bg{width:100%;height:6px;background:rgba(255,255,255,.2);border-radius:3px;margin-top:6px;}
  .conf-bar{height:6px;border-radius:3px;background:#00ff88;}

  /* ── Pattern pills ── */
  .ppill-bull{background:#d4edda;color:#155724;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}
  .ppill-bear{background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}
  .ppill-neutral{background:#fff3cd;color:#856404;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;margin:2px;}

  .badge-bull{background:#d4edda;color:#155724;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}
  .badge-bear{background:#f8d7da;color:#721c24;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}
  .badge-neutral{background:#fff3cd;color:#856404;padding:3px 10px;border-radius:20px;
    font-size:.78rem;font-weight:600;display:inline-block;}

  .section-title{
    font-family:'Syne',sans-serif!important;font-size:1.3rem;
    font-weight:700;color:#1a1a1a;
    border-bottom:2px solid #FF6B35;padding-bottom:8px;margin-bottom:20px;
  }
  .profit{color:#22863a!important;font-weight:600;}
  .loss{color:#cb2431!important;font-weight:600;}

  /* ── Ticker board ── */
  .ticker-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px;margin-bottom:24px;}
  .tc-green{
    background:linear-gradient(160deg,#0a4d1f 0%,#1a7a35 60%,#0d5e27 100%);
    border:1px solid #00c853;border-radius:12px;padding:16px 18px;
    transition:transform .15s,box-shadow .15s;
  }
  .tc-red{
    background:linear-gradient(160deg,#4d0a0a 0%,#8c1a1a 60%,#5e0d0d 100%);
    border:1px solid #ff1744;border-radius:12px;padding:16px 18px;
    transition:transform .15s,box-shadow .15s;
  }
  .tc-name{font-family:'Syne',sans-serif;font-size:.76rem;font-weight:700;
    color:#FFD700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .tc-ticker-tag{font-size:.67rem;opacity:.55;color:white;font-family:monospace;margin-bottom:4px;}
  .tc-price{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
    color:white;line-height:1.1;margin-bottom:6px;}
  .tc-change-row{display:flex;align-items:center;gap:6px;margin-bottom:4px;}
  .tc-arrow-up{color:#00ff88;font-size:1.3rem;}
  .tc-arrow-down{color:#ff6666;font-size:1.3rem;}
  .tc-chg-val{font-size:.9rem;font-weight:700;color:white;}
  .tc-chg-pct-g{font-size:.82rem;font-weight:700;color:#00ff88;}
  .tc-chg-pct-r{font-size:.82rem;font-weight:700;color:#ff9999;}
  .tc-pnl-row{font-size:.76rem;color:rgba(255,255,255,.8);
    border-top:1px solid rgba(255,255,255,.15);padding-top:6px;margin-top:6px;}
  .board-hdr-g{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    color:#00c853;border-left:4px solid #00c853;padding-left:12px;margin:20px 0 14px;}
  .board-hdr-r{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
    color:#ff1744;border-left:4px solid #ff1744;padding-left:12px;margin:20px 0 14px;}

  /* ── NSE Market Watch ── */
  .mw-ticker-row{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 16px; border-radius:8px; margin-bottom:6px;
    border-left:4px solid transparent;
    background:rgba(255,255,255,.03);
    transition:background .15s;
  }
  .mw-ticker-row:hover{background:rgba(255,255,255,.07);}
  .mw-green{border-left-color:#00c853;}
  .mw-red  {border-left-color:#ff1744;}
  .mw-name {font-family:'Syne',sans-serif;font-weight:700;font-size:.88rem;color:white;min-width:160px;}
  .mw-sym  {font-family:monospace;font-size:.72rem;color:#aaa;min-width:100px;}
  .mw-price{font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:white;min-width:90px;text-align:right;}
  .mw-chg-g{font-size:.82rem;font-weight:700;color:#00ff88;min-width:80px;text-align:right;}
  .mw-chg-r{font-size:.82rem;font-weight:700;color:#ff6666;min-width:80px;text-align:right;}
  .mw-vol  {font-size:.74rem;color:#888;min-width:90px;text-align:right;}
  .mw-sector{font-size:.7rem;color:#FF9A3C;min-width:130px;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .mw-header{
    display:flex;align-items:center;justify-content:space-between;
    padding:8px 16px;border-radius:6px;
    background:rgba(255,107,53,.15);
    font-size:.75rem;font-weight:700;color:#FF9A3C;
    margin-bottom:10px;letter-spacing:.5px;text-transform:uppercase;
  }
  .mw-stat-g{background:rgba(0,200,83,.12);border:1px solid rgba(0,200,83,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-r{background:rgba(255,23,68,.12);border:1px solid rgba(255,23,68,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-n{background:rgba(255,165,0,.1);border:1px solid rgba(255,165,0,.3);
    border-radius:10px;padding:14px 20px;text-align:center;}
  .mw-stat-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:white;}
  .mw-stat-lbl{font-size:.74rem;color:#aaa;margin-top:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Hero ──────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <h1>🇮🇳 Indian Market Research Agent</h1>
  <p>AI market intelligence • Candlestick charts • Bulkowski AI prediction engine • Groq Llama 3.3 70B</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍  Market Research", "📈  Portfolio Tracker", "📊  Ticker Board", "🌐  NSE Market Watch"])

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PORTFOLIO_FILE = "portfolio.json"

QUICK_ADD_STOCKS = [
    ("Reliance Industries", "RELIANCE.NS"),
    ("TCS",                 "TCS.NS"),
    ("Infosys",             "INFY.NS"),
    ("HDFC Bank",           "HDFCBANK.NS"),
    ("Tata Motors",         "TATAMOTORS.NS"),
    ("Zomato",              "ZOMATO.NS"),
    ("Bajaj Finance",       "BAJFINANCE.NS"),
    ("Wipro",               "WIPRO.NS"),
]


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)


def fetch_stock_snapshot(company, ticker):
    try:
        stock = yf.Ticker(ticker)

        # ── Historical data for indicators ────────────────────────────
        df = stock.history(start="2026-01-01")
        if df is None or df.empty:
            return {"error": f"No 2026 data for {ticker}", "ticker": ticker, "company": company}

        df = df.dropna(subset=["Close"])
        if len(df) < 2:
            return {"error": f"Insufficient data for {ticker}", "ticker": ticker, "company": company}

        hist_close = float(df["Close"].iloc[-1])   # last known close (may be yesterday)
        prev_close = float(df["Close"].iloc[-2])

        if np.isnan(hist_close) or np.isnan(prev_close) or hist_close <= 0:
            return {"error": f"Invalid price data for {ticker}", "ticker": ticker, "company": company}

        # ── Real-time price via fast_info ─────────────────────────────
        # fast_info.last_price reflects intraday price during market hours
        try:
            fi          = stock.fast_info
            live_price  = float(fi.last_price)
            prev_close_ = float(fi.previous_close) if fi.previous_close else prev_close

            # Sanity check — if fast_info returns garbage, fall back
            if np.isnan(live_price) or live_price <= 0:
                raise ValueError("bad fast_info price")

            cur  = live_price
            prev = prev_close_ if (prev_close_ and not np.isnan(prev_close_)) else prev_close
        except Exception:
            # Fallback to last historical close
            cur  = hist_close
            prev = prev_close

        chg = (cur - prev) / prev * 100 if prev else 0

        # ── Technical indicators (from historical df) ─────────────────
        ma20 = float(df["Close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
        ma50 = float(df["Close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
        if ma20 and np.isnan(ma20): ma20 = None
        if ma50 and np.isnan(ma50): ma50 = None

        rsi = None
        if len(df) >= 15:
            d = df["Close"].diff()
            g = d.clip(lower=0).rolling(14).mean()
            l = (-d.clip(upper=0)).rolling(14).mean()
            r = g / l.replace(0, float("nan"))
            v = (100 - 100 / (1 + r)).iloc[-1]
            rsi = float(v) if not np.isnan(v) else None

        bull, bear = 0, 0
        if ma20: bull += (1 if cur > ma20 else 0); bear += (1 if cur < ma20 else 0)
        if ma50: bull += (1 if cur > ma50 else 0); bear += (1 if cur < ma50 else 0)
        if rsi:
            if rsi < 35: bull += 1
            if rsi > 65: bear += 1
        signal = "BULLISH" if bull > bear+1 else ("BEARISH" if bear > bull+1 else "NEUTRAL")

        vol_series = df["Volume"].replace(0, np.nan).dropna()
        avg_vol    = int(vol_series.rolling(min(20, len(vol_series))).mean().iloc[-1]) if len(vol_series) >= 2 else 0
        today_vol  = int(df["Volume"].iloc[-1]) if not np.isnan(df["Volume"].iloc[-1]) else 0

        return {
            "ticker":        ticker,
            "company":       company,
            "current_price": round(cur, 2),
            "prev_close":    round(prev, 2),
            "today_change":  round(chg, 2),
            "week52_high":   round(float(df["High"].max()), 2),
            "week52_low":    round(float(df["Low"].min()), 2),
            "ma20":          round(ma20, 2) if ma20 else None,
            "ma50":          round(ma50, 2) if ma50 else None,
            "rsi":           round(rsi, 2)  if rsi  else None,
            "signal":        signal,
            "avg_volume":    avg_vol,
            "today_volume":  today_vol,
            "return_1m":     _pret(df, 21),
            "df":            df,
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker, "company": company}


def _pret(df, days):
    if len(df) < 2: return None
    lb = min(days, len(df) - 1)
    p  = float(df["Close"].iloc[-lb - 1])
    n  = float(df["Close"].iloc[-1])
    if np.isnan(p) or np.isnan(n) or p == 0: return None
    return round((n - p) / p * 100, 2)


def _signal_badge(s):
    m = {"BULLISH": ("badge-bull", "▲ BULLISH"), "BEARISH": ("badge-bear", "▼ BEARISH")}
    cls, lbl = m.get(s, ("badge-neutral", "◆ NEUTRAL"))
    return f'<span class="{cls}">{lbl}</span>'


def _ppill(p):
    t   = p.get("type", "neutral")
    cls = {"bullish": "ppill-bull", "bearish": "ppill-bear"}.get(t, "ppill-neutral")
    return f'<span class="{cls}">{p["name"]}</span>'


# ─────────────────────────── Candlestick chart builder ─────────────────────

def build_candlestick(df: pd.DataFrame, ticker: str, prediction: dict = None) -> go.Figure:
    df = df.tail(90).copy()
    df.index = pd.to_datetime(df.index)
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, showlegend=False,
        increasing_line_color="#00d084", decreasing_line_color="#ff4444",
        increasing_fillcolor="#00d084",  decreasing_fillcolor="#ff4444",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, name="MA20",
        line=dict(color="#FF9A3C", width=1.5), opacity=0.9
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=ma50, name="MA50",
        line=dict(color="#9b59b6", width=1.5, dash="dot"), opacity=0.9
    ))

    if prediction:
        tgt = prediction.get("price_target")
        dir_ = prediction.get("direction", "NEUTRAL")
        if tgt:
            tgt_color = "#00ff88" if dir_ == "BULLISH" else ("#ff4444" if dir_ == "BEARISH" else "#ffd700")
            fig.add_hline(y=tgt, line_color=tgt_color, line_dash="dash", line_width=1.5,
                          annotation_text=f"🎯 Target ₹{tgt:,.0f}",
                          annotation_position="right",
                          annotation_font_color=tgt_color)
        sup = prediction.get("support_level")
        res = prediction.get("resistance_level")
        if sup:
            fig.add_hline(y=sup, line_color="rgba(0,208,132,.4)", line_dash="dot", line_width=1,
                          annotation_text=f"Support ₹{sup:,.0f}", annotation_position="right",
                          annotation_font_color="rgba(0,208,132,.8)")
        if res:
            fig.add_hline(y=res, line_color="rgba(255,68,68,.4)", line_dash="dot", line_width=1,
                          annotation_text=f"Resistance ₹{res:,.0f}", annotation_position="right",
                          annotation_font_color="rgba(255,68,68,.8)")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)", rangeslider=dict(visible=False), tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,.05)", tickprefix="₹", tickfont=dict(size=11), side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=0, r=80, t=20, b=0),
        hovermode="x unified", height=380,
    )
    fig.update_xaxes(showspikes=True, spikecolor="rgba(255,255,255,.3)", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="rgba(255,255,255,.3)", spikethickness=1)
    return fig


# ─────────────────────────── Prediction card renderer ──────────────────────

def render_pred_card(pred: dict, cs_pats: list, ch_pats: list, cur_price: float) -> str:
    direction = pred.get("direction", "NEUTRAL")
    conf      = pred.get("confidence", 50)
    target    = pred.get("price_target", cur_price)
    tgt_pct   = pred.get("price_target_pct", 0)
    pattern   = pred.get("primary_pattern", "—")
    reasoning = pred.get("reasoning", "")
    sl        = pred.get("stop_loss")
    tf        = pred.get("timeframe", "3-5 days")
    risk      = pred.get("risk_level", "MEDIUM")
    vol_conf  = pred.get("volume_confirmation", False)

    card_cls = {"BULLISH": "pred-bull", "BEARISH": "pred-bear"}.get(direction, "pred-neutral")
    icon     = {"BULLISH": "🟢 ▲", "BEARISH": "🔴 ▼"}.get(direction, "🟡 ◆")
    sign     = "+" if tgt_pct >= 0 else ""

    all_pats = (cs_pats + ch_pats)[:4]
    pills    = " ".join(_ppill(p) for p in all_pats)

    sl_html  = f'<div class="pred-metric"><span class="pred-metric-label">Stop Loss</span>₹{sl:,.0f}</div>' if sl else ""
    vol_html = '<div class="pred-metric"><span class="pred-metric-label">Volume</span>✅ Confirmed</div>' if vol_conf else ""
    sup      = pred.get("support_level", cur_price * 0.97)
    res      = pred.get("resistance_level", cur_price * 1.03)

    return f"""
<div class="{card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div class="pred-direction">{icon} {direction}</div>
      <div class="pred-pattern">📊 Pattern: {pattern}</div>
    </div>
    <div style="text-align:right;">
      <div class="pred-target">🎯 ₹{target:,.2f}</div>
      <div class="pred-conf">{sign}{tgt_pct:.2f}% &nbsp;·&nbsp; {tf}</div>
      <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%"></div></div>
      <div class="pred-conf" style="margin-top:3px;">Confidence: {conf}%</div>
    </div>
  </div>
  <div class="pred-metrics">
    {sl_html}
    <div class="pred-metric"><span class="pred-metric-label">Risk</span>{risk}</div>
    <div class="pred-metric"><span class="pred-metric-label">Support</span>₹{sup:,.0f}</div>
    <div class="pred-metric"><span class="pred-metric-label">Resistance</span>₹{res:,.0f}</div>
    {vol_html}
  </div>
  <div class="pred-reasoning">💡 {reasoning}</div>
  <div style="margin-top:10px;">{pills}</div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — MARKET RESEARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<p class="section-title">Market Research Agent</p>', unsafe_allow_html=True)
    st.caption("Performs 5-6+ deep searches and writes a comprehensive 9-section Indian market report.")

    example_topics = [
        "⚡ Quick Commerce","🚗 Electric Vehicles","📚 EdTech","💸 UPI Payments",
        "🛍️ D2C Brands","🍳 Cloud Kitchen","🌾 Agritech","🏥 Health Insurance"
    ]
    st.markdown("**Quick Topics:**")
    cols = st.columns(4)
    selected_topic = ""
    for i, topic in enumerate(example_topics):
        if cols[i % 4].button(topic, key=f"topic_{i}"):
            selected_topic = topic.split(" ", 1)[1]

    research_topic = st.text_input(
        "Or enter a custom topic:", value=selected_topic,
        placeholder="e.g., Fintech lending, ONDC, Gaming, Space tech...",
        key="research_topic_input"
    )
    research_btn = st.button("🚀 Start Research", type="primary", key="research_btn")

    if research_btn and research_topic:
        from agent import run_market_research_agent
        st.markdown("---")
        st.markdown("**Live Research Log:**")
        log_ph = st.empty()
        prog   = st.progress(0.0, text="Initializing...")
        msgs   = []

        def _status(msg):
            ts = datetime.now().strftime("%H:%M:%S")
            msgs.append(f"[{ts}] {msg}")
            log_ph.markdown(f'<div class="status-log">{"<br>".join(msgs[-8:])}</div>', unsafe_allow_html=True)

        def _prog(v):
            prog.progress(min(v, 1.0), text=f"Researching... {int(v*100)}%")

        with st.spinner("Researching..."):
            try:
                report = run_market_research_agent(research_topic, _status, _prog)
                prog.progress(1.0, text="✅ Done!")
                st.success("Research complete!")
                st.markdown("---")
                st.markdown('<p class="section-title">Research Report</p>', unsafe_allow_html=True)
                r1, r2 = st.tabs(["📄 Formatted", "📝 Raw Markdown"])
                with r1: st.markdown(report)
                with r2: st.code(report, language="markdown")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"report_{research_topic.replace(' ','_').lower()[:40]}_{ts}.md"
                st.download_button("⬇️ Download Report (.md)",
                    data=f"# {research_topic}\n\n---\n\n{report}",
                    file_name=fn, mime="text/markdown")
            except Exception as e:
                st.error(f"Research failed: {e}")
    elif research_btn:
        st.warning("Please enter a topic.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — PORTFOLIO TRACKER
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}

    st.markdown('<p class="section-title">📈 Portfolio Tracker + AI Prediction</p>', unsafe_allow_html=True)
    st.caption("Candlestick charts · Bulkowski pattern detection · Groq AI price prediction · Live NSE/BSE prices")

    # ── Add stock ─────────────────────────────────────────────────────────
    with st.expander("➕ Add New Stock", expanded=len(st.session_state.portfolio) == 0):
        st.markdown("**Quick Add:**")
        qa_cols = st.columns(4)
        for i, (name, ticker) in enumerate(QUICK_ADD_STOCKS):
            if qa_cols[i % 4].button(f"+ {name.split()[0]}", key=f"qa_{i}"):
                st.session_state["add_name"]   = name
                st.session_state["add_ticker"] = ticker

        st.markdown("---")
        c1, c2, c3, c4 = st.columns([2, 1.5, 1, 1])
        with c1:
            add_name = st.text_input("Company Name",
                value=st.session_state.get("add_name", ""),
                placeholder="Reliance Industries", key="form_name")
        with c2:
            add_ticker = st.text_input("NSE Ticker",
                value=st.session_state.get("add_ticker", ""),
                placeholder="RELIANCE.NS", key="form_ticker")
        with c3:
            add_qty   = st.number_input("Qty", min_value=1, value=1, key="form_qty")
        with c4:
            add_price = st.number_input("Avg Buy (₹)", min_value=0.01, value=100.0,
                                        format="%.2f", key="form_price")

        if st.button("✅ Add", type="primary", key="add_btn"):
            if add_name and add_ticker:
                st.session_state.portfolio.append({
                    "id":        f"{add_ticker}_{int(time.time())}",
                    "company":   add_name,
                    "ticker":    add_ticker.upper(),
                    "quantity":  add_qty,
                    "avg_price": add_price,
                    "added_on":  datetime.now().strftime("%Y-%m-%d")
                })
                save_portfolio(st.session_state.portfolio)
                for k in ["add_name", "add_ticker"]: st.session_state.pop(k, None)
                st.success(f"Added {add_name}!")
                st.rerun()
            else:
                st.warning("Fill in company name and ticker.")

    # ── Action buttons ────────────────────────────────────────────────────
    ca, cb, cc, _ = st.columns([1, 1.2, 1, 3])
    refresh_btn  = ca.button("🔄 Refresh Prices",  key="refresh_btn")
    predict_all  = cb.button("🤖 Predict All Stocks", key="predict_all")
    export_btn   = cc.button("📥 Export CSV",     key="export_btn")

    if not st.session_state.portfolio:
        st.info("Portfolio is empty. Add stocks using the form above.")
        st.stop()

    # ── Migrate old portfolio entries missing required keys ───────────────
    for entry in st.session_state.portfolio:
        if "id"        not in entry: entry["id"]        = f"{entry.get('ticker','?')}_{int(time.time())}"
        if "company"   not in entry: entry["company"]   = entry.get("name", entry.get("ticker", "Unknown"))
        if "ticker"    not in entry: entry["ticker"]    = entry.get("symbol", "UNKNOWN")
        if "quantity"  not in entry: entry["quantity"]  = 1
        if "avg_price" not in entry: entry["avg_price"] = 0.0
        if "added_on"  not in entry: entry["added_on"]  = "2026-01-01"
    save_portfolio(st.session_state.portfolio)  # persist migrated data

    # ── Fetch live prices ─────────────────────────────────────────────────
    with st.spinner("Fetching live prices from NSE/BSE..."):
        stock_data = {}
        for entry in st.session_state.portfolio:
            stock_data[entry["id"]] = fetch_stock_snapshot(
                entry.get("company", "Unknown"),
                entry.get("ticker", "UNKNOWN")
            )

    # ── Summary metrics ───────────────────────────────────────────────────
    tot_inv = tot_cur = 0
    winners = losers  = 0
    for entry in st.session_state.portfolio:
        snap = stock_data.get(entry["id"], {})
        if "current_price" in snap:
            iv = entry["quantity"] * entry["avg_price"]
            cv = entry["quantity"] * snap["current_price"]
            tot_inv += iv; tot_cur += cv
            if cv >= iv: winners += 1
            else:         losers  += 1

    tot_pnl  = tot_cur - tot_inv
    pnl_pct  = (tot_pnl / tot_inv * 100) if tot_inv else 0

    st.markdown("---")
    st.markdown('<p class="section-title">Portfolio Summary</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Invested", f"₹{tot_inv:,.0f}")
    m2.metric("Current Value",  f"₹{tot_cur:,.0f}")
    m3.metric("Total P&L",      f"₹{tot_pnl:+,.0f}", delta=f"{pnl_pct:+.2f}%")
    m4.metric("Winners 🟢", str(winners))
    m5.metric("Losers 🔴",  str(losers))

    # ── Holdings ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">Holdings</p>', unsafe_allow_html=True)

    to_remove = []

    for entry in st.session_state.portfolio:
        snap     = stock_data.get(entry["id"], {})
        has_data = "current_price" in snap
        df_stock = snap.get("df")
        pred_key = entry["id"]

        st.markdown(f"#### {entry['company']}  `{entry['ticker']}`")

        if not has_data:
            st.error(f"⚠️ {snap.get('error', 'Data unavailable')}")
            if st.button("🗑️ Remove", key=f"rm_{pred_key}"): to_remove.append(pred_key)
            st.divider(); continue

        # ── Predict button ────────────────────────────────────────────
        _, pb_col, _ = st.columns([3, 2, 3])
        run_pred = pb_col.button(
            f"🤖 Run AI Prediction",
            key=f"pred_{pred_key}",
            use_container_width=True
        )

        if run_pred or predict_all:
            if df_stock is not None and not df_stock.empty:
                with st.spinner(f"Running Bulkowski AI analysis for {entry['company']}..."):
                    try:
                        from prediction_engine import run_full_prediction
                        result = run_full_prediction(
                            entry["company"], entry["ticker"], snap, df_stock
                        )
                        st.session_state.predictions[pred_key] = result
                    except Exception as e:
                        st.session_state.predictions[pred_key] = {"error": str(e)}

        # ── Prediction card ABOVE chart ───────────────────────────────
        pred_result = st.session_state.predictions.get(pred_key)
        if pred_result:
            if "error" in pred_result:
                st.warning(f"Prediction error: {pred_result['error']}")
            else:
                pred   = pred_result["prediction"]
                cs_pat = pred_result.get("candlestick_patterns", [])
                ch_pat = pred_result.get("chart_patterns", [])
                st.markdown(
                    render_pred_card(pred, cs_pat, ch_pat, snap["current_price"]),
                    unsafe_allow_html=True
                )

        # ── Metrics + Chart ───────────────────────────────────────────
        col_m, col_c = st.columns([1, 2.2])

        with col_m:
            price   = snap["current_price"]
            chg     = snap["today_change"]
            inv_val = entry["quantity"] * entry["avg_price"]
            cur_val = entry["quantity"] * price
            pnl     = cur_val - inv_val
            pnl_p   = (pnl / inv_val * 100) if inv_val else 0
            pnl_cls = "profit" if pnl >= 0 else "loss"

            st.metric("Live Price", f"₹{price:,.2f}", delta=f"{chg:+.2f}%")
            ca2, cb2 = st.columns(2)
            ca2.metric("RSI",  f"{snap['rsi']:.1f}" if snap.get("rsi") else "—")
            cb2.metric("Qty",  str(entry["quantity"]))
            st.markdown(
                f"**P&L:** <span class='{pnl_cls}'>₹{pnl:+,.0f} ({pnl_p:+.2f}%)</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Avg buy: ₹{entry['avg_price']:,.2f}")
            ma_parts = []
            if snap.get("ma20"): ma_parts.append(f"MA20 ₹{snap['ma20']:,.0f}")
            if snap.get("ma50"): ma_parts.append(f"MA50 ₹{snap['ma50']:,.0f}")
            if ma_parts: st.caption(" · ".join(ma_parts))
            st.caption(f"52W: ₹{snap['week52_low']:,.0f} – ₹{snap['week52_high']:,.0f}")
            st.markdown(_signal_badge(snap["signal"]), unsafe_allow_html=True)

        with col_c:
            if df_stock is not None and not df_stock.empty:
                pred_for_chart = None
                if pred_result and "error" not in pred_result:
                    pred_for_chart = pred_result["prediction"]
                fig = build_candlestick(df_stock, entry["ticker"], pred_for_chart)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{pred_key}")
            else:
                st.info("No chart data available.")

        if st.button("🗑️ Remove from portfolio", key=f"remove_{pred_key}"):
            to_remove.append(pred_key)

        st.divider()

    # ── Process removals ──────────────────────────────────────────────────
    if to_remove:
        st.session_state.portfolio = [e for e in st.session_state.portfolio if e["id"] not in to_remove]
        save_portfolio(st.session_state.portfolio)
        st.rerun()

    # ── Export CSV ────────────────────────────────────────────────────────
    if export_btn:
        rows = []
        for entry in st.session_state.portfolio:
            snap = stock_data.get(entry["id"], {})
            if "current_price" in snap:
                iv  = entry["quantity"] * entry["avg_price"]
                cv  = entry["quantity"] * snap["current_price"]
                pnl = cv - iv
                pr  = st.session_state.predictions.get(entry["id"])
                pd_ = pr["prediction"] if pr and "error" not in pr else {}
                rows.append({
                    "Company":        entry["company"],
                    "Ticker":         entry["ticker"],
                    "Qty":            entry["quantity"],
                    "Avg Buy (₹)":    entry["avg_price"],
                    "Live (₹)":       snap["current_price"],
                    "Today %":        snap["today_change"],
                    "RSI":            snap.get("rsi", ""),
                    "MA20":           snap.get("ma20", ""),
                    "MA50":           snap.get("ma50", ""),
                    "Signal":         snap.get("signal", ""),
                    "AI Direction":   pd_.get("direction", ""),
                    "AI Target (₹)":  pd_.get("price_target", ""),
                    "AI Confidence":  pd_.get("confidence", ""),
                    "Pattern":        pd_.get("primary_pattern", ""),
                    "Invested (₹)":   round(iv, 2),
                    "Current (₹)":    round(cv, 2),
                    "P&L (₹)":        round(pnl, 2),
                    "P&L %":          round((pnl/iv*100) if iv else 0, 2),
                })
        if rows:
            df_exp = pd.DataFrame(rows)
            st.download_button(
                "⬇️ Download CSV",
                data=df_exp.to_csv(index=False),
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — LIVE TICKER BOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = load_portfolio()
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}
    if "board_pred" not in st.session_state:
        st.session_state.board_pred = {}

    st.markdown('<p class="section-title">📊 Live Ticker Board</p>', unsafe_allow_html=True)
    st.caption("Live MCX-style green/red cards for your portfolio • Click Predict for AI analysis")

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks in the Portfolio Tracker tab.")
        st.stop()

    # ── Controls ─────────────────────────────────────────────────────────
    bc1, bc2, _ = st.columns([1, 1.5, 4])
    board_refresh = bc1.button("🔄 Refresh", key="board_refresh")
    board_pred_all = bc2.button("🤖 Predict All", key="board_pred_all")

    # ── Fetch data ────────────────────────────────────────────────────────
    with st.spinner("Fetching live prices..."):
        board_data = {}
        for entry in st.session_state.portfolio:
            eid = entry.get("id", entry.get("ticker", "?"))
            board_data[eid] = fetch_stock_snapshot(
                entry.get("company", "Unknown"),
                entry.get("ticker", "UNKNOWN")
            )

    # ── Predict all if requested ──────────────────────────────────────────
    if board_pred_all:
        with st.spinner("Running AI predictions for all stocks..."):
            for entry in st.session_state.portfolio:
                eid  = entry.get("id", entry.get("ticker"))
                snap = board_data.get(eid, {})
                df_s = snap.get("df")
                if df_s is not None and not df_s.empty and "current_price" in snap:
                    try:
                        from prediction_engine import run_full_prediction
                        result = run_full_prediction(
                            entry.get("company", "Unknown"),
                            entry.get("ticker", "UNKNOWN"),
                            snap, df_s
                        )
                        st.session_state.board_pred[eid] = result
                    except Exception as e:
                        st.session_state.board_pred[eid] = {"error": str(e)}

    # ── Separate into gainers and losers ─────────────────────────────────
    gainers = []
    losers  = []

    for entry in st.session_state.portfolio:
        eid  = entry.get("id", entry.get("ticker", "?"))
        snap = board_data.get(eid, {})
        if "current_price" not in snap:
            continue

        cur      = snap["current_price"]
        avg      = entry.get("avg_price", 0)
        qty      = entry.get("quantity", 1)
        invested = avg * qty
        current  = cur * qty
        pnl      = current - invested
        pnl_pct  = (pnl / invested * 100) if invested else 0
        today    = snap.get("today_change", 0)

        card = {
            "id":       eid,
            "entry":    entry,
            "snap":     snap,
            "cur":      cur,
            "today":    today,
            "pnl":      pnl,
            "pnl_pct":  pnl_pct,
            "invested": invested,
            "current":  current,
        }
        if pnl >= 0:
            gainers.append(card)
        else:
            losers.append(card)

    # Sort by P&L %
    gainers.sort(key=lambda x: x["pnl_pct"], reverse=True)
    losers.sort(key=lambda x: x["pnl_pct"])

    # ── Summary bar ───────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Gainers", str(len(gainers)))
    s2.metric("🔴 Losers",  str(len(losers)))
    tot_pnl = sum(c["pnl"] for c in gainers + losers)
    tot_inv = sum(c["invested"] for c in gainers + losers)
    s3.metric("Total P&L", f"₹{tot_pnl:+,.0f}",
              delta=f"{(tot_pnl/tot_inv*100) if tot_inv else 0:+.2f}%")
    s4.metric("Total Stocks", str(len(gainers) + len(losers)))

    st.markdown("---")

    # ── Helper: build one ticker card HTML ───────────────────────────────
    def _ticker_card_html(card: dict, is_green: bool) -> str:
        entry   = card["entry"]
        snap    = card["snap"]
        name    = entry.get("company", "Unknown")
        ticker  = entry.get("ticker", "—")
        cur     = card["cur"]
        today   = card["today"]
        pnl     = card["pnl"]
        pnl_pct = card["pnl_pct"]

        cls      = "tc-green" if is_green else "tc-red"
        arrow    = '<span class="tc-arrow-up">▲</span>' if today >= 0 else '<span class="tc-arrow-down">▼</span>'
        pct_cls  = "tc-chg-pct-g" if today >= 0 else "tc-chg-pct-r"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_p_s  = "+" if pnl_pct >= 0 else ""

        # Shorten name for card
        short_name = name if len(name) <= 18 else name[:17] + "…"

        return f"""
<div class="{cls}">
  <div class="tc-name">{short_name}</div>
  <div class="tc-ticker-tag">{ticker}</div>
  <div class="tc-price">₹{cur:,.2f}</div>
  <div class="tc-change-row">
    {arrow}
    <span class="tc-chg-val">{abs(today):.2f}</span>
    <span class="{pct_cls}">&nbsp;{today:+.2f}%</span>
  </div>
  <div class="tc-pnl-row">
    P&amp;L: {pnl_sign}₹{pnl:,.0f} &nbsp;|&nbsp; {pnl_p_s}{pnl_pct:.2f}%
  </div>
</div>"""

    # ── GAINERS SECTION ───────────────────────────────────────────────────
    if gainers:
        st.markdown(
            f'<div class="board-hdr-g">🟢 GAINERS ({len(gainers)} stocks)</div>',
            unsafe_allow_html=True
        )

        # Render cards in rows of 4
        for row_start in range(0, len(gainers), 4):
            row_cards = gainers[row_start:row_start + 4]
            cols = st.columns(len(row_cards))
            for i, card in enumerate(row_cards):
                with cols[i]:
                    st.markdown(_ticker_card_html(card, is_green=True), unsafe_allow_html=True)

                    eid = card["id"]
                    pred_res = st.session_state.board_pred.get(eid)

                    # Predict button
                    if st.button("🤖 Predict", key=f"bpred_{eid}", use_container_width=True):
                        snap = card["snap"]
                        df_s = snap.get("df")
                        if df_s is not None and not df_s.empty:
                            with st.spinner("Analyzing..."):
                                try:
                                    from prediction_engine import run_full_prediction
                                    result = run_full_prediction(
                                        card["entry"].get("company", "Unknown"),
                                        card["entry"].get("ticker", "UNKNOWN"),
                                        snap, df_s
                                    )
                                    st.session_state.board_pred[eid] = result
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    # Show mini prediction result below card
                    if pred_res and "error" not in pred_res:
                        pred = pred_res["prediction"]
                        dir_ = pred.get("direction", "NEUTRAL")
                        conf = pred.get("confidence", 0)
                        tgt  = pred.get("price_target", 0)
                        pat  = pred.get("primary_pattern", "—")
                        d_color = {"BULLISH": "#00ff88", "BEARISH": "#ff4444"}.get(dir_, "#ffd700")
                        st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;
     margin-top:4px;border-left:3px solid {d_color};font-size:.78rem;color:white;">
  <b style="color:{d_color}">{dir_}</b> · ₹{tgt:,.0f} · {conf}%<br>
  <span style="opacity:.7">{pat}</span>
</div>""", unsafe_allow_html=True)
                    elif pred_res and "error" in pred_res:
                        st.caption(f"⚠️ {pred_res['error'][:40]}")

    else:
        st.markdown('<div class="board-hdr-g">🟢 GAINERS — None today</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOSERS SECTION ────────────────────────────────────────────────────
    if losers:
        st.markdown(
            f'<div class="board-hdr-r">🔴 LOSERS ({len(losers)} stocks)</div>',
            unsafe_allow_html=True
        )

        for row_start in range(0, len(losers), 4):
            row_cards = losers[row_start:row_start + 4]
            cols = st.columns(len(row_cards))
            for i, card in enumerate(row_cards):
                with cols[i]:
                    st.markdown(_ticker_card_html(card, is_green=False), unsafe_allow_html=True)

                    eid = card["id"]
                    pred_res = st.session_state.board_pred.get(eid)

                    if st.button("🤖 Predict", key=f"bpred_{eid}", use_container_width=True):
                        snap = card["snap"]
                        df_s = snap.get("df")
                        if df_s is not None and not df_s.empty:
                            with st.spinner("Analyzing..."):
                                try:
                                    from prediction_engine import run_full_prediction
                                    result = run_full_prediction(
                                        card["entry"].get("company", "Unknown"),
                                        card["entry"].get("ticker", "UNKNOWN"),
                                        snap, df_s
                                    )
                                    st.session_state.board_pred[eid] = result
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    if pred_res and "error" not in pred_res:
                        pred = pred_res["prediction"]
                        dir_ = pred.get("direction", "NEUTRAL")
                        conf = pred.get("confidence", 0)
                        tgt  = pred.get("price_target", 0)
                        pat  = pred.get("primary_pattern", "—")
                        d_color = {"BULLISH": "#00ff88", "BEARISH": "#ff4444"}.get(dir_, "#ffd700")
                        st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;
     margin-top:4px;border-left:3px solid {d_color};font-size:.78rem;color:white;">
  <b style="color:{d_color}">{dir_}</b> · ₹{tgt:,.0f} · {conf}%<br>
  <span style="opacity:.7">{pat}</span>
</div>""", unsafe_allow_html=True)
                    elif pred_res and "error" in pred_res:
                        st.caption(f"⚠️ {pred_res['error'][:40]}")
    else:
        st.markdown('<div class="board-hdr-r">🔴 LOSERS — None today</div>', unsafe_allow_html=True)

    # ── Stocks with no data ───────────────────────────────────────────────
    no_data = [
        e for e in st.session_state.portfolio
        if "current_price" not in board_data.get(e.get("id", e.get("ticker", "")), {})
    ]
    if no_data:
        st.markdown("---")
        st.caption(f"⚠️ No data: {', '.join(e.get('ticker','?') for e in no_data)}")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — NSE MARKET WATCH  (real-time, auto-refresh like Zerodha)
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    from nse_stocks import NSE_STOCKS, NIFTY50, NIFTY50_NAMES

    st.markdown('<p class="section-title">🌐 NSE Market Watch</p>', unsafe_allow_html=True)
    st.caption("Real-time NSE prices • Auto-refreshes like Zerodha • No page reload needed")

    # ── Static controls (outside fragment — never reset on refresh) ───────
    f1, f2, f3, f4 = st.columns([1.8, 1.5, 1.2, 1])

    universe_choice = f1.selectbox(
        "Index / Universe",
        ["⭐ Nifty 50", "🏦 Banking & Finance", "💻 IT & Technology",
         "🚗 Automobile & EV", "⚡ Energy & Power", "🧪 Chemicals & Pharma",
         "🛒 FMCG & Consumer", "🏥 Healthcare & Diagnostics",
         "🏗️ Infrastructure & Real Estate", "🔩 Metals & Mining",
         "🏭 Manufacturing & Industrials", "📡 Telecom & Media",
         "📦 Logistics & Transport", "🛍️ Retail & D2C"],
        key="mw_universe"
    )
    filter_move  = f2.selectbox("Show",
        ["All Stocks", "🟢 Gainers Only", "🔴 Losers Only", "🔥 Top Movers (>2%)"],
        key="mw_filter")
    search_query = f3.text_input("Search", placeholder="e.g. Reliance", key="mw_search")
    sort_by      = f4.selectbox("Sort by", ["Change %", "Price", "Name"], key="mw_sort")

    rb1, rb2, rb3, rb4 = st.columns([1, 1.2, 1.2, 4])
    load_btn      = rb1.button("🚀 Load", type="primary", key="mw_load")
    interval_secs = rb2.selectbox("Auto-refresh", [15, 30, 60, 120, 300],
                                  index=1, key="mw_interval",
                                  format_func=lambda x: f"Every {x}s")
    live_toggle   = rb3.toggle("⚡ Live Mode", value=False, key="mw_live")
    if live_toggle:
        rb4.markdown(
            '<span style="color:#00ff88;font-weight:700;font-size:.9rem">' +
            '🔴 LIVE — prices auto-updating</span>', unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────
    for _k, _v in [("mw_history", {}), ("mw_base", []), ("mw_loaded", False),
                   ("mw_pred", {}), ("mw_universe_c", "")]:
        if _k not in st.session_state: st.session_state[_k] = _v

    def _build_tl(choice):
        if "Nifty 50" in choice:
            return [(t, NIFTY50_NAMES.get(t, t.replace(".NS","")), "Nifty 50") for t in NIFTY50]
        for sk, stocks in NSE_STOCKS.items():
            if sk in choice or choice in sk:
                return [(f"{s}.NS", n, sk) for s, n in stocks]
        return []

    # ── Full history load — ONLY on explicit Load click or universe change ─
    # live_toggle change must NOT trigger a reload
    universe_changed = (universe_choice != st.session_state.mw_universe_c)
    should_load = load_btn or (not st.session_state.mw_loaded and not live_toggle) or (universe_changed and st.session_state.mw_loaded)

    if should_load:
        tl = _build_tl(universe_choice)
        st.session_state.mw_universe_c = universe_choice
        prog = st.progress(0.0, text=f"Loading {len(tl)} stocks (history, RSI, 52W)...")
        history, base = {}, []
        for idx, (sym, name, sector) in enumerate(tl):
            prog.progress((idx+1)/len(tl), text=f"{sym} ({idx+1}/{len(tl)})")
            try:
                df_t = yf.Ticker(sym).history(start="2026-01-01")
                if df_t is None or df_t.empty or len(df_t) < 2: continue
                df_t = df_t.dropna(subset=["Close"])
                if len(df_t) < 2: continue
                rsi = None
                if len(df_t) >= 15:
                    try:
                        d = df_t["Close"].diff()
                        g = d.clip(lower=0).rolling(14).mean()
                        l = (-d.clip(upper=0)).rolling(14).mean()
                        r = g / l.replace(0, float("nan"))
                        v = (100 - 100/(1+r)).iloc[-1]
                        rsi = round(float(v), 1) if not np.isnan(v) else None
                    except: pass
                vol = int(df_t["Volume"].iloc[-1]) if not np.isnan(df_t["Volume"].iloc[-1]) else 0
                prev_c = float(df_t["Close"].iloc[-2])   # yesterday's close for % calc
                history[sym] = {
                    "df": df_t,
                    "prev_close": prev_c,
                    "hi52": round(float(df_t["High"].max()), 2),
                    "lo52": round(float(df_t["Low"].min()), 2),
                    "rsi": rsi, "volume": vol,
                }
                base.append({"ticker": sym, "name": name, "sector": sector})
            except: continue
        prog.progress(1.0, text=f"✅ {len(base)} stocks loaded. Toggle Live Mode to auto-refresh prices.")
        st.session_state.mw_history = history
        st.session_state.mw_base    = base
        st.session_state.mw_loaded  = True

    # ── LIVE FRAGMENT ─────────────────────────────────────────────────────
    # @st.fragment(run_every=N) re-runs ONLY this block every N seconds.
    # Uses a single yf.download(period="1d") batch call for ALL tickers at once
    # — one HTTP request instead of 50 sequential fast_info calls. Much faster.
    @st.fragment(run_every=(interval_secs if live_toggle else None))
    def _live_watch():
        history  = st.session_state.mw_history
        base     = st.session_state.mw_base
        s_query  = st.session_state.get("mw_search", "")
        s_filter = st.session_state.get("mw_filter", "All Stocks")
        s_sort   = st.session_state.get("mw_sort", "Change %")

        if not base:
            st.info("Click 🚀 Load to fetch market data first.")
            return

        now_str = datetime.now().strftime("%H:%M:%S")

        # ── BATCH price fetch — single API call for all tickers ───────
        # yf.download(period="2d") gets today + yesterday in one request
        tickers_str = " ".join(item["ticker"] for item in base)
        live_prices = {}   # ticker -> (current_price, prev_close)

        try:
            raw = yf.download(
                tickers_str,
                period="2d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker"
            )
            # Parse the multi-ticker DataFrame
            # Structure: columns are MultiIndex (field, ticker) or (ticker, field)
            for item in base:
                sym = item["ticker"]
                try:
                    # Try (ticker, field) first
                    if hasattr(raw.columns, 'levels'):
                        lvl0 = raw.columns.get_level_values(0).unique().tolist()
                        lvl1 = raw.columns.get_level_values(1).unique().tolist()
                        # Detect structure
                        if sym in lvl0:
                            df_s = raw[sym]
                        elif sym in lvl1:
                            df_s = raw.xs(sym, level=1, axis=1)
                        else:
                            raise KeyError(sym)
                    else:
                        df_s = raw  # single ticker

                    df_s = df_s.dropna(subset=["Close"])
                    if len(df_s) >= 2:
                        cur  = float(df_s["Close"].iloc[-1])
                        prev = float(df_s["Close"].iloc[-2])
                        if not np.isnan(cur) and cur > 0:
                            live_prices[sym] = (cur, prev)
                except Exception:
                    pass

        except Exception:
            pass

        # ── Build rows — merge batch prices with cached history ────────
        rows = []
        for item in base:
            sym  = item["ticker"]
            hist = history.get(sym, {})
            if not hist: continue

            hist_prev = hist["prev_close"]

            if sym in live_prices:
                cur, prev = live_prices[sym]
            else:
                # Fallback: fast_info for this single stock only
                try:
                    fi  = yf.Ticker(sym).fast_info
                    lp  = float(fi.last_price)
                    pc  = float(fi.previous_close) if fi.previous_close else hist_prev
                    if np.isnan(lp) or lp <= 0: raise ValueError()
                    cur, prev = lp, (pc if not np.isnan(pc) else hist_prev)
                except:
                    cur, prev = hist_prev, hist_prev

            if prev == 0 or np.isnan(cur): continue
            chg = (cur - prev) / prev * 100
            rows.append({
                "ticker":  sym, "name": item["name"], "sector": item["sector"],
                "price":   round(cur, 2),
                "chg_abs": round(cur - prev, 2),
                "chg_pct": round(chg, 2),
                "volume":  hist.get("volume", 0),
                "hi52":    hist.get("hi52", 0),
                "lo52":    hist.get("lo52", 0),
                "rsi":     hist.get("rsi"),
            })

        # filters
        data = list(rows)
        if s_query:
            q = s_query.lower()
            data = [d for d in data if q in d["name"].lower() or q in d["ticker"].lower()]
        if "Gainers"    in s_filter: data = [d for d in data if d["chg_pct"] > 0]
        elif "Losers"   in s_filter: data = [d for d in data if d["chg_pct"] < 0]
        elif "Top Movers" in s_filter: data = [d for d in data if abs(d["chg_pct"]) >= 2]
        if s_sort == "Change %": data.sort(key=lambda x: x["chg_pct"], reverse=True)
        elif s_sort == "Price":  data.sort(key=lambda x: x["price"], reverse=True)
        elif s_sort == "Name":   data.sort(key=lambda x: x["name"])

        if not data:
            st.info("No stocks match filter.")
            return

        gainers_c = sum(1 for d in data if d["chg_pct"] > 0)
        losers_c  = sum(1 for d in data if d["chg_pct"] < 0)
        unch_c    = len(data) - gainers_c - losers_c
        top_g = max(data, key=lambda x: x["chg_pct"])
        top_l = min(data, key=lambda x: x["chg_pct"])

        pulse = "#00ff88" if live_toggle else "#aaa"
        anim  = "animation:pulse 1s infinite;" if live_toggle else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">' +
            f'<span style="width:10px;height:10px;border-radius:50%;background:{pulse};display:inline-block;{anim}"></span>' +
            f'<span style="color:#aaa;font-size:.8rem;">Updated: {now_str} IST &nbsp;·&nbsp; {len(data)} stocks &nbsp;·&nbsp; ' +
            (f"Auto-refreshing every {interval_secs}s" if live_toggle else "Manual mode — enable Live Mode") +
            '</span></div>', unsafe_allow_html=True)

        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        sc1.markdown(f'<div class="mw-stat-g"><div class="mw-stat-val">{gainers_c}</div><div class="mw-stat-lbl">🟢 Advancing</div></div>', unsafe_allow_html=True)
        sc2.markdown(f'<div class="mw-stat-r"><div class="mw-stat-val">{losers_c}</div><div class="mw-stat-lbl">🔴 Declining</div></div>', unsafe_allow_html=True)
        sc3.markdown(f'<div class="mw-stat-n"><div class="mw-stat-val">{unch_c}</div><div class="mw-stat-lbl">⬜ Unchanged</div></div>', unsafe_allow_html=True)
        sc4.markdown(f'<div class="mw-stat-g"><div class="mw-stat-val" style="font-size:.9rem">{top_g["name"][:14]}</div><div class="mw-stat-lbl">🏆 Top Gainer +{top_g["chg_pct"]:.2f}%</div></div>', unsafe_allow_html=True)
        sc5.markdown(f'<div class="mw-stat-r"><div class="mw-stat-val" style="font-size:.9rem">{top_l["name"][:14]}</div><div class="mw-stat-lbl">📉 Top Loser {top_l["chg_pct"]:.2f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="mw-header">
  <span style="min-width:175px">Company</span>
  <span style="min-width:105px">Symbol</span>
  <span style="min-width:100px;text-align:right">Price (₹)</span>
  <span style="min-width:85px;text-align:right">Change ₹</span>
  <span style="min-width:80px;text-align:right">Chg %</span>
  <span style="min-width:55px;text-align:right">RSI</span>
  <span style="min-width:85px;text-align:right">Volume</span>
  <span style="min-width:140px;text-align:right">Sector</span>
  <span style="min-width:40px;text-align:right">AI</span>
</div>""", unsafe_allow_html=True)

        for row in data:
            is_up     = row["chg_pct"] >= 0
            row_cls   = "mw-green" if is_up else "mw-red"
            chg_cls   = "mw-chg-g" if is_up else "mw-chg-r"
            arrow     = "▲" if is_up else "▼"
            rsi_val   = row.get("rsi")
            rsi_str   = f"{rsi_val}" if rsi_val else "—"
            rsi_color = "#ff6666" if (rsi_val and rsi_val > 70) else ("#00ff88" if (rsi_val and rsi_val < 30) else "#aaa")
            vol       = row["volume"]
            vol_str   = f"{vol/1e6:.1f}M" if vol>1e6 else (f"{vol/1e3:.0f}K" if vol>1e3 else str(vol))
            sec_s     = row["sector"].split(" ",1)[1][:18] if " " in row["sector"] else row["sector"][:18]

            col_row, col_btn = st.columns([10, 1])
            with col_row:
                st.markdown(f"""<div class="mw-ticker-row {row_cls}">
  <span class="mw-name">{row["name"][:24]}</span>
  <span class="mw-sym">{row["ticker"].replace(".NS","")}</span>
  <span class="mw-price">₹{row["price"]:,.2f}</span>
  <span class="{chg_cls}">{arrow} {abs(row["chg_abs"]):.2f}</span>
  <span class="{chg_cls}">{row["chg_pct"]:+.2f}%</span>
  <span style="min-width:55px;text-align:right;font-size:.82rem;font-weight:700;color:{rsi_color}">{rsi_str}</span>
  <span class="mw-vol">{vol_str}</span>
  <span class="mw-sector">{sec_s}</span>
</div>""", unsafe_allow_html=True)

            with col_btn:
                sc = row["ticker"].replace(".","_").replace("-","_")
                if st.button("🤖", key=f"mwp_{sc}", help=f"AI Predict {row['name']}"):
                    with st.spinner(f"Analyzing {row['name']}..."):
                        try:
                            from prediction_engine import run_full_prediction
                            df_t = history.get(row["ticker"], {}).get("df")
                            if df_t is not None and not df_t.empty:
                                result = run_full_prediction(row["name"], row["ticker"], {
                                    "current_price": row["price"], "today_change": row["chg_pct"],
                                    "rsi": row.get("rsi"), "ma20": None, "ma50": None,
                                    "week52_high": row["hi52"], "week52_low": row["lo52"],
                                    "return_1m": None, "avg_volume": row["volume"], "today_volume": row["volume"],
                                }, df_t)
                                st.session_state.mw_pred[row["ticker"]] = result
                        except Exception as e:
                            st.session_state.mw_pred[row["ticker"]] = {"error": str(e)}

            pr = st.session_state.mw_pred.get(row["ticker"])
            if pr and "error" not in pr:
                p   = pr["prediction"]
                d_  = p.get("direction","NEUTRAL")
                dc  = {"BULLISH":"#00ff88","BEARISH":"#ff4444"}.get(d_,"#ffd700")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.04);border-radius:8px;padding:10px 16px;' +
                    f'margin:0 0 8px 60px;border-left:3px solid {dc};font-size:.8rem;color:white;">' +
                    f'<b style="color:{dc}">{d_}</b> · 🎯 ₹{p.get("price_target",0):,.2f} · {p.get("confidence",0)}% · {p.get("timeframe","")}' +
                    f'<br><span style="opacity:.75">📊 {p.get("primary_pattern","—")} · {p.get("reasoning","")[:110]}</span></div>',
                    unsafe_allow_html=True)
            elif pr and "error" in pr:
                st.caption(f"⚠️ {pr['error'][:60]}")

        st.markdown("---")
        live_label = f"🔴 Auto-refreshing every {interval_secs}s" if live_toggle else "⚪ Enable Live Mode for auto-refresh"
        st.caption(f"NSE via yfinance fast_info · {live_label}")

    _live_watch()
