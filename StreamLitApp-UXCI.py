"""
QBO Seasonal Forecast Dashboard - Streamlit wrapper
------------------------------------------------------
Wraps the seasonal-naive-with-growth forecasting pipeline (QBOForecast) in
a client-facing dashboard. Same forecasting logic, but the data-coverage
diagnostics and fallback notices - which the original script only printed
to a console the client would never see - are now visible in the UI, since
a client needs to know when a number is a true seasonal projection vs. a
flat run-rate fallback.

CREDENTIALS: reads from Streamlit secrets first (st.secrets), falling back
to a local .env for local development - same pattern as the fluctuation
dashboard. Never commit API_Keys.env or .streamlit/secrets.toml with real
values.

ENVIRONMENT: sandbox vs. production is a config value, not a code edit.
Set [qbo] environment = "production" in secrets.toml (or QBO_Environment
in the .env) to point at a live company; anything else defaults to sandbox.

SUPPORT: admin@otnnamadim.com (single-tenant deployment for now).

Run locally with:  streamlit run app.py
"""

import os
import re
import json
import time
import calendar
import logging
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="QBO Seasonal Forecast", page_icon="📈", layout="wide")

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SUPPORT_EMAIL = "admin@otnnamadim.com"
REQUEST_TIMEOUT = (5, 30)  # (connect, read) seconds on every outbound call

# ---------------------------------------------------------------------------
# Logging: full detail goes to the log file; users see sanitized messages.
# Add qbo_dashboard.log to .gitignore (alongside .qbo_tokens.json).
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=Path(__file__).parent / "qbo_dashboard.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("qbo_forecast")


# ---------------------------------------------------------------------------
# Credentials: st.secrets first, .env fallback for local dev.
# The optional "environment" key ("sandbox" | "production") selects which
# QBO endpoints this deployment talks to - no code edits between environments.
# ---------------------------------------------------------------------------
from streamlit.errors import StreamlitSecretNotFoundError


def load_credentials() -> dict:
    try:
        if "qbo" in st.secrets:
            return {
                "client_id": st.secrets["qbo"]["client_id"],
                "client_secret": st.secrets["qbo"]["client_secret"],
                "refresh_token": st.secrets["qbo"]["refresh_token"],
                "realm_id": st.secrets["qbo"]["realm_id"],
                "environment": st.secrets["qbo"].get("environment", "sandbox"),
            }
    except StreamlitSecretNotFoundError:
        pass   # no secrets.toml at all -> fall through to .env

    from dotenv import load_dotenv
    env_path = Path(__file__).parent / "API_Keys.env"
    load_dotenv(env_path, override=True)
    creds = {
        "client_id": os.getenv("Client_ID"),
        "client_secret": os.getenv("Client_Secret"),
        "refresh_token": os.getenv("Refresh_Token"),
        "realm_id": os.getenv("Realm_ID"),
        "environment": os.getenv("QBO_Environment", "sandbox"),
    }
    if not all(v for k, v in creds.items() if k != "environment"):
        st.error("No QBO credentials found. Set them in .streamlit/secrets.toml "
                 f"(deployed) or API_Keys.env (local). Support: {SUPPORT_EMAIL}")
        st.stop()
    return creds


CREDS = load_credentials()

ENVIRONMENT = str(CREDS.get("environment", "sandbox")).strip().lower()
IS_PRODUCTION = ENVIRONMENT == "production"

BASE_URL = ("https://quickbooks.api.intuit.com" if IS_PRODUCTION
            else "https://sandbox-quickbooks.api.intuit.com")
DISCOVERY_URL = (
    "https://developer.api.intuit.com/.well-known/openid_configuration"
    if IS_PRODUCTION else
    "https://developer.api.intuit.com/.well-known/openid_sandbox_configuration"
)
FALLBACK_TOKEN_ENDPOINT = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"

log.info("Starting dashboard in %s mode (base=%s)", ENVIRONMENT, BASE_URL)


@st.cache_data(ttl=86400, show_spinner=False)  # re-fetch once a day
def get_token_endpoint(discovery_url: str) -> str:
    """Latest token endpoint from Intuit's discovery document, with a safe
    fallback so a discovery outage can't take down token refresh."""
    try:
        resp = requests.get(discovery_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["token_endpoint"]
    except Exception:
        log.warning("Discovery document unavailable; using fallback token endpoint")
        return FALLBACK_TOKEN_ENDPOINT


# ---------------------------------------------------------------------------
# Token store: persists the rotating token pair. Secrets only seed first run.
# ---------------------------------------------------------------------------
TOKEN_FILE = Path(__file__).parent / ".qbo_tokens.json"   # add to .gitignore!


def _load_tokens() -> dict:
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())
    # First run: seed from secrets/.env
    return {"access_token": None,
            "refresh_token": CREDS["refresh_token"],
            "expires_at": 0}


def _save_tokens(tokens: dict) -> None:
    TOKEN_FILE.write_text(json.dumps(tokens))


def get_access_token(force_refresh: bool = False) -> str:
    tokens = _load_tokens()

    # Reuse the current access token if it has >60s of life left
    if (not force_refresh and tokens.get("access_token")
            and time.time() < tokens.get("expires_at", 0) - 60):
        return tokens["access_token"]

    resp = requests.post(
        get_token_endpoint(DISCOVERY_URL),
        headers={"Accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token",
              "refresh_token": tokens["refresh_token"]},
        auth=HTTPBasicAuth(CREDS["client_id"], CREDS["client_secret"]),
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code == 400 and "invalid_grant" in resp.text:
        log.error("Refresh token invalid_grant; re-authorization required")
        st.error("The QuickBooks connection has expired and needs to be "
                 "re-authorized. Generate a new refresh token and update "
                 f"the stored credentials. Support: {SUPPORT_EMAIL}")
        st.stop()
    resp.raise_for_status()

    new = resp.json()
    tokens.update(
        access_token=new["access_token"],
        # THE critical line: keep the rotated refresh token
        refresh_token=new.get("refresh_token", tokens["refresh_token"]),
        expires_at=time.time() + new.get("expires_in", 3600),
    )
    _save_tokens(tokens)
    return tokens["access_token"]


# ---------------------------------------------------------------------------
# Fetch + parse (same logic as QBOForecast)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_monthly_report(report_name: str, start_date: str, end_date: str) -> dict:
    url = f"{BASE_URL}/v3/company/{CREDS['realm_id']}/reports/{report_name}"
    params = {"start_date": start_date, "end_date": end_date,
              "accounting_method": "Accrual", "summarize_column_by": "Month",
              "minorversion": "75"}

    def _call(token):
        return requests.get(url, params=params,
                            headers={"Authorization": f"Bearer {token}",
                                     "Accept": "application/json"},
                            timeout=REQUEST_TIMEOUT)

    response = _call(get_access_token())
    if response.status_code == 401:                      # token died mid-flight
        response = _call(get_access_token(force_refresh=True))  # refresh, retry once
    tid = response.headers.get("intuit_tid", "n/a")
    log.info("QBO %s %s intuit_tid=%s", report_name, response.status_code, tid)
    if not response.ok:
        log.error("QBO error body (tid=%s): %s", tid, response.text[:500])
    response.raise_for_status()
    return response.json()


def parse_monthly_report(report_json: dict) -> pd.DataFrame:
    columns = report_json.get("Columns", {}).get("Column", [])
    col_meta = []
    for col in columns:
        m = re.match(r"([A-Za-z]{3})\s+(\d{4})", col.get("ColTitle", ""))
        col_meta.append((m.group(1), int(m.group(2))) if m else None)

    records = []

    def extract_line(col_data, category, is_summary):
        if not col_data:
            return
        name = col_data[0].get("value")
        for i, cell in enumerate(col_data[1:], start=1):
            if i >= len(col_meta) or col_meta[i] is None:
                continue
            month, year = col_meta[i]
            try:
                val = float(cell.get("value", 0) or 0)
            except ValueError:
                val = 0.0
            records.append({"Line_Item": name, "Category": category, "Month": month,
                             "Year": year, "Value": val, "Is_Summary": is_summary})

    def walk(rows_json, category="Unknown"):
        if not rows_json or "Row" not in rows_json:
            return
        for row in rows_json["Row"]:
            if row.get("type") == "Section":
                group_name = row.get("Header", {}).get("ColData", [{}])[0].get("value", category)
                walk(row.get("Rows", {}), category=group_name)
                if "Summary" in row:
                    extract_line(row["Summary"].get("ColData", []), group_name, True)
            elif row.get("type") == "Data":
                extract_line(row.get("ColData", []), category, False)

    walk(report_json.get("Rows", {}))
    return pd.DataFrame(records)


def get_data_coverage(df: pd.DataFrame, expected_months: list) -> dict:
    if df.empty:
        return {"nonzero_months": [], "missing": expected_months}
    nonzero_months = sorted(df[df["Value"] != 0]["Month"].unique().tolist(),
                             key=lambda m: MONTH_ORDER.index(m))
    return {"nonzero_months": nonzero_months,
            "missing": [m for m in expected_months if m not in nonzero_months]}


# ---------------------------------------------------------------------------
# Forecast logic (identical to QBOForecast, with two correctness upgrades):
#  - Line items are keyed on (Category, Line_Item), so two accounts that
#    share a leaf name under different parents no longer merge into one series.
#  - The seasonal method requires the prior-year and current-year YTD totals
#    to share the same sign. A loss-to-profit (or profit-to-loss) year would
#    otherwise produce a negative growth factor that flips the sign of every
#    forecasted month; those items fall back to run-rate, and the reason is
#    disclosed in the UI alongside the sparse-data fallbacks.
# ---------------------------------------------------------------------------
KEY_COLS = ["Category", "Line_Item"]


def build_seasonal_forecast(df_prior: pd.DataFrame, df_current: pd.DataFrame,
                             last_actual_month: int, current_year: int,
                             min_prior_months: int = 3) -> tuple:
    actual_months = MONTH_ORDER[:last_actual_month]
    future_months = MONTH_ORDER[last_actual_month:]

    ytd_current = (df_current[df_current["Month"].isin(actual_months)]
                   .groupby(KEY_COLS)["Value"].sum())
    ytd_prior = (df_prior[df_prior["Month"].isin(actual_months)]
                 .groupby(KEY_COLS)["Value"].sum())
    growth = pd.DataFrame({"current": ytd_current, "prior": ytd_prior}).fillna(0.0)
    growth["factor"] = np.where(growth["prior"] != 0,
                                growth["current"] / growth["prior"], 1.0)
    # Seasonal is only trustworthy when both YTD totals point the same way.
    growth["same_sign"] = np.sign(growth["current"]) == np.sign(growth["prior"])
    growth_map = growth["factor"].to_dict()
    sign_ok_map = growth["same_sign"].to_dict()
    prior_ytd_map = growth["prior"].to_dict()

    run_rate_map = (df_current[df_current["Month"].isin(actual_months)]
                     .groupby(KEY_COLS)["Value"].mean().to_dict())

    prior_future_nonzero = df_prior[df_prior["Month"].isin(future_months) & (df_prior["Value"] != 0)]
    coverage_map = prior_future_nonzero.groupby(KEY_COLS)["Month"].nunique().to_dict()
    prior_future_lookup = (df_prior[df_prior["Month"].isin(future_months)]
                            .set_index(KEY_COLS + ["Month"])["Value"].to_dict())

    all_keys = (set(map(tuple, df_current[KEY_COLS].drop_duplicates().values))
                | set(map(tuple, df_prior[KEY_COLS].drop_duplicates().values)))

    forecast_rows = []
    fallback_reasons = {}  # (category, line_item) -> reason string
    for key in all_keys:
        category, line_item = key
        coverage_ok = coverage_map.get(key, 0) >= min_prior_months
        sign_ok = bool(sign_ok_map.get(key, True)) and prior_ytd_map.get(key, 0.0) != 0.0

        if coverage_ok and sign_ok:
            use_seasonal, method = True, "Seasonal"
        elif not coverage_ok:
            use_seasonal, method = False, "Run-Rate Fallback"
            fallback_reasons[key] = (f"fewer than {min_prior_months} months of "
                                     "prior-year data in the forecast window")
        else:
            use_seasonal, method = False, "Run-Rate (sign change)"
            fallback_reasons[key] = ("prior-year and current-year YTD totals have "
                                     "opposite signs (e.g. loss last year, profit this "
                                     "year), so a YoY growth factor would flip signs")

        factor = growth_map.get(key, 1.0)
        flat_val = run_rate_map.get(key, 0.0)
        for month in future_months:
            value = (prior_future_lookup.get((category, line_item, month), 0.0) * factor
                     if use_seasonal else flat_val)
            forecast_rows.append({"Line_Item": line_item, "Category": category,
                                   "Month": month, "Year": current_year, "Value": value,
                                   "Is_Summary": False, "Actual_Forecast": "Forecast",
                                   "Method": method})

    actual_rows = df_current[df_current["Month"].isin(actual_months)].copy()
    actual_rows["Actual_Forecast"] = "Actual"
    actual_rows["Method"] = "Actual"

    combined = pd.concat([actual_rows, pd.DataFrame(forecast_rows)], ignore_index=True, sort=False)
    combined["Month"] = pd.Categorical(combined["Month"], categories=MONTH_ORDER, ordered=True)
    return combined.sort_values(["Category", "Line_Item", "Month"]), fallback_reasons


def make_actual_forecast_chart(df: pd.DataFrame, metrics: list, title: str) -> go.Figure:
    fig = go.Figure()
    for metric in metrics:
        sub = df[df["Line_Item"] == metric].sort_values("Month")
        actual_part = sub[sub["Actual_Forecast"] == "Actual"]
        forecast_part = sub[sub["Actual_Forecast"] == "Forecast"]
        bridge = pd.concat([actual_part.tail(1), forecast_part])
        fig.add_trace(go.Scatter(x=actual_part["Month"], y=actual_part["Value"],
                                  mode="lines+markers", name=f"{metric} (Actual)", legendgroup=metric))
        fig.add_trace(go.Scatter(x=bridge["Month"], y=bridge["Value"],
                                  mode="lines+markers", name=f"{metric} (Forecast)",
                                  line=dict(dash="dash"), legendgroup=metric))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Amount ($)", template="plotly_white")
    fig.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    return fig


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")
current_year = st.sidebar.number_input("Forecast year", value=date.today().year, step=1)
prior_year = current_year - 1
today = date.today()

# January edge case: with no closed month of the forecast year yet, the old
# default (12) meant "everything is an actual" and forecast nothing. Default
# to month 1 and say so, instead of silently producing an empty forecast.
january_gap = (today.year == current_year and today.month == 1)
default_last_month = 12 if today.year != current_year else max(1, today.month - 1)
last_actual_month = st.sidebar.slider("Last closed month (actuals confirmed through)", 1, 12,
                                       value=default_last_month)
if january_gap:
    st.sidebar.caption(
        f"⚠️ It's January: no {current_year} month has closed yet, so YoY growth "
        "factors rest on very little data. Treat early-year forecasts as directional."
    )
last_day = calendar.monthrange(current_year, last_actual_month)[1]
min_prior_months = st.sidebar.slider(
    "Min. prior-year months required for seasonal method", 1, 6, value=3,
    help="Line items with fewer nonzero prior-year months in the forecast window "
         "fall back to a flat run-rate instead of a seasonal projection."
)
refresh = st.sidebar.button("Refresh data", type="primary")
if refresh:
    fetch_monthly_report.clear()

st.title("📈 Seasonal Forecast Dashboard")
caption_env = "" if IS_PRODUCTION else " · sandbox data"
st.caption(f"Forecasting {current_year} through year-end · actuals confirmed through "
           f"{MONTH_ORDER[last_actual_month - 1]} {current_year}{caption_env}")

# ---------------------------------------------------------------------------
# Fetch, forecast, and render
# ---------------------------------------------------------------------------
try:
    with st.spinner("Pulling P&L and Cash Flow data from QuickBooks..."):
        pl_prior = parse_monthly_report(fetch_monthly_report("ProfitAndLoss", f"{prior_year}-01-01", f"{prior_year}-12-31"))
        pl_current = parse_monthly_report(fetch_monthly_report(
            "ProfitAndLoss", f"{current_year}-01-01", f"{current_year}-{last_actual_month:02d}-{last_day:02d}"))
        cf_prior = parse_monthly_report(fetch_monthly_report("CashFlow", f"{prior_year}-01-01", f"{prior_year}-12-31"))
        cf_current = parse_monthly_report(fetch_monthly_report(
            "CashFlow", f"{current_year}-01-01", f"{current_year}-{last_actual_month:02d}-{last_day:02d}"))

    pl_forecast, pl_fallbacks = build_seasonal_forecast(
        pl_prior, pl_current, last_actual_month, current_year, min_prior_months)
    cf_forecast, cf_fallbacks = build_seasonal_forecast(
        cf_prior, cf_current, last_actual_month, current_year, min_prior_months)

    # -----------------------------------------------------------------
    # Data coverage - visible to the client, not buried in a console
    # -----------------------------------------------------------------
    with st.expander("📋 Data coverage check (what's driving this forecast)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**P&L — {prior_year} prior year**")
            cov = get_data_coverage(pl_prior, MONTH_ORDER)
            st.write(f"Months with activity: {', '.join(cov['nonzero_months']) or 'none'}")
            if cov["missing"]:
                st.warning(f"Missing: {', '.join(cov['missing'])}")
        with col2:
            st.markdown(f"**Cash Flow — {prior_year} prior year**")
            cov = get_data_coverage(cf_prior, MONTH_ORDER)
            st.write(f"Months with activity: {', '.join(cov['nonzero_months']) or 'none'}")
            if cov["missing"]:
                st.warning(f"Missing: {', '.join(cov['missing'])}")

        all_fallbacks = {**pl_fallbacks, **cf_fallbacks}
        if all_fallbacks:
            lines = [f"- **{item}** ({category}): {reason}"
                     for (category, item), reason in sorted(all_fallbacks.items())]
            st.info(
                f"**{len(all_fallbacks)} line item(s)** use a flat run-rate average "
                "instead of a true seasonal projection:\n\n" + "\n".join(lines)
            )

    # -----------------------------------------------------------------
    # P&L chart
    # -----------------------------------------------------------------
    st.subheader("Profit & Loss Forecast")
    key_pl_metrics = ["Total Income", "Total Expenses", "Net Income"]
    fig_pl = make_actual_forecast_chart(pl_forecast, key_pl_metrics,
                                         f"{current_year} P&L: Actual + Seasonal Forecast")
    st.plotly_chart(fig_pl, use_container_width=True)

    # -----------------------------------------------------------------
    # Prior year vs current year trend comparison (real data, not illustrative)
    # -----------------------------------------------------------------
    st.subheader(f"{prior_year} Baseline vs {current_year} (Actual + Forecast)")
    st.caption(
        f"Shows {prior_year}'s real monthly Total Income directly against {current_year}'s "
        f"combined actual-through-{MONTH_ORDER[last_actual_month - 1]} plus forecasted remainder - "
        f"this is the actual growth factor your data produces, not an assumption."
    )
    comparison_metric = st.selectbox(
        "Metric", ["Total Income", "Total Expenses", "Net Income"], index=0, key="trend_compare_metric"
    )
    prior_line = (pl_prior[pl_prior["Line_Item"] == comparison_metric]
                  .set_index("Month")["Value"].reindex(MONTH_ORDER))
    current_line = (pl_forecast[pl_forecast["Line_Item"] == comparison_metric]
                     .set_index("Month")["Value"].reindex(MONTH_ORDER))

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=MONTH_ORDER, y=prior_line.values, mode="lines+markers",
                                    name=f"{prior_year}", line=dict(color="#1f77b4", width=2)))
    fig_trend.add_trace(go.Scatter(x=MONTH_ORDER, y=current_line.values, mode="lines+markers",
                                    name=f"{current_year}", line=dict(color="#d62728", width=2)))
    fig_trend.update_layout(
        title=f"{comparison_metric}: {prior_year} vs {current_year}",
        xaxis_title="Month", yaxis_title="Amount ($)", template="plotly_white",
    )
    fig_trend.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------------------------------------------
    # Cash Flow chart
    # -----------------------------------------------------------------
    st.subheader("Cash Flow Forecast")
    key_cf_metrics = ["Net cash provided by operating activities", "Net increase in cash"]
    available_cf = [m for m in key_cf_metrics if m in cf_forecast["Line_Item"].unique()]
    if available_cf:
        fig_cf = make_actual_forecast_chart(cf_forecast, available_cf,
                                             f"{current_year} Cash Flow: Actual + Seasonal Forecast")
        st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info("Standard Cash Flow line items not found for this company's report labels.")

    # -----------------------------------------------------------------
    # Full-year summary table
    # -----------------------------------------------------------------
    st.subheader("Full-Year Summary")
    fy_summary = (pl_forecast[pl_forecast["Line_Item"].isin(key_pl_metrics)]
                  .groupby("Line_Item")["Value"].sum().rename(f"FY{current_year} Forecast").reset_index())
    fy_prior = (pl_prior[pl_prior["Line_Item"].isin(key_pl_metrics)]
                .groupby("Line_Item")["Value"].sum().rename(f"FY{prior_year} Actual"))
    fy_summary = fy_summary.merge(fy_prior, on="Line_Item", how="left")
    fy_summary["YoY Change %"] = (
        (fy_summary[f"FY{current_year} Forecast"] - fy_summary[f"FY{prior_year} Actual"])
        / fy_summary[f"FY{prior_year} Actual"] * 100
    )
    st.dataframe(
        fy_summary.style.format({
            f"FY{current_year} Forecast": "${:,.2f}",
            f"FY{prior_year} Actual": "${:,.2f}",
            "YoY Change %": "{:+.1f}%",
        }),
        use_container_width=True,
    )

    with st.expander("View underlying monthly data"):
        tab1, tab2 = st.tabs(["P&L", "Cash Flow"])
        with tab1:
            st.dataframe(pl_forecast[["Category", "Line_Item", "Month", "Value", "Actual_Forecast", "Method"]],
                         use_container_width=True)
        with tab2:
            st.dataframe(cf_forecast[["Category", "Line_Item", "Month", "Value", "Actual_Forecast", "Method"]],
                         use_container_width=True)

except requests.exceptions.HTTPError:
    log.exception("QBO HTTP error")
    st.error("QuickBooks returned an error. The details have been logged; "
             f"please try Refresh, or contact {SUPPORT_EMAIL} if it persists.")
except Exception:
    log.exception("Unexpected error")
    st.error("Something went wrong loading the dashboard. The details have been "
             f"logged. Contact {SUPPORT_EMAIL} if it persists.")

# ---------------------------------------------------------------------------
# Support footer (always visible)
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Questions or issues? Contact {SUPPORT_EMAIL}")                "client_id": st.secrets["qbo"]["client_id"],
                "client_secret": st.secrets["qbo"]["client_secret"],
                "refresh_token": st.secrets["qbo"]["refresh_token"],
                "realm_id": st.secrets["qbo"]["realm_id"],
            }
    except StreamlitSecretNotFoundError:
        pass   # no secrets.toml at all -> fall through to .env

    from dotenv import load_dotenv
    env_path = Path(__file__).parent / "API_Keys.env"
    load_dotenv(env_path, override=True)
    creds = {
        "client_id": os.getenv("Client_ID"),
        "client_secret": os.getenv("Client_Secret"),
        "refresh_token": os.getenv("Refresh_Token"),
        "realm_id": os.getenv("Realm_ID"),
    }
    if not all(creds.values()):
        st.error("No QBO credentials found. Set them in .streamlit/secrets.toml "
                 "(deployed) or API_Keys.env (local).")
        st.stop()
    return creds


CREDS = load_credentials()

# ---------------------------------------------------------------------------
# Token store: persists the rotating token pair. Secrets only seed first run.

TOKEN_FILE = Path(__file__).parent / ".qbo_tokens.json"   # add to .gitignore!

def _load_tokens() -> dict:
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())
    # First run: seed from secrets/.env
    return {"access_token": None,
            "refresh_token": CREDS["refresh_token"],
            "expires_at": 0}

def _save_tokens(tokens: dict) -> None:
    TOKEN_FILE.write_text(json.dumps(tokens))

def get_access_token(force_refresh: bool = False) -> str:
    tokens = _load_tokens()

    # Reuse the current access token if it has >60s of life left
    if (not force_refresh and tokens.get("access_token")
            and time.time() < tokens.get("expires_at", 0) - 60):
        return tokens["access_token"]

    resp = requests.post(
        get_token_endpoint(),
        headers={"Accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token",
              "refresh_token": tokens["refresh_token"]},
        auth=HTTPBasicAuth(CREDS["client_id"], CREDS["client_secret"]),
    )

    if resp.status_code == 400 and "invalid_grant" in resp.text:
        st.error("The QuickBooks connection has expired and needs to be "
                 "re-authorized. Generate a new refresh token and update "
                 "the stored credentials.")
        st.stop()
    resp.raise_for_status()

    new = resp.json()
    tokens.update(
        access_token=new["access_token"],
        # THE critical line: keep the rotated refresh token
        refresh_token=new.get("refresh_token", tokens["refresh_token"]),
        expires_at=time.time() + new.get("expires_in", 3600),
    )
    _save_tokens(tokens)
    return tokens["access_token"]


# ---------------------------------------------------------------------------
# Fetch + parse (same logic as QBOForecast)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_monthly_report(report_name: str, start_date: str, end_date: str) -> dict:
    url = f"{BASE_URL}/v3/company/{CREDS['realm_id']}/reports/{report_name}"
    params = {"start_date": start_date, "end_date": end_date,
              "accounting_method": "Accrual", "summarize_column_by": "Month",
              "minorversion": "75"}

    def _call(token):
        return requests.get(url, params=params,
                            headers={"Authorization": f"Bearer {token}",
                                     "Accept": "application/json"})

    response = _call(get_access_token())
    if response.status_code == 401:                      # token died mid-flight
        response = _call(get_access_token(force_refresh=True))  # refresh, retry once
    response.raise_for_status()
    return response.json()

def parse_monthly_report(report_json: dict) -> pd.DataFrame:
    columns = report_json.get("Columns", {}).get("Column", [])
    col_meta = []
    for col in columns:
        m = re.match(r"([A-Za-z]{3})\s+(\d{4})", col.get("ColTitle", ""))
        col_meta.append((m.group(1), int(m.group(2))) if m else None)

    records = []

    def extract_line(col_data, category, is_summary):
        if not col_data:
            return
        name = col_data[0].get("value")
        for i, cell in enumerate(col_data[1:], start=1):
            if i >= len(col_meta) or col_meta[i] is None:
                continue
            month, year = col_meta[i]
            try:
                val = float(cell.get("value", 0) or 0)
            except ValueError:
                val = 0.0
            records.append({"Line_Item": name, "Category": category, "Month": month,
                             "Year": year, "Value": val, "Is_Summary": is_summary})

    def walk(rows_json, category="Unknown"):
        if not rows_json or "Row" not in rows_json:
            return
        for row in rows_json["Row"]:
            if row.get("type") == "Section":
                group_name = row.get("Header", {}).get("ColData", [{}])[0].get("value", category)
                walk(row.get("Rows", {}), category=group_name)
                if "Summary" in row:
                    extract_line(row["Summary"].get("ColData", []), group_name, True)
            elif row.get("type") == "Data":
                extract_line(row.get("ColData", []), category, False)

    walk(report_json.get("Rows", {}))
    return pd.DataFrame(records)


def get_data_coverage(df: pd.DataFrame, expected_months: list) -> dict:
    if df.empty:
        return {"nonzero_months": [], "missing": expected_months}
    nonzero_months = sorted(df[df["Value"] != 0]["Month"].unique().tolist(),
                             key=lambda m: MONTH_ORDER.index(m))
    return {"nonzero_months": nonzero_months,
            "missing": [m for m in expected_months if m not in nonzero_months]}


# ---------------------------------------------------------------------------
# Forecast logic (identical to QBOForecast)
# ---------------------------------------------------------------------------
def build_seasonal_forecast(df_prior: pd.DataFrame, df_current: pd.DataFrame,
                             last_actual_month: int, current_year: int,
                             min_prior_months: int = 3) -> tuple:
    actual_months = MONTH_ORDER[:last_actual_month]
    future_months = MONTH_ORDER[last_actual_month:]

    ytd_current = df_current[df_current["Month"].isin(actual_months)].groupby("Line_Item")["Value"].sum()
    ytd_prior = df_prior[df_prior["Month"].isin(actual_months)].groupby("Line_Item")["Value"].sum()
    growth = pd.DataFrame({"current": ytd_current, "prior": ytd_prior}).fillna(0.0)
    growth["factor"] = np.where(growth["prior"] != 0, growth["current"] / growth["prior"], 1.0)
    growth_map = growth["factor"].to_dict()

    run_rate_map = (df_current[df_current["Month"].isin(actual_months)]
                     .groupby("Line_Item")["Value"].mean().to_dict())

    prior_future_nonzero = df_prior[df_prior["Month"].isin(future_months) & (df_prior["Value"] != 0)]
    coverage_map = prior_future_nonzero.groupby("Line_Item")["Month"].nunique().to_dict()
    prior_future_lookup = (df_prior[df_prior["Month"].isin(future_months)]
                            .set_index(["Line_Item", "Month"])["Value"].to_dict())

    all_line_items = set(df_current["Line_Item"]).union(df_prior["Line_Item"])
    categories = df_current[["Line_Item", "Category"]].drop_duplicates().set_index("Line_Item")["Category"].to_dict()
    categories.update(df_prior[["Line_Item", "Category"]].drop_duplicates().set_index("Line_Item")["Category"].to_dict())

    forecast_rows, fallback_items = [], []
    for line_item in all_line_items:
        use_seasonal = coverage_map.get(line_item, 0) >= min_prior_months
        if not use_seasonal:
            fallback_items.append(line_item)
        factor = growth_map.get(line_item, 1.0)
        flat_val = run_rate_map.get(line_item, 0.0)
        for month in future_months:
            if use_seasonal:
                value = prior_future_lookup.get((line_item, month), 0.0) * factor
                method = "Seasonal"
            else:
                value = flat_val
                method = "Run-Rate Fallback"
            forecast_rows.append({"Line_Item": line_item, "Category": categories.get(line_item, "Unknown"),
                                   "Month": month, "Year": current_year, "Value": value,
                                   "Is_Summary": False, "Actual_Forecast": "Forecast", "Method": method})

    actual_rows = df_current[df_current["Month"].isin(actual_months)].copy()
    actual_rows["Actual_Forecast"] = "Actual"
    actual_rows["Method"] = "Actual"

    combined = pd.concat([actual_rows, pd.DataFrame(forecast_rows)], ignore_index=True, sort=False)
    combined["Month"] = pd.Categorical(combined["Month"], categories=MONTH_ORDER, ordered=True)
    return combined.sort_values(["Line_Item", "Month"]), fallback_items


def make_actual_forecast_chart(df: pd.DataFrame, metrics: list, title: str) -> go.Figure:
    fig = go.Figure()
    for metric in metrics:
        sub = df[df["Line_Item"] == metric].sort_values("Month")
        actual_part = sub[sub["Actual_Forecast"] == "Actual"]
        forecast_part = sub[sub["Actual_Forecast"] == "Forecast"]
        bridge = pd.concat([actual_part.tail(1), forecast_part])
        fig.add_trace(go.Scatter(x=actual_part["Month"], y=actual_part["Value"],
                                  mode="lines+markers", name=f"{metric} (Actual)", legendgroup=metric))
        fig.add_trace(go.Scatter(x=bridge["Month"], y=bridge["Value"],
                                  mode="lines+markers", name=f"{metric} (Forecast)",
                                  line=dict(dash="dash"), legendgroup=metric))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Amount ($)", template="plotly_white")
    fig.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    return fig


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Settings")
current_year = st.sidebar.number_input("Forecast year", value=date.today().year, step=1)
prior_year = current_year - 1
today = date.today()
default_last_month = (today.month - 1) if today.year == current_year and today.month > 1 else 12
last_actual_month = st.sidebar.slider("Last closed month (actuals confirmed through)", 1, 12,
                                       value=default_last_month)
last_day = calendar.monthrange(current_year, last_actual_month)[1]
min_prior_months = st.sidebar.slider(
    "Min. prior-year months required for seasonal method", 1, 6, value=3,
    help="Line items with fewer nonzero prior-year months in the forecast window "
         "fall back to a flat run-rate instead of a seasonal projection."
)
refresh = st.sidebar.button("Refresh data", type="primary")
if refresh:
    fetch_monthly_report.clear()

st.title("📈 Seasonal Forecast Dashboard")
st.caption(f"Forecasting {current_year} through year-end · actuals confirmed through "
           f"{MONTH_ORDER[last_actual_month - 1]} {current_year}")

# ---------------------------------------------------------------------------
# Fetch, forecast, and render
# ---------------------------------------------------------------------------
try:
    with st.spinner("Pulling P&L and Cash Flow data from QuickBooks..."):
        pl_prior = parse_monthly_report(fetch_monthly_report("ProfitAndLoss", f"{prior_year}-01-01", f"{prior_year}-12-31"))
        pl_current = parse_monthly_report(fetch_monthly_report(
            "ProfitAndLoss", f"{current_year}-01-01", f"{current_year}-{last_actual_month:02d}-{last_day:02d}"))
        cf_prior = parse_monthly_report(fetch_monthly_report("CashFlow", f"{prior_year}-01-01", f"{prior_year}-12-31"))
        cf_current = parse_monthly_report(fetch_monthly_report(
            "CashFlow", f"{current_year}-01-01", f"{current_year}-{last_actual_month:02d}-{last_day:02d}"))

    pl_forecast, pl_fallback_items = build_seasonal_forecast(
        pl_prior, pl_current, last_actual_month, current_year, min_prior_months)
    cf_forecast, cf_fallback_items = build_seasonal_forecast(
        cf_prior, cf_current, last_actual_month, current_year, min_prior_months)

    # -----------------------------------------------------------------
    # Data coverage - visible to the client, not buried in a console
    # -----------------------------------------------------------------
    with st.expander("📋 Data coverage check (what's driving this forecast)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**P&L — {prior_year} prior year**")
            cov = get_data_coverage(pl_prior, MONTH_ORDER)
            st.write(f"Months with activity: {', '.join(cov['nonzero_months']) or 'none'}")
            if cov["missing"]:
                st.warning(f"Missing: {', '.join(cov['missing'])}")
        with col2:
            st.markdown(f"**Cash Flow — {prior_year} prior year**")
            cov = get_data_coverage(cf_prior, MONTH_ORDER)
            st.write(f"Months with activity: {', '.join(cov['nonzero_months']) or 'none'}")
            if cov["missing"]:
                st.warning(f"Missing: {', '.join(cov['missing'])}")

        all_fallback = set(pl_fallback_items + cf_fallback_items)
        if all_fallback:
            st.info(
                f"**{len(all_fallback)} line item(s)** had fewer than {min_prior_months} months of "
                f"prior-year data in the forecast window, so they use a flat run-rate average "
                f"instead of a true seasonal projection: " + ", ".join(sorted(all_fallback))
            )

    # -----------------------------------------------------------------
    # P&L chart
    # -----------------------------------------------------------------
    st.subheader("Profit & Loss Forecast")
    key_pl_metrics = ["Total Income", "Total Expenses", "Net Income"]
    fig_pl = make_actual_forecast_chart(pl_forecast, key_pl_metrics,
                                         f"{current_year} P&L: Actual + Seasonal Forecast")
    st.plotly_chart(fig_pl, use_container_width=True)

    # -----------------------------------------------------------------
    # Prior year vs current year trend comparison (real data, not illustrative)
    # -----------------------------------------------------------------
    st.subheader(f"{prior_year} Baseline vs {current_year} (Actual + Forecast)")
    st.caption(
        f"Shows {prior_year}'s real monthly Total Income directly against {current_year}'s "
        f"combined actual-through-{MONTH_ORDER[last_actual_month - 1]} plus forecasted remainder - "
        f"this is the actual growth factor your data produces, not an assumption."
    )
    comparison_metric = st.selectbox(
        "Metric", ["Total Income", "Total Expenses", "Net Income"], index=0, key="trend_compare_metric"
    )
    prior_line = (pl_prior[pl_prior["Line_Item"] == comparison_metric]
                  .set_index("Month")["Value"].reindex(MONTH_ORDER))
    current_line = (pl_forecast[pl_forecast["Line_Item"] == comparison_metric]
                     .set_index("Month")["Value"].reindex(MONTH_ORDER))

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=MONTH_ORDER, y=prior_line.values, mode="lines+markers",
                                    name=f"{prior_year}", line=dict(color="#1f77b4", width=2)))
    fig_trend.add_trace(go.Scatter(x=MONTH_ORDER, y=current_line.values, mode="lines+markers",
                                    name=f"{current_year}", line=dict(color="#d62728", width=2)))
    fig_trend.update_layout(
        title=f"{comparison_metric}: {prior_year} vs {current_year}",
        xaxis_title="Month", yaxis_title="Amount ($)", template="plotly_white",
    )
    fig_trend.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------------------------------------------
    # Cash Flow chart
    # -----------------------------------------------------------------
    st.subheader("Cash Flow Forecast")
    key_cf_metrics = ["Net cash provided by operating activities", "Net increase in cash"]
    available_cf = [m for m in key_cf_metrics if m in cf_forecast["Line_Item"].unique()]
    if available_cf:
        fig_cf = make_actual_forecast_chart(cf_forecast, available_cf,
                                             f"{current_year} Cash Flow: Actual + Seasonal Forecast")
        st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info("Standard Cash Flow line items not found for this company's report labels.")

    # -----------------------------------------------------------------
    # Full-year summary table
    # -----------------------------------------------------------------
    st.subheader("Full-Year Summary")
    fy_summary = (pl_forecast[pl_forecast["Line_Item"].isin(key_pl_metrics)]
                  .groupby("Line_Item")["Value"].sum().rename(f"FY{current_year} Forecast").reset_index())
    fy_prior = (pl_prior[pl_prior["Line_Item"].isin(key_pl_metrics)]
                .groupby("Line_Item")["Value"].sum().rename(f"FY{prior_year} Actual"))
    fy_summary = fy_summary.merge(fy_prior, on="Line_Item", how="left")
    fy_summary["YoY Change %"] = (
        (fy_summary[f"FY{current_year} Forecast"] - fy_summary[f"FY{prior_year} Actual"])
        / fy_summary[f"FY{prior_year} Actual"] * 100
    )
    st.dataframe(
        fy_summary.style.format({
            f"FY{current_year} Forecast": "${:,.2f}",
            f"FY{prior_year} Actual": "${:,.2f}",
            "YoY Change %": "{:+.1f}%",
        }),
        use_container_width=True,
    )

    with st.expander("View underlying monthly data"):
        tab1, tab2 = st.tabs(["P&L", "Cash Flow"])
        with tab1:
            st.dataframe(pl_forecast[["Line_Item", "Month", "Value", "Actual_Forecast", "Method"]],
                         use_container_width=True)
        with tab2:
            st.dataframe(cf_forecast[["Line_Item", "Month", "Value", "Actual_Forecast", "Method"]],
                         use_container_width=True)

except requests.exceptions.HTTPError as http_err:
    st.error(f"HTTP error from QuickBooks API: {http_err}")
except Exception as err:
    st.error(f"Unexpected error: {err}")
