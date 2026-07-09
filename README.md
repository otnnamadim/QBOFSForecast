# QBOFSForecast

Forecasts QuickBooks Online Income Statement (P&L) and Cash Flow through year-end using a **seasonal naive + growth** method — a simple, explainable approach suited for CFO-style reporting rather than a statistical/ML model.

## How it works

1. Pulls monthly actuals from the QuickBooks Online Reports API for both the prior year (Jan–Dec) and the current year (Jan through the last fully closed month), for **Profit & Loss** and **Cash Flow**.
2. For each line item, calculates a year-over-year growth factor: this year's year-to-date total ÷ the same months' total last year.
3. Projects each remaining month of the current year as: *last year's actual for that month × growth factor.* This carries forward whatever seasonal shape existed last year (a slow Q1, a Q4 spike, etc.) while scaling it to how the business is actually trending this year.
4. Falls back to a flat run-rate (average of this year's actual months) for any line item with fewer than 3 months of prior-year data in the forecast window — this keeps sparse or newly-created accounts from projecting to zero.
5. Outputs interactive actual-vs-forecast charts (Plotly) for Total Income, Total Expenses, and Net Income, plus key Cash Flow metrics, and a full-year summary table comparing the forecast total to last year's actual.

## What this is *not*

This is a **seasonal-naive-with-growth model**, not a statistical or machine-learning forecast. It assumes this year's month-to-month pattern will resemble last year's — a reasonable, explainable baseline for a forecast conversation, but not a substitute for judgment, and it can be distorted by one-off anomalies in either year's history (e.g. a large one-time transaction).

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `requests`, `python-dotenv`, `plotly`

```bash
pip install pandas numpy requests python-dotenv plotly
```

## Setup

1. Create a QuickBooks Online app at [Intuit Developer](https://developer.intuit.com) and obtain your Client ID, Client Secret, and a Refresh Token (via OAuth 2.0) for the company you want to forecast.
2. Create a file named `API_Keys.env` in the same directory as the script:

    ```env
    Client_ID=your-client-id
    Client_Secret=your-client-secret
    Refresh_Token=your-refresh-token
    Realm_ID=your-company-realm-id
    ```

3. **Do not commit `API_Keys.env`.** Add it to `.gitignore` before your first commit.

By default, `BASE_URL` points at the QuickBooks **sandbox** environment (`sandbox-quickbooks.api.intuit.com`). Swap to `https://quickbooks.api.intuit.com` for a production company.

## Usage

```bash
python QBOForecast
```

The script prints a data-coverage summary for both years (so you can see up front whether a line item will use the seasonal method or the run-rate fallback), then opens interactive P&L and Cash Flow charts in your browser, followed by a full-year forecast summary table in the console.

## Example output

- **P&L chart:** Total Income, Total Expenses, and Net Income, actuals as solid lines transitioning to dashed forecast lines through December.
- **Cash Flow chart:** Net cash provided by operating activities and net increase in cash, same actual → forecast treatment.
- **Console summary:** full-year forecast total vs. prior-year actual and % change, for each key P&L metric.

## Limitations

- Requires at least some prior-year transaction history in QuickBooks to forecast meaningfully; sparse sandbox/test companies will lean heavily on the run-rate fallback.
- Cash Flow line-item labels (`Net cash provided by operating activities`, etc.) follow QuickBooks' standard report labels — if your chart of accounts produces different labels, update `key_cf_metrics` in the script.
- Demoed here against a QuickBooks Online **sandbox** company, not an actual live, client account.
