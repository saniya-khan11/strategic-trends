

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import datetime as dt

# ----------------------------------------
# NIFTY 50 STOCK NAME → TICKER MAPPING
# ----------------------------------------
NIFTY50_MAP = {
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "apollo hospitals": "APOLLOHOSP.NS",
    "asian paints": "ASIANPAINT.NS",
    "axis bank": "AXISBANK.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "bharat electronics": "BEL.NS",
    "bharat petroleum": "BPCL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "cipla": "CIPLA.NS",
    "coal india": "COALINDIA.NS",
    "divis labs": "DIVISLAB.NS",
    "dr reddy": "DRREDDY.NS",
    "eicher motors": "EICHERMOT.NS",
    "grasim": "GRASIM.NS",
    "hcl tech": "HCLTECH.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc life": "HDFCLIFE.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hindalco": "HINDALCO.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "icici bank": "ICICIBANK.NS",
    "itc": "ITC.NS",
    "indusind bank": "INDUSINDBK.NS",
    "infosys": "INFY.NS",
    "jsw steel": "JSWSTEEL.NS",
    "kotak bank": "KOTAKBANK.NS",
    "l&t": "LT.NS",
    "ltimindtree": "LTIM.NS",
    "maruti": "MARUTI.NS",
    "m&m": "M&M.NS",
    "nestle": "NESTLEIND.NS",
    "ntpc": "NTPC.NS",
    "ongc": "ONGC.NS",
    "powergrid": "POWERGRID.NS",
    "reliance": "RELIANCE.NS",
    "sbi": "SBIN.NS",
    "sbi life": "SBILIFE.NS",
    "shriram finance": "SHRIRAMFIN.NS",
    "sun pharma": "SUNPHARMA.NS",
    "tata consumer": "TATACONSUM.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "tcs": "TCS.NS",
    "tech mahindra": "TECHM.NS",
    "titan": "TITAN.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "wipro": "WIPRO.NS"
}

# ----------------------------------------
# FUNCTION: MAP USER INPUT → TICKERS
# ----------------------------------------
def map_to_tickers(user_stocks):
    tickers = []
    invalid = []

    for stock in user_stocks:
        key = stock.strip().lower()
        if key in NIFTY50_MAP:
            tickers.append(NIFTY50_MAP[key])
        else:
            invalid.append(stock)

    return tickers, invalid

# ----------------------------------------
# MAIN FUNCTION: CALCULATE VAR
# ----------------------------------------
def calculate_var(user_stocks):

    tickers, invalid = map_to_tickers(user_stocks)

    if len(tickers) == 0:
        return {"error": "No valid Nifty 50 stocks selected"}

    # Equal weights
    weights = np.array([1/len(tickers)] * len(tickers))

    start = dt.datetime(2020, 1, 1)
    end   = dt.datetime.today()

    prices = yf.download(tickers, start=start, end=end)['Close']

    if prices.empty:
        return {"error": "Could not fetch stock data"}

    prices = prices.dropna().ffill().bfill()

    logR = np.log(prices / prices.shift(1)).dropna()

    portR = logR.dot(weights)

    mu = portR.mean()
    sigma = portR.std()

    z = stats.norm.ppf(0.05)

    var = -(mu + sigma * z)

    return {
        "VaR": round(var, 5),
        "Mean Return": round(mu, 5),
        "Volatility": round(sigma, 5),
        "Selected Stocks": tickers,
        "Invalid Stocks": invalid
    }