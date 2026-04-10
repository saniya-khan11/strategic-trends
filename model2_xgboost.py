
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import torch
import optuna
from datetime import datetime, timedelta

from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# ======================================================
# CONFIG
# ======================================================
NEWS_API_KEY = "YOUR_API_KEY"

STOCKS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS",
    "ICICIBANK.NS","HINDUNILVR.NS","ITC.NS","SBIN.NS"
]

START = "2020-01-01"
END = datetime.today().strftime('%Y-%m-%d')

FEATURES = [
    'Open','High','Low','Close','Volume',
    'EMA','SMA','momentum','breakout',
    'RSI','ema_diff','acceleration','vol_spike',
    'MACD','MACD_signal','MACD_diff',
    'bb_width','volatility','trend_strength',
    'returns','lag1','lag2','lag3','sentiment'
]

# ======================================================
# NEWS FETCH
# ======================================================
news_cache = {}

def fetch_news(company, date):
    key = f"{company}_{date}"
    if key in news_cache:
        return news_cache[key]

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company,
        "from": date,
        "to": date,
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params).json()
    articles = response.get("articles", [])

    texts = [
        (a["title"] + " " + str(a["description"]))
        for a in articles if a["title"]
    ]

    news_cache[key] = texts
    return texts

# ======================================================
# SENTIMENT MODEL (FinBERT)
# ======================================================
class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def get_score(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return (probs[:,2] - probs[:,0]).detach().numpy()

# ======================================================
# STOCK DATA
# ======================================================
def fetch_stock_data(ticker):
    df = yf.download(ticker, start=START, end=END)

    df = df[['Open','High','Low','Close','Volume']].copy()

    df['EMA'] = EMAIndicator(df['Close'], window=10).ema_indicator()
    df['SMA'] = SMAIndicator(df['Close'], window=10).sma_indicator()
    df['ema_diff'] = df['EMA'] - df['SMA']
    df['momentum'] = df['Close'] - df['Close'].shift(5)
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['vol_spike'] = df['Volume'] / df['Volume'].rolling(10).mean()
    df['trend_strength'] = df['EMA'] - df['SMA']

    df['breakout'] = (df['Close'] > df['High'].rolling(10).max().shift(1)).astype(int)

    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    bb = BollingerBands(df['Close'])
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()

    df['volatility'] = df['Close'].pct_change().rolling(10).std()
    df['acceleration'] = df['Close'].diff(2)

    df['returns'] = df['Close'].pct_change()
    df['lag1'] = df['returns'].shift(1)
    df['lag2'] = df['returns'].shift(2)
    df['lag3'] = df['returns'].shift(3)

    df.dropna(inplace=True)
    return df

# ======================================================
# SENTIMENT ADD
# ======================================================
def add_sentiment(df, stock, sentiment_model):
    sentiments = []

    cutoff = datetime.today() - timedelta(days=30)

    for date in df.index:
        if date < cutoff:
            sentiments.append(0)
            continue

        news = fetch_news(stock, date.strftime("%Y-%m-%d"))
        if news:
            sentiments.append(np.mean(sentiment_model.get_score(news)))
        else:
            sentiments.append(0)

    df['sentiment'] = sentiments
    return df

# ======================================================
# LABELS
# ======================================================
def create_labels(df):
    df['future_return'] = df['Close'].shift(-5) / df['Close'] - 1
    df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
    threshold = 0.8 * df['volatility_20']

    df['signal'] = np.where(df['future_return'] >= threshold, 1, 0)
    df.dropna(inplace=True)
    return df

# ======================================================
# BUILD DATASET
# ======================================================
def build_dataset():
    sentiment_model = SentimentAnalyzer()
    all_data = []

    for stock in STOCKS:
        df = fetch_stock_data(stock)
        df = add_sentiment(df, stock, sentiment_model)
        df = create_labels(df)
        df['stock'] = stock
        all_data.append(df)

    return pd.concat(all_data)

# ======================================================
# TRAINING
# ======================================================
def prepare_data(df):
    X = df[FEATURES]
    y = df['signal']
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def balance_data(X, y):
    sm = SMOTE()
    return sm.fit_resample(X, y)

def train_model(X, y):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective='binary:logistic'
    )
    model.fit(X, y)
    return model

# ======================================================
# MAIN PIPELINE
# ======================================================
df = build_dataset()

X_train, X_test, y_train, y_test = prepare_data(df)
X_train, y_train = balance_data(X_train, y_train)

model = train_model(X_train, y_train)

# ======================================================
# GENERATE FINAL SIGNALS (IMPORTANT)
# ======================================================
latest_probs = model.predict_proba(X_test.iloc[-len(STOCKS):])[:, 1]

results = []

for i, stock in enumerate(STOCKS):

    prob = latest_probs[i]

    if prob > 0.62:
        signal = "BUY"
    elif prob < 0.38:
        signal = "SELL"
    else:
        signal = "HOLD"

    results.append({
        "Stock": stock,
        "Signal": signal,
        "Confidence": round(prob, 2),
        "Date": datetime.today().strftime("%Y-%m-%d")
    })

results_df = pd.DataFrame(results)

# ======================================================
# SAVE OUTPUT (KEY PART)
# ======================================================
results_df.to_csv("model2_output.csv", index=False)

print("✅ Signals saved to model2_output.csv")