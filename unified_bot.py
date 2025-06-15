
import ccxt, ta, numpy as np, pandas as pd, sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time

# ✅ OKX API CONFIG — replace these
exchange = ccxt.okx({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'password': 'YOUR_PASSPHRASE',
    'enableRateLimit': True
})

# ✅ Constants
symbol = 'BTC/USDT'
features = ['ema_20', 'rsi', 'macd', 'atr']
seq_len = 30
usdt_per_trade = 10

# ✅ Fetch OHLCV Data
def fetch_data(timeframe='5m', limit=1200):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ✅ Technical Indicators
def add_indicators(df):
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df.dropna(inplace=True)
    return df

# ✅ Label for supervised learning
def label_data(df):
    df['label'] = 0
    df.loc[df['close'].shift(-1) > df['close'], 'label'] = 1
    df.loc[df['close'].shift(-1) < df['close'], 'label'] = 2
    df.dropna(inplace=True)
    return df

# ✅ Sequence maker
def create_sequences(df, features, seq_len=30):
    X, y = [], []
    for i in range(len(df) - seq_len - 1):
        seq = df[features].iloc[i:i+seq_len].values
        label = df['label'].iloc[i + seq_len]
        X.append(seq)
        y.append(label)
    return np.array(X), to_categorical(y, 3)

# ✅ Model trainer
def train_model():
    df = fetch_data(limit=1000)
    df = add_indicators(df)
    df = label_data(df)
    df[features] = StandardScaler().fit_transform(df[features])
    X, y = create_sequences(df, features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    model.save('lstm_model.h5')
    print("✅ Model trained & saved as lstm_model.h5")

    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("Accuracy:", round(accuracy_score(y_true, y_pred_labels) * 100, 2), "%")
    print(classification_report(y_true, y_pred_labels, target_names=["WAIT", "BUY", "SELL"]))

# ✅ Live prediction + trade
def predict_and_trade():
    model = load_model('lstm_model.h5')
    df = fetch_data(limit=seq_len+1)
    df = add_indicators(df)[-seq_len:]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    X = np.expand_dims(X, axis=0)

    probs = model.predict(X, verbose=0)[0]
    label = np.argmax(probs)
    confidence = float(np.max(probs))
    close_price = df['close'].iloc[-1]

    action = ['WAIT', 'BUY', 'SELL'][label]
    print(f"🔍 Signal: {action} | Confidence: {confidence:.2f} | Price: {close_price:.2f}")

    if action == 'BUY':
        exchange.create_market_buy_order(symbol, None, {'sz': str(usdt_per_trade)})
        print("✅ Market BUY order placed.")
    elif action == 'SELL':
        exchange.create_market_sell_order(symbol, None, {'sz': str(usdt_per_trade)})
        print("✅ Market SELL order placed.")

# ✅ Backtesting
def backtest():
    model = load_model('lstm_model.h5')
    df = fetch_data(limit=1200)
    df = add_indicators(df)
    trades = []

    for i in range(seq_len, len(df) - 2):
        df_block = df.iloc[i - seq_len:i]
        current_close = df['close'].iloc[i]
        future_close = df['close'].iloc[i + 2]

        X = StandardScaler().fit_transform(df_block[features])
        X = np.expand_dims(X, axis=0)
        pred = model.predict(X, verbose=0)[0]
        label = np.argmax(pred)
        confidence = float(np.max(pred))

        if label == 1:
            profit = (future_close - current_close) / current_close
            trades.append({'type': 'BUY', 'profit': profit, 'confidence': confidence})
        elif label == 2:
            profit = (current_close - future_close) / current_close
            trades.append({'type': 'SELL', 'profit': profit, 'confidence': confidence})

    results = pd.DataFrame(trades)
    print(f"Total Trades: {len(results)}")
    print(f"Win Rate: {(results['profit'] > 0).mean() * 100:.2f}%")
    print(f"Avg Profit: {results['profit'].mean() * 100:.2f}%")
    print(f"Total Return: {results['profit'].sum() * 100:.2f}%")

# ✅ Command Line Controller
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'predict'
    if mode == 'train':
        train_model()
    elif mode == 'predict':
        predict_and_trade()
    elif mode == 'backtest':
        backtest()
    else:
        print("❌ Invalid mode. Use: train | predict | backtest")
