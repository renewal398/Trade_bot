import ccxt
import pandas as pd
import numpy as np
import ta
import time
import logging
import json
import os
import smtplib
from email.mime.text import MIMEText

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log", mode="a")
    ]
)

# -------------------------------
# Parameter Loader & Default Parameters
# -------------------------------
def load_parameters(config_file="config.json"):
    """
    Load parameters from a JSON configuration file.
    If the file is not found, default parameters are returned.
    """
    defaults = {
        "emaFastLen": 50,
        "emaSlowLen": 200,
        "rsiLen": 14,
        "stochLen": 14,
        "bbLen": 20,
        "bbMult": 2.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_threshold_long": 55,
        "rsi_threshold_short": 45,
        "volAvg_window": 20,
        "stoch_smooth_k": 3,
        "stoch_smooth_d": 3,
        # Email settings
        "sender_email": "renewal39@gmail.com",  # Replace with your sender email
        "sender_password": "iqgq ygia kfsx ybrw",                   # Replace with your sender email or app password
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
    }
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                params = json.load(f)
            logging.info("Parameters loaded from config file.")
            defaults.update(params)
        except Exception as e:
            logging.error(f"Error loading parameters from {config_file}: {e}")
    else:
        logging.info("Config file not found. Using default parameters.")
    return defaults

# Load parameters and assign to variables
params = load_parameters()
emaFastLen = params["emaFastLen"]
emaSlowLen = params["emaSlowLen"]
rsiLen     = params["rsiLen"]
stochLen   = params["stochLen"]
bbLen      = params["bbLen"]
bbMult     = params["bbMult"]

# -------------------------------
# Setup exchange (MEXC example)
# -------------------------------
exchange = ccxt.mexc({'enableRateLimit': True})

# -------------------------------
# List of Symbols to Check
# -------------------------------
symbols = [
    "MOEW/USDT",
    "URO/USDT",
    "AI16Z/USDT",
    "FLOCK/USDT",
    "SUI/USDT",
    "AVAAI/USDT",
    "PIPPIN/USDT",
    "YNE/USDT",
    "UFD/USDT",
    "PEPU/USDT",
    "GORK/USDT",   # Gork AI Agent (GORK)
    "SOON/USDT",   # Soon (SOON)
    "MAI/USDT",    # MuxyAI (MAI)
    "STAR/USDT",   # Sponstar (STAR)
    "GOONC/USDT",  # Gooncoin (GOONC)
    "SUN/USDT",    # Sun Token (SUN)
    "SHIB/USDT",   # Shiba Inu (SHIB)
    "PEPE/USDT",   # Pepe (PEPE)
    "SOL/USDT",    # Solana (SOL)
    "XRP/USDT",    # Ripple (XRP)
    "ADA/USDT",    # Cardano (ADA)
    "BTC/USDT",    # Bitcoin (BTC)
    "ETH/USDT",    # Ethereum (ETH)
    "BNB/USDT",    # Binance Coin (BNB)
    "PENGU/USDT"   # Pudgy Penguins (PENGU)
]

# -------------------------------
# Function: Fetch Data
# -------------------------------
def fetch_data(symbol, timeframe='1h', limit=300):
    """
    Fetch historical OHLCV data with error handling.
    Returns a pandas DataFrame with the relevant columns.
    """
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"Fetched {len(df)} records for {symbol} on timeframe {timeframe}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# -------------------------------
# Function: Compute Indicators
# -------------------------------
def compute_indicators(df):
    """
    Compute technical indicators similar to the Pine script.
    Adds EMAs, MACD, RSI, Stochastic RSI, Bollinger Bands, volume filter,
    and price action confirmation to the DataFrame.
    """
    try:
        # === EMAs ===
        df['emaFast'] = df['close'].ewm(span=emaFastLen, adjust=False).mean()
        df['emaSlow'] = df['close'].ewm(span=emaSlowLen, adjust=False).mean()

        # === MACD ===
        macd = ta.trend.MACD(
            df['close'], 
            window_slow=params["macd_slow"], 
            window_fast=params["macd_fast"], 
            window_sign=params["macd_signal"]
        )
        df['macdLine'] = macd.macd()
        df['signalLine'] = macd.macd_signal()
        df['macdHist'] = macd.macd_diff()

        # === RSI ===
        rsi_indicator = ta.momentum.RSIIndicator(df['close'], window=rsiLen)
        df['rsi'] = rsi_indicator.rsi()

        # === True Stochastic RSI ===
        df['rsi_min'] = df['rsi'].rolling(window=stochLen).min()
        df['rsi_max'] = df['rsi'].rolling(window=stochLen).max()
        df['stochRSI'] = np.where(
            (df['rsi_max'] - df['rsi_min']) == 0,
            0,
            (df['rsi'] - df['rsi_min']) / (df['rsi_max'] - df['rsi_min'])
        )
        df['k'] = df['stochRSI'].rolling(window=params["stoch_smooth_k"]).mean()
        df['d'] = df['k'].rolling(window=params["stoch_smooth_d"]).mean()

        # === Bollinger Bands ===
        df['basis'] = df['close'].rolling(window=bbLen).mean()
        df['std'] = df['close'].rolling(window=bbLen).std()
        df['upperBB'] = df['basis'] + bbMult * df['std']
        df['lowerBB'] = df['basis'] - bbMult * df['std']

        # === Volume Filter ===
        df['volAvg'] = df['volume'].rolling(window=params["volAvg_window"]).mean()

        # === Price Action Confirmation ===
        df['prevHigh'] = df['high'].shift(1)
        df['prevLow'] = df['low'].shift(1)
        df['bullishBreakout'] = df['close'] > df['prevHigh']
        df['bearishBreakdown'] = df['close'] < df['prevLow']

        logging.info("Technical indicators computed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error computing indicators: {e}")
        return df

# -------------------------------
# Function: Check Signals
# -------------------------------
def check_signals(df):
    """
    Determine if the latest bar triggers a buy or sell signal.
    Returns a tuple of booleans for (longSignal, shortSignal).
    """
    try:
        if df is None or df.empty:
            logging.warning("DataFrame is empty. Cannot check signals.")
            return False, False
        
        latest = df.iloc[-1]
        
        longCondition = (
            latest['close'] > latest['emaFast'] and
            latest['close'] > latest['emaSlow'] and
            latest['macdLine'] > latest['signalLine'] and
            latest['macdHist'] > 0 and
            latest['rsi'] > params["rsi_threshold_long"] and
            latest['k'] > latest['d'] and
            latest['close'] > latest['basis'] and
            latest['volume'] > latest['volAvg'] and
            latest['bullishBreakout']
        )

        shortCondition = (
            latest['close'] < latest['emaFast'] and
            latest['close'] < latest['emaSlow'] and
            latest['macdLine'] < latest['signalLine'] and
            latest['macdHist'] < 0 and
            latest['rsi'] < params["rsi_threshold_short"] and
            latest['k'] < latest['d'] and
            latest['close'] < latest['basis'] and
            latest['volume'] > latest['volAvg'] and
            latest['bearishBreakdown']
        )

        logging.info(f"Signals checked: Long signal: {longCondition}, Short signal: {shortCondition}")
        return longCondition, shortCondition
    except Exception as e:
        logging.error(f"Error checking signals: {e}")
        return False, False

# -------------------------------
# Function: Send Alert via Email
# -------------------------------
def send_alert(message):
    """
    Sends an email alert to the designated recipient.
    For this example, the recipient is set to prakashsteve5@gmail.com.
    """
    try:
        subject = "Trading Bot Alert"
        sender_email = params.get("sender_email")
        sender_password = params.get("sender_password")
        recipient_email = "prakashsteve5@gmail.com"
        smtp_server = params.get("smtp_server")
        smtp_port = params.get("smtp_port")

        if not sender_email or not sender_password:
            logging.error("Sender email credentials are not set. Check your configuration.")
            return

        # Create the email message
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        # Send the email via SMTP with TLS
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()

        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Error sending email alert: {e}")

# -------------------------------
# Main Execution Flow (Looping Through Symbols)
# -------------------------------
def main():
    timeframe = '1h'
    limit = 300

    for symbol in symbols:
        try:
            # Fetch data for the symbol
            df = fetch_data(symbol, timeframe, limit)
            if df is None or df.empty:
                logging.error(f"No data fetched for {symbol}. Skipping.")
                continue

            # Compute technical indicators
            df = compute_indicators(df)

            # Check for signals
            longSignal, shortSignal = check_signals(df)

            # Get the last close price for calculations
            last_close = df.iloc[-1]['close']
            # Read percentage values from config (with defaults)
            tp_pct = params.get("take_profit_pct", 0.04)
            sl_pct = params.get("stop_loss_pct", 0.02)
            leverage = params.get("leverage", 1)

            # For long (buy) signal:
            if longSignal:
                take_profit = last_close * (1 + tp_pct)
                stop_loss = last_close * (1 - sl_pct)
                # Formula: Profit % = ((TakeProfit - Entry Price) / Entry Price) * Leverage * 100%
                profit_formula = (f"Profit formula (Long): ((TakeProfit - Entry Price) / Entry Price) * Leverage * 100%.\n"
                                  f"In numbers: (({take_profit:.2f} - {last_close:.2f}) / {last_close:.2f}) * {leverage} * 100 = "
                                  f"{((take_profit - last_close) / last_close) * leverage * 100:.2f}%")
                message = (f"Buy signal triggered for {symbol} at price {last_close:.2f}.\n"
                           f"Take Profit: {take_profit:.2f}, Stop Loss: {stop_loss:.2f}.\n"
                           f"{profit_formula}")
                send_alert(message)
            
            # For short (sell) signal:
            elif shortSignal:
                take_profit = last_close * (1 - tp_pct)
                stop_loss = last_close * (1 + sl_pct)
                # Formula for short: Profit % = ((Entry Price - TakeProfit) / Entry Price) * Leverage * 100%
                profit_formula = (f"Profit formula (Short): ((Entry Price - TakeProfit) / Entry Price) * Leverage * 100%.\n"
                                  f"In numbers: (({last_close:.2f} - {take_profit:.2f}) / {last_close:.2f}) * {leverage} * 100 = "
                                  f"{((last_close - take_profit) / last_close) * leverage * 100:.2f}%")
                message = (f"Sell signal triggered for {symbol} at price {last_close:.2f}.\n"
                           f"Take Profit: {take_profit:.2f}, Stop Loss: {stop_loss:.2f}.\n"
                           f"{profit_formula}")
                send_alert(message)
            else:
                logging.info(f"No signal for {symbol} at {df.iloc[-1]['timestamp']} (Close: {last_close:.2f})")
            
            # Sleep briefly between requests to honor rate limits
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")


if __name__ == '__main__':
    main()
