
import yfinance as yf
from datetime import date, datetime, timedelta


def get_yahoo_data(days = 500, symbols = ['VIX']) :

    today = date.today()
    start_date = datetime.now() - timedelta(days = days)

    df = {}
    for ticker in symbols :
        df[ticker] = yf.download(ticker , start_date, today)
        print(ticker)
        print(df[ticker].head())

    return df