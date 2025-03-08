import yfinance as yf
df = yf.download('035420.KS', start='2023-06-01', end='2024-06-30')
print(df.head())
