"""
Fetch External Data
-------------------
Downloads sample earnings transcripts from Hugging Face and 
correlates them with stock data from Yahoo Finance.
"""
import os
import time
import requests
import yfinance as yf
from datasets import load_dataset
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/earnings_transcripts")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_transcripts(limit=3):
    """Download sample earnings transcripts."""
    print(f"Downloading {limit} samples from Hugging Face...")
    
    # Validated datasets
    datasets_to_try = [
        "financial_phrasebank", # Prioritize reliable dataset
        "jlh-ibm/earnings_call", 
        "kurry/sp500_earnings_transcripts"
    ]
    
    dataset = None
    for ds_name in datasets_to_try:
        try:
            print(f"Trying to load dataset: {ds_name}")
            # Use 'sentences_allagree' config for financial_phrasebank if needed
            config = "sentences_allagree" if "phrasebank" in ds_name else None
            
            # Enable trust_remote_code to handle legacy datasets with python scripts
            dataset = load_dataset(ds_name, config, split="train", streaming=True, trust_remote_code=True)
            # Verify we can actually get an item
            next(iter(dataset))
            print(f"✅ Successfully connected to {ds_name}")
            break
        except Exception as e:
            print(f"⚠️ Failed to load {ds_name}: {e}")
            dataset = None
    
    if not dataset:
        print("❌ Could not load any datasets. Switching to Yahoo Finance News Fallback...")
        # Fallback: Fetch news titles as text
        fallback_tickers = ["NVDA", "TSLA", "AAPL"]
        fallback_samples = []
        for ticker in fallback_tickers:
            try:
                print(f"Fetching news for {ticker}...")
                yf_ticker = yf.Ticker(ticker)
                news = yf_ticker.news
                if news:
                    lines = [f"News Briefing for {ticker}"]
                    for n in news:
                        lines.append(f"Title: {n.get('title', '')}")
                        lines.append(f"Publisher: {n.get('publisher', '')}")
                    
                    content = "\n\n".join(lines)
                    fname = f"{ticker}_News_{datetime.now().strftime('%Y-%m-%d')}.txt"
                    path = os.path.join(DATA_DIR, fname)
                    with open(path, "w") as f:
                        f.write(content)
                    print(f"  Saved news transcript: {fname}")
                    fallback_samples.append({"ticker": ticker, "date": datetime.now().strftime('%Y-%m-%d'), "file": path})
            except Exception as e:
                print(f"  Failed news fetch for {ticker}: {e}")
        return fallback_samples
    
    samples = []
    print("Processing items...")
    
    for i, item in enumerate(dataset):
        if len(samples) >= limit:
            break
        
        # Normalize fields across different datasets
        text = item.get("text") or item.get("sentence") or item.get("content") or ""
        ticker = item.get("ticker") or item.get("symbol") or "UNKNOWN"
        date_str = item.get("date") or datetime.now().strftime("%Y-%m-%d")
        
        # Filter for substantial content
        if len(text) < 500: 
            continue
            
        # Clean specific dataset quirks
        if isinstance(text, list):
            text = " ".join(text)
            
        # Save to file
        filename = f"{ticker}_{date_str}_{i}.txt"
        path = os.path.join(DATA_DIR, filename)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"  Saved {filename} ({len(text)} chars)")
        samples.append({"ticker": ticker, "date": date_str, "file": path})
        
    return samples

def fetch_stock_data(ticker, event_date_str):
    """Get stock data around the event date."""
    if ticker in ["UNKNOWN", ""]:
        print(f"  Skipping stock fetch for invalid ticker: {ticker}")
        return None
        
    print(f"Fetching stock data for {ticker} around {event_date_str}...")
    
    try:
        # Handle date parsing
        try:
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        except:
            event_date = datetime.now()
            
        start_date = event_date - timedelta(days=7)
        end_date = event_date + timedelta(days=7)
        
        # Configure yfinance to avoid 429s (simulate browser)
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        # Fetch
        ticker_obj = yf.Ticker(ticker, session=session)
        history = ticker_obj.history(start=start_date, end=end_date)
        
        if history.empty:
            print(f"  ⚠️ No stock data found for {ticker}")
            return None
            
        return history
        
    except Exception as e:
        print(f"  ❌ Error fetching stock data: {e}")
        return None

if __name__ == "__main__":
    print("Starting External Data Fetch Job...")
    samples = fetch_transcripts(limit=3)
    
    print(f"Found {len(samples)} transcripts. Fetching stock data...")
    for sample in samples:
        stock_data = fetch_stock_data(sample['ticker'], sample['date'])
        if stock_data is not None and not stock_data.empty:
            # Save stock data summary
            csv_path = sample['file'].replace(".txt", "_stock.csv")
            stock_data.to_csv(csv_path)
            print(f"  ✅ Saved stock data to {os.path.basename(csv_path)}")
            
        # Rate limit protection
        time.sleep(2)
        
    print("Job Complete.")
