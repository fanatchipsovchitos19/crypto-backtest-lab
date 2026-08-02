import pandas as pd
from typing import Optional
import httpx
import time
import os
from tqdm import tqdm
from src.config import DATA_DIR

class BinanceDataLoader:
    BASE_URLS = [
        "https://api.binance.com/api/v3",
        "https://api1.binance.com/api/v3",
        "https://api2.binance.com/api/v3",
        "https://api3.binance.com/api/v3",
        "https://api.binance.us/api/v3",
    ]
    INTERVAL_MS = {"1m":60_000,"5m":300_000,"15m":900_000,"30m":1_800_000,"1h":3_600_000,"4h":14_400_000,"1d":86_400_000,"1w":604_800_000}
    
    def __init__(self, proxy=None):
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    
    def fetch_ohlcv(self, symbol, interval="1h", start=None, end=None, use_cache=True):
        if use_cache:
            cached = self._load_from_cache(symbol, interval)
            if cached is not None and len(cached) > 0:
                if start: cached = cached[cached.index >= start]
                if end: cached = cached[cached.index < end]
                if len(cached) > 0:
                    print(f"  Загружено из кэша: {len(cached)} свечей")
                    return cached
        
        print(f"Загрузка {symbol} {interval} с Binance API...")
        start_ts = int(pd.Timestamp(start).timestamp() * 1000) if start else None
        end_ts = int(pd.Timestamp(end).timestamp() * 1000) if end else int(time.time() * 1000)
        if start_ts is None: start_ts = end_ts - 90 * 24 * 3600 * 1000
        
        all_klines, chunk_start, chunk_count = [], start_ts, 0
        max_chunks = 200
        total_duration = end_ts - chunk_start
        chunk_duration = 1000 * self.INTERVAL_MS.get(interval, 3_600_000)
        n_chunks = min(max_chunks, max(1, total_duration // chunk_duration + 1))
        
        working_url = None
        for url in self.BASE_URLS:
            try:
                with httpx.Client(timeout=10.0, follow_redirects=True) as c:
                    r = c.get(f"{url}/ping")
                    if r.status_code == 200: working_url = url; break
            except: continue
        
        if working_url is None:
            raise ConnectionError("Не удалось подключиться к Binance API.\nВключите VPN и попробуйте снова.")
        
        with tqdm(total=n_chunks, desc=f"Загрузка {symbol}") as pbar:
            while chunk_start < end_ts and chunk_count < max_chunks:
                chunk_count += 1
                params = {"symbol":symbol,"interval":interval,"startTime":chunk_start,"endTime":end_ts,"limit":1000}
                try:
                    with httpx.Client(timeout=30.0, follow_redirects=True) as c:
                        response = c.get(f"{working_url}/klines", params=params)
                except Exception as e:
                    if chunk_count <= 3: print("  Сетевая ошибка, повтор..."); time.sleep(5); continue
                    else: raise ConnectionError(str(e))
                
                if response.status_code == 451:
                    for url in self.BASE_URLS:
                        if url != working_url:
                            try:
                                with httpx.Client(timeout=10.0, follow_redirects=True) as c2:
                                    if c2.get(f"{url}/ping").status_code == 200:
                                        working_url = url
                                        response = c2.get(f"{url}/klines", params=params)
                                        break
                            except: continue
                    if response.status_code == 451: raise ConnectionError("Binance API недоступен. Включите VPN.")
                if response.status_code == 429: print("  Rate limit, ждём 15 сек..."); time.sleep(15); continue
                if response.status_code != 200: print(f"  Ошибка API: {response.status_code}"); time.sleep(5); continue
                
                klines = response.json()
                if not klines: break
                all_klines.extend(klines)
                chunk_start = klines[-1][0] + 1
                pbar.update(1)
                time.sleep(0.3)
        
        if not all_klines: raise ValueError(f"Не удалось загрузить данные для {symbol} {interval}.")
        df = self._klines_to_dataframe(all_klines)
        self._save_to_cache(df, symbol, interval)
        return df
    
    def _klines_to_dataframe(self, klines):
        cols = ["timestamp","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(klines, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for c in ["open","high","low","close","volume","quote_volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()
        return df[~df.index.duplicated(keep="first")]
    
    def _get_cache_path(self, symbol, interval): return self.cache_dir / f"{symbol}_{interval}.parquet"
    def _load_from_cache(self, symbol, interval):
        p = self._get_cache_path(symbol, interval)
        if p.exists():
            try: return pd.read_parquet(p)
            except: pass
        return None
    def _save_to_cache(self, df, symbol, interval):
        try: df.to_parquet(self._get_cache_path(symbol, interval))
        except Exception as e: print(f"  Кэш не сохранён: {e}")