"""
Data loading and preprocessing utilities for the NIFTY-50 Investment
Intelligence Platform.

Handles:
  - Loading the raw consolidated CSV
  - Consolidating symbols that were renamed over the years into a single,
    continuous series (e.g. INFOSYSTCH -> INFY)
  - Mapping each company to a sector / industry
  - Cleaning and basic feature derivation (returns)
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "NIFTY50_all.csv")

# ---------------------------------------------------------------------------
# Symbol consolidation map
# Several NIFTY-50 constituents changed their NSE trading symbol at some
# point in the dataset's history (corporate name changes, mergers, demergers).
# We map the OLD symbol -> the CURRENT symbol so that each company has one
# continuous price history. This is a manual mapping based on publicly known
# corporate history of these NSE-listed companies.
# ---------------------------------------------------------------------------
SYMBOL_CONSOLIDATION = {
    "HINDALC0": "HINDALCO",       # typo'd / old code -> Hindalco Industries
    "HINDLEVER": "HINDUNILVR",     # Hindustan Lever -> Hindustan Unilever
    "INFOSYSTCH": "INFY",          # Infosys Technologies -> Infosys
    "HEROHONDA": "HEROMOTOCO",     # Hero Honda -> Hero MotoCorp
    "TELCO": "TATAMOTORS",         # Tata Engineering & Locomotive Co -> Tata Motors
    "TISCO": "TATASTEEL",          # Tata Iron & Steel Co -> Tata Steel
    "JSWSTL": "JSWSTEEL",          # JSW Steel old code
    "BHARTI": "BHARTIARTL",        # Bharti Tele-Ventures -> Bharti Airtel
    "MUNDRAPORT": "ADANIPORTS",    # Mundra Port -> Adani Ports & SEZ
    "KOTAKMAH": "KOTAKBANK",       # Kotak Mahindra Finance -> Kotak Mahindra Bank
    "BAJAUTOFIN": "BAJFINANCE",    # Bajaj Auto Finance -> Bajaj Finance
    "UNIPHOS": "UPL",              # United Phosphorus -> UPL Limited
    "UTIBANK": "AXISBANK",         # UTI Bank -> Axis Bank
    "ZEETELE": "ZEEL",             # Zee Telefilms -> Zee Entertainment
    "SSLT": "VEDL",                # Sterlite Industries -> Vedanta Limited
}

# ---------------------------------------------------------------------------
# Sector / Industry mapping (manually curated, NIFTY-50 constituents)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "ADANIPORTS": ("Services", "Transport Infrastructure"),
    "ASIANPAINT": ("Consumer Goods", "Paints"),
    "AXISBANK": ("Financial Services", "Banking"),
    "BAJAJ-AUTO": ("Automobile", "2/3 Wheelers"),
    "BAJAJFINSV": ("Financial Services", "Financial Services - Diversified"),
    "BAJFINANCE": ("Financial Services", "NBFC"),
    "BHARTIARTL": ("Telecom", "Telecom Services"),
    "BPCL": ("Energy", "Oil & Gas - Refining & Marketing"),
    "BRITANNIA": ("Consumer Goods", "Food Products"),
    "CIPLA": ("Pharmaceuticals", "Pharmaceuticals"),
    "COALINDIA": ("Energy", "Mining - Coal"),
    "DRREDDY": ("Pharmaceuticals", "Pharmaceuticals"),
    "EICHERMOT": ("Automobile", "2/3 Wheelers"),
    "GAIL": ("Energy", "Gas Transmission/Marketing"),
    "GRASIM": ("Manufacturing", "Cement & Diversified"),
    "HCLTECH": ("Information Technology", "IT Services"),
    "HDFC": ("Financial Services", "Housing Finance"),
    "HDFCBANK": ("Financial Services", "Banking"),
    "HEROMOTOCO": ("Automobile", "2/3 Wheelers"),
    "HINDALCO": ("Manufacturing", "Aluminium / Metals"),
    "HINDUNILVR": ("Consumer Goods", "FMCG"),
    "ICICIBANK": ("Financial Services", "Banking"),
    "INDUSINDBK": ("Financial Services", "Banking"),
    "INFY": ("Information Technology", "IT Services"),
    "IOC": ("Energy", "Oil & Gas - Refining & Marketing"),
    "ITC": ("Consumer Goods", "FMCG / Tobacco"),
    "JSWSTEEL": ("Manufacturing", "Iron & Steel"),
    "KOTAKBANK": ("Financial Services", "Banking"),
    "LT": ("Manufacturing", "Construction & Engineering"),
    "M&M": ("Automobile", "Cars & Utility Vehicles / Tractors"),
    "MARUTI": ("Automobile", "Cars & Utility Vehicles"),
    "NESTLEIND": ("Consumer Goods", "FMCG / Food Products"),
    "NTPC": ("Energy", "Power Generation"),
    "ONGC": ("Energy", "Oil & Gas - Exploration"),
    "POWERGRID": ("Energy", "Power Transmission"),
    "RELIANCE": ("Energy", "Oil & Gas - Refining & Conglomerate"),
    "SBIN": ("Financial Services", "Banking"),
    "SESAGOA": ("Manufacturing", "Mining / Metals"),
    "SHREECEM": ("Manufacturing", "Cement"),
    "SSLT": ("Manufacturing", "Mining / Metals"),
    "SUNPHARMA": ("Pharmaceuticals", "Pharmaceuticals"),
    "TATAMOTORS": ("Automobile", "Cars & Commercial Vehicles"),
    "TATASTEEL": ("Manufacturing", "Iron & Steel"),
    "TCS": ("Information Technology", "IT Services"),
    "TECHM": ("Information Technology", "IT Services"),
    "TITAN": ("Consumer Goods", "Watches/Jewellery - Consumer Durables"),
    "ULTRACEMCO": ("Manufacturing", "Cement"),
    "UPL": ("Manufacturing", "Agro Chemicals"),
    "VEDL": ("Manufacturing", "Mining / Metals"),
    "ZEEL": ("Services", "Media & Entertainment"),
}

COMPANY_NAMES = {
    "ADANIPORTS": "Adani Ports & SEZ",
    "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BAJAJFINSV": "Bajaj Finserv",
    "BAJFINANCE": "Bajaj Finance",
    "BHARTIARTL": "Bharti Airtel",
    "BPCL": "Bharat Petroleum Corp.",
    "BRITANNIA": "Britannia Industries",
    "CIPLA": "Cipla",
    "COALINDIA": "Coal India",
    "DRREDDY": "Dr. Reddy's Laboratories",
    "EICHERMOT": "Eicher Motors",
    "GAIL": "GAIL India",
    "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies",
    "HDFC": "Housing Development Finance Corp.",
    "HDFCBANK": "HDFC Bank",
    "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",
    "INFY": "Infosys",
    "IOC": "Indian Oil Corp.",
    "ITC": "ITC Limited",
    "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra",
    "MARUTI": "Maruti Suzuki India",
    "NESTLEIND": "Nestle India",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil & Natural Gas Corp.",
    "POWERGRID": "Power Grid Corp. of India",
    "RELIANCE": "Reliance Industries",
    "SBIN": "State Bank of India",
    "SESAGOA": "Sesa Goa (Vedanta)",
    "SHREECEM": "Shree Cement",
    "SSLT": "Sterlite (Vedanta)",
    "SUNPHARMA": "Sun Pharmaceutical Industries",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "TCS": "Tata Consultancy Services",
    "TECHM": "Tech Mahindra",
    "TITAN": "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "UPL": "UPL Limited",
    "VEDL": "Vedanta Limited",
    "ZEEL": "Zee Entertainment Enterprises",
}


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw NIFTY-50 CSV."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def consolidate_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Map renamed/legacy symbols onto their current symbol."""
    df = df.copy()
    df["Symbol"] = df["Symbol"].replace(SYMBOL_CONSOLIDATION)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: sort, drop duplicate (Symbol, Date), fill small gaps."""
    df = df.copy()
    df = df.sort_values(["Symbol", "Date"])
    df = df.drop_duplicates(subset=["Symbol", "Date"], keep="last")

    # If two legacy-symbol rows exist for the same date after consolidation
    # (rare, around transition dates), keep the later filing.
    df = df.drop_duplicates(subset=["Symbol", "Date"], keep="last")

    # Fill missing Trades / Deliverable Volume / %Deliverable with 0 / NaN-safe
    for col in ["Trades", "Deliverable Volume", "%Deliverble"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Drop rows with non-positive prices (data errors)
    for col in ["Open", "High", "Low", "Close", "Prev Close"]:
        df = df[df[col] > 0]

    return df.reset_index(drop=True)


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily simple & log returns per symbol."""
    df = df.copy()
    df["Return"] = df.groupby("Symbol")["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df.groupby("Symbol")["Close"].shift(1))
    return df


def get_sector(symbol: str):
    return SECTOR_MAP.get(symbol, ("Unknown", "Unknown"))


def get_company_name(symbol: str):
    return COMPANY_NAMES.get(symbol, symbol)


def load_processed_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Full pipeline: load -> consolidate -> clean -> returns."""
    df = load_raw_data(path)
    df = consolidate_symbols(df)
    df = clean_data(df)
    df = add_returns(df)
    return df


def get_symbol_list(df: pd.DataFrame):
    return sorted(df["Symbol"].unique())


if __name__ == "__main__":
    data = load_processed_data()
    print("Total rows:", len(data))
    print("Symbols:", len(data["Symbol"].unique()))
    print(data["Symbol"].unique())
    print(data.groupby("Symbol")["Date"].agg(["min", "max", "count"]))
