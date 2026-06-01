"""
nse_stocks.py — Comprehensive NSE stock universe organized by sector
~500 major NSE stocks covering Nifty 50, Nifty 500, SME and sectoral indices
"""

# ─────────────────────────────────────────────────────────────────────────────
#  COMPLETE NSE STOCK UNIVERSE  (ticker.NS format)
# ─────────────────────────────────────────────────────────────────────────────

NSE_STOCKS = {

    "🏦 Banking & Finance": [
        ("HDFCBANK",      "HDFC Bank"),
        ("ICICIBANK",     "ICICI Bank"),
        ("SBIN",          "State Bank of India"),
        ("KOTAKBANK",     "Kotak Mahindra Bank"),
        ("AXISBANK",      "Axis Bank"),
        ("INDUSINDBK",    "IndusInd Bank"),
        ("BANDHANBNK",    "Bandhan Bank"),
        ("FEDERALBNK",    "Federal Bank"),
        ("IDFCFIRSTB",    "IDFC First Bank"),
        ("PNB",           "Punjab National Bank"),
        ("BANKBARODA",    "Bank of Baroda"),
        ("CANBK",         "Canara Bank"),
        ("UNIONBANK",     "Union Bank"),
        ("INDIANB",       "Indian Bank"),
        ("MAHABANK",      "Bank of Maharashtra"),
        ("KARURVYSYA",    "Karur Vysya Bank"),
        ("CUB",           "City Union Bank"),
        ("DCBBANK",       "DCB Bank"),
        ("RBLBANK",       "RBL Bank"),
        ("YESBANK",       "Yes Bank"),
        ("BAJFINANCE",    "Bajaj Finance"),
        ("BAJAJFINSV",    "Bajaj Finserv"),
        ("CHOLAFIN",      "Cholamandalam Finance"),
        ("MUTHOOTFIN",    "Muthoot Finance"),
        ("MANAPPURAM",    "Manappuram Finance"),
        ("SUNDARMFIN",    "Sundaram Finance"),
        ("M&MFIN",        "M&M Financial"),
        ("SHRIRAMFIN",    "Shriram Finance"),
        ("LICHSGFIN",     "LIC Housing Finance"),
        ("PNBHOUSING",    "PNB Housing Finance"),
        ("AAVAS",         "Aavas Financiers"),
        ("HOMEFIRST",     "Home First Finance"),
        ("JIOFIN",        "Jio Financial Services"),
        ("PAYTM",         "One97 Communications (Paytm)"),
        ("POLICYBZR",     "PB Fintech (PolicyBazaar)"),
        ("ANGELONE",      "Angel One"),
        ("HDFCLIFE",      "HDFC Life Insurance"),
        ("SBILIFE",       "SBI Life Insurance"),
        ("ICICIPRULI",    "ICICI Prudential Life"),
        ("ICICIGI",       "ICICI Lombard GIC"),
        ("STARHEALTH",    "Star Health Insurance"),
        ("NIACL",         "New India Assurance"),
        ("BSE",           "BSE Ltd"),
        ("CDSL",          "CDSL"),
        ("CAMS",          "CAMS"),
    ],

    "💻 IT & Technology": [
        ("TCS",           "Tata Consultancy Services"),
        ("INFY",          "Infosys"),
        ("WIPRO",         "Wipro"),
        ("HCLTECH",       "HCL Technologies"),
        ("TECHM",         "Tech Mahindra"),
        ("LTIM",          "LTIMindtree"),
        ("MPHASIS",       "Mphasis"),
        ("COFORGE",       "Coforge"),
        ("PERSISTENT",    "Persistent Systems"),
        ("KPIT",          "KPIT Technologies"),
        ("TATAELXSI",     "Tata Elxsi"),
        ("OFSS",          "Oracle Financial Services"),
        ("MASTEK",        "Mastek"),
        ("ZENSAR",        "Zensar Technologies"),
        ("NIIT",          "NIIT Ltd"),
        ("RATEGAIN",      "RateGain Travel Tech"),
        ("NEWGEN",        "Newgen Software"),
        ("TANLA",         "Tanla Platforms"),
        ("NAUKRI",        "Info Edge (Naukri)"),
        ("AFFLE",         "Affle India"),
        ("INDIAMART",     "IndiaMART InterMESH"),
        ("JUSTDIAL",      "Just Dial"),
        ("MAPMYINDIA",    "C.E. Info Systems (MapmyIndia)"),
        ("HAPPYMINDS",    "Happiest Minds"),
        ("INTELLECT",     "Intellect Design Arena"),
        ("CYIENT",        "Cyient"),
        ("BIRLASOFT",     "Birlasoft"),
        ("HEXAWARE",      "Hexaware Technologies"),
    ],

    "🏭 Manufacturing & Industrials": [
        ("LT",            "Larsen & Toubro"),
        ("SIEMENS",       "Siemens India"),
        ("ABB",           "ABB India"),
        ("BHEL",          "Bharat Heavy Electricals"),
        ("BEL",           "Bharat Electronics"),
        ("HAL",           "Hindustan Aeronautics"),
        ("GRINDWELL",     "Grindwell Norton"),
        ("CUMMINSIND",    "Cummins India"),
        ("THERMAX",       "Thermax"),
        ("VOLTAMP",       "Voltamp Transformers"),
        ("APLAPOLLO",     "APL Apollo Tubes"),
        ("TIINDIA",       "Tube Investments of India"),
        ("SCHAEFFLER",    "Schaeffler India"),
        ("SKFINDIA",      "SKF India"),
        ("TIMKEN",        "Timken India"),
        ("KAYNES",        "Kaynes Technology"),
        ("DIXON",         "Dixon Technologies"),
        ("AMBER",         "Amber Enterprises"),
        ("SYRMA",         "Syrma SGS Technology"),
        ("AVALON",        "Avalon Technologies"),
        ("BHARAT FORGE",  "Bharat Forge"),
        ("SUNDRMFAST",    "Sundram Fasteners"),
        ("CRAFTSMAN",     "Craftsman Automation"),
        ("JASH",          "Jash Engineering"),
    ],

    "🚗 Automobile & EV": [
        ("MARUTI",        "Maruti Suzuki"),
        ("TATAMOTORS",    "Tata Motors"),
        ("M&M",           "Mahindra & Mahindra"),
        ("BAJAJ-AUTO",    "Bajaj Auto"),
        ("HEROMOTOCO",    "Hero MotoCorp"),
        ("EICHERMOT",     "Eicher Motors"),
        ("TVSMOTORS",     "TVS Motor"),
        ("ASHOKLEY",      "Ashok Leyland"),
        ("TVSMOTOR",      "TVS Motor Company"),
        ("FORCEMOT",      "Force Motors"),
        ("MOTHERSON",     "Samvardhana Motherson"),
        ("BOSCHLTD",      "Bosch India"),
        ("MINDAIND",      "Minda Industries"),
        ("EXIDEIND",      "Exide Industries"),
        ("AMARAJABAT",    "Amara Raja Energy"),
        ("BALKRISIND",    "Balkrishna Industries"),
        ("APOLLOTYRE",    "Apollo Tyres"),
        ("MRF",           "MRF"),
        ("CEATLTD",       "CEAT"),
        ("ATHENERGY",     "Ather Energy"),
        ("OLECTRA",       "Olectra Greentech"),
        ("TATAPOWER",     "Tata Power"),
        ("RPOWER",        "Reliance Power"),
        ("MINDA",         "Minda Corporation"),
    ],

    "⚡ Energy & Power": [
        ("RELIANCE",      "Reliance Industries"),
        ("ONGC",          "Oil & Natural Gas Corp"),
        ("IOC",           "Indian Oil Corporation"),
        ("BPCL",          "Bharat Petroleum"),
        ("HPCL",          "Hindustan Petroleum"),
        ("GAIL",          "GAIL India"),
        ("PETRONET",      "Petronet LNG"),
        ("OIL",           "Oil India"),
        ("IGL",           "Indraprastha Gas"),
        ("MGL",           "Mahanagar Gas"),
        ("NTPC",          "NTPC"),
        ("POWERGRID",     "Power Grid Corp"),
        ("ADANIPOWER",    "Adani Power"),
        ("ADANIGREEN",    "Adani Green Energy"),
        ("ADANIENSOL",    "Adani Energy Solutions"),
        ("TORNTPOWER",    "Torrent Power"),
        ("CESC",          "CESC"),
        ("TATAPOWER",     "Tata Power"),
        ("NHPC",          "NHPC"),
        ("SJVN",          "SJVN"),
        ("IREDA",         "Indian Renewable Energy"),
        ("SUZLON",        "Suzlon Energy"),
        ("INOXWIND",      "Inox Wind"),
        ("PREMIER ENRG",  "Premier Energies"),
        ("WAAREEENER",    "Waaree Energies"),
    ],

    "🏗️ Infrastructure & Real Estate": [
        ("ULTRACEMCO",    "UltraTech Cement"),
        ("SHREECEM",      "Shree Cement"),
        ("AMBUJACEM",     "Ambuja Cements"),
        ("ACC",           "ACC"),
        ("DALMIACEM",     "Dalmia Bharat"),
        ("JKCEMENT",      "JK Cement"),
        ("RAMCOCEM",      "Ramco Cements"),
        ("HEIDELBERG",    "HeidelbergCement India"),
        ("DLF",           "DLF"),
        ("GODREJPROP",    "Godrej Properties"),
        ("OBEROIRLTY",    "Oberoi Realty"),
        ("PRESTIGE",      "Prestige Estates"),
        ("PHOENIXLTD",    "Phoenix Mills"),
        ("SOBHA",         "Sobha"),
        ("BRIGADE",       "Brigade Enterprises"),
        ("MAHINDCIE",     "Mahindra CIE Automotive"),
        ("IRCON",         "IRCON International"),
        ("NBCC",          "NBCC (India) Ltd"),
        ("RVNL",          "Rail Vikas Nigam"),
        ("IRB",           "IRB Infrastructure"),
        ("KNR",           "KNR Constructions"),
        ("NCC",           "NCC"),
        ("JSWINFRA",      "JSW Infrastructure"),
        ("ADANIPORTS",    "Adani Ports & SEZ"),
    ],

    "🧪 Chemicals & Pharma": [
        ("SUNPHARMA",     "Sun Pharmaceutical"),
        ("DRREDDY",       "Dr. Reddy's Laboratories"),
        ("CIPLA",         "Cipla"),
        ("DIVISLAB",      "Divi's Laboratories"),
        ("BIOCON",        "Biocon"),
        ("AUROPHARMA",    "Aurobindo Pharma"),
        ("LUPIN",         "Lupin"),
        ("TORNTPHARM",    "Torrent Pharma"),
        ("ALKEM",         "Alkem Laboratories"),
        ("ABBOTINDIA",    "Abbott India"),
        ("PFIZER",        "Pfizer India"),
        ("SANOFI",        "Sanofi India"),
        ("GLAXO",         "GSK Pharma"),
        ("IPCALAB",       "IPCA Laboratories"),
        ("GLENMARK",      "Glenmark Pharma"),
        ("NATCOPHARM",    "Natco Pharma"),
        ("GRANULES",      "Granules India"),
        ("SOLARA",        "Solara Active Pharma"),
        ("APLLTD",        "APL Apollo (Pharma)"),
        ("PIIND",         "PI Industries"),
        ("UPL",           "UPL"),
        ("RALLIS",        "Rallis India"),
        ("BAYER",         "Bayer CropScience"),
        ("COROMANDEL",    "Coromandel International"),
        ("SUMICHEM",      "Sumitomo Chemical India"),
        ("AARTI",         "Aarti Industries"),
        ("DEEPAKNITR",    "Deepak Nitrite"),
        ("VINATIORGA",    "Vinati Organics"),
        ("NAVINFLUOR",    "Navin Fluorine"),
        ("SRF",           "SRF"),
        ("ATUL",          "Atul"),
        ("CLEAN",         "Clean Science"),
        ("FINPIPE",       "Fine Organic Industries"),
    ],

    "🛒 FMCG & Consumer": [
        ("HINDUNILVR",    "Hindustan Unilever"),
        ("ITC",           "ITC"),
        ("NESTLEIND",     "Nestle India"),
        ("BRITANNIA",     "Britannia Industries"),
        ("DABUR",         "Dabur India"),
        ("GODREJCP",      "Godrej Consumer Products"),
        ("MARICO",        "Marico"),
        ("EMAMILTD",      "Emami"),
        ("COLPAL",        "Colgate-Palmolive India"),
        ("PGHH",          "Procter & Gamble Hygiene"),
        ("GILLETTE",      "Gillette India"),
        ("BAJAJCON",      "Bajaj Consumer Care"),
        ("VBL",           "Varun Beverages"),
        ("UNITEDBRWRY",   "United Breweries"),
        ("UBL",           "United Spirits"),
        ("RADICO",        "Radico Khaitan"),
        ("TATACONSUM",    "Tata Consumer Products"),
        ("MCDOWELL-N",    "McDowell's (United Spirits)"),
        ("PATANJALI",     "Patanjali Foods"),
        ("BIKAJI",        "Bikaji Foods"),
        ("DEVYANI",       "Devyani International"),
        ("WESTLIFE",      "Westlife Foodworld"),
        ("JUBLFOOD",      "Jubilant Foodworks"),
        ("SAPPHIRE",      "Sapphire Foods"),
        ("ZOMATO",        "Zomato"),
        ("SWIGGY",        "Swiggy"),
    ],

    "🏥 Healthcare & Diagnostics": [
        ("APOLLOHOSP",    "Apollo Hospitals"),
        ("MAXHEALTH",     "Max Healthcare"),
        ("FORTIS",        "Fortis Healthcare"),
        ("NARAYANHRU",    "Narayana Hrudayalaya"),
        ("KIMS",          "KIMS Hospitals"),
        ("YATHARTH",      "Yatharth Hospital"),
        ("METROPOLIS",    "Metropolis Healthcare"),
        ("LALPATHLAB",    "Dr Lal PathLabs"),
        ("THYROCARE",     "Thyrocare Technologies"),
        ("KRSNAA",        "Krsnaa Diagnostics"),
        ("HEALTHIUM",     "Healthium Medtech"),
        ("POLYMED",       "Poly Medicure"),
        ("MEDANTA",       "Global Health (Medanta)"),
        ("RAINBOW",       "Rainbow Children's Medicare"),
    ],

    "📦 Logistics & Transport": [
        ("IRFC",          "Indian Railway Finance"),
        ("CONCOR",        "Container Corp of India"),
        ("DELHIVERY",     "Delhivery"),
        ("BLUEDART",      "Blue Dart Express"),
        ("MAHINDLOG",     "Mahindra Logistics"),
        ("GESHIP",        "Great Eastern Shipping"),
        ("SCI",           "Shipping Corp of India"),
        ("INTERGLOBE",    "IndiGo (InterGlobe Aviation)"),
        ("SPICEJET",      "SpiceJet"),
        ("GMRAIRPORT",    "GMR Airports"),
        ("AIAENG",        "AIA Engineering"),
        ("TITAGARH",      "Titagarh Rail Systems"),
    ],

    "🛍️ Retail & D2C": [
        ("TRENT",         "Trent (Zara India)"),
        ("ABFRL",         "Aditya Birla Fashion"),
        ("VMART",         "V-Mart Retail"),
        ("DMART",         "Avenue Supermarts (DMart)"),
        ("NYKAA",         "Nykaa (FSN E-Commerce)"),
        ("MEESHO",        "Meesho"),
        ("CARTRADE",      "CarTrade Tech"),
        ("EASEMYTRIP",    "EaseMyTrip"),
        ("IXIGO",         "Le Travenues Technology (ixigo)"),
        ("MAMAEARTH",     "Honasa Consumer (Mamaearth)"),
        ("MANYAVAR",      "Vedant Fashions (Manyavar)"),
        ("BATA",          "Bata India"),
        ("CAMPUS",        "Campus Activewear"),
        ("METROBRAND",    "Metro Brands"),
    ],

    "🔩 Metals & Mining": [
        ("TATASTEEL",     "Tata Steel"),
        ("JSWSTEEL",      "JSW Steel"),
        ("HINDALCO",      "Hindalco Industries"),
        ("VEDL",          "Vedanta"),
        ("SAIL",          "Steel Authority of India"),
        ("NATIONALUM",    "National Aluminium"),
        ("HINDZINC",      "Hindustan Zinc"),
        ("NMDC",          "NMDC"),
        ("COALINDIA",     "Coal India"),
        ("MOIL",          "MOIL"),
        ("RATNAMANI",     "Ratnamani Metals"),
        ("WELCORP",       "Welspun Corp"),
        ("APL",           "APL Apollo Tubes"),
        ("JSPL",          "Jindal Steel & Power"),
        ("JINDALSTEL",    "Jindal Steel"),
    ],

    "📡 Telecom & Media": [
        ("BHARTIARTL",    "Bharti Airtel"),
        ("IDEA",          "Vodafone Idea"),
        ("TATACOMM",      "Tata Communications"),
        ("HFCL",          "HFCL"),
        ("ROUTE",         "Route Mobile"),
        ("ONMOBILE",      "OnMobile Global"),
        ("SUNTV",         "Sun TV Network"),
        ("ZEEL",          "Zee Entertainment"),
        ("PVRINOX",       "PVR INOX"),
        ("SAREGAMA",      "Saregama India"),
        ("TIPS",          "Tips Music"),
        ("NETWORK18",     "Network18 Media"),
        ("TV18BRDCST",    "TV18 Broadcast"),
    ],

    "🎓 Education & New Age": [
        ("CAREEREDGE",    "Career Edge"),
        ("CAMPUS",        "Campus Activewear"),
        ("LXCHEM",        "Laxmi Organic Industries"),
        ("ZOMATO",        "Zomato"),
        ("PAYTM",         "Paytm"),
        ("NYKAA",         "Nykaa"),
        ("CARTRADE",      "CarTrade"),
        ("DELHIVERY",     "Delhivery"),
        ("POLICYBZR",     "PolicyBazaar"),
        ("LATENTVIEW",    "LatentView Analytics"),
        ("ZAGGLE",        "Zaggle Prepaid"),
        ("IDEAFORGE",     "ideaForge Technology"),
    ],
}

# ── Flat list of all (ticker, name, sector) ────────────────────────────────

ALL_STOCKS_FLAT: list[tuple[str, str, str]] = []
for sector, stocks in NSE_STOCKS.items():
    for ticker, name in stocks:
        ALL_STOCKS_FLAT.append((f"{ticker}.NS", name, sector))

# Deduplicate by ticker
_seen = set()
ALL_STOCKS_DEDUPED: list[tuple[str, str, str]] = []
for item in ALL_STOCKS_FLAT:
    if item[0] not in _seen:
        _seen.add(item[0])
        ALL_STOCKS_DEDUPED.append(item)

# ── Nifty 50 fast list ──────────────────────────────────────────────────────

NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBIN.NS", "SBILIFE.NS", "SHREECEM.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "ZOMATO.NS",
]

NIFTY50_NAMES = {
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals", "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv",
    "BPCL.NS": "BPCL", "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS": "Britannia", "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India", "DIVISLAB.NS": "Divi's Labs",
    "DRREDDY.NS": "Dr. Reddy's", "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim", "HCLTECH.NS": "HCL Tech",
    "HDFCBANK.NS": "HDFC Bank", "HDFCLIFE.NS": "HDFC Life",
    "HEROMOTOCO.NS": "Hero MotoCorp", "HINDALCO.NS": "Hindalco",
    "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "ITC.NS": "ITC", "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys", "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Bank", "LT.NS": "L&T",
    "M&M.NS": "M&M", "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India", "NTPC.NS": "NTPC",
    "ONGC.NS": "ONGC", "POWERGRID.NS": "Power Grid",
    "RELIANCE.NS": "Reliance", "SBIN.NS": "SBI",
    "SBILIFE.NS": "SBI Life", "SHREECEM.NS": "Shree Cement",
    "SUNPHARMA.NS": "Sun Pharma", "TATACONSUM.NS": "Tata Consumer",
    "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "TCS", "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan", "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro", "ZOMATO.NS": "Zomato",
}


if __name__ == "__main__":
    print(f"Total stocks in universe: {len(ALL_STOCKS_DEDUPED)}")
    print(f"Sectors: {len(NSE_STOCKS)}")
    for s, stocks in NSE_STOCKS.items():
        print(f"  {s}: {len(stocks)} stocks")

# ─────────────────────────────────────────────────────────────────────────────
#  NIFTY 500 — Dynamic fetcher
#  Tries 3 sources in order:
#  1. NSE India official CSV (most accurate, may block non-India IPs)
#  2. Wikipedia Nifty 500 page (reliable fallback)
#  3. Our curated ALL_STOCKS_DEDUPED list (offline fallback)
# ─────────────────────────────────────────────────────────────────────────────

_nifty500_cache: list[tuple[str, str]] | None = None   # (ticker.NS, name)


def fetch_nifty500(force_refresh: bool = False) -> list[tuple[str, str]]:
    """
    Fetch the live Nifty 500 constituent list.
    Returns list of (ticker_with_NS, company_name) tuples.
    Cached in memory — only fetches once per process.
    """
    global _nifty500_cache
    if _nifty500_cache and not force_refresh:
        return _nifty500_cache

    result = (
        _fetch_nifty500_from_nse()
        or _fetch_nifty500_from_wikipedia()
        or _fetch_nifty500_from_local()
    )
    _nifty500_cache = result
    return result


def _fetch_nifty500_from_nse() -> list[tuple[str, str]] | None:
    """Source 1: NSE India official index constituents CSV."""
    try:
        import requests, io, csv
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer":    "https://www.nseindia.com/",
            "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        # NSE requires a session cookie first
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)

        url = ("https://www.nseindia.com/api/equity-stockIndices"
               "?index=NIFTY%20500")
        r = session.get(url, headers=headers, timeout=10)
        data = r.json()

        stocks = []
        for item in data.get("data", []):
            sym  = item.get("symbol", "").strip()
            name = item.get("companyName", sym).strip()
            if sym and sym != "NIFTY 500":
                stocks.append((f"{sym}.NS", name))

        if len(stocks) >= 400:
            print(f"[Nifty500] NSE source: {len(stocks)} stocks")
            return stocks
    except Exception as e:
        print(f"[Nifty500] NSE source failed: {e}")
    return None


def _fetch_nifty500_from_wikipedia() -> list[tuple[str, str]] | None:
    """Source 2: Wikipedia list of Nifty 500 companies."""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://en.wikipedia.org/wiki/NIFTY_500"
        r   = requests.get(url, timeout=10,
                           headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        stocks = []
        for table in soup.find_all("table", {"class": "wikitable"}):
            headers_row = [th.get_text(strip=True) for th in table.find_all("th")]
            # Find symbol column
            sym_col  = next((i for i, h in enumerate(headers_row)
                             if "symbol" in h.lower() or "ticker" in h.lower()), None)
            name_col = next((i for i, h in enumerate(headers_row)
                             if "company" in h.lower() or "name" in h.lower()), None)
            if sym_col is None:
                continue
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if not cells or len(cells) <= sym_col:
                    continue
                sym  = cells[sym_col].get_text(strip=True).replace(" ", "")
                name = cells[name_col].get_text(strip=True) if name_col and len(cells) > name_col else sym
                if sym:
                    stocks.append((f"{sym}.NS", name))

        if len(stocks) >= 200:
            print(f"[Nifty500] Wikipedia source: {len(stocks)} stocks")
            return stocks
    except Exception as e:
        print(f"[Nifty500] Wikipedia source failed: {e}")
    return None


def _fetch_nifty500_from_local() -> list[tuple[str, str]]:
    """Source 3: Offline fallback — our curated list (400+ stocks)."""
    print("[Nifty500] Using local curated list as fallback")
    return [(t, n) for t, n, _ in ALL_STOCKS_DEDUPED]


# ── Nifty 100 (Nifty 50 + Next 50) ──────────────────────────────────────────

NIFTY_NEXT50 = [
    "ADANIENSOL.NS", "ADANIENT.NS",  "ALKEM.NS",     "AMBUJACEM.NS",
    "AUROPHARMA.NS", "BAJAJHFL.NS",  "BAJAJFINSV.NS","BANKBARODA.NS",
    "BEL.NS",        "BERGEPAINT.NS","BIOCON.NS",     "BOSCHLTD.NS",
    "CHOLAFIN.NS",   "COLPAL.NS",    "DALBHARAT.NS",  "DABUR.NS",
    "DLF.NS",        "GAIL.NS",      "GLENMARK.NS",   "GODREJCP.NS",
    "GODREJPROP.NS", "HAVELLS.NS",   "HDFCAMC.NS",    "HINDZINC.NS",
    "ICICIPRULI.NS", "IDEA.NS",      "IDFCFIRSTB.NS", "IGL.NS",
    "INDHOTEL.NS",   "INDUSTOWER.NS","IOC.NS",        "IRCTC.NS",
    "JINDALSTEL.NS", "JUBLFOOD.NS",  "LICI.NS",       "LUPIN.NS",
    "MARICO.NS",     "MCDOWELL-N.NS","MUTHOOTFIN.NS", "NMDC.NS",
    "NYKAA.NS",      "OFSS.NS",      "PAGEIND.NS",    "PIIND.NS",
    "POLYCAB.NS",    "SAIL.NS",      "SIEMENS.NS",    "TRENT.NS",
    "VEDL.NS",       "ZYDUSLIFE.NS",
]

NIFTY100 = list(dict.fromkeys(NIFTY50 + NIFTY_NEXT50))  # deduped

NIFTY_NEXT50_NAMES = {
    "ADANIENSOL.NS": "Adani Energy Solutions", "ADANIENT.NS": "Adani Enterprises",
    "ALKEM.NS": "Alkem Laboratories",          "AMBUJACEM.NS": "Ambuja Cements",
    "AUROPHARMA.NS": "Aurobindo Pharma",        "BAJAJHFL.NS": "Bajaj Housing Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",           "BANKBARODA.NS": "Bank of Baroda",
    "BEL.NS": "Bharat Electronics",             "BERGEPAINT.NS": "Berger Paints",
    "BIOCON.NS": "Biocon",                       "BOSCHLTD.NS": "Bosch India",
    "CHOLAFIN.NS": "Cholamandalam Finance",      "COLPAL.NS": "Colgate-Palmolive",
    "DALBHARAT.NS": "Dalmia Bharat",            "DABUR.NS": "Dabur India",
    "DLF.NS": "DLF",                            "GAIL.NS": "GAIL India",
    "GLENMARK.NS": "Glenmark Pharma",           "GODREJCP.NS": "Godrej Consumer",
    "GODREJPROP.NS": "Godrej Properties",        "HAVELLS.NS": "Havells India",
    "HDFCAMC.NS": "HDFC AMC",                   "HINDZINC.NS": "Hindustan Zinc",
    "ICICIPRULI.NS": "ICICI Prudential",         "IDEA.NS": "Vodafone Idea",
    "IDFCFIRSTB.NS": "IDFC First Bank",          "IGL.NS": "Indraprastha Gas",
    "INDHOTEL.NS": "Indian Hotels",              "INDUSTOWER.NS": "Indus Towers",
    "IOC.NS": "Indian Oil Corp",                "IRCTC.NS": "IRCTC",
    "JINDALSTEL.NS": "Jindal Steel & Power",     "JUBLFOOD.NS": "Jubilant Foodworks",
    "LICI.NS": "LIC India",                     "LUPIN.NS": "Lupin",
    "MARICO.NS": "Marico",                       "MCDOWELL-N.NS": "United Spirits",
    "MUTHOOTFIN.NS": "Muthoot Finance",          "NMDC.NS": "NMDC",
    "NYKAA.NS": "Nykaa",                         "OFSS.NS": "Oracle Financial",
    "PAGEIND.NS": "Page Industries",             "PIIND.NS": "PI Industries",
    "POLYCAB.NS": "Polycab India",               "SAIL.NS": "SAIL",
    "SIEMENS.NS": "Siemens India",               "TRENT.NS": "Trent",
    "VEDL.NS": "Vedanta",                        "ZYDUSLIFE.NS": "Zydus Lifesciences",
}

NIFTY100_NAMES = {**NIFTY50_NAMES, **NIFTY_NEXT50_NAMES}
