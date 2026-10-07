"""Sharpe core: the Tech 1 + Tech 2 logic from the team's Colab notebook, as a reusable module.

Pipeline for one trading day and a set of holdings:
  prices -> two-factor return split (market / sector / company-specific, sigma, z)
  news (Alpha Vantage) + macro news + SEC filings (EDGAR)
  LLM classify -> LLM score (impact, new info, thesis relevance) -> attention rule (Python) -> LLM summary
Plus a portfolio overview and an optional what-if optimizer over the same holdings.
"""
import os, re, json, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import html
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

EST_DAYS = 120          # estimation window (>= 100 trading days, per the Return Split Method)
SIGMA_FLAG = 2.0
K_COMPANY, K_MACRO = 6, 5
MACRO_TOPICS = ['economy_macro', 'economy_monetary']
AV_TZ = 'America/New_York'
UA = {'User-Agent': 'Mozilla/5.0'}
_lock = threading.Lock()


def env(name, default=''):
    return os.environ.get(name, default).strip()


# ----------------------------------------------------------------------------- config
SUFFIX = {'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation', 'co', 'co.', 'company', 'ltd', 'ltd.', 'plc',
          'holdings', 'group', 'limited', 'new', 'class', 'a', 'b', 'the'}
GENERIC_FIRST = {'taiwan', 'united', 'bank', 'general', 'american', 'johnson', 'berkshire', 'procter', 'home', 'wells', 'goldman'}


def short_name(name):
    """Distinctive search/regex key from a company name: 'Taiwan Semiconductor Manufacturing Co Ltd' -> 'Taiwan Semiconductor'."""
    base = re.split(r'\s*\(', name)[0]
    words = [w for w in re.findall(r"[A-Za-z0-9&.'-]+", base)]
    while len(words) > 1 and words[-1].lower().strip('.,') in SUFFIX:
        words.pop()
    if not words:
        return ''
    if (words[0].lower() in GENERIC_FIRST or len(words[0].strip('.')) <= 2) and len(words) > 1:
        return ' '.join(words[:2])
    return words[0]


KNOWN_ALIAS = {'JPM': ['JPMorgan'], 'XOM': ['Exxon'], 'GOOG': ['Google', 'Alphabet'], 'GOOGL': ['Google', 'Alphabet'], 'META': ['Meta Platforms'],
               'TSM': ['TSMC'], 'BRK-B': ['Berkshire Hathaway'], 'UNH': ['UnitedHealth'], 'CAT': ['Caterpillar'], 'COST': ['Costco']}


def name_variants(h):
    """Strings that identify a company in headlines: short name, first two words, the two joined, plus known aliases."""
    base = re.split(r'\s*\(', h['name'])[0]
    words = [w for w in re.findall(r"[A-Za-z0-9&.'-]+", base)]
    while len(words) > 1 and words[-1].lower().strip('.,') in SUFFIX:
        words.pop()
    out = [short_name(h['name'])]
    if len(words) >= 2:
        out += [' '.join(words[:2])] + ([''.join(words[:2])] if len(words) == 2 else [])
    out += KNOWN_ALIAS.get(h['ticker'], [])
    out += re.findall(r'\(([^)]+)\)', h['name'])
    return [v for v in dict.fromkeys(o for o in out if len(o) >= 3)]


def config_loaded():
    return os.path.exists(os.path.join(HERE, 'holdings.json'))


def load_config():
    path = os.path.join(HERE, 'holdings.json')
    if not os.path.exists(path):
        path = os.path.join(HERE, 'holdings.default.json')
    with open(path) as f:
        cfg = json.load(f)
    for h in cfg['holdings']:
        h.setdefault('name', h['ticker'])
        h.setdefault('thesis', '')
        h.setdefault('chain', '')
        h.setdefault('guide', ['', '', ''])
        if not h.get('pattern'):
            nm = re.escape(short_name(h['name']).lower())
            h['pattern'] = (rf'\b{nm}\b|' if nm else '') + rf'\b{re.escape(h["ticker"].lower())}\b'
    with open(os.path.join(HERE, 'holdings.default.json')) as f:
        dflt = json.load(f)
    cfg['calendar'] = dflt.get('calendar', {})
    cfg['context'] = dflt.get('context', [])
    cfg.setdefault('market', 'SPY')
    return cfg


def save_config(cfg):
    with open(os.path.join(HERE, 'holdings.json'), 'w') as f:
        json.dump(cfg, f, indent=1)


# ----------------------------------------------------------------------------- prices
def _cache_path(kind, key):
    return os.path.join(CACHE_DIR, f'{kind}_{hashlib.md5(key.encode()).hexdigest()[:16]}.json')


def fetch_prices(tickers, max_age_hours=6):
    """Dividend-adjusted daily closes from Yahoo's chart endpoint. Cached for a few hours."""
    tickers = sorted(set(tickers))
    path = _cache_path('px', ','.join(tickers))
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_hours * 3600:
        px = pd.read_json(path, orient='split')
        px.index = pd.to_datetime(px.index)
        return px
    cols = {}
    for t in tickers:
        r = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{t}',
                         params={'range': '2y', 'interval': '1d'}, headers=UA, timeout=30)
        r.raise_for_status()
        res = r.json()['chart']['result']
        if not res:
            raise ValueError(f'No price data for {t}')
        res = res[0]
        ts = pd.to_datetime(res['timestamp'], unit='s').normalize()
        cols[t] = pd.Series(res['indicators']['adjclose'][0]['adjclose'], index=ts)
    px = pd.DataFrame(cols).dropna()
    px = px[~px.index.duplicated(keep='last')]
    px.to_json(path, orient='split', date_format='iso')
    return px


# ----------------------------------------------------------------------------- return split
def split_day(rets, tk, etf, market, day, est_days=EST_DAYS):
    """stock = a + b1*SPY + b2*(ETF - SPY) + resid, fit on the est_days BEFORE `day`.
    market part = b1*SPY, sector part = b2*(ETF-SPY), company part = total - market - sector (alpha stays in it).
    z = company part / residual std."""
    hist = rets.loc[rets.index < day].tail(est_days)
    if len(hist) < 100:
        raise ValueError(f'Only {len(hist)} trading days before {day.date()} (need >= 100)')
    y = hist[tk].values
    X = np.column_stack([np.ones(len(hist)), hist[market].values, (hist[etf] - hist[market]).values])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = resid.std(ddof=3)
    r_s, r_m, r_e = rets.loc[day, tk], rets.loc[day, market], rets.loc[day, etf]
    m = coef[1] * r_m
    s = coef[2] * (r_e - r_m)
    c = r_s - m - s
    return dict(ret_pct=100 * r_s, market_part_pct=100 * m, sector_part_pct=100 * s, company_part_pct=100 * c,
                sigma_pct=100 * sigma, z=float(c / sigma), big_move=bool(abs(c / sigma) >= SIGMA_FLAG),
                beta_mkt=float(coef[1]), beta_sector=float(coef[2]), r2=float(1 - resid.var() / y.var()),
                spy_ret_pct=100 * r_m, etf_ret_pct=100 * r_e)


# ----------------------------------------------------------------------------- Alpha Vantage news
class Budget:
    """Tracks uncached Alpha Vantage calls so a run cannot burn the free daily quota."""
    def __init__(self, limit=25):
        self.limit, self.used = limit, 0


def av_get(name, params, budget, log):
    key = env('ALPHAVANTAGE_API_KEY')
    path = _cache_path('av', name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if not key:
        raise RuntimeError('ALPHAVANTAGE_API_KEY not set')
    if budget.used >= budget.limit:
        raise RuntimeError(f'Stopped: {budget.limit} Alpha Vantage calls used this run')
    r = requests.get('https://www.alphavantage.co/query', params=dict(params, apikey=key), timeout=60)
    budget.used += 1
    data = r.json()
    msg = data.get('Note') or data.get('Information') or data.get('Error Message')
    if msg and len(data) <= 2:       # rate-limit message: never cache it
        log(f'Alpha Vantage: {msg[:140]}')
        if 'premium' in msg.lower():
            raise RuntimeError('Alpha Vantage says NEWS_SENTIMENT is a premium endpoint for this API key, so no news was retrieved')
        time.sleep(12)
        return None
    with open(path, 'w') as f:
        json.dump(data, f)
    time.sleep(float(env('AV_PAUSE', '12')))
    return data


def to_trading_day(ts_et, trading_days):
    """Map an ET timestamp to the trading day whose return it can affect (before 4pm -> that day; else next)."""
    d = ts_et.normalize().tz_localize(None)
    if d in trading_days and ts_et.hour < 16:
        return d
    later = trading_days[trading_days > d]
    return later[0] if len(later) else pd.NaT


def parse_feed(data):
    rows = []
    for a in (data or {}).get('feed', []):
        rows.append(dict(title=a.get('title', '').strip(), url=a.get('url'), time_published=a.get('time_published'),
                         source=a.get('source'), summary=a.get('summary', '').strip(),
                         ticker_sentiment=a.get('ticker_sentiment', [])))
    return rows


def norm_title(t):
    return re.sub(r'[^a-z0-9 ]', '', t.lower()).strip()


# ----------------------------------------------------------------------------- Google News RSS (keyless fallback)
GN_MACRO = {'economy_macro': '(economy OR GDP OR "jobs report" OR payrolls OR CPI OR inflation OR "retail sales" OR housing)',
            'economy_monetary': '("Federal Reserve" OR FOMC OR "interest rates" OR "Treasury yields" OR "rate hike" OR "rate cut")'}


def gn_search(query, d_from, d_to):
    """Headlines from Google News RSS between two dates (before: is exclusive). No key, no relevance scores,
    short snippets: weaker than Alpha Vantage News & Sentiment, but works on any account."""
    path = _cache_path('gn', f'{query}|{d_from}|{d_to}')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    r = requests.get('https://news.google.com/rss/search', headers=UA, timeout=30,
                     params={'q': f'{query} after:{d_from} before:{d_to}', 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
    r.raise_for_status()
    rows = []
    for it in ET.fromstring(r.content).iter('item'):
        try:
            ts = pd.to_datetime(it.findtext('pubDate'), utc=True).tz_convert(AV_TZ)
        except Exception:
            continue
        title, src = (it.findtext('title') or '').strip(), (it.findtext('source') or '').strip()
        if src and title.endswith(' - ' + src):
            title = title[:-len(src) - 3]
        desc = ' '.join(re.sub(r'<[^>]+>', ' ', html.unescape(it.findtext('description') or '')).split())
        if desc.lower().startswith(title.lower()[:40]):   # Google repeats the headline + publisher: not a real summary
            desc = ''
        rows.append(dict(title=title, url=(it.findtext('link') or '').strip(), time_published=ts.strftime('%Y%m%dT%H%M%S'),
                         source=src, summary=desc, ticker_sentiment=[]))
    with open(path, 'w') as f:
        json.dump(rows, f)
    time.sleep(1)
    return rows


def gn_terms(h):
    terms = [h['ticker']] + [f'"{v}"' if ' ' in v else v for v in name_variants(h)]
    return ' OR '.join(dict.fromkeys(terms))


def frame_news(rows, trading_days):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ts = pd.to_datetime(df.time_published, format='%Y%m%dT%H%M%S')
    df['ts_et'] = ts.dt.tz_localize(AV_TZ, ambiguous='NaT', nonexistent='NaT')
    df = df[df.ts_et.notna()].copy()
    df['trading_day'] = df.ts_et.apply(lambda t: to_trading_day(t, trading_days))
    return df


# ----------------------------------------------------------------------------- SEC EDGAR
ITEM_IMPACT = {'2.02': 3, '1.01': 3, '2.01': 3, '7.01': 2, '8.01': 2, '5.07': 1}
FILING_FORMS = {'8-K', '8-K/A', '6-K', '6-K/A'}


def sec_headers():
    return {'User-Agent': f'Sharpe portfolio monitor {env("SEC_EMAIL", "sharpe@example.com")}'}


def sec_cik_map():
    path = _cache_path('sec', 'company_tickers')
    if not os.path.exists(path):
        r = requests.get('https://www.sec.gov/files/company_tickers.json', headers=sec_headers(), timeout=60)
        r.raise_for_status()
        with open(path, 'w') as f:
            f.write(r.text)
    with open(path) as f:
        return {v['ticker'].upper(): int(v['cik_str']) for v in json.load(f).values()}


def sec_filings(tk, cik, trading_days, day, ttl_hours=3):
    path = _cache_path('sec', f'sub{cik}')
    if not (os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl_hours * 3600):
        r = requests.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json', headers=sec_headers(), timeout=60)
        r.raise_for_status()
        with open(path, 'w') as f:
            f.write(r.text)
        time.sleep(0.2)
    with open(path) as f:
        rec = pd.DataFrame(json.load(f)['filings']['recent'])
    rec = rec[rec.form.isin(FILING_FORMS)].copy()
    out = []
    for _, x in rec.iterrows():
        acc = pd.Timestamp(x.acceptanceDateTime).tz_convert('America/New_York')
        if to_trading_day(acc, trading_days) != day:
            continue
        codes = [c.strip() for c in str(x.get('items', '')).split(',') if c.strip() and c.strip() != '9.01']
        imp = max([ITEM_IMPACT.get(c, 3 if c == '5.02' else 2) for c in codes], default=None)
        out.append(dict(form=x.form, items=codes, impact=imp, description=x.get('primaryDocDescription', ''),
                        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{x.accessionNumber.replace('-', '')}/{x.primaryDocument}"))
    return out


# ----------------------------------------------------------------------------- LLM
def llm_provider():
    if env('OPENROUTER_API_KEY'):
        return 'openrouter'
    if env('ANTHROPIC_API_KEY'):
        return 'anthropic'
    return None


def parse_json(text):
    m = re.search(r'\{.*\}', text or '', flags=re.S)
    return json.loads(m.group(0)) if m else None


def llm(system, user, max_tokens=400):
    """Cached, temperature-0 JSON call. Returns a dict or None."""
    prov = llm_provider()
    if not prov:
        return None
    model = env('SHARPE_MODEL', 'anthropic/claude-haiku-4.5' if prov == 'openrouter' else 'claude-haiku-4-5-20251001')
    path = _cache_path('llm', f'{model}\n{system}\n{user}')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = None
    for attempt in range(3):
        try:
            if prov == 'openrouter':
                r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                                  headers={'Authorization': f'Bearer {env("OPENROUTER_API_KEY")}'}, timeout=90,
                                  json=dict(model=model, temperature=0, max_tokens=max_tokens,
                                            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))
                r.raise_for_status()
                text = r.json()['choices'][0]['message']['content']
            else:
                r = requests.post('https://api.anthropic.com/v1/messages', timeout=90,
                                  headers={'x-api-key': env('ANTHROPIC_API_KEY'), 'anthropic-version': '2023-06-01'},
                                  json=dict(model=model, temperature=0, max_tokens=max_tokens, system=system,
                                            messages=[{'role': 'user', 'content': user}]))
                r.raise_for_status()
                text = r.json()['content'][0]['text']
            out = parse_json(text)
            if out is not None:
                break
        except Exception:
            time.sleep(2 * (attempt + 1))
    if out is not None:
        with open(path, 'w') as f:
            json.dump(out, f)
    return out


CLASSIFY = '''You classify financial news for ONE portfolio holding. Read the article and decide what kind of news it is FOR THIS HOLDING.

Labels:
- "company": specifically about the holding itself: its results, guidance, products, deals, legal or regulatory actions against it, management changes, analyst rating changes ON the holding, large changes in who owns its shares.
- "sector": affects the holding's industry or its value chain (a customer, supplier, or key input), e.g. new chip export rules, a cloud provider's AI capex for NVDA/TSM, oil supply for XOM. If it is about a customer, supplier, or key input, set value_chain=true and vc_tag to one of: "Customer (<ticker or name>)", "Supplier (<name>)", "Key input", "Geopolitics → oil", "Power demand", "Financing".
- "macro": economy-wide news: inflation (CPI, PPI), jobs, retail sales, housing, Fed decisions or expectations, Treasury yields, broad oil or geopolitical shocks. Set channel to one of: Inflation, Labor, Growth/Consumer, Policy, Energy, Global.
- "not_relevant": the holding is only mentioned in passing; the holding acts as an analyst, broker, underwriter, or investor on ANOTHER company (e.g. "JPMorgan raises price target on XYZ" is not news about JPMorgan); "stocks to watch" lists; the article is mainly about another company with no clear link to the holding.

Return JSON only:
{"label": "...", "value_chain": true|false, "vc_tag": string|null, "channel": string|null, "reason": "<= 15 words"}'''

SCORE = '''You score one news item for one portfolio holding using the team's written rubric.

Impact (company news; for sector or macro news, judge the effect on this holding):
3 = could change earnings estimates, guidance, or risk (earnings surprise, guidance change, major regulatory action, acquisition, large lawsuit)
2 = notable but unlikely to change estimates (analyst rating change, product launch, mid-sized contract)
1 = routine (conference appearance, recap article, minor personnel news, routine institutional share purchases or sales)

New information:
3 = first report of a new fact
2 = meaningful update to a known story
1 = repeat or recap of something already reported (check the earlier headlines if given)

Thesis relevance (does it touch the reason we own the stock?):
3 = directly supports or threatens the thesis
2 = indirectly related
1 = unrelated

Also write one_line: one plain-English sentence (<= 25 words) saying what happened. Do not claim it moved the stock. Do not use em dashes.
Return JSON only: {"impact": 1-3, "new_info": 1-3, "thesis_relevance": 1-3, "one_line": "..."}'''

SUMMARIZE = '''You write one row of a portfolio monitoring dashboard for one holding on one day.
The attention level was set by a Python rule. You cannot change it.
Rules:
- Describe only what was reported, using only the facts in the items.
- Never mention the stock's price, its move, or whether it rose, fell, gained, or outperformed. Never link a news item to the price.
- Macro lines quote actual vs expected when the calendar gives it.
- Plain English. No em dashes.
Return JSON only:
{"headline": "<= 70 characters: the main news for this holding today (not the price)",
 "company": one sentence or null, "sector": one sentence or null, "macro": one sentence or null,
 "why": "two sentences on how the items relate to why we own the stock"}
Return null for a layer with no item.'''

WHY_RULE = {'High': 'Large company-specific move with new, high-impact company news.',
            'Medium': 'Large move without matching news, or high-impact news the market has not priced.',
            'Low': 'Move mostly explained by the market and sector.'}

PRICE_VERB = r'(rose|rises|fell|falls|gain(s|ed)?|drop(s|ped)?|slid(es)?|rall(y|ies|ied)|jump\w*|climb\w*|tumbl\w*|surg\w*|sank|sinks?|edged|slump\w*|soar\w*)'


def describes_price(o, h):
    """Guardrail from the notebook: reject summaries that describe the price move."""
    tk = h['ticker']
    txt = ' '.join(str(o.get(k) or '') for k in ('headline', 'company', 'sector', 'macro', 'why'))
    subj = rf'(shares?|stock|{tk}|{re.escape(h["name"].split(" ")[0])})'
    return (bool(re.search(rf'\b{subj}\b[^.;]{{0,25}}\b{PRICE_VERB}\b', txt, flags=re.I)) or
            bool(re.search(r'\b(outperform\w*|underperform\w*)\b', txt, flags=re.I)) or
            bool(re.search(rf'\b{tk}\b (up|down)\b', txt)))


def attention(z, co):
    """Attention rule (identical to the notebook and dashboard)."""
    big = abs(z) >= SIGMA_FLAG
    if big and co and co['impact'] >= 2 and co['newinfo'] >= 2:
        return 'High'
    if big:
        return 'Medium'
    if co and co['impact'] == 3:
        return 'Medium'
    return 'Low'


# ----------------------------------------------------------------------------- analysis
def _mentions(h, text):
    return bool(re.search(h['pattern'], str(text).lower()))


def cal_text(entry):
    """Calendar entry -> plain text for the LLM prompts."""
    if not entry:
        return 'n/a'
    if isinstance(entry, str):
        return entry
    ev = ' '.join(f"{e['time']} {e['name']}: {e['result']}." for e in entry.get('events', []))
    return (entry.get('summary', '') + ' ' + ev).strip()


MACRO_KW = re.compile(r'\b(cpi|inflation|fed|federal reserve|fomc|rate (hike|cut|decision)|interest rate|jobs|payroll|'
                      r'unemployment|retail sales|treasury|yield|gdp|housing|ppi|economy|economic|tariff|oil)\b', re.I)


def analyze_day(day_str, cfg, log=print, progress=lambda f: None):
    """Run the full pipeline for one trading day. Returns a JSON-able dict."""
    warnings = []
    hs, market = cfg['holdings'], cfg.get('market', 'SPY')
    etfs = sorted({h['etf'] for h in hs})
    px = fetch_prices([h['ticker'] for h in hs] + [market] + etfs)
    rets = px.pct_change().iloc[1:]
    days = rets.index
    day = pd.Timestamp(day_str)
    if day not in days:
        raise ValueError(f'{day_str} is not a trading day in the price data (latest: {days.max().date()})')
    progress(0.05)

    # --- return split
    log('Splitting returns (market / sector / company-specific)…')
    rows = []
    for h in hs:
        s = split_day(rets, h['ticker'], h['etf'], market, day)
        rows.append(dict(h=h, split=s, co=None, sector=None, macro=None, counts=[0, 0, 0]))
    progress(0.15)

    # --- news
    news_ok = True
    cands, scored = pd.DataFrame(), pd.DataFrame()
    macro_titles = []
    prev = days[days < day].max()
    t_from = f"{prev:%Y%m%d}T1600"
    t_to = f"{day:%Y%m%d}T1600"
    if True:
        budget = Budget()
        feeds, macro_rows = {}, []
        blocked = os.path.join(CACHE_DIR, 'av_news_blocked')
        use_av = bool(env('ALPHAVANTAGE_API_KEY')) and not os.path.exists(blocked)
        if use_av:
            try:
                for i, h in enumerate(hs):
                    log(f'Fetching news for {h["ticker"]} (Alpha Vantage)…')
                    d = av_get(f'news_{h["ticker"]}_{t_from}_{t_to}',
                               dict(function='NEWS_SENTIMENT', tickers=h['ticker'], time_from=t_from, time_to=t_to,
                                    sort='EARLIEST', limit=1000), budget, log)
                    feeds[h['ticker']] = parse_feed(d)
                    progress(0.15 + 0.1 * (i + 1) / len(hs))
                for topic in MACRO_TOPICS:
                    log(f'Fetching macro news ({topic})…')
                    d = av_get(f'macro_{topic}_{t_from}_{t_to}',
                               dict(function='NEWS_SENTIMENT', topics=topic, time_from=t_from, time_to=t_to,
                                    sort='EARLIEST', limit=1000), budget, log)
                    macro_rows += parse_feed(d)
            except Exception as e:
                feeds, macro_rows = {}, []
                if 'premium' in str(e).lower():
                    open(blocked, 'w').write(str(e))      # remember, so we do not waste a call every run
                warnings.append(f'{e}.')
                use_av = False
        if not use_av:
            if not env('ALPHAVANTAGE_API_KEY'):
                warnings.append('No Alpha Vantage key.')
            warnings.append('News comes from Google News RSS (keyless): headlines only, no per-ticker relevance or sentiment scores, so '
                            'it is noisier than Alpha Vantage News & Sentiment.')
            d_from, d_to = f'{prev:%Y-%m-%d}', f'{day + timedelta(days=1):%Y-%m-%d}'
            try:
                for i, h in enumerate(hs):
                    log(f'Fetching news for {h["ticker"]} (Google News)…')
                    feeds[h['ticker']] = gn_search(f'({gn_terms(h)})', d_from, d_to)
                    progress(0.15 + 0.1 * (i + 1) / len(hs))
                for topic in MACRO_TOPICS:
                    log(f'Fetching macro news ({topic})…')
                    macro_rows += gn_search(GN_MACRO[topic], d_from, d_to)
            except Exception as e:
                warnings.append(f'Google News fetch stopped early: {e}')
        # own feeds
        pool_rows = []
        for tk, feed in feeds.items():
            for a in feed:
                tsm = {x['ticker']: x for x in a['ticker_sentiment']}
                try:
                    rel = float(tsm.get(tk, {}).get('relevance_score', 'nan'))
                except ValueError:
                    rel = float('nan')
                pool_rows.append(dict(a, ticker=tk, relevance_score=rel))
        pool = frame_news(pool_rows, days)
        if not pool.empty:
            pool = pool[pool.trading_day == day].copy()
        macro = frame_news(macro_rows, days)
        if not macro.empty:
            macro = macro[macro.trading_day == day].copy()
            macro = macro.drop_duplicates('url').copy()
            macro['tk'] = macro.title.map(norm_title)
            macro = macro.drop_duplicates('tk')

        # candidate selection: up to K_COMPANY per holding (named in headline first, then AV relevance)
        cand = []
        uniq = pool.drop_duplicates('url') if not pool.empty else pool
        for h in hs:
            tk = h['ticker']
            if pool.empty:
                continue
            own = pool[pool.ticker == tk].copy()
            own['via'] = 'own feed'
            extra = uniq[(uniq.ticker != tk) & (uniq.title.map(lambda t: _mentions(h, t)) |
                                                 uniq.summary.map(lambda t: _mentions(h, str(t)[:300])))].copy()
            extra = extra[~extra.url.isin(own.url)]
            extra['via'] = 'other holding feed'
            extra['relevance_score'] = np.nan
            both = pd.concat([own, extra])
            both = both.assign(ticker=tk, kind='company_feed')
            both['named'] = both.title.map(lambda t: _mentions(h, t))
            both['tk'] = both.title.map(norm_title)
            both = both.drop_duplicates('tk')
            both = both.sort_values(['named', 'relevance_score'], ascending=False).head(K_COMPANY)
            cand.append(both)
        if not macro.empty:
            def single_company(r):
                return any(_mentions(h, r.title) for h in hs)
            m = macro[macro.apply(lambda r: bool(MACRO_KW.search(r.title + ' ' + str(r.summary))) and not single_company(r), axis=1)]
            m = m.head(K_MACRO)
            macro_titles = m.title.tolist()
            for h in hs:
                cand.append(m.assign(ticker=h['ticker'], kind='macro_feed', via='macro feed', relevance_score=np.nan))
        cands = pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()

    # --- LLM classify + score
    cal_entry = cfg.get('calendar', {}).get(day_str)
    ctx_cal = cal_text(cal_entry)
    if not cands.empty:
        if not llm_provider():
            warnings.append('No OPENROUTER_API_KEY / ANTHROPIC_API_KEY: found articles are listed but not classified or scored.')
            cands = pd.DataFrame()
        else:
            hmap = {h['ticker']: h for h in hs}
            cands['cid'] = range(len(cands))
            log(f'Classifying {len(cands)} article-holding pairs with the LLM…')

            def classify(r):
                h = hmap[r.ticker]
                u = (f'Holding: {h["ticker"]} ({h["name"]})\nValue chain: {h["chain"]}\n\n'
                     f'Article title: {r.title}\nPublisher: {r.source}\nSummary: {str(r.summary)[:700]}')
                return llm(CLASSIFY, u, 200) or {}
            with ThreadPoolExecutor(8) as ex:
                outs = list(ex.map(classify, cands.itertuples()))
            cands['label'] = [str(o.get('label', 'error')).strip().lower().replace(' ', '_') for o in outs]
            cands.loc[~cands.label.isin(['company', 'sector', 'macro', 'not_relevant']), 'label'] = 'error'
            cands['value_chain'] = [bool(o.get('value_chain')) for o in outs]
            cands['vc_tag'] = [o.get('vc_tag') for o in outs]
            cands['channel'] = [o.get('channel') for o in outs]
            progress(0.55)

            rel = cands[cands.label.isin(['company', 'sector', 'macro'])]
            log(f'Scoring {len(rel)} relevant items…')

            def score(r):
                h = hmap[r.ticker]
                g3, g2, g1 = (list(h.get('guide') or []) + ['', '', ''])[:3]
                prior = cands[(cands.ticker == r.ticker) & (cands.kind == 'company_feed') & (cands.cid < r.cid)].title.head(12).tolist()
                u = (f'Holding: {h["ticker"]} ({h["name"]})\nThesis: {h["thesis"]}\nItem type: {r.label}\n'
                     f'Value chain: {h["chain"]}\nThesis relevance examples: 3 = {g3}; 2 = {g2}; 1 = {g1}\n'
                     f'Macro calendar today: {ctx_cal}\n')
                if prior:
                    u += 'Earlier headlines for this holding today:\n' + '\n'.join(f'- {t}' for t in prior) + '\n'
                u += f'\nArticle title: {r.title}\nPublisher: {r.source}\nSummary: {str(r.summary)[:700]}'
                return llm(SCORE, u, 200) or {}
            with ThreadPoolExecutor(8) as ex:
                souts = list(ex.map(score, rel.itertuples()))

            def clip(v):
                try:
                    return int(min(3, max(1, int(v))))
                except Exception:
                    return None
            sc = pd.DataFrame({'cid': rel.cid.values, 'impact': [clip(o.get('impact')) for o in souts],
                               'new_info': [clip(o.get('new_info')) for o in souts],
                               'thesis_rel': [clip(o.get('thesis_relevance')) for o in souts],
                               'one_line': [o.get('one_line') for o in souts]})
            scored = cands.merge(sc, on='cid', how='left')
    progress(0.75)

    # --- SEC filings
    sec_ok = True
    try:
        ciks = sec_cik_map()
    except Exception as e:
        sec_ok = False
        warnings.append(f'SEC EDGAR unavailable: {e}')
    # --- assemble per holding
    for r in rows:
        h = r['h']
        tk = h['ticker']
        g = scored[scored.ticker == tk] if not scored.empty else scored
        co = None
        if sec_ok and tk in ciks:
            try:
                f = sec_filings(tk, ciks[tk], days, day)
            except Exception as e:
                f = []
                warnings.append(f'SEC filings for {tk} failed: {e}')
            if f:
                fi = f[0]
                co = dict(type=f"SEC {fi['form'].replace('/A', '')}", impact=fi['impact'] or 2, newinfo=3,
                          title=fi['description'] or fi['form'], publisher='SEC EDGAR', url=fi['url'],
                          text='Items ' + ', '.join(fi['items']) if fi['items'] else None)
        sect = se = ma = None
        co_row = None
        if len(g):
            comp = g[g.label == 'company'].dropna(subset=['impact'])
            if not comp.empty:
                co_row = comp.sort_values(['impact', 'new_info', 'thesis_rel'], ascending=False).iloc[0]
                if co is None:
                    co = dict(type='News', impact=int(co_row.impact), newinfo=int(co_row.new_info), title=co_row.title,
                              publisher=co_row.source, url=co_row.url, text=co_row.one_line)
            sect_df = g[(g.label == 'sector') & (g.thesis_rel >= 2)]
            mac_df = g[(g.label == 'macro') & (g.thesis_rel >= 2)]
            if not sect_df.empty:
                se = sect_df.assign(vc=sect_df.value_chain.astype(int)).sort_values(['thesis_rel', 'vc', 'impact'], ascending=False).iloc[0]
                sect = dict(text=se.one_line, vc=se.vc_tag if se.value_chain else None, title=se.title, publisher=se.source, url=se.url)
            if not mac_df.empty:
                ma = mac_df.sort_values(['thesis_rel', 'impact'], ascending=False).iloc[0]
                macro_ = dict(text=ma.one_line, ch=ma.channel, title=ma.title, publisher=ma.source, url=ma.url)
            else:
                macro_ = None
        else:
            macro_ = None
        thesis = max([int(x.thesis_rel) for x in (co_row, se, ma) if x is not None and pd.notna(x.thesis_rel)] or [1])
        r.update(co=co, sector=sect, macro=macro_, thesis=thesis,
                 counts=[int(len(g)), int((g.label != 'not_relevant').sum()) if len(g) else 0,
                         int(sum(x is not None for x in (co, sect, macro_)))])
        r['level'] = attention(r['split']['z'], co)

    # --- summaries
    if llm_provider() and not scored.empty:
        log('Writing summaries…')

        def summ(r):
            h = r['h']
            items = {k: (r[k] and {kk: r[k].get(kk) for kk in ('title', 'text', 'ch', 'vc', 'type', 'impact', 'newinfo')})
                     for k in ('co', 'sector', 'macro')}
            items = {'company': items['co'], 'sector': items['sector'], 'macro': items['macro']}
            z = r['split']['z']
            user = json.dumps(dict(holding=h['ticker'], name=h['name'], thesis=h['thesis'], day=day_str, macro_calendar=ctx_cal,
                                   company_specific_move='large (>= 2 sigma)' if abs(z) >= 2 else 'within normal range',
                                   attention=r['level'], attention_rule=WHY_RULE[r['level']], items=items), ensure_ascii=False)
            o = llm(SUMMARIZE, user, 400)
            if o and describes_price(o, h):
                o2 = llm(SUMMARIZE, user + '\n\nYour last answer described the stock price or its move. Rewrite it about the news only.', 400)
                o = o2 if o2 and not describes_price(o2, h) else None
            return o or {}
        with ThreadPoolExecutor(5) as ex:
            souts = list(ex.map(summ, rows))
    else:
        souts = [{} for _ in rows]
    out_rows = []
    for r, o in zip(rows, souts):
        h = r['h']
        fallback = (r['co'] or r['sector'] or r['macro'] or {}).get('title') or 'No relevant new information'
        for k, key in (('co', 'company'), ('sector', 'sector'), ('macro', 'macro')):
            if r[k] is not None and o.get(key):
                r[k]['text'] = o[key]
            if r[k] is not None and not r[k].get('text'):
                r[k]['text'] = r[k].get('title')
        out_rows.append(dict(ticker=h['ticker'], name=h['name'], etf=h['etf'], thesis_text=h['thesis'], level=r['level'],
                             headline=o.get('headline') or fallback[:90], why=o.get('why') or '', thesis=r['thesis'],
                             split=r['split'], company=r['co'], sector=r['sector'], macro=r['macro'], counts=r['counts']))
    order = {'High': 0, 'Medium': 1, 'Low': 2}
    out_rows.sort(key=lambda x: (order[x['level']], -x['thesis'], -abs(x['split']['z'])))
    progress(1.0)
    if not cal_entry:
        summ_line = None
        if llm_provider() and macro_titles:
            o = llm('Write ONE sentence (<= 30 words) starting with "Macro today:" that summarizes the economy-wide news in these headlines '
                    'for a portfolio brief. Use only facts in the headlines. Do not mention any stock price or move. Plain English, no em dashes. '
                    'Return JSON only: {"summary": "..."}', json.dumps(macro_titles), 150)
            summ_line = (o or {}).get('summary')
        cal_entry = dict(summary=summ_line or 'Macro today: no scheduled-release calendar for this day.', events=[], generated=bool(summ_line))
    nxt = next_earnings([h['ticker'] for h in hs])
    for x in out_rows:
        x['next_earnings'] = nxt.get(x['ticker'])
    return dict(day=day_str, calendar=cal_entry, context=cfg.get('context', []), rows=out_rows, warnings=warnings,
                latest_day=str(days.max().date()), llm=llm_provider(), news=news_ok, stress=False)


# ----------------------------------------------------------------------------- portfolio
def optimize(tickers, lo=0.05, hi=0.40, rf=0.036, holdout=20, lookback=504, current=None):
    """Max-Sharpe long-only weights. Trained on everything except the last `holdout` trading days."""
    from scipy.optimize import minimize
    n = len(tickers)
    if n < 2:
        raise ValueError('Need at least 2 tickers')
    if lo * n > 1 or hi * n < 1:
        raise ValueError(f'Infeasible bounds for {n} assets: need {1/n:.0%} between min and max weight')
    px = fetch_prices(tickers + ['SPY'])
    r = px.pct_change().dropna().tail(lookback)
    tr = r.iloc[:-holdout] if holdout else r
    oos = r.iloc[-holdout:] if holdout else r.iloc[0:0]
    mu, cov = tr[tickers].mean() * 252, tr[tickers].cov() * 252

    def stats(w):
        a, v = float(w @ mu.values), float(np.sqrt(w @ cov.values @ w))
        return a, v, (a - rf) / v
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    b = [(lo, hi)] * n
    x0 = np.ones(n) / n
    w = minimize(lambda w: -stats(w)[2], x0, bounds=b, constraints=cons, method='SLSQP').x
    w = np.clip(w, lo, hi); w = w / w.sum()
    mv = minimize(lambda w: stats(w)[1], x0, bounds=b, constraints=cons, method='SLSQP').x
    mv = np.clip(mv, lo, hi); mv = mv / mv.sum()
    fr = []
    for t in np.linspace(mu.min() * 0.9, mu.max() * 0.9, 40):
        c2 = cons + [{'type': 'eq', 'fun': lambda w, t=t: w @ mu.values - t}]
        q = minimize(lambda w: w @ cov.values @ w, x0, bounds=b, constraints=c2, method='SLSQP')
        if q.success:
            fr.append([float(np.sqrt(q.fun)), float(t)])

    def perf(pr):
        if len(pr) < 2:
            return None
        cum = (1 + pr).cumprod()
        ann, vol = pr.mean() * 252, pr.std() * np.sqrt(252)
        return dict(ret=float(cum.iloc[-1] - 1), vol=float(vol), sharpe=float((ann - rf) / vol) if vol else None,
                    mdd=float((cum / cum.cummax() - 1).min()))
    series = {'Max-Sharpe mix': r[tickers] @ w, 'Equal weight': r[tickers] @ x0, 'Min variance': r[tickers] @ mv, 'SPY': r['SPY']}
    cur = None
    if current and all(t in current for t in tickers):
        cw = np.array([current[t] for t in tickers], dtype=float)
        cw = cw / cw.sum()
        cur = dict(zip(tickers, map(float, cw)))
        series = dict(Current=r[tickers] @ cw, **series)
    curves = {k: [round(float(v), 4) for v in (1 + s).cumprod() / (1 + s.iloc[0])] for k, s in series.items()}
    comp = {k: dict(train=perf(s.loc[tr.index]), holdout=perf(s.loc[oos.index]) if len(oos) else None) for k, s in series.items()}
    pw = w @ cov.values @ w
    rc = (w * (cov.values @ w)) / pw
    a, v, sh = stats(w)
    return dict(tickers=tickers, current=cur, weights=dict(zip(tickers, map(float, w))), min_var=dict(zip(tickers, map(float, mv))),
                expected=dict(ret=a, vol=v, sharpe=sh), risk=dict(zip(tickers, map(float, rc))), frontier=fr,
                assets={t: dict(ret=float(mu[t]), vol=float(np.sqrt(cov.loc[t, t])), sharpe=float((mu[t] - rf) / np.sqrt(cov.loc[t, t]))) for t in tickers},
                dates=[str(d.date()) for d in r.index], curves=curves, compare=comp,
                train=[str(tr.index[0].date()), str(tr.index[-1].date())],
                holdout=[str(oos.index[0].date()), str(oos.index[-1].date())] if len(oos) else None,
                rf=rf, lo=lo, hi=hi)


# ----------------------------------------------------------------------------- portfolio CSV
SECTOR_ETF = {'Technology': 'XLK', 'Communication Services': 'XLC', 'Financial Services': 'XLF', 'Financials': 'XLF',
              'Energy': 'XLE', 'Healthcare': 'XLV', 'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP',
              'Industrials': 'XLI', 'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Basic Materials': 'XLB'}
COLS = {'ticker': ['ticker', 'symbol', 'tickers', 'stock'], 'shares': ['shares', 'quantity', 'qty', 'units'],
        'weight': ['weight', 'allocation', 'pct', 'percent', 'weight_pct', 'portfolio_weight'],
        'value': ['value', 'market_value', 'marketvalue', 'amount'], 'name': ['name', 'company', 'company_name', 'description'],
        'etf': ['sector_etf', 'etf', 'benchmark', 'sector_benchmark'], 'thesis': ['thesis', 'why', 'reason', 'rationale'],
        'chain': ['value_chain', 'chain', 'notes']}
MAX_HOLDINGS = 15
KNOWN_ETF = {'NVDA': 'SMH', 'TSM': 'SMH', 'ASML': 'SMH', 'QCOM': 'SMH', 'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'C': 'XLF', 'GS': 'XLF', 'MS': 'XLF', 'V': 'XLF', 'MA': 'XLF', 'BRK-B': 'XLF',
             'AAPL': 'XLK', 'MSFT': 'XLK', 'ORCL': 'XLK', 'CRM': 'XLK', 'AMD': 'SMH', 'AVGO': 'SMH', 'INTC': 'SMH', 'MU': 'SMH',
             'GOOG': 'XLC', 'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY',
             'COST': 'XLP', 'WMT': 'XLP', 'PG': 'XLP', 'KO': 'XLP', 'XOM': 'XLE', 'CVX': 'XLE', 'UNH': 'XLV', 'JNJ': 'XLV', 'LLY': 'XLV',
             'CAT': 'XLI', 'GE': 'XLI', 'BA': 'XLI'}


def yahoo_profile(tk, tries=2):
    for _ in range(tries):
      try:
        r = requests.get('https://query1.finance.yahoo.com/v1/finance/search', headers=UA, timeout=15,
                         params={'q': tk, 'quotesCount': 5, 'newsCount': 0})
        for q in r.json().get('quotes', []):
            if q.get('symbol', '').upper() == tk:
                return q
      except Exception:
        time.sleep(1)
    return None


def parse_portfolio_csv(text):
    """CSV text -> (holdings, warnings). Only a ticker column is required. Optional: shares | weight | value, name,
    sector_etf, thesis, value_chain. Missing names and sector ETFs are looked up on Yahoo."""
    import csv, io
    warnings = []
    text = text.lstrip('\ufeff')
    lines = text.splitlines()
    # brokerage exports start with title lines ("Positions for account ... as of ..."): find the real header row
    hdr = next((i for i, l in enumerate(lines[:30]) if re.search(r'\b(symbol|ticker)\b', l, re.I) and l.count(',') >= 1), 0)
    text = '\n'.join(lines[hdr:])
    rdr = csv.DictReader(io.StringIO(text))
    if not rdr.fieldnames:
        raise ValueError('The file is empty')
    norm = {f: re.sub(r'[^a-z0-9_]', '', f.strip().lower().replace(' ', '_').replace('%', 'pct')) for f in rdr.fieldnames}
    col = {}
    for key, aliases in COLS.items():
        for f, n in norm.items():
            if n in aliases and key not in col:
                col[key] = f
    if 'ticker' not in col:
        raise ValueError(f'No ticker column found (columns: {", ".join(rdr.fieldnames)}). Name one of them "ticker" or "symbol".')

    def num(v):
        v = str(v).strip()
        neg = v.startswith('(') and v.endswith(')')
        try:
            x = float(v.strip('()').replace('$', '').replace(',', '').replace('%', '').strip())
            return -x if neg else x
        except ValueError:
            return None
    rows = {}
    for i, row in enumerate(rdr, start=2):
        tk = (row.get(col['ticker']) or '').strip().upper().replace(' ', '')
        if not tk:
            continue
        if tk.startswith(('CASH', 'ACCOUNTTOTAL', 'TOTAL', 'USD', '$', 'PENDING', 'MARGIN')) or 'TOTAL' in tk:
            warnings.append(f'Skipped "{(row.get(col["ticker"]) or "").strip()}" (cash and total rows are not analysed).')
            continue
        if not re.fullmatch(r'[A-Z0-9.\-^]{1,10}', tk):
            warnings.append(f'Row {i}: "{tk}" does not look like a ticker; skipped.')
            continue
        g = lambda k: (row.get(col[k]) or '').strip() if k in col else ''
        if (num(g('shares')) or 0) < 0 or (num(g('value')) or 0) < 0:
            warnings.append(f'{tk}: short position skipped (Sharpe analyses long holdings only).')
            continue
        r = rows.setdefault(tk, dict(ticker=tk, shares=0.0, weight=None, value=None, name='', etf='', thesis='', chain=''))
        if tk in rows and r['shares'] and 'shares' in col and num(g('shares')):
            warnings.append(f'{tk}: appears more than once; shares added together.')
        if num(g('shares')) is not None:
            r['shares'] += num(g('shares'))
        for k in ('weight', 'value'):
            if num(g(k)) is not None:
                r[k] = (r[k] or 0) + num(g(k))
        for k in ('name', 'etf', 'thesis', 'chain'):
            if g(k) and not r[k]:
                r[k] = g(k).upper() if k == 'etf' else g(k)
    if not rows:
        raise ValueError('No holdings found in the file')
    if len(rows) > MAX_HOLDINGS:
        warnings.append(f'{len(rows)} holdings found; only the first {MAX_HOLDINGS} are analysed (news and LLM cost grow with each).')
        rows = dict(list(rows.items())[:MAX_HOLDINGS])
    hs = list(rows.values())

    # enrich: name + sector ETF
    with ThreadPoolExecutor(6) as ex:
        profiles = list(ex.map(lambda h: yahoo_profile(h['ticker']), hs))
    for h, q in zip(hs, profiles):
        if q is None:
            warnings.append(f'{h["ticker"]}: not found on Yahoo Finance. Check the symbol.')
        if not h['name'] or (h['name'].isupper() and q):   # brokerage exports shout: prefer Yahoo's name
            h['name'] = (q or {}).get('shortname') or (q or {}).get('longname') or h['name'] or h['ticker']
        if not h['etf']:
            sec, ind = (q or {}).get('sectorDisp'), (q or {}).get('industryDisp') or ''
            if h['ticker'] in KNOWN_ETF:
                h['etf'] = KNOWN_ETF[h['ticker']]
            elif 'Semiconductor' in ind:
                h['etf'] = 'SMH'
            elif sec in SECTOR_ETF:
                h['etf'] = SECTOR_ETF[sec]
            else:
                h['etf'] = 'SPY'
                if (q or {}).get('quoteType') == 'ETF':
                    warnings.append(f'{h["ticker"]}: is an ETF; compared against SPY only.')
                else:
                    warnings.append(f'{h["ticker"]}: sector not found, so no sector benchmark was assigned (SPY used). '
                                    'Add a sector_etf column (e.g. XLP for consumer staples) for a proper sector split.')

    cash_like = [h for h in hs if re.search(r'\bmoney\b|\bcash\b|\bsweep\b', h['name'], re.I)]
    for h in cash_like:
        warnings.append(f'{h["ticker"]} ({h["name"]}) looks like a money-market/cash fund, so it was left out.')
    hs = [h for h in hs if h not in cash_like]
    # drop anything Yahoo cannot price (money-market funds, options, delisted symbols) rather than failing everything
    def priceable(h):
        try:
            fetch_prices([h['ticker']])
            return True
        except Exception:
            return False
    with ThreadPoolExecutor(6) as ex:
        ok = list(ex.map(priceable, hs))
    for h, good in zip(hs, ok):
        if not good:
            warnings.append(f'{h["ticker"]}: no price history on Yahoo (money-market fund, option or invalid symbol?), so it was left out.')
    hs = [h for h, good in zip(hs, ok) if good]
    if not hs:
        raise ValueError('None of the tickers could be priced')
    # weights: explicit weight > value > shares x latest price > equal
    try:
        px = fetch_prices([h['ticker'] for h in hs] + ['SPY'] + sorted({h['etf'] for h in hs}))
        last = px.iloc[-1]
    except Exception as e:
        last = None
        warnings.append(f'Could not price holdings ({e}); weights are equal.')
    vals = []
    for h in hs:
        if h['weight'] is not None:
            vals.append(h['weight'])
        elif h['value'] is not None:
            vals.append(h['value'])
        elif h['shares'] and last is not None and h['ticker'] in last.index:
            vals.append(h['shares'] * float(last[h['ticker']]))
        else:
            vals.append(None)
    if all(v is None for v in vals) or any(v is None for v in vals):
        if any(v is not None for v in vals):
            warnings.append('Some rows have no shares/weight/value; using equal weights.')
        vals = [1.0] * len(hs)
    tot = sum(vals)
    for h, v in zip(hs, vals):
        h['weight'] = v / tot
        h.pop('value', None)
        h['shares'] = h['shares'] or None
    return hs, warnings


# ----------------------------------------------------------------------------- earnings
def next_earnings(tickers):
    """Next scheduled report date per ticker from Alpha Vantage's earnings calendar (one call, cached 12h)."""
    key = env('ALPHAVANTAGE_API_KEY')
    path = _cache_path('av', f'earn_cal_{pd.Timestamp.now():%Y%m%d}')
    try:
        if not os.path.exists(path):
            if not key:
                return {}
            r = requests.get('https://www.alphavantage.co/query', timeout=60,
                             params=dict(function='EARNINGS_CALENDAR', horizon='6month', apikey=key))
            if not r.text.startswith('symbol'):
                return {}
            open(path, 'w').write(r.text)
        import io
        df = pd.read_csv(path)
        df = df[df.symbol.isin(tickers)].sort_values('reportDate').drop_duplicates('symbol')
        return dict(zip(df.symbol, df.reportDate))
    except Exception:
        return {}


EARN_SYS = '''You extract facts from an earnings call transcript excerpt for a portfolio monitor.
Use only the excerpt. Return JSON only:
{"guidance": "Raised" | "Maintained" | "Cut" | "Not given",
 "tone": "Positive" | "Neutral" | "Cautious" | "Negative",
 "tone_shift": true|false (management sounds clearly more upbeat or more cautious than a routine quarter),
 "metrics": "key numbers in <= 20 words",
 "qa_theme": "top analyst question theme in <= 12 words",
 "new_vs_last": "what management said that sounds new this quarter, <= 20 words",
 "summary": ["results vs expectations", "guidance and outlook", "management priorities", "main analyst concern"],
 "new_info": 1-3, "thesis_relevance": 1-3,
 "one_line": "one plain-English sentence on results and guidance, no em dashes"}'''
EARN_KEYS = r'guidance|outlook|expect|forecast|raise|margin|demand|revenue|next quarter|full.year|capex|capital expenditure|backlog|net interest income|production'


def _retrieve(segs, budget=9000):
    """Pick the transcript segments that talk about guidance/outlook, plus the first analyst question (Lab 3 approach)."""
    scored = []
    for i, sg in enumerate(segs):
        c = str(sg.get('content') or '')
        scored.append((len(re.findall(EARN_KEYS, c.lower())), i, sg, 'analyst' in str(sg.get('title') or '').lower()))
    top = sorted(scored, key=lambda x: -x[0])[:10]
    first_q = next((x for x in scored if x[3]), None)
    if first_q and first_q not in top:
        top.append(first_q)
    text = ''
    for _, _, sg, _ in sorted(top, key=lambda x: x[1]):
        chunk = f"[{sg.get('speaker')}, {sg.get('title')}] {str(sg.get('content'))[:2500]}\n"
        if len(text) + len(chunk) > budget:
            break
        text += chunk
    return text


def _quarter_label(tk, fiscal_end):
    fe = pd.Timestamp(fiscal_end)
    if tk == 'NVDA':   # NVDA's fiscal year ends late January; Alpha Vantage labels by fiscal year
        return f'{fe.year + 1}Q{((fe.month - 2) % 12) // 3 + 1}'
    return f'{fe.year}Q{(fe.month - 1) // 3 + 1}'


def earnings_impact(co):
    """Impact rule: guidance raised/cut or |EPS surprise| >= 5% -> 3; tone shift -> 2; else 1."""
    if co['guidance'] in ('Raised', 'Cut'):
        return 3, 'guidance ' + co['guidance'].lower()
    if co['surprise'] is not None and abs(co['surprise']) >= 5:
        return 3, f"large EPS surprise ({co['surprise']:+.1f}%)"
    if co['toneShift']:
        return 2, 'guidance maintained, tone shift vs last call'
    return 1, 'in line, no change in guidance or tone'


def analyze_earnings(cfg, log=print, progress=lambda f: None):
    """Out-of-window stress test: each holding's most recent earnings report, scored on its reaction day."""
    warnings = []
    hs, market = cfg['holdings'], cfg.get('market', 'SPY')
    px = fetch_prices([h['ticker'] for h in hs] + [market] + sorted({h['etf'] for h in hs}))
    rets = px.pct_change().iloc[1:]
    days = rets.index
    key = env('ALPHAVANTAGE_API_KEY')
    if not key:
        raise ValueError('The earnings stress test needs ALPHAVANTAGE_API_KEY (earnings and transcripts).')
    budget = Budget()
    rows = []
    for i, h in enumerate(hs):
        tk = h['ticker']
        try:
            log(f'Fetching earnings for {tk}…')
            d = av_get(f'earnings_{tk}', dict(function='EARNINGS', symbol=tk), budget, log)
            q = [x for x in (d or {}).get('quarterlyEarnings', []) if pd.Timestamp(x['reportedDate']) < days.max()]
            if not q:
                warnings.append(f'{tk}: no earnings report found.')
                continue
            q = sorted(q, key=lambda x: x['reportedDate'])[-1]
            rep = pd.Timestamp(q['reportedDate'])
            after = str(q.get('reportTime', '')).lower().startswith('post')
            react = days[days > rep][0] if (after or rep not in days) and (days > rep).any() else rep
            if react not in days:
                warnings.append(f'{tk}: reaction day {react.date()} is outside the price data.')
                continue
            sp = split_day(rets, tk, h['etf'], market, react)
            segs = None
            qtr = _quarter_label(tk, q['fiscalDateEnding'])
            log(f'Fetching {tk} {qtr} call transcript…')
            t = av_get(f'transcript_{tk}_{qtr}', dict(function='EARNINGS_CALL_TRANSCRIPT', symbol=tk, quarter=qtr), budget, log)
            if t and t.get('transcript'):
                segs = t['transcript']
            def num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            rep_eps, est_eps, sur = num(q.get('reportedEPS')), num(q.get('estimatedEPS')), num(q.get('surprisePercentage'))
            o = None
            if segs and llm_provider():
                o = llm(EARN_SYS, f"Company: {h['name']}\nThesis: {h['thesis']}\n\nTranscript excerpt:\n{_retrieve(segs)}", 700)
            if not o:
                o = dict(guidance='Not given', tone='n/a', tone_shift=False, metrics=f'Reported EPS {rep_eps} vs {est_eps} estimated',
                         qa_theme='Transcript for this call not available' if not segs else 'n/a', new_vs_last='n/a', summary=[],
                         one_line=f'EPS {rep_eps} vs {est_eps} estimated; ' + ('call transcript not available from the data source.' if not segs else 'call not analysed (no LLM key).'))
                if not segs:
                    warnings.append(f'{tk}: Alpha Vantage has no transcript for {qtr}; scored on EPS only.')
            co = dict(type='Earnings call', timing='After close' if after else 'Before open', surprise=sur, guidance=o.get('guidance', 'Not given'),
                      tone=o.get('tone', 'Neutral'), toneShift=bool(o.get('tone_shift')), metrics=o.get('metrics'), qa=o.get('qa_theme'),
                      newVsLast=o.get('new_vs_last'), summary=o.get('summary') or [], newinfo=int(o.get('new_info') or 3),
                      text=o.get('one_line'), title=f'{tk} earnings call ({q["reportedDate"]})', publisher='Alpha Vantage', url=None)
            co['impact'], co['impactWhy'] = earnings_impact(co)
            lvl = attention(sp['z'], co)
            g = co['guidance'].lower() + ' guidance' if co['guidance'] != 'Not given' else 'guidance not stated'
            rows.append(dict(ticker=tk, name=h['name'], etf=h['etf'], thesis_text=h['thesis'], level=lvl,
                             headline=(f"Earnings call: EPS {sur:+.1f}% vs estimate, {g}" if sur is not None else 'Earnings call'),
                             why=f"Earnings calls are the most direct read on the thesis. {co['text'] or ''}".strip(),
                             thesis=int(o.get('thesis_relevance') or 3), split=sp, company=co, sector=None, macro=None, counts=[1, 1, 1],
                             reaction_day=str(react.date())))
        except Exception as e:
            warnings.append(f'{tk}: earnings step failed: {e}')
        progress(0.1 + 0.85 * (i + 1) / len(hs))
    order = {'High': 0, 'Medium': 1, 'Low': 2}
    rows.sort(key=lambda x: (order[x['level']], -x['thesis'], -abs(x['split']['z'])))
    nxt = next_earnings([h['ticker'] for h in hs])
    for x in rows:
        x['next_earnings'] = nxt.get(x['ticker'])
    cal = dict(summary="Out-of-window check: each holding's most recent earnings call. This tests whether the system flags company-driven moves "
                       "when a company reports.",
               events=[dict(time='Latest', name="Most recent earnings calls (each holding's own report date)", ch='Company',
                            result='Report-day returns, transcripts from the earnings-call source')])
    return dict(day='earnings', calendar=cal, context=[], rows=rows, warnings=warnings, stress=True, llm=llm_provider(), news=True)


# ----------------------------------------------------------------------------- portfolio overview
def overview(holdings, lookback=252):
    """Plain portfolio view: positions, performance vs SPY, risk, concentration. Uses current weights held constant."""
    tks = [h['ticker'] for h in holdings]
    etfs = sorted({h['etf'] for h in holdings})
    px = fetch_prices(tks + ['SPY'] + etfs)
    rets_full = px.pct_change().iloc[1:]
    r = px.pct_change().dropna().tail(lookback)
    wts = np.array([h.get('weight') or 0 for h in holdings], dtype=float)
    wts = wts / wts.sum() if wts.sum() > 0 else np.ones(len(tks)) / len(tks)
    port = r[tks] @ wts
    spy = r['SPY']

    def stat(x):
        cum = (1 + x).cumprod()
        return dict(ret=float(cum.iloc[-1] - 1), vol=float(x.std() * np.sqrt(252)), mdd=float((cum / cum.cummax() - 1).min()))
    ps, ss = stat(port), stat(spy)
    beta = float(np.cov(port, spy)[0, 1] / spy.var())
    cov = r[tks].cov() * 252
    pv = float(wts @ cov.values @ wts)
    rc = (wts * (cov.values @ wts)) / pv
    corr = r[tks].corr()
    pairs = [(tks[i], tks[j], float(corr.iloc[i, j])) for i in range(len(tks)) for j in range(i + 1, len(tks))]
    top_pair = max(pairs, key=lambda x: x[2]) if pairs else None
    last = px.iloc[-1]
    pos = []
    for h, w in zip(holdings, wts):
        t = h['ticker']
        s_ = px[t]
        try:
            sp = split_day(rets_full, t, h['etf'], 'SPY', rets_full.index[-1])
            zz = sp['z']
        except Exception:
            zz = 0.0
        pos.append(dict(ticker=t, name=h['name'], etf=h['etf'], z=float(zz), zday=str(rets_full.index[-1].date()), weight=float(w), shares=h.get('shares'), price=float(last[t]),
                        d1=float(s_.iloc[-1] / s_.iloc[-2] - 1), d5=float(s_.iloc[-1] / s_.iloc[-6] - 1),
                        m1=float(s_.iloc[-1] / s_.iloc[-22] - 1), period=float(s_.iloc[-1] / s_.iloc[-len(r) - 1] - 1),
                        risk=float(rc[tks.index(t)])))
    pos.sort(key=lambda x: -x['weight'])
    sect = {}
    for p_ in pos:
        sect[p_['etf']] = sect.get(p_['etf'], 0) + p_['weight']
    cur = {k: [round(float(v), 4) for v in (1 + s_).cumprod()] for k, s_ in {'Your portfolio': port, 'SPY': spy}.items()}
    return dict(positions=pos, portfolio=ps, spy=ss, beta=beta, top3=float(sum(p_['weight'] for p_ in pos[:3])),
                by_etf=dict(sorted(sect.items(), key=lambda x: -x[1])), top_pair=top_pair, days=len(r),
                dates=[str(d.date()) for d in r.index], curves=cur, start=str(r.index[0].date()), end=str(r.index[-1].date()))
