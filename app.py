"""Sharpe: portfolio monitoring agent.

Stateless JSON API (the browser holds the portfolio and sends it with each request), so the same code runs locally
(`python app.py`) and on Vercel's serverless Python runtime (`api/index.py` imports `app`).
"""
import hmac
import os

from flask import Flask, jsonify, request, send_from_directory

import core

# optional .env loader for local runs (Vercel injects real environment variables)
_env = os.path.join(core.HERE, '.env')
if os.path.exists(_env):
    for line in open(_env):
        line = line.split('#')[0].strip() if not line.strip().startswith('#') else ''
        if '=' in line:
            k, v = line.split('=', 1)
            if v.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

PUBLIC = os.path.join(core.HERE, 'public')
app = Flask(__name__, static_folder=None)


@app.before_request
def gate():
    """If SHARPE_PASSWORD is set, every data endpoint needs it in the X-Access-Code header (the UI asks for it once)."""
    pw = core.env('SHARPE_PASSWORD')
    if pw and request.path.startswith('/api/') and request.path != '/api/config':
        if not hmac.compare_digest(request.headers.get('X-Access-Code', '').encode(), pw.encode()):
            return jsonify(error='Access code required or incorrect.'), 401


def api(fn):
    """Run a handler and turn exceptions into a JSON error the UI can show."""
    try:
        return jsonify(fn(request.get_json(silent=True) or {}))
    except Exception as e:                                   # noqa: BLE001
        return jsonify(error=str(e)), 400


# ---- static files (served by the CDN on Vercel; by Flask locally)
@app.get('/')
def index():
    return send_from_directory(PUBLIC, 'index.html')


@app.get('/<path:p>')
def static_files(p):
    if p.startswith('api/'):
        return jsonify(error='not found'), 404
    return send_from_directory(PUBLIC, p)


# ---- API
@app.get('/api/config')
def get_config():
    return jsonify(auth=bool(core.env('SHARPE_PASSWORD')),
                   status=dict(av=bool(core.env('ALPHAVANTAGE_API_KEY')), llm=core.llm_provider(),
                               model=core.env('SHARPE_MODEL') or None))


@app.post('/api/upload')
def upload():
    """Parse a portfolio CSV (multipart `file` or JSON {csv}); returns holdings and warnings, stores nothing."""
    try:
        if request.files.get('file'):
            text = request.files['file'].read().decode('utf-8-sig', errors='replace')
        else:
            text = (request.get_json(silent=True) or {}).get('csv', '')
        hs, warns = core.parse_portfolio_csv(text)
        return jsonify(holdings=hs, warnings=warns)
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.get('/api/days')
def days():
    try:
        px = core.fetch_prices(['SPY'])
        return jsonify(days=[str(d.date()) for d in px.index[-60:]], latest=str(px.index[-1].date()))
    except Exception as e:                                   # noqa: BLE001
        return jsonify(error=str(e)), 400


# Brief for one day = prep (splits) + macro + one request per holding; the browser assembles the page.
@app.post('/api/day/prep')
def day_prep():
    return api(lambda b: core.prep_day(b['day'], b['holdings'], b.get('market', 'SPY')))


@app.post('/api/day/macro')
def day_macro():
    return api(lambda b: core.macro_day(b['day'], b['prev'], b['holdings']))


@app.post('/api/day/holding')
def day_holding():
    return api(lambda b: core.holding_day(b['day'], b['prev'], b['holding'], b['split'], b.get('macro_items', []),
                                          b.get('cal_text', 'n/a')))


@app.post('/api/earn/holding')
def earn_holding():
    return api(lambda b: core.earnings_holding(b['holding'], b.get('market', 'SPY')))


@app.post('/api/next')
def nxt():
    return api(lambda b: dict(next=core.next_earnings([h['ticker'] for h in b['holdings']])))


@app.post('/api/overview')
def overview():
    return api(lambda b: core.overview(core.normalize_holdings(b['holdings'])))


@app.post('/api/optimize')
def optimize():
    return api(lambda b: core.optimize(b['holdings'], lo=float(b.get('lo', .05)), hi=float(b.get('hi', .40)),
                                       rf=float(b.get('rf', .036)), holdout=int(b.get('holdout', 20))))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5057)), debug=False)
