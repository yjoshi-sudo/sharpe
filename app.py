"""Sharpe: portfolio monitoring agent. Run: python app.py  ->  http://127.0.0.1:5057"""
import os, threading, uuid, traceback
from flask import Flask, jsonify, request, render_template, send_file
import core

# optional .env loader (no extra dependency)
_env = os.path.join(core.HERE, '.env')
if os.path.exists(_env):
    for line in open(_env):
        line = line.split('#')[0].strip() if not line.strip().startswith('#') else ''
        if '=' in line:
            k, v = line.split('=', 1)
            if v.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

app = Flask(__name__)
JOBS = {}


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/config')
def get_config():
    cfg = core.load_config()
    return jsonify(cfg=cfg, loaded=core.config_loaded(), status=dict(av=bool(core.env('ALPHAVANTAGE_API_KEY')), llm=core.llm_provider(),
                                        model=core.env('SHARPE_MODEL') or None,
                                        news_blocked=os.path.exists(os.path.join(core.CACHE_DIR, 'av_news_blocked'))))


@app.get('/sample.csv')
def sample_csv():
    return send_file(os.path.join(core.HERE, 'sample_portfolio.csv'), mimetype='text/csv', as_attachment=True,
                     download_name='sample_portfolio.csv')


@app.get('/sample_brokerage.csv')
def sample_brokerage():
    return send_file(os.path.join(core.HERE, 'sample_brokerage_export.csv'), mimetype='text/csv', as_attachment=True,
                     download_name='sample_brokerage_export.csv')


@app.post('/api/upload')
def upload():
    """Parse a portfolio CSV (multipart file or JSON {csv}) and return holdings for review. Nothing is saved yet."""
    try:
        if request.files.get('file'):
            text = request.files['file'].read().decode('utf-8-sig', errors='replace')
        else:
            text = (request.get_json(silent=True) or {}).get('csv', '')
        hs, warns = core.parse_portfolio_csv(text)
        return jsonify(holdings=hs, warnings=warns)
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post('/api/config')
def set_config():
    cfg = request.get_json()
    for h in cfg['holdings']:
        h['ticker'] = h['ticker'].strip().upper()
        h['etf'] = h['etf'].strip().upper()
    cfg.setdefault('calendar', core.load_config().get('calendar', {}))
    cfg.setdefault('market', 'SPY')
    core.save_config(cfg)
    return jsonify(ok=True, cfg=core.load_config())


@app.post('/api/analyze')
def analyze():
    day = request.get_json().get('day')
    fn = core.analyze_earnings if day == 'earnings' else None
    cfg = core.load_config()
    job = dict(id=uuid.uuid4().hex[:8], status='running', progress=0.0, log=[], result=None, error=None)
    JOBS[job['id']] = job

    def run():
        try:
            job['result'] = (fn(cfg, log=job['log'].append, progress=lambda f: job.update(progress=f)) if fn else
                             core.analyze_day(day, cfg, log=job['log'].append, progress=lambda f: job.update(progress=f)))
            job['status'] = 'done'
        except Exception as e:
            traceback.print_exc()
            job['error'], job['status'] = str(e), 'error'
    threading.Thread(target=run, daemon=True).start()
    return jsonify(id=job['id'])


@app.get('/api/job/<jid>')
def job(jid):
    j = JOBS.get(jid)
    return (jsonify(j) if j else (jsonify(error='unknown job'), 404))


@app.get('/api/days')
def days():
    cfg = core.load_config()
    px = core.fetch_prices([h['ticker'] for h in cfg['holdings']] + [cfg['market']] + sorted({h['etf'] for h in cfg['holdings']}))
    return jsonify(days=[str(d.date()) for d in px.index[-60:]], latest=str(px.index[-1].date()))


@app.get('/api/overview')
def overview():
    try:
        return jsonify(core.overview(core.load_config()['holdings']))
    except Exception as e:
        return jsonify(error=str(e)), 400


@app.post('/api/optimize')
def optimize():
    b = request.get_json() or {}
    cfg = core.load_config()
    tickers = b.get('tickers') or [h['ticker'] for h in cfg['holdings']]
    try:
        return jsonify(core.optimize(tickers, lo=float(b.get('lo', .05)), hi=float(b.get('hi', .40)),
                                     rf=float(b.get('rf', .036)), holdout=int(b.get('holdout', 20)),
                                     current={h['ticker']: h.get('weight') for h in cfg['holdings'] if h.get('weight')}))
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5057)), debug=False)
