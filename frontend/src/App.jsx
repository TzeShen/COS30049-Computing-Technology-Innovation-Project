import React, { Component, useEffect, useMemo, useState } from 'react';
import {
  ArrowRight, ShieldCheck, LockKeyhole, Link2, AlertTriangle, CheckCircle2,
  Clock3, Trash2, Activity, Info, ArrowLeft, Sun, Moon, RotateCcw, Search,
  SlidersHorizontal
} from 'lucide-react';
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, ReferenceLine
} from 'recharts';
import Topbar from './components/Topbar.jsx';
import { analyzeUrl } from './services/api';

function validateUrl(value) {
  if (!value.trim()) return 'Please enter a URL to analyse.';
  if (value.length > 2048) return 'The URL must be 2,048 characters or fewer.';
  if (/[\u0000-\u001f\u007f]/.test(value)) return 'Control characters are not allowed.';
  try {
    const parsed = new URL(value);
    if (!['http:', 'https:'].includes(parsed.protocol)) return 'Only HTTP and HTTPS URLs are supported.';
    if (!parsed.hostname || (!parsed.hostname.includes('.') && parsed.hostname.toLowerCase() !== 'localhost')) {
      return 'Enter a complete URL with a valid domain name.';
    }
    if (parsed.username || parsed.password) return 'URLs containing embedded usernames or passwords are not supported.';
    return '';
  } catch {
    return 'Enter a valid URL, including https:// or http://.';
  }
}
function statusFor(result) {
  const label = String(result.status || '').toLowerCase();
  if (label === 'lower predicted risk' || label === 'safe') return 'Safe';
  if (label === 'potential phishing' || label === 'dangerous') return 'Dangerous';
  if (label === 'risky') return 'Risky';
  const score = Number(result.trust_score ?? 0);
  return score >= 70 ? 'Safe' : score >= 40 ? 'Risky' : 'Dangerous';
}
function Badge({ status }) {
  return <span className={'badge ' + String(status).toLowerCase()}><i />{status}</span>;
}
function HighlightedUrl({ url, reasons = [] }) {
  // URL is rendered as React text, never as HTML.
  let content = url;
  try {
    const parsed = new URL(url);
    const host = parsed.hostname;
    const hostReasons = reasons.filter(r => r.flagged && ['IP Address', 'Suspicious TLD', 'Subdomain Count'].includes(r.title));
    if (hostReasons.length) {
      const prefix = `${parsed.protocol}//`;
      const rest = url.slice(prefix.length);
      const start = rest.toLowerCase().indexOf(host.toLowerCase());
      if (start >= 0) {
        content = <>{prefix}{rest.slice(0, start)}<mark className="url-highlight" title={hostReasons.map(r => r.title).join(', ')}>{host}</mark>{rest.slice(start + host.length)}</>;
      }
    }
  } catch { /* Keep the original URL as plain text. */ }
  return <div className="highlighted-url" aria-label={'Submitted URL: ' + url}>{content}</div>;
}
function routeFor(page) {
  return ({ home: '/', result: '/result', dashboard: '/dashboard', about: '/about', 'server-error': '/error', notfound: '/not-found' })[page] || '/';
}
function pageFor(path) {
  return ({ '/': 'home', '/result': 'result', '/dashboard': 'dashboard', '/about': 'about', '/error': 'server-error' })[path] || 'notfound';
}
function Privacy() {
  return <div className="privacy"><LockKeyhole size={16} /><span>Only URL text is analysed. Checkie never opens the destination webpage.</span></div>;
}
function Home({ history, onAnalyze, busy, onClear, failedUrl, onRetry }) {
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All');
  async function submit(event) {
    event.preventDefault();
    const message = validateUrl(url);
    if (message) { setError(message); return; }
    setError('');
    await onAnalyze(url.trim());
  }
  const shown = history.filter(item => {
    const matchesText = item.url.toLowerCase().includes(query.toLowerCase());
    const matchesFilter = filter === 'All' || statusFor(item) === filter;
    return matchesText && matchesFilter;
  });
  return <main>
    <section className="hero">
      <div className="eyebrow"><span className="live-dot" /> OFFLINE URL STRUCTURE ANALYSIS</div>
      <h1 className="gradient">Phishing URL Detector</h1>
      <p className="hero-copy">Check suspicious links before opening them. Get a clear risk assessment based on the structure of a URL — without visiting the website.</p>
      <form className="url-form" onSubmit={submit} noValidate>
        <Link2 className="url-icon" size={19} />
        <label className="sr-only" htmlFor="url-input">URL to analyse</label>
        <input id="url-input" value={url} maxLength={2048} onChange={e => { setUrl(e.target.value); setError(''); }}
          placeholder="Paste a URL here (e.g. https://example.com)" autoComplete="url" aria-describedby="url-help url-error" />
        <button className="primary" disabled={busy} type="submit">
          {busy ? <><span className="spinner" /> Analysing…</> : <>Analyse URL <ArrowRight size={17} /></>}
        </button>
      </form>
      <p id="url-error" className={'form-hint ' + (error ? 'error' : '')} role="alert">{error}</p>
      <p id="url-help" className="form-hint">Accepts complete HTTP or HTTPS URLs. Maximum 2,048 characters.</p>
      {busy && <div className="scan-state" role="status" aria-live="polite"><span className="spinner" /> Checking URL structure and requesting the detection model…</div>}
      {failedUrl && !busy && <div className="retry-panel" role="alert"><span>Previous analysis failed for <b>{failedUrl}</b>.</span><button className="secondary" onClick={onRetry}><RotateCcw size={15} /> Retry</button></div>}
    </section>
    <section className="history section">
      <div className="section-head"><div><span className="kicker">YOUR ACTIVITY</span><h2>Recent URL checks</h2></div>
        <div className="history-tools"><span className="count">{history.length} this session</span>{history.length > 0 && <button className="text-button" onClick={onClear} type="button"><Trash2 size={13} /> Clear history</button>}</div>
      </div>
      {history.length > 0 && <div className="history-filters">
        <label className="search-box"><Search size={15} /><span className="sr-only">Search session history</span><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search checked URLs" /></label>
        <label className="filter-box"><SlidersHorizontal size={15} /><span className="sr-only">Filter results</span><select value={filter} onChange={e => setFilter(e.target.value)}><option>All</option><option>Safe</option><option>Risky</option><option>Dangerous</option></select></label>
      </div>}
      {shown.length ? <div className="table-wrap"><table><thead><tr><th>URL</th><th>Trust score</th><th>Prediction</th><th>Time</th></tr></thead>
        <tbody>{shown.slice(0, 30).map(item => <tr key={item.id}><td className="url-text" title={item.url}>{item.url}</td><td>{Number(item.trust_score).toFixed(1)}/100</td><td><Badge status={statusFor(item)} /></td><td>{new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td></tr>)}</tbody>
      </table></div> : <div className="empty"><Clock3 size={20} />{history.length ? 'No checks match your search or filter.' : 'Your analysed URLs will appear here.'}</div>}
    </section>
    <Privacy />
  </main>;
}
function ContributionChart({ reasons, contributions }) {
  // Prefer genuine per-prediction attribution values if a future API response supplies them.
  // The currently bundled API only supplies rule-based reasons, so the fallback is labelled accordingly.
  const hasContributions = Array.isArray(contributions) && contributions.length > 0;
  const data = hasContributions
    ? contributions.map(item => ({ name: item.feature || item.name || 'Feature', value: Number(item.contribution ?? item.value ?? 0), detail: item.detail || 'Contribution to this prediction' }))
    : reasons.map(item => ({ name: item.title, value: item.flagged ? 1 : 0, detail: item.detail }));
  return <section className="chart-card">
    <h2>{hasContributions ? 'Feature contributions to this prediction' : 'Detected URL indicators'}</h2>
    <p className="subtext">{hasContributions ? 'Local feature attribution values returned by the API' : 'Which explanatory URL checks were flagged for this URL'}</p>
    {data.length ? <div className="chart contribution-chart"><ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 18, left: 18, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
        <XAxis type="number" stroke="var(--muted)" />
        <YAxis type="category" dataKey="name" width={112} stroke="var(--muted)" tick={{ fontSize: 11 }} />
        <Tooltip
  contentStyle={{
    backgroundColor: 'var(--panel)',
    border: '1px solid var(--line)',
    borderRadius: 8}}labelStyle={{ color: 'var(--text)' }}itemStyle={{ color: 'var(--text)' }}cursor={{ fill: 'var(--text)', fillOpacity: 0.08 }}formatter={(value, name, props) => [hasContributions ? Number(value).toFixed(4) : props.payload.detail,hasContributions ? 'Contribution' : 'Indicator']}labelFormatter={label => label}/>
        <ReferenceLine x={0} stroke="var(--muted)" />
        <Bar dataKey="value" name={hasContributions ? 'Contribution' : 'Flagged indicator'} fill="#e3a83f" radius={[0, 6, 6, 0]} />
      </BarChart>
    </ResponsiveContainer></div> : <div className="chart-empty">The API did not return contribution or indicator data for this check.</div>}
    <p className="chart-disclaimer">{hasContributions ? 'Contribution values are shown as returned by the backend; check the backend documentation for the attribution method and sign convention.' : 'The current API returns rule-based indicator flags, not per-feature model contribution weights. This chart does not claim that a flagged indicator caused the model prediction.'}</p>
  </section>;
}
function Result({ result, onBack }) {
  if (!result) return <main className="empty-result"><Info size={30} /><h2>No analysis yet</h2><p>Submit a URL from the home page first.</p><button className="primary" onClick={onBack}>Back to home</button></main>;
  const trust = Math.max(0, Math.min(100, Number(result.trust_score ?? 0)));
  const status = statusFor(result);
  const reasons = Array.isArray(result.indicators) ? result.indicators : [];
  const phishingProbability = Math.max(0, Math.min(1, Number(result.phishing_probability ?? (1 - trust / 100))));
  const confidence = Math.max(phishingProbability, 1 - phishingProbability) * 100;
  return <main className="result-page">
    <div className="result-title"><span className="kicker">ANALYSIS COMPLETE</span><h1 className="gradient">URL Analysis Result</h1><HighlightedUrl url={result.url} reasons={reasons} /></div>
    <section className={'score-panel ' + status.toLowerCase()}>
      <div className="score-top"><div><span className="kicker">TRUST SCORE</span><div className="score">{trust.toFixed(1)}<small>/100</small></div><Badge status={status} /></div>
        <div className="score-ring" style={{ '--score': trust + '%' }} aria-label={`Trust score ${trust.toFixed(1)} out of 100`}><div><strong>{trust.toFixed(0)}%</strong><span>Trust</span></div></div>
      </div>
      <p className="score-description">{status === 'Safe' ? 'The model estimates a lower phishing probability for this URL.' : status === 'Risky' ? 'The model identifies characteristics that deserve closer inspection.' : 'The model estimates a high phishing probability for this URL.'} This is a prediction from URL structure only; it cannot confirm whether a website is safe.</p>
      <div className="result-metrics">
        <div><span>Prediction</span><strong><Badge status={status} /></strong></div>
        <div><span>Model confidence</span><strong>{confidence.toFixed(1)}%</strong><small>Probability of the predicted class</small></div>
        <div><span>Phishing probability</span><strong>{(phishingProbability * 100).toFixed(1)}%</strong><small>Estimated probability of phishing</small></div>
      </div>
      <div className="probability"><div><span>Estimated phishing probability</span><strong>{(phishingProbability * 100).toFixed(1)}%</strong></div><div className="track"><span style={{ width: (phishingProbability * 100) + '%' }} /></div></div>
    </section>
    <section className="signals"><div className="section-head"><div><span className="kicker">MODEL EXPLANATION</span><h2>Why this result was given</h2><p className="subtext">Signals identified by the backend's URL checks</p></div></div>
      {reasons.length ? <div className="signal-list">{reasons.map((reason, index) => <div className={'signal ' + (reason.flagged ? 'flagged' : 'clear')} key={index}>
        <span className="signal-icon">{reason.flagged ? <AlertTriangle size={20} /> : <CheckCircle2 size={20} />}</span>
        <div><strong>{reason.title}</strong><span>{reason.detail}</span></div><span className="signal-state">{reason.flagged ? 'Flagged' : 'Clear'}</span>
      </div>)}</div> : <div className="empty"><ShieldCheck size={20} />No explanation signals were returned by the backend.</div>}
    </section>
    <div className="result-visuals"><ContributionChart reasons={reasons} contributions={result.contributions} /></div>
    <p className="explanation-note">A flagged indicator is not necessarily proof of phishing. A clear indicator does not guarantee safety. The current backend response provides rule-based indicator explanations, not SHAP or other per-prediction feature attribution scores.</p>
    <div className="actions"><button className="primary" onClick={onBack}>Check another URL <ArrowRight size={17} /></button></div><Privacy />
  </main>;
}
function Dashboard({ history }) {
  // Match the two classification labels returned by the backend model.
  const distribution = [
    { name: 'Lower predicted risk', value: history.filter(x => statusFor(x) === 'Safe').length, color: '#35c99a' },
    { name: 'Potential phishing', value: history.filter(x => statusFor(x) === 'Dangerous').length, color: '#ed6676' }
  ];
  const scores = [...history].reverse().map((item, index) => ({ name: `Check ${index + 1}`, trust: Number(item.trust_score || 0), url: item.url, prediction: statusFor(item) }));
  return <main className="dashboard">
    <div className="page-heading"><span className="kicker">SESSION OVERVIEW</span><h1 className="gradient">Analysis Dashboard</h1><p className="subtext">A comparison of URLs analysed during this browser session. History is kept in memory and clears when the page is refreshed.</p></div>
    <div className="stats"><div><Activity /><span>Total checks</span><strong>{history.length}</strong></div><div><ShieldCheck /><span>Lower predicted risk</span><strong>{distribution[0].value}</strong></div><div><AlertTriangle /><span>Potential phishing</span><strong>{distribution[1].value}</strong></div></div>
    <div className="charts">
      <section className="chart-card"><h2>Result distribution</h2><p className="subtext">How URLs were classified in this session</p>{history.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution.filter(d => d.value > 0)} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={88} label>{distribution.filter(d => d.value > 0).map(d => <Cell key={d.name} fill={d.color} />)}</Pie><Tooltip /><Legend /></PieChart></ResponsiveContainer></div> : <div className="chart-empty">Analyse URLs to populate this chart.</div>}</section>
      <section className="chart-card"><h2>Trust score comparison</h2><p className="subtext">Compare the trust score of each check in this session</p>{history.length ? <div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={scores} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" /><XAxis dataKey="name" stroke="var(--muted)" /><YAxis domain={[0, 100]} stroke="var(--muted)" /><Tooltip formatter={value => [`${Number(value).toFixed(1)}/100`, 'Trust score']} labelFormatter={(label, payload) => payload?.[0]?.payload?.url || label} /><ReferenceLine y={70} stroke="#35c99a" strokeDasharray="4 4" /><Bar dataKey="trust" name="Trust score" fill="#557dff" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer></div> : <div className="chart-empty">Analyse URLs to populate this chart.</div>}</section>
    </div><Privacy />
  </main>;
}
function About() {
  return <main className="about"><span className="kicker">ABOUT CHECKIE</span><h1 className="gradient">How it works</h1><p className="subtext">Checkie estimates URL-level phishing risk by analysing characteristics in the address itself.</p>
    <div className="steps"><article><b>01</b><h2>Submit a URL</h2><p>Paste a complete HTTP or HTTPS address. The frontend validates the input before sending it to the API.</p></article><article><b>02</b><h2>Extract URL features</h2><p>The Python backend parses the URL string and calculates lexical, hostname and structural features.</p></article><article><b>03</b><h2>Review the assessment</h2><p>The trained classifier returns a trust score, phishing probability and explanatory indicators.</p></article></div>
    <div className="privacy-box"><LockKeyhole /><div><strong>No destination browsing</strong><p>Checkie never fetches, opens or executes the submitted URL. It analyses the string only.</p></div></div>
  </main>;
}
function ServerErrorPage({ message, code, onRetry, onHome, busy }) {
  return <main className="not-found server-error-page" role="alert">
    <span className="kicker">500 · SERVER ERROR</span>
    <h1 className="gradient">We couldn't finish that check.</h1>
    <p className="subtext">{message || 'An unexpected server error occurred. Please try again.'}</p>
    <p className="form-hint">Error reference: {code || 'INTERNAL_SERVER_ERROR'}</p>
    <div className="actions">
      <button className="primary" onClick={onRetry} disabled={busy}><RotateCcw size={16} /> {busy ? 'Retrying…' : 'Try again'}</button>
      <button className="secondary" onClick={onHome}>Back to home</button>
    </div>
  </main>;
}
function NotFound({ onHome }) {
  return <main className="not-found"><span className="kicker">404 · PAGE NOT FOUND</span><h1 className="gradient">This page wandered off.</h1><p className="subtext">The page you requested does not exist. Return to Checkie to analyse a URL.</p><button className="primary" onClick={onHome}><ArrowLeft size={16} /> Back to home</button></main>;
}
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(error, info) { console.error('Checkie UI error:', error, info); }
  render() {
    if (this.state.hasError) return <div className="fatal-fallback" role="alert"><AlertTriangle size={32} /><h1>Something went wrong</h1><p>Checkie could not display this page. You can try reloading the application.</p><button className="primary" onClick={() => window.location.assign('/')}>Reload Checkie</button></div>;
    return this.props.children;
  }
}
function AppContent() {
  const [page, setPage] = useState(() => pageFor(window.location.pathname));
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [apiError, setApiError] = useState('');
  const [apiErrorCode, setApiErrorCode] = useState('');
  const [history, setHistory] = useState([]);
  const [failedUrl, setFailedUrl] = useState('');
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('checkie-theme') || 'dark'; } catch { return 'dark'; }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('checkie-theme', theme); } catch { /* Theme still works for this page view. */ }
  }, [theme]);
  function navigate(next) {
    const path = routeFor(next);
    if (window.location.pathname !== path) window.history.pushState({}, '', path);
    setPage(next); setApiError(''); setApiErrorCode('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  useEffect(() => {
    const handle = () => { setPage(pageFor(window.location.pathname)); setApiError(''); };
    window.addEventListener('popstate', handle);
    return () => window.removeEventListener('popstate', handle);
  }, []);
  async function run(url) {
    setBusy(true); setApiError(''); setFailedUrl(url);
    try {
      const data = await analyzeUrl(url);
      const item = { ...data, id: `${Date.now()}-${Math.random()}`, created_at: new Date().toISOString() };
      setResult(item); setHistory(old => [item, ...old].slice(0, 30)); setFailedUrl('');
      navigate('result');
    } catch (error) {
      setApiError(error.message || 'The analysis could not be completed.');
      setApiErrorCode(error.code || (error.status === 500 ? 'INTERNAL_SERVER_ERROR' : 'API_ERROR'));
      if (error.status >= 500) {
        if (window.location.pathname !== '/error') window.history.pushState({}, '', '/error');
        setPage('server-error');
      }
    } finally { setBusy(false); }
  }
  const retry = () => { if (failedUrl) run(failedUrl); };
  return <div className="app">
    <Topbar page={page} onNavigate={navigate} theme={theme} onToggleTheme={() => setTheme(value => value === 'dark' ? 'light' : 'dark')} />
    {apiError && page !== 'server-error' && <div className="global-error" role="alert"><AlertTriangle size={17} /><span>{apiError}</span><button className="retry-inline" onClick={retry} disabled={busy}><RotateCcw size={14} /> Retry</button><button onClick={() => setApiError('')} aria-label="Dismiss error">×</button></div>}
    {page === 'home' && <Home history={history} onAnalyze={run} busy={busy} onClear={() => setHistory([])} failedUrl={failedUrl} onRetry={retry} />}
    {page === 'server-error' && <ServerErrorPage message={apiError} code={apiErrorCode} onRetry={retry} onHome={() => navigate('home')} busy={busy} />}
    {page === 'result' && <Result result={result} onBack={() => navigate('home')} />}
    {page === 'dashboard' && <Dashboard history={history} />}
    {page === 'about' && <About />}
    {page === 'notfound' && <NotFound onHome={() => navigate('home')} />}
    <footer><span>CHECKIE</span><span>Offline-first URL risk analysis</span><span>For awareness, not a safety guarantee</span></footer>
  </div>;
}
export default function App() { return <ErrorBoundary><AppContent /></ErrorBoundary>; }
