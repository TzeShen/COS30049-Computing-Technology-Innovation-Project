import {useMemo,useState} from 'react';
import {ArrowRight,ShieldCheck,LockKeyhole,Link2,AlertTriangle,CheckCircle2,Clock3,Trash2,Activity,Globe,RefreshCw,Info} from 'lucide-react';
import {ResponsiveContainer,PieChart,Pie,Cell,Tooltip,Legend,BarChart,Bar,XAxis,YAxis,CartesianGrid} from 'recharts';
import Topbar from './components/Topbar.jsx';
import {analyzeUrl} from './services/api';

function validateUrl(value){
  if(!value.trim()) return 'Please enter a URL to analyse.';
  if(value.length>2048) return 'The URL must be 2,048 characters or fewer.';
  if(/[\u0000-\u001f\u007f]/.test(value)) return 'Control characters are not allowed.';
  try{
    const u=new URL(value);
    if(!['http:','https:'].includes(u.protocol)) return 'Only HTTP and HTTPS URLs are supported.';
    if(!u.hostname || !u.hostname.includes('.')) return 'Enter a complete URL with a domain name.';
    return '';
  }catch{return 'Enter a valid URL, including https:// or http://.';}
}
function statusFor(result){
  if(result.status) return result.status;
  const s=Number(result.trust_score??0);
  return s>=70?'Safe':s>=40?'Risky':'Dangerous';
}
function Badge({status}){return <span className={'badge '+status.toLowerCase()}><i/>{status}</span>}
function Privacy(){return <div className="privacy"><LockKeyhole size={16}/><span>Analysed locally from URL text. Checkie never opens the destination webpage.</span></div>}
function Home({history,onAnalyze,busy}){
 const [url,setUrl]=useState(''); const [error,setError]=useState('');
 async function submit(e){e.preventDefault();const err=validateUrl(url);if(err){setError(err);return;}setError('');await onAnalyze(url.trim());}
 return <main>
  <section className="hero">
   <div className="eyebrow"><span className="live-dot"/> OFFLINE URL STRUCTURE ANALYSIS</div>
   <h1 className="gradient">Phishing URL Detector</h1>
   <p className="hero-copy">Check suspicious links before opening them. Get a clear risk assessment based on the structure of a URL — without visiting the website.</p>
   <form className="url-form" onSubmit={submit} noValidate>
    <Link2 className="url-icon" size={19}/><label className="sr-only" htmlFor="url">URL to analyse</label>
    <input id="url" value={url} maxLength={2048} onChange={e=>{setUrl(e.target.value);setError('');}} placeholder="Paste a URL here (e.g. https://example.com)" autoComplete="url"/>
    <button className="primary" disabled={busy}>{busy?<><span className="spinner"/> Analysing</>:<>Analyse URL <ArrowRight size={17}/></>}</button>
   </form>
   <p className={'form-hint '+(error?'error':'')} role="alert">{error||'Only the URL text is inspected. No webpage is fetched.'}</p>
  </section>
  <section className="history section">
   <div className="section-head"><div><span className="kicker">YOUR ACTIVITY</span><h2>Recent URL checks</h2></div><span className="count">{history.length} this session</span></div>
   {history.length?<div className="table-wrap"><table><thead><tr><th>URL</th><th>Trust score</th><th>Result</th><th>Time</th></tr></thead><tbody>{history.slice(0,8).map((x,i)=><tr key={x.id||i}><td className="url-text" title={x.url}>{x.url}</td><td>{Number(x.trust_score).toFixed(1)}/100</td><td><Badge status={statusFor(x)}/></td><td>{new Date(x.created_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</td></tr>)}</tbody></table></div>:<div className="empty"><Clock3 size={20}/><span>Your analysed URLs will appear here.</span></div>}
  </section><Privacy/>
 </main>;
}
function Result({result,onBack}){
 if(!result)return <main className="empty-result"><Info size={30}/><h2>No analysis yet</h2><p>Submit a URL from the home page first.</p><button className="primary" onClick={onBack}>Back to home</button></main>;
 const trust=Number(result.trust_score??0), status=statusFor(result);
 const reasons=Array.isArray(result.reasons)?result.reasons:[];
 const probability=Number(result.phishing_probability??(1-trust/100));
 const confidence=Math.max(0,Math.min(100,probability*100));
 return <main className="result-page">
  <div className="result-title"><span className="kicker">ANALYSIS COMPLETE</span><h1 className="gradient">URL Analysis Result</h1><div className="checked-url">{result.url}</div></div>
  <section className={'score-panel '+status.toLowerCase()}>
   <div className="score-top"><div><span className="kicker">TRUST SCORE</span><div className="score">{trust.toFixed(1)}<small>/100</small></div><Badge status={status}/></div><div className="score-ring" style={{'--score':trust+'%'}}><div><strong>{trust.toFixed(0)}%</strong><span>Trust</span></div></div></div>
   <p className="score-description">{status==='Safe'?'No strong suspicious signals were identified by the model.':status==='Risky'?'This URL has some characteristics that deserve a closer look.':'This URL shows strong signals associated with phishing.'}</p>
   <div className="probability"><div><span>Estimated phishing probability</span><strong>{confidence.toFixed(1)}%</strong></div><div className="track"><span style={{width:confidence+'%'}}/></div></div>
  </section>
  <section className="signals"><div className="section-head"><div><span className="kicker">MODEL EXPLANATION</span><h2>Why this result was given</h2><p className="subtext">Signals identified in the submitted URL</p></div></div>
   {reasons.length?<div className="signal-list">{reasons.map((r,i)=><div className={'signal '+(r.flagged?'flagged':'clear')} key={i}><span className="signal-icon">{r.flagged?<AlertTriangle size={20}/>:<CheckCircle2 size={20}/>}</span><div><strong>{r.title}</strong><span>{r.detail}</span></div><span className="signal-state">{r.flagged?'Flagged':'Clear'}</span></div>)}</div>:<div className="empty"><ShieldCheck size={20}/>No strong suspicious signals were returned by the backend.</div>}
  </section>
  <div className="actions"><button className="primary" onClick={onBack}>Check another URL <ArrowRight size={17}/></button></div><Privacy/>
 </main>;
}
function Dashboard({history}){
 const distribution=[{name:'Safe',value:history.filter(x=>statusFor(x)==='Safe').length,color:'#35d39a'},{name:'Risky',value:history.filter(x=>statusFor(x)==='Risky').length,color:'#f4bd50'},{name:'Dangerous',value:history.filter(x=>statusFor(x)==='Dangerous').length,color:'#ff6477'}];
 const scores=history.slice(0,10).map((x,i)=>({name:`Check ${i+1}`,trust:Number(x.trust_score||0),url:x.url}));
 return <main className="dashboard"><div className="page-heading"><span className="kicker">SESSION OVERVIEW</span><h1 className="gradient">Analysis Dashboard</h1><p className="subtext">A summary of URLs analysed during this browser session.</p></div>
  <div className="stats"><div><Activity/><span>Total checks</span><strong>{history.length}</strong></div><div><ShieldCheck/><span>Safe</span><strong>{distribution[0].value}</strong></div><div><AlertTriangle/><span>Risky or dangerous</span><strong>{distribution[1].value+distribution[2].value}</strong></div></div>
  <div className="charts"><section className="chart-card"><h2>Result distribution</h2><p className="subtext">How the checked URLs were classified</p>{history.length?<div className="chart"><ResponsiveContainer><PieChart><Pie data={distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>{distribution.map(d=><Cell key={d.name} fill={d.color}/>)}</Pie><Tooltip/><Legend/></PieChart></ResponsiveContainer></div>:<div className="chart-empty">Analyse URLs to populate this chart.</div>}</section>
  <section className="chart-card"><h2>Trust score by check</h2><p className="subtext">Compare trust scores across recent checks</p>{history.length?<div className="chart"><ResponsiveContainer><BarChart data={scores}><CartesianGrid strokeDasharray="3 3" stroke="#2a354b"/><XAxis dataKey="name" stroke="#aab4c5"/><YAxis domain={[0,100]} stroke="#aab4c5"/><Tooltip formatter={v=>[`${v}/100`,'Trust score']} labelFormatter={(label,p)=>p?.[0]?.payload?.url||label}/><Bar dataKey="trust" fill="#557dff" radius={[6,6,0,0]}/></BarChart></ResponsiveContainer></div>:<div className="chart-empty">Analyse URLs to populate this chart.</div>}</section></div><Privacy/>
 </main>;
}
function About(){return <main className="about"><span className="kicker">ABOUT CHECKIE</span><h1 className="gradient">How it works</h1><p className="subtext">Checkie estimates URL-level phishing risk by analysing characteristics in the address itself.</p><div className="steps"><article><b>01</b><h2>Submit a URL</h2><p>Paste a complete HTTP or HTTPS address. The frontend checks basic input validity before sending it to the API.</p></article><article><b>02</b><h2>Extract URL features</h2><p>The Python backend parses the URL string and calculates lexical, hostname, structural and obfuscation features.</p></article><article><b>03</b><h2>Review the assessment</h2><p>The trained classifier returns a trust score and explanatory signals. The result is an estimate, not a guarantee of safety.</p></article></div><div className="privacy-box"><LockKeyhole/><div><strong>No destination browsing</strong><p>Checkie must never fetch, open or execute the submitted URL. It analyses the string only.</p></div></div></main>}
export default function App(){
 const [page,setPage]=useState('home'),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[apiError,setApiError]=useState(''),[history,setHistory]=useState([]);
 async function run(url){setBusy(true);setApiError('');try{const data=await analyzeUrl(url);const item={...data,id:Date.now(),created_at:new Date().toISOString()};setResult(item);setHistory(old=>[item,...old].slice(0,30));setPage('result');}catch(e){setApiError(e.message);}finally{setBusy(false);}}
 return <div className="app"><Topbar page={page} onNavigate={setPage}/>{apiError&&<div className="global-error" role="alert"><AlertTriangle size={17}/>{apiError}<button onClick={()=>setApiError('')} aria-label="Dismiss error">×</button></div>}
 {page==='home'&&<Home history={history} onAnalyze={run} busy={busy}/>}
 {page==='result'&&<Result result={result} onBack={()=>setPage('home')}/>}
 {page==='dashboard'&&<Dashboard history={history}/>}
 {page==='about'&&<About/>}
 <footer><span>CHECKIE</span><span>Offline-first URL risk analysis</span><span>For awareness, not a safety guarantee</span></footer>
 </div>;
}