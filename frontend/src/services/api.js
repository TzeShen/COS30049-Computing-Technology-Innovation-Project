const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

// Expected FastAPI contract:
// POST /api/analyze  { "url": "https://..." }
// Response: { url, trust_score, phishing_probability, status, reasons: [{title, flagged, detail}] }
// The UI keeps session history locally; no URL is ever opened by this frontend.
export async function analyzeUrl(url) {
  let response;
  try {
    response = await fetch(`${BASE}/api/analyze`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url})
    });
  } catch {
    throw new Error('Cannot reach the Checkie API. Start the FastAPI backend and try again.');
  }
  const body = await response.json().catch(()=>({}));
  if (!response.ok) {
    const detail = body.detail;
    throw new Error(typeof detail === 'string' ? detail : detail?.message || body.message || `The API returned an error (${response.status}).`);
  }
  if (typeof body.trust_score !== 'number' || typeof body.phishing_probability !== 'number') {
    throw new Error('The API response was incomplete. Check that the frontend is connected to the matching Checkie FastAPI backend.');
  }
  return body;
}