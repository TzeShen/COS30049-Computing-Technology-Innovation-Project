const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 20000;

// POST /api/analyze { url }; backend returns trust_score, phishing_probability,
// status and reasons. Submitted URLs are sent as JSON text and are never opened.
export async function analyzeUrl(url) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(`${BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ url }),
      signal: controller.signal
    });
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The analysis request timed out. Check the API and try again.');
    throw new Error('Cannot reach the Checkie API. Start the FastAPI backend and try again.');
  } finally {
    window.clearTimeout(timeout);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === 'string' ? detail : detail?.message || body.message;
    throw new Error(message || `The API returned an error (${response.status}).`);
  }
  if (!body || typeof body.trust_score !== 'number' || typeof body.phishing_probability !== 'number' || !body.url) {
    throw new Error('The API response was incomplete. Check that the frontend is connected to the matching Checkie FastAPI backend.');
  }
  if (body.phishing_probability < 0 || body.phishing_probability > 1 || body.trust_score < 0 || body.trust_score > 100) {
    throw new Error('The API returned an invalid prediction range. Please check the backend response.');
  }
  return body;
}
