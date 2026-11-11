const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 20000;

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
    if (error.name === 'AbortError') {
      const timeoutError = new Error('The analysis request timed out. Check the API and try again.');
      timeoutError.status = 0;
      throw timeoutError;
    }
    const networkError = new Error('Cannot reach the Checkie API. Start the FastAPI backend and try again.');
    networkError.status = 0;
    throw networkError;
  } finally {
    window.clearTimeout(timeout);
  }

  // Read the response body even for non-2xx statuses. fetch() does not reject
  // for HTTP 4xx/5xx responses; response.ok must be checked explicitly.
  const responseText = await response.text();
  let body = {};
  if (responseText) {
    try {
      body = JSON.parse(responseText);
    } catch {
      body = { message: responseText };
    }
  }

  if (!response.ok) {
    const apiError = body?.error;
    const detail = body?.detail;
    const message = apiError?.message
      || (typeof detail === 'string' ? detail : detail?.message)
      || body?.message
      || `The API returned an error (${response.status}).`;
    const error = new Error(message);
    error.status = response.status;
    error.code = apiError?.code || (response.status === 500 ? 'INTERNAL_SERVER_ERROR' : 'API_ERROR');
    throw error;
  }

  if (!body || typeof body.trust_score !== 'number'
      || typeof body.phishing_probability !== 'number' || !body.url) {
    const error = new Error('The API response was incomplete. Check that the frontend is connected to the matching Checkie FastAPI backend.');
    error.status = 502;
    error.code = 'INVALID_API_RESPONSE';
    throw error;
  }
  if (body.phishing_probability < 0 || body.phishing_probability > 1
      || body.trust_score < 0 || body.trust_score > 100) {
    const error = new Error('The API returned an invalid prediction range. Please check the backend response.');
    error.status = 502;
    error.code = 'INVALID_API_RESPONSE';
    throw error;
  }
  return body;
}
