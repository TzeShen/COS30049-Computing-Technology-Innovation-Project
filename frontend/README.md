# Checkie React frontend

This is the React frontend scaffold for the Checkie phishing URL detector. It is designed to connect to a FastAPI backend while reusing the existing Python model and URL feature extraction in the supplied project.

## Start the frontend

1. Install Node.js (LTS).
2. In this folder run:

   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and set `VITE_API_BASE_URL` to your FastAPI address.
4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open `http://127.0.0.1:5173`.

## API contract expected by this frontend

`POST /api/analyze`

Request:

```json
{"url":"https://example.com/login"}
```

Response expected:

```json
{
  "url": "https://example.com/login",
  "trust_score": 91.2,
  "phishing_probability": 0.088,
  "status": "Safe",
  "reasons": [
    {"title":"IP Address","flagged":false,"detail":"Not Detected"},
    {"title":"URL Length","flagged":false,"detail":"Normal (24 characters)"}
  ]
}
```

`status` can be `Safe`, `Risky`, or `Dangerous`. The `reasons` array can be generated using the existing Flask app's `_build_cards()` logic in `src/phishing_detector/app.py`.

## Integration notes

- The current ZIP contains a Flask app, not a FastAPI API. This frontend expects the new FastAPI endpoint above.
- The current prediction code in `src/phishing_detector/predict_url.py` returns `trust_score`, `phishing_probability`, `top_signals`, and `raw_features`. The FastAPI backend should adapt that result into the response contract above.
- The frontend never fetches or opens the submitted URL. It sends the URL string only to the configured Checkie API.
- Session history is currently held in React state and resets when the page is refreshed. It is not persisted to a server.
- The dashboard contains two distinct session visualisations: classification distribution and trust score per check.
- For production/assessment, remove external font imports if the application must work with no internet access at all; system font fallbacks are already provided.
