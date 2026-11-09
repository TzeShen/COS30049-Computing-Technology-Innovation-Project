# Checkie React frontend

Improved React + Vite interface for the Checkie URL-structure phishing detector. It sends the submitted URL string to the FastAPI API; it does **not** open or fetch the destination website.

## Run locally

1. Install Node.js LTS.
2. Open a terminal in this `Checkie-React-Frontend` folder.
3. Install dependencies: `npm install`
4. Copy `.env.example` to `.env` (the default API URL is `http://127.0.0.1:8000`).
5. Start the frontend: `npm run dev`
6. Open `http://127.0.0.1:5173`. Start the FastAPI backend separately before analysing URLs.

Build a production bundle with `npm run build`; preview it with `npm run preview`.

## FastAPI contract

The frontend calls `POST /api/analyze` with `{ "url": "https://example.com/login" }`. The response must contain numeric `trust_score` and `phishing_probability` (a fraction from 0 to 1), plus `status` and `reasons` (an array of `{ "title", "flagged", "detail" }`). This matches the supplied Checkie FastAPI backend. Set `VITE_API_BASE_URL` if the backend uses another address.

## Improvements in this version

- Browser URL paths for Home, Result, Dashboard and How it works; browser back/forward is supported. Unknown paths show a 404 fallback page.
- Flagged hostname-related signals highlight the hostname in the submitted URL. The original URL remains plain text rendered by React (never interpreted as HTML).
- Clear session history control. History remains in memory for the current tab/session and is not stored on a server.
- Dashboard charts include clearer axis context, legends/tooltips and empty states.
- More explicit model limitations and explanation of the highlighted hostname.
- Removed the external Google Fonts import so the interface does not request a font from the internet.
- API response validation and clearer connection/response errors.

## Security notes

- Never add `dangerouslySetInnerHTML` for submitted URL content.
- Do not add code that fetches, previews, resolves or opens the submitted URL.
- Configure backend CORS to allow only the intended frontend origin.
- Session history is local React state and is cleared when the page is refreshed.
