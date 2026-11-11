# Checkie frontend

## Run on this computer

Requirements: Node.js (LTS) and npm.

1. Open PowerShell in the repository's `frontend` folder.
2. Install the locked dependencies:

   ```powershell
   npm.cmd ci
   ```

3. Start the frontend:

   ```powershell
   npm.cmd run dev
   ```

4. Open `http://127.0.0.1:5173` in Chrome, Edge or Firefox.
5. Start the FastAPI backend separately (see the repository root README).

The frontend sends URL text to `POST /api/analyze`. It never opens or fetches the submitted destination.

## Run on a phone browser over Wi-Fi

This is a responsive web app, not a separately installed native Android/iOS app. To test it on a phone:

1. Connect the phone and computer to the same Wi-Fi network.
2. Find the computer's IPv4 address with `ipconfig` in PowerShell (look under the active Wi-Fi adapter).
3. In the backend PowerShell terminal, set the allowed frontend origins, replacing `192.168.1.20` with the computer's actual IPv4 address:

   ```powershell
   $env:CHECKIE_FRONTEND_ORIGINS="http://127.0.0.1:5173,http://localhost:5173,http://192.168.1.20:5173"
   ```

4. Start FastAPI so it listens on the network:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. In `frontend/.env.local`, set the API address (replace the IP):

   ```env
   VITE_API_BASE_URL=http://192.168.1.20:8000
   ```

6. Start Vite (it is configured to listen on `0.0.0.0`):

   ```powershell
   npm.cmd run dev -- --host 0.0.0.0
   ```

7. On the phone, open `http://192.168.1.20:5173`.

Keep both terminals running. Allow Python/Node through Windows Firewall on a private network if prompted. If the phone cannot connect, check both devices are on the same Wi-Fi, verify the IP address, and check firewall rules. Do not use `127.0.0.1` as the phone's API address: on the phone, that points back to the phone itself.

For a phone demo, keep the backend and frontend on the same trusted private Wi-Fi. This development setup uses HTTP and is not suitable for public internet deployment.

## Error handling and test cases

- A valid URL such as `https://www.example.org` returns HTTP 200 with the model result (`trust_score`, `phishing_probability`, `status`, `indicators`/`reasons`, and local EBM contributions). The frontend displays the analysis result page.
- Invalid URL syntax sent directly to the API returns HTTP 422 with `error.code = INVALID_URL`; the frontend's own input validation catches ordinary invalid entries first and displays an inline message, without sending the request.
- A Pydantic request-body error (for example, missing `url`) returns HTTP 422 with `error.code = VALIDATION_ERROR`.
- An unexpected backend exception returns HTTP 500 with `error.code = INTERNAL_SERVER_ERROR`; the frontend shows a dedicated server-error page with a retry action. Internal exception details are logged server-side and are not exposed to the browser.
- If the backend is unreachable, the frontend displays a connection error and retry action. This is a network failure, not an HTTP 500 response.

### Verify the 500 error page in the browser

For a temporary local test only, add `raise RuntimeError("Temporary 500 test")` inside `analyze()` in `backend/app/main.py`, immediately before the `score_url(...)` call. Restart FastAPI, submit a syntactically valid URL in the React app, and confirm that the browser navigates to `/error` and displays the safe server-error message and `INTERNAL_SERVER_ERROR` reference. The internal exception text must not appear in the browser. Remove the temporary `raise` line and restart FastAPI after the test.

Use FastAPI Swagger UI at `http://127.0.0.1:8000/docs` to send direct API requests and inspect the documented request/response schemas and examples. Test a valid JSON body for HTTP 200, an invalid URL (or a missing `url` field) for HTTP 422, and use the temporary test above for HTTP 500. Do not use a real suspicious link by opening it; Checkie analyses the string only.

## Build and dependency audit

```powershell
npm.cmd run build
npm.cmd audit
```

`npm audit` requires internet access. These commands should be run before submission; do not report them as passed unless they complete successfully.
