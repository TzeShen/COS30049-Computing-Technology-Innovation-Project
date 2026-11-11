# Checkie — Phishing URL Detector

Checkie estimates phishing risk from the text of a URL and shows a trust score and URL indicators. The FastAPI backend loads the saved PhiUSIIL EBM and its 26 features at startup. It does not open the submitted URL or fetch the destination. A lower predicted risk does not establish that a site is safe.

## Run locally

Install Git, Python 3.11, and Node.js with npm. Installing dependencies requires internet access. The saved model is included, so you do not need to download the training datasets or retrain the model to run the app.

If you do not already have the repository, clone it:

```sh
git clone https://github.com/TzeShen/COS30049-Computing-Technology-Innovation-Project.git
cd COS30049-Computing-Technology-Innovation-Project
```

### Terminal 1: Start the FastAPI backend

On Windows (PowerShell), from the repository root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.3"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On macOS or Linux, from the repository root:

```sh
cd backend
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade "pip==26.3"
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
./.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Visit [the health endpoint](http://127.0.0.1:8000/api/health). It should report `status: "ready"`, `model_loaded: true`, and `feature_count: 26`.

### Terminal 2: Start the React frontend

Open a second terminal in the repository root. On Windows (PowerShell):

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

On macOS or Linux:

```sh
cd frontend
npm ci
npm run dev
```

Keep both terminals running and open [http://127.0.0.1:5173](http://127.0.0.1:5173). Use that exact address because the backend allows requests from that frontend origin. If the app cannot reach the API, check that the backend is still running on port 8000 and the frontend is on port 5173.

## Try the dashboard

Paste these addresses into Checkie one at a time:

| URL | Expected model result |
| --- | --- |
| `https://www.example.org` | Lower predicted risk |
| `http://192.0.2.1/verify-account` | Potential phishing |

Open **Dashboard** without reloading the page. A fresh session with these two checks should show **2 total checks, 1 lower predicted risk, and 1 potential phishing**. History resets when the page reloads.

## API

| Method and route | Purpose |
| --- | --- |
| `GET /api/health` | Checks whether the saved model is ready. |
| `POST /api/analyze` | Accepts JSON such as `{"url":"https://www.example.org"}` and returns a trust score, prediction, model contributions, and URL indicators. |
| `GET /docs` | Opens the interactive FastAPI documentation. |

The current app uses `backend/` and `frontend/`. The root `src/` contains earlier project work. Flask is not needed to run this version.