# Contributing to CoverWise

## Prerequisites

- Python 3.11+
- A Google Cloud project with Vertex AI enabled
- A [CMS API key](https://developer.cms.gov/) for plan data

## Local setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/ritwiksharan/CoverWise.git
cd CoverWise

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your values

# 5. Authenticate with Google Cloud
gcloud auth application-default login

# 6. Run the server
cd backend
uvicorn main:app --reload --port 8080
```

The app will be available at `http://localhost:8080`.

## Docker (alternative)

```bash
cp .env.example .env  # fill in your values
docker compose up --build
```

## Environment variables

| Variable | Description |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID with Vertex AI enabled |
| `GOOGLE_CLOUD_REGION` | GCP region (default: `us-central1`) |
| `CMS_API_KEY` | CMS marketplace API key |
| `FORCE_OPEN_ENROLLMENT` | Set to `true` to bypass enrollment period checks |
