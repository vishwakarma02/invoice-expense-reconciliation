# Invoice Expense Reconciliation

## Setup

Run these commands from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
New-Item -ItemType File -Force .env
```

Add the following settings to the `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL=gpt-6-astra
```

## Test the OpenAI Connection

Run this command from the project directory:

```powershell
python test_openai_connection.py
```

## Vendor Dispute Correspondence Agent

The correspondence API validates dispute evidence and creates a vendor email
draft for human review. It does not send, reply to, or forward email. When Azure
OpenAI is unavailable or not configured, it uses a deterministic template.

Copy `.env.example` to `.env` and set `AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT` to enable Azure OpenAI
drafting. Set `LLM_ENABLED=false` to use only the deterministic template.

Run the API from the project directory:

```powershell
uvicorn src.correspondence.agent:app --reload
```

The health check is at `http://127.0.0.1:8000/health`; submit draft requests to
`POST http://127.0.0.1:8000/v1/vendor-disputes/draft`. The request schema is
available at `http://127.0.0.1:8000/docs`.

Run the correspondence tests with:

```powershell
python -m pytest tests/test_correspondence_agent.py
```