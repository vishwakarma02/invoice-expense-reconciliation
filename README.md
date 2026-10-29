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

## Tesseract OCR Installation (Windows)

Tesseract OCR is required to extract text from scanned invoices and image-based documents. Install it separately from the Python dependencies.

### 1. Download Tesseract OCR

1. Open the [Tesseract OCR Windows installer page](https://github.com/UB-Mannheim/tesseract/wiki).
2. Download the latest **64-bit Windows installer**.
3. Run the installer and follow the setup instructions.
4. Ensure that **English language data** is selected.
5. Install using the default location: `C:\Program Files\Tesseract-OCR`.

### 2. Add Tesseract to the Windows PATH

This step allows you to run Tesseract from any terminal.

1. Open the Windows Start menu and search for **Edit environment variables for your account**.
2. Select `Path` under User variables and click **Edit**.
3. Click **New** and add `C:\Program Files\Tesseract-OCR`.
4. Click **OK** to save all changes.
5. Close and reopen PowerShell or your terminal.

If you installed Tesseract in a different location, use your actual installation directory.

### 3. Verify the installation

Open a new PowerShell terminal and run:

```powershell
tesseract --version
```

The command should display the installed Tesseract version. To verify that English language data is available, run:

```powershell
tesseract --list-langs
```

Ensure `eng` appears in the output.

### 4. Configure the project

In the project root, add the following setting to your `.env` file:

```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed elsewhere, update the path accordingly. The Python ingestion agent uses this setting to locate the Tesseract executable through `pytesseract`.

### 5. Verify Python access

Activate the project's virtual environment, then run:

```powershell
python -c "from dotenv import load_dotenv; import os; import pytesseract; load_dotenv(); pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract'); print(pytesseract.get_tesseract_version())"
```

If the command prints the Tesseract version without an error, Python can access the OCR engine.

### Troubleshooting

- **`tesseract` is not recognized:** Verify the installation directory is in `PATH`, then reopen the terminal.
- **Python cannot find Tesseract:** Check that `TESSERACT_CMD` in `.env` points to the actual `tesseract.exe` file.
- **English language data is missing:** Rerun the installer or install the English language data, then verify using `tesseract --list-langs`.

**Note:** Tesseract is an external application. Installing the Python package `pytesseract` alone does not install the Tesseract OCR engine.