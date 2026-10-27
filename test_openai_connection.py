
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")

if not api_key or api_key == "your_actual_api_key_here":
    raise RuntimeError(
        "Set your actual OPENAI_API_KEY in the .env file."
    )

if not model:
    raise RuntimeError("Set OPENAI_MODEL in the .env file.")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model=model,
    input=(
        "Reply with exactly this text: "
        "OpenAI connection successful"
    ),
)

print("Model:", model)
print("Response:", response.output_text)
