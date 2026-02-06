from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64, io, os, json
from pypdf import PdfReader
from docx import Document
from openai import AzureOpenAI
from fastapi.middleware.cors import CORSMiddleware

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-10-21",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = FastAPI(title="Document Field Extraction API")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractRequest(BaseModel):
    file_base64: str
    fields: List[str]

# -------- TEXT EXTRACTION --------
def extract_text(file_bytes: bytes) -> str:
    # PDF
    if file_bytes[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(file_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)

    # DOCX
    try:
        doc = Document(io.BytesIO(file_bytes))
        return " ".join(p.text for p in doc.paragraphs)
    except:
        pass

    # TXT fallback
    return file_bytes.decode("utf-8", errors="ignore")

# -------- API --------
@app.post("/extract-fields")
def extract_fields(req: ExtractRequest):
    try:
        file_bytes = base64.b64decode(req.file_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64")

    document_text = extract_text(file_bytes)
    if not document_text.strip():
        return {}

    prompt = f"""
Extract ONLY the following fields from the document text.

RULES:
- Use ONLY the document text
- Do NOT guess or infer
- If a field is not found, return empty string
- Return STRICT JSON only (key-value pairs)
- No explanations, no extra text

Fields:
{req.fields}

Document:
{document_text}
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid JSON from model")
