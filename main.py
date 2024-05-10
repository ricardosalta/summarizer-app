from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import os
import openai

from pydantic import BaseModel

class Request(BaseModel):
    url: str

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_TOKEN")

if groq_api_key:
    openai.api_base = "https://api.groq.com/openai/v1"
    openai.api_key = groq_api_key
    language_model = "llama3-8b-8192"
else:
    openai.api_key = openai_api_key
    language_model = "gpt-3.5-turbo-0125"

@app.post("/summarize")
async def summarize(request: Request):
    url = request.url
    page_content = get_page_content(url)
    summary = generate_summary(page_content)
    return JSONResponse(content=jsonable_encoder(summary))

def get_page_content(url):
    import requests
    from bs4 import BeautifulSoup

    page = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    })
    soup = BeautifulSoup(page.content, "html.parser")
    return soup.get_text()

def generate_summary(page_content):
    completion = openai.ChatCompletion.create(
        model=language_model,
        messages=[
            {"role": "system", "content": "You are a summarization assistant. The user provides text dumped from Python BeautifulSoup.get_text() and you summarize it. You will be as concise as possible while maintaining accuracy and grammatical correctness."},
            {"role": "user", "content": page_content},
        ]
    )
    return completion.choices[0].message["content"].strip()
