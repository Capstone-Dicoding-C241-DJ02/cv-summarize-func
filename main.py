from google.cloud import storage
from transformers import T5Tokenizer, T5ForConditionalGeneration
from io import BytesIO
import fitz
import functions_framework
import requests
import os

PATH_TO_MODEL = "model"
PATH_TO_TOKENIZER = "tokenizer"

model = T5ForConditionalGeneration.from_pretrained(PATH_TO_MODEL)
tokenizer = T5Tokenizer.from_pretrained(PATH_TO_TOKENIZER)

storage_client = storage.Client()

def read_pdf(bucket_name, file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    contents = blob.download_as_bytes()
    bytes_stream = BytesIO(contents)

    document = fitz.open("pdf", bytes_stream)
    return document.load_page(0).get_text()
    


def summarize_resume(resume_text):

    # Preprocess input text
    inputs = tokenizer.encode("summarize: " + resume_text, return_tensors='pt', max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



@functions_framework.cloud_event
def send_summarize_result(cloud_event):
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]
    
    resume_text = read_pdf(bucket, name)
    summary = summarize_resume(resume_text)
    data = {
        "cv_name": name,
        "summarized_cv": summary
    }
    print(summary)
    URL = os.getenv("API_URL", "http://localhost:8000/candidates/summarize/cv")
    print(URL)
    response = requests.patch(URL, json=data)
    response.raise_for_status()
    print(response.json())

 