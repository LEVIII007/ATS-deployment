# from flask import Flask, request, jsonify
# from flask_cors import CORS
from supabase import create_client, Client
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import httpx
import logging
import fitz
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# app = Flask(__name__)
# CORS(app) 
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
print(SUPABASE_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def extract_text_from_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    pdf_data = BytesIO(response.content) 
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text() 
    logging.debug(f"Text extracted successfully: {text[:20000]}")  # Log the first 1000 characters for brevity
    return text


def compute_similarity(text, job_description ):#, model, util
    # Compute embeddings for the text and job description
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    content_embedding = model.encode(text, convert_to_tensor=True)
    # Compute cosine similarity between the embeddings
    score = util.cos_sim(jd_embedding, content_embedding)
    # Extract the float value from the tensor
    similarity_score = score.item() 
    return similarity_score


def get_embeddings(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings.cpu().tolist()  # Convert tensor to list



def process_pdfs(job_description,category):
        logging.debug(f"process pdfs started")
        urls = fetch_urls(category)
        logging.debug(f"urls fetched successfully{urls}")
        for entry in urls:
            pdf_text = extract_text_from_pdf(entry['url'])
            score = compute_similarity(pdf_text, job_description)
            # Update the database with the similarity score
            supabase_client.table(category).update({'score': score}).eq('id', entry['id']).execute()
            logging.debug(f"updated score to database successfully")
# def row_count(name):
#     table_name = name
#     logging.debug(f"table name is {table_name}")
#     if not table_name:
#         return jsonify({'status': 'failure', 'message': 'Table name is required'}), 400

#     try:
#         # Fetch the row count
#         response = supabase_client.table(table_name).select("*", count="exact").execute()
#         if response.count is not None:
#             logging.debug(f"response fetched of row count {response.count}")
#             row_count = response.count
#             return row_count
#         else:
#             return jsonify({'status': 'failure', 'message': 'Failed to fetch row count'})
#     except Exception as e:
#         logging.debug(f"Failed to fetch row count: {e}")
#         return jsonify({'status': 'failure', 'message': 'Failed to fetch row count'})
    


def extract_pdf_content(id,category):
    data = request.json
    
    # Fetch the row with the URL and ID from Supabase
    response = supabase_client.table(category).select('url').eq('id', id).execute()
    if response.data:
        logging.debug("URL and ID fetched successfully")
        pdf_url = response.data[0]['url']

        # Download and process the PDF (same as before)
        pdf_response = httpx.get(pdf_url)
        pdf_response.raise_for_status()
        pdf_content = pdf_response.content

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        text_content = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()

        return text_content
    else:
        logging.debug("No data found for the given category")
        return jsonify({'status': 'failure', 'message': 'No data found'})



def compute_score(content,description):
    return 1