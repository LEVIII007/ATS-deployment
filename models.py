from dotenv import load_dotenv
import os
import logging
import fitz
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

from langchain.embeddings import GoogleGenerativeAIEmbeddings as GeminiEmbeddings
from langchain import GoogleGenerativeAI as Gemini

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_text_from_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    pdf_data = BytesIO(response.content)
    document = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    logging.debug(f"Text extracted successfully: {text[:20000]}")  # Log the first 20000 characters for brevity
    return text

# def get_embeddings(text):
#     gemini = Gemini(api_key=os.getenv("GEMINI_API_KEY"))
#     embedder = GeminiEmbeddings(gemini)
#     try:
#         embeddings = embedder.embed([text])
#         return embeddings[0]  # Return the first (and only) embedding
#     except Exception as e:
#         logging.error(f"Error getting embeddings: {e}")
#         return []

from google.api_core import retry


def make_embed_text_fn(model):
    @retry.Retry(timeout=300.0)
    def embed_fn(text: str) -> list[float]:
        embedding = genai.embed_content(model=model, content=text, task_type="classification")
        return embedding['embedding']
    return embed_fn

def get_embeddings(text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = 'models/embedding-001'
    try:
        embed_fn = make_embed_text_fn(model)
        embeddings = embed_fn(text)
        return embeddings
    except Exception as e:
        logging.error(f"Error getting embeddings: {e}")
        return []


# def compute_similarity(text, job_description):
#     # Compute embeddings for the text and job description
#     jd_embedding = get_embeddings(job_description)
#     content_embedding = get_embeddings(text)
    
#     # Ensure embeddings are numpy arrays
#     jd_embedding = np.array(jd_embedding).reshape(1, -1)
#     content_embedding = np.array(content_embedding).reshape(1, -1)
    
#     # Compute cosine similarity between the embeddings
#     similarity_score = cosine_similarity(jd_embedding, content_embedding)[0][0]
#     return similarity_score

from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(text, job_description):
    # Compute embeddings for the text and job description
    jd_embedding = get_embeddings(job_description)
    content_embedding = get_embeddings(text)
    
    # Ensure embeddings are numpy arrays
    if not jd_embedding or not content_embedding:
        logging.error("One or both embeddings are empty, cannot compute similarity")
        return 0.0  # or any default value or handling you prefer
    
    jd_embedding = np.array(jd_embedding).reshape(1, -1)
    content_embedding = np.array(content_embedding).reshape(1, -1)
    
    # Compute cosine similarity between the embeddings
    similarity_score = cosine_similarity(jd_embedding, content_embedding)[0][0]
    return similarity_score

def process_pdfs(job_description, category):
    logging.debug(f"process pdfs started")
    urls = fetch_urls(category)
    logging.debug(f"urls fetched successfully{urls}")
    for entry in urls:
        pdf_text = extract_text_from_pdf(entry['url'])
        score = compute_similarity(pdf_text, job_description)
        # Update the database with the similarity score
        supabase_client.table(category).update({'score': score}).eq('id', entry['id']).execute()
        logging.debug(f"updated score to database successfully")

def extract_pdf_content(id, category):
    data = request.json
    
    # Fetch the row with the URL and ID from Supabase
    response = supabase_client.table(category).select('url').eq('id', id).execute()
    if response.data:
        logging.debug("URL and ID fetched successfully")
        pdf_url = response.data[0]['url']

        # Download and process the PDF
        pdf_response = requests.get(pdf_url)
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

def compute_score(content, description):
    return 1

# Example usage
if __name__ == "__main__":
    job_description = "Example job description"
    category = "example_category"
    process_pdfs(job_description, category)
