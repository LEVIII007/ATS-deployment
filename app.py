import os
import base64
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from supabase import create_client, Client
from sqlalchemy import create_engine
from dotenv import load_dotenv
import fitz
import logging
from io import BytesIO
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from langchain.memory import ConversationBufferWindowMemory
import google.generativeai as genai

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask application and CORS
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Constants for Supabase and LLM
DATABASE_URL = os.getenv('DATABASE_URL')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
API_KEY_GEMINI = os.getenv('API_TOKEN_GEMINI')

# Initialize Supabase client
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize the conversation buffer memory
memory = ConversationBufferWindowMemory(k=50)

# Function to extract text from PDF file
def extract_text_from_pdf(file_stream):
    try:
        doc = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

# # Function to get embeddings of text using Gemini embeddings
# def get_embeddings(text):
#     genai.configure(api_key=API_KEY_GEMINI)
#     # model = genai.Model('gemini-1.0-pro')
#     model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     try:
#         response = model.embed_text(text)
#         embeddings = response["embeddings"]
#         return embeddings
#     except Exception as e:
#         logging.error(f"Error getting embeddings: {e}")
#         return []


# import google.generativeai as genai
from google.api_core import retry

def make_embed_text_fn(model):
    @retry.Retry(timeout=300.0)
    def embed_fn(text: str) -> list[float]:
        # Set the task_type to CLASSIFICATION.
        embedding = genai.embed_content(model=model, content=text, task_type="classification")
        return embedding['embedding']

    return embed_fn

# Function to get embeddings of text using Gemini embeddings
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

# Function to create input text for LLM
def create_input_text(all_pdf_texts, number, query_text):
    input_text = "You are a hiring manager at a company. You have received multiple resumes for a job opening regarding the job description. Now you have to answer the Query with these documents I am giving to you:\n\n"
    input_text += f"Query:\n{query_text}\n"
    input_text += f"Number of candidates you have to output:\n{number}\n"
    for i, pdf_text in enumerate(all_pdf_texts, 1):
        input_text += f"Document {i}:\n{pdf_text}\n\n"
    return input_text

# Function to get response from LLM
def get_response_from_llm(input_text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.Model('gemini-1.0-pro')
    output = model.generate_text(input_text)
    response = output["text"]
    return response

# Route for file upload
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        # Before saving the details in the database table, first we'll truncate the table
        trunc = supabase_client.table('resumes').delete().neq("id", 0).execute()
        logging.debug(f"Truncated table: {trunc}") 
        data = request.json
        job_description = data.get('jobDesc')
        logging.debug(f"Received job description: {job_description}")
        job_description_text = "You are a hiring manager at a company. Your work is to judge the resumes on the basis of the job description. You have to figure out some key points from the job description which I can easily check in the candidates' resume. I'll give you the job description. These are the key points I needed from the job description: 1. Qualifications required for the job. 2. Skills required for this job. 3. Preferred skills. 4. Candidates' roles and responsibilities"
        job_description_text += f"\n\nJob Description:\n{job_description}"

        # Get response from LLM
        job_description_response = get_response_from_llm(job_description_text)
        logging.debug(f"Response from LLM: {job_description_response}")
        job_description_response = job_description_response.replace("*", "")
        logging.debug(f"Response from LLM after removing *: {job_description_response}")
        job_description_embeddings = get_embeddings(job_description_response)

        files = data.get('files', [])
        logging.debug(f"Received {len(files)} files")

        for file in files:
            file_content = base64.b64decode(file['content'])
            content = extract_text_from_pdf(BytesIO(file_content))

            if content:
                content_embeddings = get_embeddings(content)
                score = cosine_similarity([job_description_embeddings], [content_embeddings])[0][0]
                logging.debug(f"Computed similarity score: {score}")

                # Save data to Supabase
                res = supabase_client.table('resumes').insert({
                    'resumetext': content, 
                    'score': score, 
                    'embedding': json.dumps(content_embeddings.tolist())
                }).execute()
                logging.debug(f"Saved data to database: {res}")

            else:
                logging.warning("Empty content extracted from PDF")

        return jsonify({'status': 'success', 'message': 'Uploaded file(s) successfully'}), 200

    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({'error': 'Failed to upload file'}), 500

# Route for generating response from prompt
@app.route('/api/prompt', methods=['POST'])
def prompt():
    try:
        data = request.json
        query = data.get('prompt')
        number = data.get('shortlistedCand')
        logging.debug(f"query: {query} and shortlisted candidate: {number}")

        # Fetch top 100 resumes and their embeddings
        response = supabase_client.table('resumes').select('resumetext', 'embedding', 'score').order('score', desc=True).limit(100).execute()

        # Check if the response has data
        if response.data:
            # Extract resume content and embeddings
            resume_content = [row['resumetext'] for row in response.data]
            top_n_embeddings = [json.loads(row['embedding']) for row in response.data]
            resume_scores = [row['score'] for row in response.data]
            resume_score_100 = [score * 100.0 for score in resume_scores]
            logging.debug(f"resume score 100: {resume_score_100}")
            logging.debug(f"resume score length: {len(resume_score_100)}")
        else:
            logging.error("No data found in the response.")
            return jsonify({'error': 'No resumes found'}), 500

        # Calculate similarities and retrieve top N resumes
        query_embedding = get_embeddings(query)
        similarities = cosine_similarity([query_embedding], top_n_embeddings)[0]

        # Multiply the similarity score with 100 and then again multiplied with the previous score
        similarities = [score * 100.0 for score in similarities]
        updated_similarity_score = [a + (b * 1.5) for a, b in zip(similarities, resume_score_100)]
        logging.debug(f"updated Similarities score: {updated_similarity_score}")

        # Get top N indices
        top_indices = np.argsort(updated_similarity_score)[::-1][:int(number)]
        logging.debug(f"Top indices are: {top_indices}")

        # Retrieve the content of the top N resumes
        all_pdf_texts = [resume_content[i] for i in top_indices]
        logging.debug(f"Top N resume content: {all_pdf_texts}")

        # Add query and resume content to memory
        memory.add_context(query)
        memory.add_context(all_pdf_texts)

        # Create input text for LLM
        input_text = create_input_text(all_pdf_texts, number, query)
        logging.debug(f"Input text for LLM: {input_text}")

        # Get response from LLM
        final_response = get_response_from_llm(input_text)
        logging.debug(f"Final response from LLM: {final_response}")

        return jsonify({'response': final_response})

    except Exception as e:
        logging.error(f"An error occurred during prompt processing: {e}")
        return jsonify({'error': 'Failed to process prompt'}), 500

# Route for chatting
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        logging.debug(f"Received message: {user_message}")

        # Append the user's message to memory
        memory.append(user_message)
        logging.debug(f"Current memory buffer: {memory.buffer}")

        # Create input text for the LLM
        chat_input_text = "\n".join(memory.buffer)
        logging.debug(f"Input text for chat LLM: {chat_input_text}")

        chat_response = get_chat_response_from_llm(chat_input_text)

        # Append the assistant's response to memory
        memory.append(chat_response)
        logging.debug(f"Updated memory buffer: {memory.buffer}")

        return jsonify({'response': chat_response}), 200

    except Exception as e:
        logging.error(f"Error handling chat: {e}")
        return jsonify({'error': 'Failed to handle chat'}), 500

def get_chat_response_from_llm(chat_input_text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.Model('gemini-1.0-pro')
    output = model.generate_text(chat_input_text)
    response = output["text"]
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)
