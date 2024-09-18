from flask import Flask, request, render_template, jsonify
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import validators
import docx2txt
import os
import pickle
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask_cors import CORS
from better_profanity import profanity  # Import the profanity filter
load_dotenv()

app = Flask(__name__)
CORS(app)


groq_api_key = os.environ['GROQ_API_KEY']

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2', llm_model='llama-3.1-8b-instant'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.memory = ConversationBufferWindowMemory(k=5)  # Adjustable memory length
        self.groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=llm_model
        )
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            memory=self.memory
        )

    def extract_text_from_pdfs(self, pdf_files):
        texts = []
        try:
            for pdf_file in pdf_files:
                text = ""
                pdf = pdfium.PdfDocument(pdf_file)
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    textpage = page.get_textpage()
                    text += textpage.get_text_range()
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_word(self, docx_files):
        texts = []
        try:
            for docx_file in docx_files:
                text = docx2txt.process(docx_file)
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_txt(self, txt_files):
        texts = []
        try:
            for txt_file in txt_files:
                text = txt_file.read().decode("utf-8")
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def fetch_url_content(self, urls):
        contents = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                if len(text.split()) < 50:
                    text = ' '.join([div.get_text() for div in soup.find_all('div') if len(div.get_text().split()) > 50])
                contents.append(text)
            except requests.exceptions.RequestException as e:
                return str(e), False
        return contents, True

    def vectorize_content(self, texts):
        sentences = []
        for text in texts:
            sentences.extend(text.split('. '))
        
        embeddings = self.model.encode(sentences)
        self.index = create_faiss_index(embeddings)
        
        # Save embeddings and sentences to a pickle file
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump((embeddings, sentences), f)

    def load_embeddings(self):
        # Load embeddings and sentences from the pickle file
        with open('embeddings.pkl', 'rb') as f:
            embeddings, sentences = pickle.load(f)
        self.index = create_faiss_index(embeddings)
        return sentences

    def retrieve_relevant_content(self, query, k=3):
        if self.index is None:
            raise ValueError("Embeddings have not been loaded yet.")
        
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=k)
        sentences = self.load_embeddings()
        relevant_sentences = [sentences[i] for i in I[0]]
        return relevant_sentences

    def generate_answer(self, context, query):
        user_message = f'{context}\n\n{query}'
        response = self.conversation(user_message)
        return response['response']

    def chat_with_llm(self, user_message):
        response = self.conversation(user_message)
        return response['response']

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_documents', methods=['POST'])
def process_documents():
    rag_system = RAGSystem()
    input_type = request.form.get('input_type')

    if input_type == "PDFs":
        pdf_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_pdfs(pdf_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "Word Files":
        docx_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_word(docx_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "TXT Files":
        txt_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_txt(txt_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "URLs":
        urls = request.form.get("urls").splitlines()
        url_list = [url.strip() for url in urls if url.strip()]
        if not all(validators.url(url) for url in url_list):
            return jsonify({'error': "Please enter valid URLs."}), 400

        texts, success = rag_system.fetch_url_content(url_list)
        if not success:
            return jsonify({'error': texts}), 400

    else:
        return jsonify({'error': "Invalid input type."}), 400

    rag_system.vectorize_content(texts)
    return jsonify({'message': "Documents processed and embeddings generated successfully."})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    query = request.form.get('query')

    # Profanity check before processing the query
    if profanity.contains_profanity(query):
        return jsonify({'error': "Please use appropriate language to ask your question."}), 400
    if not query:
        return jsonify({'error': "Please provide a query."}), 400

    rag_system = RAGSystem()
    rag_system.load_embeddings()  # Load pre-computed embeddings
    relevant_chunks = rag_system.retrieve_relevant_content(query, k=3)
    combined_context = ' '.join(relevant_chunks)
    answer = rag_system.generate_answer(combined_context, query)

    return jsonify({
        'relevant_chunks': relevant_chunks,
        'answer': answer
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message')

    # Profanity check before processing the message
    if profanity.contains_profanity(user_message):
        return jsonify({'response': "Please use appropriate language to chat."}), 400
    
    if not user_message:
        return jsonify({'error': "Please enter a message to chat with the LLM."}), 400

    rag_system = RAGSystem()
    response = rag_system.chat_with_llm(user_message)
    return jsonify({'response': response})


