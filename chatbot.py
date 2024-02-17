# import logging
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv, find_dotenv
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_qa_with_sources_chain
# from langchain.chains import ConversationalRetrievalChain

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv(find_dotenv(), override=True)

# app = Flask(__name__)
# CORS(app)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=20, length_function=len
# )
# embeddings_model = OpenAIEmbeddings()

# class CompanyData:
#     def __init__(self):
#         self.companies = {}

#     def add_company(self, company_id, data_path):
#         logger.info(f"Adding company {company_id} with data from {data_path}")
#         raw_documents = TextLoader(data_path).load()
#         documents = text_splitter.split_documents(raw_documents)
#         db = FAISS.from_documents(documents, embeddings_model)
#         retriever = db.as_retriever()
#         llm_src = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")
#         retrieval_qa = ConversationalRetrievalChain.from_llm(
#             llm_src,
#             retriever,
#             return_source_documents=True,
#         )
#         self.companies[company_id] = retrieval_qa

#     def get_company(self, company_id):
#         logger.info(f"Retrieving company {company_id}")
#         return self.companies.get(company_id)

# company_data = CompanyData()
# company_data.add_company("gp", "./companies/gp.txt")
# company_data.add_company("pinnacle", "./companies/new-pinnacle.txt")
# company_data.add_company("hillside", "./companies/new-hillside.txt")
# company_data.add_company("happy", "./companies/new-happy.txt")
# company_data.add_company("handyman", "./companies/new-handyman.txt")


# @app.route('/ask', methods=['POST'])
# def ask():
#     data = request.get_json()
#     logger.info(f"Received request data: {data}")
#     company_id = data.get('company_id')
#     retrieval_qa = company_data.get_company(company_id)
#     if retrieval_qa is None:
#         logger.error(f"Invalid company ID: {company_id}")
#         return jsonify({"error": "Invalid company ID"}), 400
#     chat_history = data.get('chat_history', [])
#     messages = data.get('messages', [])
#     question = next((message['text'] for message in reversed(messages) if message['role'] == 'user'), '')
#     logger.info(f"Processing question: {question}")
#     output = retrieval_qa({
#         "question": question,
#         "chat_history": chat_history
#     })
#     logger.info(f"Returning answer: {output['answer']}")
#     return jsonify({
#         "text": output['answer'],
#     })

# if __name__ == '__main__':
#     app.run()





import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import tool
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
CORS(app)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)
embeddings_model = OpenAIEmbeddings()



class CompanyData:
    def __init__(self):
        self.companies = {}
        self.appointment_links = {
            "pinnacle": "https://pinnacle.vet/?televetWidgetSelectRequestType=RequestAppointment",
            "hillside": "https://book.your.vet/?org=hillside&locationId=76",
            "happy": "https://happypetsvethospital.com/request-appointment/",
            "handyman": "https://handymanconnection.com/find-a-location/",
        }

    def add_company(self, company_id, data_path):
        logger.info(f"Adding company {company_id} with data from {data_path}")
        raw_documents = TextLoader(data_path).load()
        documents = text_splitter.split_documents(raw_documents)
        db = FAISS.from_documents(documents, embeddings_model)
        retriever = db.as_retriever()
        llm_src = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")
        retrieval_qa = ConversationalRetrievalChain.from_llm(
            llm_src,
            retriever,
            return_source_documents=True,
        )
        self.companies[company_id] = retrieval_qa

    def get_company(self, company_id):
        logger.info(f"Retrieving company {company_id}")
        return self.companies.get(company_id)

company_data = CompanyData()
company_data.add_company("gp", "./companies/gp.txt")
company_data.add_company("pinnacle", "./companies/new-pinnacle.txt")
company_data.add_company("hillside", "./companies/new-hillside.txt")
company_data.add_company("happy", "./companies/new-happy.txt")
company_data.add_company("handyman", "./companies/new-handyman.txt")


@tool
def check_booking(query: str, company_id:str) -> str:
     """Returns the link to book an appointment, when asked how to schedule or book an apppointment"""
     return company_data.appointment_links.get(company_id, "No appointment link found.")

    
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    logger.info(f"Received request data: {data}")
    company_id = data.get('company_id')
    retrieval_qa = company_data.get_company(company_id)
    if retrieval_qa is None:
        logger.error(f"Invalid company ID: {company_id}")
        return jsonify({"error": "Invalid company ID"}), 400
    chat_history = data.get('chat_history', [])
    messages = data.get('messages', [])
    question = next((message['text'] for message in reversed(messages) if message['role'] == 'user'), '')
    logger.info(f"Processing question: {question}")
    
    # Use the custom tool
    appointment_link = check_booking.run({"query": question, "company_id": company_id})
    if appointment_link:
        return jsonify({"text": appointment_link})
    
    output = retrieval_qa({
        "question": question,
        "chat_history": chat_history
    })
    logger.info(f"Returning answer: {output['answer']}")
    return jsonify({
        "text": output['answer'],
    })

if __name__ == '__main__':
    app.run()