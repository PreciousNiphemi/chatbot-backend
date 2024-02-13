from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.document_loaders  import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain

load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
CORS(app)  # Initialize CORS with your app instance

# Step 1
raw_documents = TextLoader("./data.txt").load()

# Step 2
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=20, length_function=len
)
documents = text_splitter.split_documents(raw_documents)

# Step 3
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings_model)

# Step 4
retriever = db.as_retriever()

# Step 5
llm_src = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")
qa_chain = create_qa_with_sources_chain(llm_src)
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    print("Received data:", data)  
    chat_history = data.get('chat_history', [])
    messages = data.get('messages', [])
    # Find the last user message
    question = next((message['text'] for message in reversed(messages) if message['role'] == 'user'), '')

    output = retrieval_qa({
        "question": question,
        "chat_history": chat_history
    })

    print("OUTPUT DATA:", output['answer'])  
    return jsonify({
        "text": output['answer'],
    })

if __name__ == '__main__':
    app.run(debug=True)