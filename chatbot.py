# import dotenv
# import os
# from langchain_community.document_loaders  import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_qa_with_sources_chain
# from langchain.chains import ConversationalRetrievalChain
# from flask import Flask, request, jsonify


# dotenv.load_dotenv()

# app = Flask(__name__)
# # Step 1
# raw_documents = TextLoader("./data.txt").load()

# # Step 2
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500, chunk_overlap=20, length_function=len
# )
# documents = text_splitter.split_documents(raw_documents)

# # Step 3
# embeddings_model = OpenAIEmbeddings(openai_api_key="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")
# db = FAISS.from_documents(documents, embeddings_model)

# # Step 4
# retriever = db.as_retriever()

# # Step 5
# llm_src = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview", openai_api_key="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")
# qa_chain = create_qa_with_sources_chain(llm_src)
# retrieval_qa = ConversationalRetrievalChain.from_llm(
#     llm_src,
#     retriever,
#     return_source_documents=True,
# )

# # Output
# output = retrieval_qa({
#     "question": "What are the portfolio companies?",
#     "chat_history": []
# })
# print(f"Question: {output['question']}")
# print(f"Answer: {output['answer']}")
# print(f"Source: {output['source_documents'][0].metadata['source']}")



from flask import Flask, request, jsonify
import dotenv
import os
from langchain_community.document_loaders  import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain

dotenv.load_dotenv()

app = Flask(__name__)

# Step 1
raw_documents = TextLoader("./data.txt").load()

# Step 2
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=20, length_function=len
)
documents = text_splitter.split_documents(raw_documents)

# Step 3
embeddings_model = OpenAIEmbeddings(openai_api_key="")
db = FAISS.from_documents(documents, embeddings_model)

# Step 4
retriever = db.as_retriever()

# Step 5
llm_src = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview", openai_api_key="")
qa_chain = create_qa_with_sources_chain(llm_src)
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    chat_history = data.get('chat_history', [])
    question = data.get('question', '')

    output = retrieval_qa({
        "question": question,
        "chat_history": chat_history
    })

    return jsonify({
        "answer": output['answer'],
    })

if __name__ == '__main__':
    app.run(debug=True)