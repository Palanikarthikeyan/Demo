import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

DATA_PATH = "C:\\Users\\karth\\GenRocket-Demo\\mydata"
documents = []

for file in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(filepath)

    elif file.endswith(".txt"):
        loader = TextLoader(filepath)

    elif file.endswith(".csv"):
        loader = CSVLoader(filepath)

    elif file.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(filepath)

    elif file.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(filepath)

    else:
        continue

    documents.extend(loader.load())




text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(docs)}")


# step 3 embedding object
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# step 4 stores to vectordb
vectorstore  = FAISS.from_documents(docs,embeddings)

# step 5 create retrievalobject
retriever_obj = vectorstore.as_retriever()

# step 6 create llm object
#llm_obj = ChatGroq(model="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY"))

llm_obj = ChatGroq(model="llama-3.1-8b-instant",api_key="<KEY>")

# step 7 prompt - Strict prompt - Strict RAG
my_prompt = PromptTemplate(
    input_variables=["context","question","history"],
    template="""
    You are a strict retrieval QA assistant.
    user following context and chat history to answer the question
    if the answer is not present in the context and history reply exactly with:
    "I don't know, the document doesnot contain this information."
    Context:
    {context}
    Chat history:
    Question:
    {question}
    Answer:
    """
)

# step 8: QAChain
rag_chain = RetrievalQA.from_chain_type(llm = llm_obj,
                                        retriever=retriever_obj,
                                        chain_type_kwargs={"prompt":my_prompt})



# Step 9: Chat history
chat_history = {}

def f1(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

obj = RunnableWithMessageHistory(rag_chain,f1,input_messages_key="query",history_messages_key="history")



#  Step 10: Create session ID 
session_id="session-1"
query1="what is attention in transformer?"
# Step 11: user query
response = obj.invoke({"query":query1},config={"configurable":{"session_id":session_id}})
print(response)




#  Step 10: Create session ID 
session_id="session-2"
query1="What is Synthetic data"
# Step 11: user query
response = obj.invoke({"query":query1},config={"configurable":{"session_id":session_id}})
print(response)


#  Step 10: Create session ID 
session_id="session-3"
query1="How to test GenRocket Automation?"
# Step 11: user query
response = obj.invoke({"query":query1},config={"configurable":{"session_id":session_id}})
print(response)



#  Step 10: Create session ID 
session_id="session-4"
query1="what is class?"
# Step 11: user query
response = obj.invoke({"query":query1},config={"configurable":{"session_id":session_id}})
print(response)


print(f"Loaded {len(documents)} documents")
