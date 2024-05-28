# Alternative is to use hugggingface transformers
# https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html

# IMPORTS -----------------------------------------------------------------------------
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
#from llama_index.llms.ollama import Ollama
#from langchain_community import hub
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain import embeddings
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
import ollama
import llama_index.llms
import time
# pip install llama-index-llms-ollama
# from llama_index.llms.ollama import Ollama

# LLM Model -------------------------------------------------------------------------------
#llm = ollama.load_model('phi')>
llm = Ollama(model="phi")
#print(llm.invoke('What is candy?'))
#llm = ChatOllama(model="llama3")


# Creating database ----------------------------------------------------------------------
"""
def load_pdf(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf('./')
print('-> PDF EXTRACTED')

#llm = ChatOllama(model='llama3', format="json", temperature=0)

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(extracted_data)
print('-> DOCUMENT SPLIT')
# Embed

persist_directory = 'plaintext_db'
vectordb = Chroma.from_documents(documents=splits,embedding=OllamaEmbeddings(model='all-minilm'),persist_directory=persist_directory)

# Wrapper on sqlite3 in backend
vectordb.persist()

"""
# Hugging face transformer alternative --------------------------------------------------------------------
from langchain_community.llms import CTransformers
#llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
 #                 model_type="llama",
   #               config={'max_new_tokens':512,
     #                     'temperature':0.8})



# Embeddings and retrievers -------------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model='all-minilm')
vectordb = Chroma(persist_directory='database',embedding_function=embeddings)
retriever1 = vectordb.as_retriever(search_type='mmr',search_kwargs={'fetch_k':11})
retriever2 = vectordb.as_retriever(search_type='similarity',search_kwargs={'k':3})
retriever3 = vectordb.as_retriever(search_type='similarity_score_threshold',search_kwargs={'score_threshold':0.7})


# Prompt template and chains ------------------------------------------------------------------------------
prompt_template = """
Contituition - Follow these rules strictly:
1) Your are an LLM for a RAG application
2) Answer the question strictly using only the given information and no external knowledge
3) Refine, summarize, aggregate the information to give a concise precise answer
4) The output should be 1 final well phrased answer statement

Information Database ( Only source to be used): 
{context}

Question: {question}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate.from_template(prompt_template)

question = 'What are the major AI companies and their models'

chain1 = (
    {"context": retriever1 | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain2 = (
    {"context": retriever2 | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain3 = (
    {"context": retriever3 | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print('-> CHAIN MADE')

# GENERATION -----------------------------------------------------------------------------------------------
start = time.time()
print('Answer:' + chain1.invoke(question))
print('Time:' + str(time.time() - start))

print('Answer:' + chain2.invoke(question))
print('Time:' + str(time.time() - start))

print('Answer:' + chain3.invoke(question))
print('Time:' + str(time.time() - start))
