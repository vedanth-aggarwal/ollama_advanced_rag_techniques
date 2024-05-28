from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='all-minilm')

llm = Ollama(model="phi")

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

print('--> GEN QUERIES')
from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
vectordb = Chroma(persist_directory='db1',embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type='mmr')
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

#docs = retrieval_chain.invoke({"question":question})

print('--->FINAL STAGE')
from operator import itemgetter
#from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"question":question}))

'''
 Task decomposition involves breaking down complex tasks into smaller, more manageable subtasks
   that an artificial intelligence agent can accomplish individually and then combine to complete
     the overall goal.
'''

'''
 Task decomposition refers to breaking down a complex task into smaller subtasks that can be
   solved by different tools or approaches, including natural language processing and machine learning
     models. In the context of LLM agents, this means providing them with specific instructions on how
       to approach a given task using available resources and capabilities. For example, an agent may
         need to search for information in a database, read text from multiple sources, or perform
           calculations based on user inputs. By decomposing these tasks into smaller components,
             the agent can more efficiently allocate its resources and achieve better results.
'''