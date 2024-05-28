from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='all-minilm')

llm = Ollama(model="phi")
vectordb = Chroma(persist_directory='db1',embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type='mmr')

# More abstract broader questions
# HyDE document genration
template = """Please write a very short brief scientific paper passage to answer the question
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser()
)

# Run
question = "What is task decomposition for LLM agents?"
generate_docs_for_retrieval.invoke({"question":question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever
retireved_docs = retrieval_chain.invoke({"question":question})
#retireved_docs

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
print(retriever.get_relevant_documents('Task decomposition'))
print('--> FINAL CHAIN')
final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"context":retireved_docs,"question":question}))

'''
 In AI systems, task decomposition refers to breaking down complex tasks into smaller, more manageable
   sub-tasks that can be executed by individual components or agents working together.
     This allows for greater efficiency and scalability in achieving a larger goal.
'''
'''
Task decomposition refers to the process of breaking down a complicated task into smaller, 
manageable subgoals for an LLM agent. This helps the agent handle complex tasks more efficiently
 and effectively. The model can use techniques such as Chained Thoughts (CoT) or Tree of Thoughts (ToT)
   to explore multiple reasoning possibilities at each step.
'''