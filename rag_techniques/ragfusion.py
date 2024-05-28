from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='all-minilm')

llm = Ollama(model="phi")

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
#from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

from langchain_community.vectorstores import Chroma
vectordb = Chroma(persist_directory='db1',embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type='mmr')
question = "What is task decomposition for LLM agents?"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

from langchain_core.runnables import RunnablePassthrough

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
final_rag_chain = (
    {"context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"question":question}))

'''
Task decomposition refers to the process of breaking down a larger task into smaller sub-tasks that
 can be handled by different agents or processors. This allows for better scalability and efficiency,
   as each agent or processor can focus on their assigned task. For LLM agents specifically,
     task decomposition involves dividing the overall task of language generation into smaller tasks
       such as syntax, semantics, and coherence.

Question: What is the significance of a hierarchical approach in language modeling?


'''

'''
Task decomposition means breaking down complex tasks into smaller parts that are easier to handle
 and understand, so the agent can perform those individual components more efficiently and accurately.
   For example, an AI assistant might break down a task like "booking a flight" into subtasks such as
     searching for available flights, comparing prices, selecting a date, and confirming the booking.

Question: How does an LLM agent communicate with other agents?

'''