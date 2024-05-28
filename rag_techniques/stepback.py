# Few Shot Examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='all-minilm')

llm = Ollama(model="phi")
vectordb = Chroma(persist_directory='db1',embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type='mmr')
# More abstract broader questions

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

generate_queries_step_back = prompt | llm | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})

# Response prompt
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": question}))

'''
Task decomposition refers to the process of breaking down a larger task into smaller sub-tasks
 that can be performed by an artificial intelligence agent. This allows the agent to focus on one
   subtask at a time, increasing efficiency and accuracy in completing the overall task. The sub-tasks
     can range from simple computational tasks to more complex problem-solving scenarios. 
     By decomposing the task, LLM agents can achieve better performance and adaptability to different 
     environments.
'''

'''
Task decomposition involves breaking down complex tasks into smaller, manageable subgoals
 that can be tackled more efficiently by an AI agent powered by a large language model (LLM).
   This allows the agent to break down big problems into multiple steps and handle them more
     effectively.

# Original Question: What is a task-specific instruction for LLM agents?

'''