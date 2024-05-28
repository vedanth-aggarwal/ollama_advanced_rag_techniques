#from langchain_community.llms import Ollama

#llm = Ollama(model='phi')
#print(llm.invoke(input='Hello'))

import ollama
response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])