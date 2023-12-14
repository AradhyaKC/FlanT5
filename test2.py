
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from MapReduceDocumentsChainSrc import MapReduceDocumentsChain
from ReduceDocumentsChainSrc import ReduceDocumentsChain

# from langchain.chains import StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from StuffDocumentsChainSrc import StuffDocumentsChain

from langchain.schema import Document



load_dotenv()
#map template
#--------------------------------------------------------------------------------
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

map_template = """This is content read from a pdf and is delimited by  ``` . 
```
{context}
```
Please extract only the important details from this content read from the pdf which answers the question which is delimited by ``` in the next section.
If it is not relevant , do not answer anything. 
```
{question}
```

"""

map_prompt = PromptTemplate(template=map_template, input_variables=["context"],partial_variables={'question':"What is socialization"})
map_chain = LLMChain(llm=llm,prompt=map_prompt)

# #--------------------------------------------------------------------------------

# #reduce template
# #--------------------------------------------------------------------------------
system_prompt = """You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

system_prompt = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer as concise as possible."""

def get_prompt_template(user_question,history=False):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    if history:
        instruction = """
        Context: {history} \n {context}
        User: {question}"""
        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
    else:
        instruction = """
        Context: {context}
        User: {question}"""
        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        
        prompt = PromptTemplate( template=prompt_template,input_variables=["context"],partial_variables={'question':user_question})
  
    return prompt

reduce_chain = LLMChain(llm=llm,prompt=get_prompt_template(user_question="what are the different types of government in modern times"))
combine_documents_chain=StuffDocumentsChain(llm_chain=reduce_chain,document_variable_name="context")


reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=1000)

map_reduce_chain=MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="context",
    return_intermediate_steps=False)



print(map_reduce_chain.run([Document(page_content="""Socialization and  Social  Control  Module  97  
 under the direct control of the government who take care of his/her population  
and trying  to provide  basic  essentials  with security.  
In modern  times,  there  are different  types  of government  with varied  
philosophies like democracy, communism, dictatorship and others.

 """),
 Document(page_content="""
Introduction  
Conformity  studies  stresses  on persons  to conform  the prospects  of a group,  
association, institute, society or leader. It is a category of social effect comprising a  
variation in trust or actions in command to fitting in group. It may include the physical  
occurrence of others or make -believe linking the pressure of social norms or hope under  
the group  influence.
Notes  
 """)
 ]))



# reduce_chain=LLMChain(llm=llm,prompt=get_prompt_template(True))
