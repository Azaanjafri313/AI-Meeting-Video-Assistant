import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from core.vector_store import build_vector_store,load_vector_store,get_retriever

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=3,
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs ])

QA_SYSTEM_PROMPT = """You are an expert assistant. Answer the user's question
based ONLY on the transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from transcript:
{context}"""

def build_rag_chain(transcript:str):
    vector_store=build_vector_store(transcript)

    retriever=get_retriever(vector_store,k=4)

    llm=get_llm()

    prompt=ChatPromptTemplate.from_messages(
        [(
             "system",
            QA_SYSTEM_PROMPT,
        ),
        ("human", "{question}"),]

    )

    rag_chain=(
        {"context":retriever|RunnableLambda(format_docs),
        "question": RunnablePassthrough()
        }
        |prompt|llm|StrOutputParser()

    )

    return rag_chain


def load_rag_chain():
    vector_store=load_vector_store()

    retriever=get_retriever(vector_store,k=4)

    llm=get_llm()

    prompt=ChatPromptTemplate.from_messages(
        [(
             "system",
            QA_SYSTEM_PROMPT,
        ),
        ("human", "{question}"),]

    )

    rag_chain=(
            {"context":retriever|RunnableLambda(format_docs),
            "question": RunnablePassthrough()
            }
            |prompt|llm|StrOutputParser()
    
        )
    
    return rag_chain


def ask_question(rag_chain,question:str)->str:
    print(f"Question :{question}")
    answer=rag_chain.invoke(question)
    print(f"answer:{answer}")
    return answer