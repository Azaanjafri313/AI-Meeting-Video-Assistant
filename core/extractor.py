from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

import os 

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=3,
    )
def build_chain(system_prompt :str):
    llm=get_llm()

    return (
        RunnablePassthrough()|RunnableLambda(lambda x:{"text":x})|
        ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                ("human","{text}"),
            ]) |llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain=build_chain(
        """You are an expert meeting and content analyst. From the meeting or content transcript,
        extract all action items. for each provide:
        -Task description 
        - Owner (who is responsible)
        - Deadline (if mentioned,else write 'Not specefied')
        Format as numbered list. if none found say 'NO ACTION ITEM FOUND'  """
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting and content analyst.
 
Extract ONLY substantively important questions from the meeting or content
transcript — questions about decisions, tasks, blockers, or topics that
matter to the outcome of the discussion.
 
For each question, provide:
- Question
- Person who asked it (if mentioned)
- Status: "Unresolved" if never answered in the transcript, or a brief
  summary of the answer if it was addressed.
 
Strictly EXCLUDE:
- Small talk or pleasantries ("do you have a minute?", "how are you?")
- Rhetorical questions
- Any question that was answered immediately and has no lasting
  significance to the discussion's outcome
 
Do not infer missing information.
Format the output as a numbered list.
If no substantively important questions are found, return "NO QUESTIONS FOUND"."""
    )

    return chain.invoke(transcript)



def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting and content analyst.

Extract all important decisions that were clearly made during the meeting or content discussion.

For each decision, provide:
- Decision
- Reason or context (if mentioned)
- Person or team responsible (if mentioned)

Do not treat suggestions, opinions, or possibilities as decisions.
Do not infer missing information.
Format the output as a numbered list.
If no clear decisions are found, return "NO KEY DECISIONS FOUND"."""
    )

    return chain.invoke(transcript)


