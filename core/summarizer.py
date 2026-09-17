from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import time


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_retries=3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=500
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:

    llm = get_llm()

    if len(transcript) <= 20000:

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert content summarizer.

Summarize the following transcript professionally.

Focus on:
- Main topics discussed
- Important points
- Conclusions
- Important details

Return the summary using clear bullet points.

Do not invent information that is not present in the transcript."""
                ),
                ("human", "{text}")
            ]
        )

        chain = prompt | llm | StrOutputParser()

        return chain.invoke({
            "text": transcript
        })

    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Summarize this portion of a content transcript.

Extract only important information such as:
- Topics discussed
- Important points
- Decisions
- Action-related information

Do not invent information."""
            ),
            ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"[{i + 1}/{len(chunks)}] summarizing chunk...", flush=True)

        try:
            summary = map_chain.invoke({
                "text": chunk
            })
        except Exception as e:
            print(f"Chunk {i + 1} failed ({e}); skipping.", flush=True)
            continue

        chunk_summaries.append(summary)
        time.sleep(1)  # small buffer to stay under free-tier RPM

    combined = "\n\n".join(chunk_summaries)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert content summarizer.

Summarize the following transcript professionally.

Focus on:
- Main topics discussed
- Important points
- Conclusions
- Important details

Return the summary using clear bullet points.

Do not invent information that is not present in the transcript."""
            ),
            ("human", "{text}")
        ]
    )

    final_chain = final_prompt | llm | StrOutputParser()

    return final_chain.invoke({
        "text": combined
    })


def generate_title(transcript: str) -> str:

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the content transcript, generate a short
professional content title.

Maximum 8 words.

Return only the title.
Do not add quotes, explanation, or punctuation."""
            ),
            ("human", "{text}")
        ]
    )

    title_chain = title_prompt | llm | StrOutputParser()

    try:
        return title_chain.invoke({
            "text": transcript[:1500]
        })
    except Exception as e:
        print(f"Title generation failed ({e}); using fallback title.", flush=True)
        return "Untitled Meeting"