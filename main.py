from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize,generate_title
from core.extractor import extract_action_items,extract_questions,extract_key_decisions
from core.RAG_engine import build_rag_chain,ask_question

load_dotenv()

def run_pipeline(source :str,language :str ="english")->dict:
    print("starting AI assistant ")

    chunks= process_input(source)

    transcript=transcribe_all(chunks,language=language)

    print(f"raw transcription (first 300 charcters){transcript[:300]}")

    title = generate_title(transcript)

    print(title)

    print("summary is below ....")

    summary=summarize(transcript=transcript)

    action_items=extract_action_items(transcript=transcript)
    decisions=extract_key_decisions(transcript=transcript)
    questions=extract_questions(transcript=transcript)
    

    rag_chain=build_rag_chain(transcript=transcript)

    result = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "extract_action": action_items,
        "extract_key_decision": decisions,
        "extract_question": questions,
        "rag_chain": rag_chain,
    }

    return result
    
    


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['extract_action']}")
    print(f"\n🔑 Key Decisions:\n{result['extract_key_decision']}")
    print(f"\n❓ Open Questions:\n{result['extract_question']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")