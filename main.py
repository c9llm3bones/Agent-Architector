import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.prepro.pipeline import build_vectorstore
from src.llm.qa_chain import create_rag_chain


def main():
    load_dotenv()

    print("building index..")
    try:
        vectorstore = build_vectorstore("data")
    except Exception as e:
        print(f"error building index: {e}")
        return

    print("creating rag chain...")
    try:
        rag_chain = create_rag_chain(vectorstore, model_name="mistral-small-latest")
    except Exception as e:
        print(f"error creating rag chain: {e}")
        return

    print("ready to answer questions. type 'exit' to exit.\n")
    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            if not question:
                continue

            print("\nsearching for information and formulating an answer...\n")
            answer = rag_chain.invoke(question)
            print(f"Answer:\n{answer}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"error generating answer: {e}\n")


if __name__ == "__main__":
    main()