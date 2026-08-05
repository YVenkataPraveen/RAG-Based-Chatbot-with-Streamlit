"""CLI interface for docqa."""

import argparse
import sys

from docqa.core import DocumentQA


def main():
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Chat with any document using RAG",
    )
    parser.add_argument(
        "files", nargs="*", help="Documents to index (PDF, TXT, CSV, DOCX)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)"
    )
    args = parser.parse_args()

    try:
        qa = DocumentQA(openai_api_key=args.api_key, model=args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.files:
        print(f"Indexing {len(args.files)} file(s)...")
        stats = qa.index(args.files)
        print(f"Indexed {stats['documents']} documents, {stats['chunks']} chunks.\n")
    else:
        print("No files provided — running in general chat mode.\n")

    print("Type your questions (Ctrl+C to quit):\n")
    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            answer = qa.ask(question)
            print(f"AI:  {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
