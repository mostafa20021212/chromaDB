import chromadb
import argparse
from nltk.tokenize import sent_tokenize 

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="my_collection")

def preprocess_text(text):
    """Split the text into sentences or chunks."""
    sentences = sent_tokenize(text)
    return sentences

def main():
    parser = argparse.ArgumentParser(description="Test Chroma with large text.")
    parser.add_argument('--text', type=str, help="Large text to split and store in Chroma")
    parser.add_argument('--query', type=str, help="Query text to search in the collection")
    
    args = parser.parse_args()

    if not args.text or not args.query:
        print("Please provide both a large text and a query.")
        return

    chunks = preprocess_text(args.text)

    chunk_ids = [f"id{i}" for i in range(len(chunks))]
    collection.upsert(
        documents=chunks,
        ids=chunk_ids
    )

    results = collection.query(
        query_texts=[args.query],
        n_results=2
    )

    print("Query Results:")
    print(results)

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    main()


