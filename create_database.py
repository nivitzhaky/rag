from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from vector_store import create_vector_store_from_documents, get_collection_name

load_dotenv()

DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_pgvector(chunks, pre_delete_collection=True)


def load_documents():
    return load_documents_from_path(DATA_PATH, "*.md")


def load_documents_from_path(data_path: str, glob_pattern: str = "*.md"):
    loader = DirectoryLoader(data_path, glob=glob_pattern)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        preview_index = min(10, len(chunks) - 1)
        document = chunks[preview_index]
        print(document.page_content)
        print(document.metadata)

    return chunks


def set_context_tag(chunks: list[Document], context_tag: str | None):
    if not context_tag:
        return chunks
    for chunk in chunks:
        chunk.metadata["context_tag"] = context_tag
    return chunks


def save_to_pgvector(chunks: list[Document], pre_delete_collection: bool = True):
    create_vector_store_from_documents(chunks, pre_delete_collection=pre_delete_collection)
    print(f"Saved {len(chunks)} chunks to collection {get_collection_name()}.")


if __name__ == "__main__":
    main()
