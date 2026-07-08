from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def chunker(document):
    parent_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    child_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    parents=parent_splitter.split_documents(document)

    for parent in parents:
        unique_id = str(uuid.uuid4())
        parent.metadata["parent_id"] = unique_id

    child_chunks = []

    for parent in parents:
        children = child_splitter.split_documents([parent])

        for child in children:
            child.metadata["parent_id"] = parent.metadata["parent_id"]

        child_chunks.extend(children)

    return parents,child_chunks
