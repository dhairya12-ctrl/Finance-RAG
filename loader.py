from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path):
    loader=PyPDFLoader(path)
    documents = loader.load()
    print(f"Total Pages: {len(documents)}")
    return documents


