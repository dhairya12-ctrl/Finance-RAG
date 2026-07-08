from loader import load_pdf
from chunking import chunker
import json
from storing import create_vector_store


document=load_pdf("NVIDIA.pdf")
parents,children=chunker(document)

result_dict = {parent.metadata.get("parent_id"): parent.page_content for parent in parents}
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

vector=create_vector_store(children)