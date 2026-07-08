from query_provider import get_complete_query_pool
from getting_answer import run_multi_query_search,resolve_and_deduplicate_parents,rerank_and_select_top_k
from storing import retrieve_vector_data
from brain import getting_answer
import json

db=retrieve_vector_data()

question="Is company in loss?"
query=get_complete_query_pool(question)


total_chunks=run_multi_query_search(query,db)

# Change this line:
with open("chunks.json", "r", encoding="utf-8") as f:
    parent_store = json.load(f)

parent_chunk=resolve_and_deduplicate_parents(total_chunks,parent_store)

selected_parent=rerank_and_select_top_k(query,parent_chunk)

answer=getting_answer(query,selected_parent)
print(answer)