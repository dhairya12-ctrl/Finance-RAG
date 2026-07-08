from storing import retrieve_vector_data
from sentence_transformers import CrossEncoder

db=retrieve_vector_data()

def run_multi_query_search(query_pool: list[str],db):
    """
    Goal 1: Runs vector search for each query in the pool and collects candidate child chunks.
    """
    raw_child_chunks = []
    
    for query in query_pool:
        # Search vector DB for the top k (5) child chunks for this specific query variation
        results = db.similarity_search(query, k=5)
        
        # Accumulate child chunks into our main candidate list
        raw_child_chunks.extend(results)
        
    print(f"Total raw child chunks collected across {len(query_pool)} queries: {len(raw_child_chunks)}")
    return raw_child_chunks



def resolve_and_deduplicate_parents(raw_child_chunks: list, parent_store: dict) -> list[dict]:
    """
    Goal 2: Resolves parent IDs from child chunks and deduplicates parent documents.
    """
    seen_parent_ids = set()
    unique_parents = []

    for child_doc in raw_child_chunks:
        # 1. Extract parent_id from metadata
        parent_id = child_doc.metadata.get("parent_id")
        
        if not parent_id:
            continue  # Skip if metadata is missing parent_id
            
        # 2. Check if we've already included this parent
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            
            # 3. Retrieve the full parent block from your store
            parent_data = parent_store.get(parent_id)
            
            if parent_data:
                # Store parent details along with parent_id
                unique_parents.append({
                    "parent_id": parent_id,
                    "text": parent_data if isinstance(parent_data, str) else parent_data.get("text", ""),
                    "metadata": child_doc.metadata # Keep original metadata if needed
                })

    print(f"Deduplicated {len(raw_child_chunks)} raw child chunks into {len(unique_parents)} unique Parent blocks.")
    return unique_parents


# Initialize the model once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_and_select_top_k(
    queries_list: list[str], deduplicated_parents: list[dict], top_k: int = 4
) -> list[dict]:
    """Re-ranks unique parent blocks using the primary user query

    and returns the top_k highest-scoring blocks.
    """
    if not deduplicated_parents:
        return []

    # 1. Extract the actual original query (the first item in the list)
    # If queries_list = ["What is Q3 revenue?", "Q3 revenue details", ...], primary_query is "What is Q3 revenue?"
    primary_query = (
        queries_list[0]
        if isinstance(queries_list, list)
        else str(queries_list)
    )

    # 2. Pair the primary query with each deduplicated parent text
    query_doc_pairs = [
        (primary_query, parent["text"]) for parent in deduplicated_parents
    ]

    # 3. Compute relevance scores using the Cross-Encoder
    scores = reranker.predict(query_doc_pairs)

    # 4. Attach scores to parent records
    scored_parents = []
    for parent, score in zip(deduplicated_parents, scores):
        parent_entry = parent.copy()
        parent_entry["rerank_score"] = float(score)
        scored_parents.append(parent_entry)

    # 5. Sort descending by score and slice the top 4
    scored_parents.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_parents[:top_k]