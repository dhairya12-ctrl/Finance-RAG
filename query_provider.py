import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def expand_query(original_query: str) -> list[str]:
   
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Your job is to expand a simple user query into 3 precise search queries optimized for vector search over SEC financial filings (like 10-Ks, 10-Qs, and earnings transcripts).

Rules:
1. Generate EXACTLY 3 alternative search queries.
2. Replace casual language with formal financial jargon:
   - "money made" / "sales" -> "revenue", "net income", "top-line growth"
   - "costs" / "spending" -> "operating expenses (OpEx)", "COGS", "CapEx"
   - "future predictions" / "outlook" -> "forward-looking guidance", "segment forecasts"
   - "debts" -> "long-term debt obligations", "liabilities"
3. Tailor queries to capture key metrics like gross margins, operating expenses, tax rates, and segment revenue.
4. Output MUST be formatted STRICTLY as a raw JSON array of strings containing ONLY the 3 queries. Do NOT include markdown blocks like ```json, headers, or conversational filler.
Example output format:
["query variation 1", "query variation 2", "query variation 3"]"""),
        ("human", "User Query: {question}")
    ])

    final_prompt = prompt.invoke({"question": original_query})
    
    try:
        response = llm.invoke(final_prompt)
        raw_content = response.content.strip()
        
       
        sub_queries = json.loads(raw_content)
        
       
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            return sub_queries
        else:
            raise ValueError("Parsed output is not a list of strings")
            
    except Exception as e:
   
        print(f"[Warning] Failed to parse query expansion. Error: {e}. Falling back to original query.")
        return []
    

def get_complete_query_pool(original_query: str) -> list[str]:
    """Assemble the complete query pool (Original + 3 sub-queries)."""
    # 1. Fetch sub-queries from LLM
    sub_queries = expand_query(original_query)
    
    # 2. Combine original query + sub-queries
    query_pool = [original_query] + sub_queries
    
    # 3. Deduplicate while preserving order
    unique_query_pool = list(dict.fromkeys(query_pool))
    
    return unique_query_pool
