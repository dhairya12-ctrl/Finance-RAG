from google import genai
from google.genai import types

# 1. Initialize client
client = genai.Client()

# 2. Guardrails
SYSTEM_INSTRUCTIONS = """You are a precise corporate financial auditor.

=== STRICT RULES & BOUNDARIES ===
1. Base your answer ONLY on the provided context blocks under === RELEVANT CONTEXT ===.
2. If the answer cannot be found in the provided context, state EXACTLY:
   "Information not available in current disclosures"
3. Do NOT use outside knowledge, guess, or fabricate any data.
4. QUANTITATIVE PRECISION: Prioritize reporting exact dollar amounts, percentages, and quarter-over-quarter comparison metrics. Avoid vague summaries.
"""


def getting_answer(queries: list[str], chunks: list[dict]) -> str:
    main_query = queries[0]

    # Step 1: Accumulate all context blocks in the loop
    context_text = ""
    for idx, chunk in enumerate(chunks, 1):
        page_num = chunk.get("page_num", "N/A")
        context_text += (
            f"\n--- CONTEXT BLOCK {idx} (Page: {page_num}) ---\n{chunk['text']}\n"
        )

    # Step 2: Build final_prompt OUTSIDE the loop (after context_text is completely built)
    final_prompt = f"""=== RELEVANT CONTEXT ===
{context_text}

=== USER QUESTION ===
{main_query}

Please answer the user's question following your system instructions.
"""

    # Step 3: Call Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            temperature=0.0,  # Zero temperature for deterministic factual accuracy
        ),
    )

    return response.text