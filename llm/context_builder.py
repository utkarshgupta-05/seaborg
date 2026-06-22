"""
llm/context_builder.py

Builds the prompt for the HYBRID path.
Fuses structured data (authoritative numerical facts) with semantic data (contextual rows)
to provide a single, grounded prompt for the LLM.
"""

import pandas as pd

HYBRID_PROMPT = """You are SeaBorg, a strict ocean data analyst.

You have been provided with two types of evidence to answer the user's question:
1. AUTHORITATIVE DATA SUMMARY: This contains exact aggregations (averages, counts, ranges) calculated directly from the database. You MUST use these numbers when referring to overall statistics. Do not attempt to recalculate them from the sample rows.
2. SUPPORTING RECORDS: A small semantic sample of related float profiles. Use these to explain patterns, variations, or provide context, but NEVER contradict the authoritative summary.

RULES:
1. Answer ONLY using the provided evidence. Never use outside knowledge.
2. If the user asks for a statistic (e.g. "average temperature"), quote the Authoritative Data Summary.
3. If the user asks "why" or asks for an explanation, look at the Supporting Records to provide oceanographic context based ONLY on the data shown.
4. Do NOT hallucinate. Do NOT infer geographic regions if not stated.

AUTHORITATIVE DATA SUMMARY:
{structured_summary}

SUPPORTING RECORDS:
{semantic_context}

Question: {question}

Answer:"""

def build_hybrid_prompt(question: str, structured_summary: str, semantic_rows: pd.DataFrame, variable: str = "temp_c") -> str:
    """
    Constructs a grounded prompt containing both authoritative aggregated facts
    and specific supporting rows for context.
    """
    if semantic_rows is None or semantic_rows.empty:
        context = "No supporting records retrieved."
    else:
        # Cap at 5 rows to prevent context bloat
        limited = semantic_rows.head(5)
        bullets = []
        for _, row in limited.iterrows():
            depth = f"{row.get('depth_m', float('nan')):.0f}m" if pd.notna(row.get('depth_m')) else "N/A"
            temp = f"{row.get('temp_c', float('nan')):.1f}°C" if pd.notna(row.get('temp_c')) else "N/A"
            sal = f"{row.get('salinity', float('nan')):.2f} PSU" if pd.notna(row.get('salinity')) else "N/A"
            oxy = f"{row.get('oxygen', float('nan')):.2f}" if pd.notna(row.get('oxygen')) else "N/A"

            if variable == "salinity":
                bullet = f"• Float {row.get('float_id', 'Unknown')} | {row.get('date', 'Unknown')} | Depth: {depth} | Salinity: {sal} | (Temp: {temp})"
            elif variable == "oxygen":
                bullet = f"• Float {row.get('float_id', 'Unknown')} | {row.get('date', 'Unknown')} | Depth: {depth} | Oxygen: {oxy} | (Temp: {temp})"
            elif variable == "depth_m":
                bullet = f"• Float {row.get('float_id', 'Unknown')} | {row.get('date', 'Unknown')} | Depth: {depth} | (Temp: {temp})"
            else:
                bullet = f"• Float {row.get('float_id', 'Unknown')} | {row.get('date', 'Unknown')} | Depth: {depth} | Temp: {temp} | (Salinity: {sal})"
                
            bullets.append(bullet)
        context = "\n".join(bullets) if bullets else "No supporting records retrieved."
        
    return HYBRID_PROMPT.format(
        structured_summary=structured_summary,
        semantic_context=context,
        question=question
    )
