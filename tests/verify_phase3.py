import os
import sys

from rag.retriever import load_index, retrieve
from llm.query_engine import answer_query

def main():
    load_index()
    rows = retrieve("what is the average temperature at 200m depth?")
    answer, sql = answer_query("what is the average temperature at 200m depth?", rows)
    print("ANSWER:")
    print(answer)
    print("SQL:")
    print(sql)

if __name__ == "__main__":
    main()
