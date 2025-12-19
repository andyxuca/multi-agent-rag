from src.agents import planner, answerer
from src.retrieval import retrieve
from dotenv import load_dotenv

def main():
    load_dotenv()
    question = input("Question: ")

    plan = planner(question)
    evidence = []

    for q in plan["queries"]:
        evidence.extend(retrieve(q, k=plan["k"]))

    answer = answerer(question, evidence)
    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
