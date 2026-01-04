import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"]
)

MODEL = os.environ["DEEPSEEK_MODEL"]

def planner(question):
    system = """Return JSON only:
    { "queries": [string]}
    where "queries" are search queries derived from the question.
    """
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":question}
        ],
        temperature=0
    )
    return json.loads(r.choices[0].message.content)

def answerer(question, evidence):
    system = """Answer ONLY using the evidence.
    Cite as [doc_id:chunk_id].
    """
    user = json.dumps({
        "question": question,
        "evidence": evidence
    })

    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.2
    )
    return r.choices[0].message.content
