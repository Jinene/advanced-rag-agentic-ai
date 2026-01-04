from langgraph.graph import StateGraph
from src.search import RAGSearch
from openai import OpenAI

llm = OpenAI()
rag = RAGSearch()

def decide(state):
    q = state["query"]
    decision = llm.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"Does this need retrieval? {q}"}]
    )
    return "retrieve" if "yes" in decision.choices[0].message.content.lower() else "answer"

def retrieve(state):
    state["context"] = rag.search_and_summarize(state["query"])
    return state

def answer(state):
    return {"answer": state.get("context", "Answered without retrieval.")}

graph = StateGraph(dict)
graph.add_node("decide", decide)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)

graph.set_entry_point("decide")
graph.add_conditional_edges("decide", decide, {
    "retrieve": "retrieve",
    "answer": "answer"
})
graph.add_edge("retrieve", "answer")

agent = graph.compile()
