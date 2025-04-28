import os
from vectors import embeddings
from vectors import FAISS   
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()


class AgentState(TypedDict):
    query: str  # The current user query
    expanded_query: str  # The expanded version of the user query
    context: List[Dict[str, Any]]  # Retrieved documents (content and metadata)
    response: str  # The generated response to the user query
    precision_score: float  # The precision score of the response
    groundedness_score: float  # The groundedness score of the response
    groundedness_loop_count: int  # Counter for groundedness refinement loops
    precision_loop_count: int  # Counter for precision refinement loops
    feedback: str  # Feedback from the user
    query_feedback: str  # Feedback specifically related to the query
    groundedness_check: bool  # Indicator for groundedness check
    loop_max_iter: int  # Maximum iterations for loops

llm = ChatGroq(temperature=0,model_name="qwen-qwq-32b",api_key=os.getenv("GROQ_API_KEY"))

# Node functions with state management
def expand_query(state: AgentState):
    """Expand the initial query"""
    return {"expanded_query": f"{state['query']} (expanded using CGSM-D method)"}

def retrieve_context(state: AgentState):
    """Retrieve relevant context from vector store"""
    retriever = FAISS.load_local(
        "nutrition_vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    ).as_retriever(k=3)
    docs = retriever.get_relevant_documents(state["expanded_query"])
    return {"context": "\n".join([d.page_content for d in docs])}

def craft_response(state: AgentState):
    """Generate initial response using LLM"""
    response = llm.invoke(
        f"Context:\n{state['context']}\n\nQuestion: {state['expanded_query']}"
    ).content
    return {"response": response}

def score_groundedness(state: AgentState):
    """Score response groundedness (simplified example)"""
    return {"groundedness_score": 0.85}  # Replace with actual scoring logic

def check_precision(state: AgentState) -> dict:
    """Check precision thresholds"""
    decision = "pass" if (
        state["groundedness_score"] >= 0.8
        and state["loop_count"] < state["max_iter"]
    ) else "refine_query"
    return {"decision": decision}  # Return dict with "decision" key

def refine_query(state: AgentState):
    """Refine query for next iteration"""
    return {
        "expanded_query": f"{state['expanded_query']} [refined]",
        "loop_count": state["loop_count"] + 1
    }

#Workflow graph
workflow = StateGraph(AgentState)

# Added nodes
workflow.add_node("expand_query", expand_query)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("craft_response", craft_response)
workflow.add_node("score_groundedness", score_groundedness)
workflow.add_node("check_precision", check_precision)
workflow.add_node("refine_query", refine_query)

# Set initial workflow
workflow.set_entry_point("expand_query")

# Main flow
workflow.add_edge("expand_query", "retrieve_context")
workflow.add_edge("retrieve_context", "craft_response")
workflow.add_edge("craft_response", "score_groundedness")
workflow.add_edge("score_groundedness", "check_precision")


# Conditional edges
workflow.add_conditional_edges(
    "check_precision",
    lambda x: x["decision"],  # Directly access the "decision" key
    {"pass": END, "refine_query": "refine_query"}
)


# Refinement loop
workflow.add_edge("refine_query", "expand_query")

#Compile and validate
graph = workflow.compile()

#Execution function
def run_agent(query: str, max_iter: int = 3):
    initial_state = {
        "query": query,
        "expanded_query": "",
        "context": "",
        "response": "",
        "groundedness_score": 0.0,
        "loop_count": 0,
        "max_iter": max_iter
    }

    final_state = None
    for step in graph.stream(initial_state):
        if "__end__" in step:
            final_state = step["__end__"]
            break  # Exit loop once terminated ✅
        else:
            print(f"Step: {list(step.keys())[0]}")
            print(f"Current state: {step}\n")

    if not final_state:
        # Gracefully handle incomplete termination
        final_state = {
            "response": "Workflow terminated unexpectedly",
            **initial_state
        }

    return final_state



# Initializing query
result = run_agent("What are the vitamin D rich foods?")

# Getting final response
print("\nFinal Response:", result["response"])


