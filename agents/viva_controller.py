import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from evaluator_agent import evaluate_answer

# -------------------------------------------------------------
# 1. CENTRAL VIVA STATE SCHEMA
# -------------------------------------------------------------
class VivaState(TypedDict):
    question_bank: List[Dict[str, Any]]
    current_index: int
    total_questions: int
    total_viva_marks: float
    
    # Time Management
    start_time: float
    total_time_limit_sec: float
    elapsed_time_sec: float
    
    # Transcription Verification Sub-State
    sub_state: str  # "WAITING_FOR_ANSWER" or "PENDING_CONFIRMATION"
    pending_transcript: Optional[str]
    
    # User Command Actions: "SUBMIT_AUDIO", "NEXT", "REDO", "SKIP", "JUMP"
    last_command: Optional[str]
    jump_target_index: Optional[int]
    
    # Stored Answers Map: {question_index: {transcript, score, max_marks, feedback, status}}
    answers_map: Dict[int, Dict[str, Any]]
    
    # Overall Status: "IN_PROGRESS", "COMPLETED", "TIME_EXPIRED"
    session_status: str


# -------------------------------------------------------------
# 2. LANGGRAPH NODES
# -------------------------------------------------------------

def timer_node(state: VivaState) -> Dict[str, Any]:
    """
    Updates elapsed time and flags session if time limit has expired.
    """
    current_time = time.time()
    elapsed = current_time - state["start_time"]
    
    status = state["session_status"]
    if elapsed >= state["total_time_limit_sec"] and status == "IN_PROGRESS":
        status = "TIME_EXPIRED"
        
    return {
        "elapsed_time_sec": round(elapsed, 2),
        "session_status": status
    }


def command_handler_node(state: VivaState) -> Dict[str, Any]:
    """
    Controller & Moderator:
    Manages state transitions, commands (NEXT, REDO, SKIP, JUMP), 
    and handles answer overwriting in answers_map.
    """
    cmd = state.get("last_command")
    c_idx = state["current_index"]
    answers = dict(state.get("answers_map", {}))
    sub_state = state["sub_state"]
    pending = state.get("pending_transcript")
    q_bank = state["question_bank"]
    total_q = state["total_questions"]
    
    # Action A: Student Submits Voice Response -> Move to Verification State
    if cmd == "SUBMIT_AUDIO":
        sub_state = "PENDING_CONFIRMATION"

    # Action B: Student Confirms Transcript ("NEXT") -> Evaluate via Agent 2 & Overwrite State
    elif cmd == "NEXT" and sub_state == "PENDING_CONFIRMATION" and pending is not None:
        eval_res = evaluate_answer(
            student_answer=pending,
            question_data=q_bank[c_idx],
            total_questions=total_q
        )
        
        # Overwrite/Save evaluation in answers_map
        answers[c_idx] = {
            "status": "ANSWERED",
            "transcript": pending,
            "score": eval_res.get("score", 0.0),
            "max_marks": eval_res.get("max_marks", 0.0),
            "feedback": eval_res.get("feedback", ""),
            "matched_keywords": eval_res.get("matched_keywords", eval_res.get("matched_concepts", [])),
            "evaluation_details": eval_res
        }
        
        pending = None
        sub_state = "WAITING_FOR_ANSWER"
        
        if c_idx + 1 < total_q:
            c_idx += 1

    # Action C: Student Re-records Answer ("REDO") -> Discard Transcript
    elif cmd == "REDO":
        pending = None
        sub_state = "WAITING_FOR_ANSWER"

    # Action D: Student Skips Question ("SKIP")
    elif cmd == "SKIP":
        if c_idx not in answers:
            answers[c_idx] = {
                "status": "SKIPPED",
                "transcript": "",
                "score": 0.0,
                "max_marks": round(state["total_viva_marks"] / total_q, 2),
                "feedback": "Question skipped by student.",
                "matched_keywords": []
            }
        pending = None
        sub_state = "WAITING_FOR_ANSWER"
        if c_idx + 1 < total_q:
            c_idx += 1

    # Action E: Jump to Specific Question ("JUMP")
    elif cmd == "JUMP" and state.get("jump_target_index") is not None:
        target = state["jump_target_index"]
        if 0 <= target < total_q:
            c_idx = target
            pending = None
            sub_state = "WAITING_FOR_ANSWER"

    # Session Completion Check
    session_status = state["session_status"]
    if len(answers) >= total_q and all(v["status"] in ["ANSWERED", "SKIPPED"] for v in answers.values()):
        session_status = "COMPLETED"

    return {
        "current_index": c_idx,
        "sub_state": sub_state,
        "pending_transcript": pending,
        "answers_map": answers,
        "session_status": session_status,
        "last_command": None,
        "jump_target_index": None
    }


# -------------------------------------------------------------
# 3. ROUTING & GRAPH COMPOSITION
# -------------------------------------------------------------

def session_router(state: VivaState) -> str:
    """Routes execution: Holds state for user input or ends if session finishes."""
    if state["session_status"] in ["COMPLETED", "TIME_EXPIRED"]:
        return "end_viva"
    return "wait_for_input"


workflow = StateGraph(VivaState)

workflow.add_node("timer_check", timer_node)
workflow.add_node("handle_command", command_handler_node)

workflow.add_edge(START, "timer_check")
workflow.add_edge("timer_check", "handle_command")

workflow.add_conditional_edges(
    "handle_command",
    session_router,
    {
        "wait_for_input": END,
        "end_viva": END
    }
)

viva_controller_app = workflow.compile()


# -------------------------------------------------------------
# 4. STANDALONE VIVA SIMULATION TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Agent 3 (LangGraph Controller / Moderator) ---\n")

    # Sample Question Bank (3 Questions, 10 Minutes Total Limit)
    sample_bank = [
        {"id": 1, "question": "What is List vs Tuple?", "keywords": ["mutable", "immutable", "square brackets"]},
        {"id": 2, "question": "What is OOP?", "keywords": ["classes", "inheritance", "encapsulation"]},
        {"id": 3, "question": "What is Dictionary?", "keywords": ["key-value", "curly braces"]}
    ]

    # Initialize State
    state: VivaState = {
        "question_bank": sample_bank,
        "current_index": 0,
        "total_questions": len(sample_bank),
        "total_viva_marks": 15.0,
        "start_time": time.time(),
        "total_time_limit_sec": 600.0,
        "elapsed_time_sec": 0.0,
        "sub_state": "WAITING_FOR_ANSWER",
        "pending_transcript": None,
        "last_command": None,
        "jump_target_index": None,
        "answers_map": {},
        "session_status": "IN_PROGRESS"
    }

    # Simulation Step 1: Submit Q1 Answer
    state["pending_transcript"] = "List is mutable with square brackets and tuple is immutable."
    state["last_command"] = "SUBMIT_AUDIO"
    state = viva_controller_app.invoke(state)

    # Simulation Step 2: Confirm NEXT
    state["last_command"] = "NEXT"
    state = viva_controller_app.invoke(state)
    print(f"Q1 Evaluated Score: {state['answers_map'][0]['score']} / {state['answers_map'][0]['max_marks']}")

    # Simulation Step 3: Skip Q2
    state["last_command"] = "SKIP"
    state = viva_controller_app.invoke(state)
    print(f"Q2 Status: {state['answers_map'][1]['status']}")

    # Simulation Step 4: Answer Q3 & Confirm
    state["pending_transcript"] = "Dictionary uses key-value pairs inside curly braces."
    state["last_command"] = "SUBMIT_AUDIO"
    state = viva_controller_app.invoke(state)
    
    state["last_command"] = "NEXT"
    state = viva_controller_app.invoke(state)

    # Simulation Step 5: Jump Back to Q2 & Overwrite Skipped Question
    state["last_command"] = "JUMP"
    state["jump_target_index"] = 1
    state = viva_controller_app.invoke(state)

    state["pending_transcript"] = "OOP uses classes, inheritance, and encapsulation."
    state["last_command"] = "SUBMIT_AUDIO"
    state = viva_controller_app.invoke(state)

    state["last_command"] = "NEXT"
    state = viva_controller_app.invoke(state)

    print("\n" + "=" * 60)
    print("SESSION STATE SUMMARY (HANDOVER READY FOR AGENT 4 / SUPER AGENT)")
    print("=" * 60)
    print(f"Session Status : {state['session_status']}")
    print(f"Elapsed Time   : {state['elapsed_time_sec']} seconds")
    print(f"Answers Map    : {state['answers_map']}")