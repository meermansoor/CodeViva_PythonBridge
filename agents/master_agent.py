import os
import time
from typing import Dict, Any, List

# -------------------------------------------------------------
# EXACT IMPORTS MATCHING YOUR ACTUAL SUB-AGENTS
# -------------------------------------------------------------
# Agent 1: Speech Agent
from speech_agent import transcribe_and_verify

# Agent 2: Evaluator Agent
from evaluator_agent import evaluate_answer

# Agent 3: LangGraph Viva Controller (Importing viva_controller_app)
from viva_controller import viva_controller_app, VivaState

# Agent 4: Report Agent
from report_agent import process_teacher_dashboard_data, export_session_to_excel


# -------------------------------------------------------------
# MASTER AGENT ORCHESTRATOR CLASS
# -------------------------------------------------------------
class MasterVivaOrchestrator:
    """
    Master Agent Orchestrator:
    Coordinates speech verification, answer evaluation, LangGraph state machine,
    and final report/dashboard generation.
    """
    def __init__(self, question_bank: List[Dict[str, Any]], total_viva_marks: float = 15.0, total_time_limit_sec: float = 600.0):
        self.question_bank = question_bank
        self.total_questions = len(question_bank)
        self.total_viva_marks = total_viva_marks
        self.total_time_limit_sec = total_time_limit_sec
        
        # Central memory for all completed student records
        self.batch_session_records: List[Dict[str, Any]] = []

    def initialize_student_state(self) -> VivaState:
        """
        Creates an initial VivaState dictionary for a new student.
        """
        return {
            "question_bank": self.question_bank,
            "current_index": 0,
            "total_questions": self.total_questions,
            "total_viva_marks": self.total_viva_marks,
            "start_time": time.time(),
            "total_time_limit_sec": self.total_time_limit_sec,
            "elapsed_time_sec": 0.0,
            "sub_state": "WAITING_FOR_ANSWER",
            "pending_transcript": None,
            "last_command": None,
            "jump_target_index": None,
            "answers_map": {},
            "session_status": "IN_PROGRESS"
        }

    def process_student_action(
        self,
        current_state: VivaState,
        command: str,
        audio_or_text_input: str = None,
        jump_index: int = None
    ) -> VivaState:
        """
        Executes one turn of the student's viva interaction.
        Handles SUBMIT_AUDIO, NEXT, REDO, SKIP, and JUMP commands.
        """
        state_input = dict(current_state)
        state_input["last_command"] = command
        if jump_index is not None:
            state_input["jump_target_index"] = jump_index

        # Handle Audio/Text Input if submitting response
        if command == "SUBMIT_AUDIO" and audio_or_text_input:
            if os.path.exists(audio_or_text_input):
                speech_res = transcribe_and_verify(audio_or_text_input)
                transcript = speech_res.get("text", "")
            else:
                transcript = str(audio_or_text_input)
            
            state_input["pending_transcript"] = transcript

        # Invoke Agent 3 State Graph Engine
        updated_state = viva_controller_app.invoke(state_input)
        return updated_state

    def complete_student_viva(
        self,
        student_info: Dict[str, str],
        final_viva_state: VivaState,
        proctoring_logs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Saves student performance into session memory and returns live Teacher Dashboard payload.
        """
        if proctoring_logs is None:
            proctoring_logs = {"cheating_probability": 0.0, "proctoring_status": "CLEAN", "flags": []}

        student_record = {
            "student_info": student_info,
            "viva_state": final_viva_state,
            "proctoring_logs": proctoring_logs
        }
        self.batch_session_records.append(student_record)

        # Agent 4 Execution: Dashboard Real-time Metrics
        dashboard_payload = process_teacher_dashboard_data(self.batch_session_records)
        return dashboard_payload

    def generate_final_batch_excel_report(self, filename: str = "muet_viva_summary.xlsx") -> str:
        """
        Agent 4 Execution: Exports final Excel file for the whole class batch.
        """
        dashboard_payload = process_teacher_dashboard_data(self.batch_session_records)
        excel_path = export_session_to_excel(dashboard_payload, output_path=filename)
        return excel_path


# -------------------------------------------------------------
# INTEGRATION TEST RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Master Agent Orchestration Pipeline ---\n")

    # Sample Question Bank
    sample_bank = [
        {"id": 1, "question": "What is List vs Tuple?", "keywords": ["mutable", "immutable", "square brackets"]},
        {"id": 2, "question": "What is OOP?", "keywords": ["classes", "inheritance", "encapsulation"]},
        {"id": 3, "question": "What is Dictionary?", "keywords": ["key-value", "curly braces"]}
    ]

    orchestrator = MasterVivaOrchestrator(question_bank=sample_bank, total_viva_marks=15.0)

    # Student 1 Simulation
    student1_info = {"roll_no": "20CS002", "name": "Marie Curie"}
    proctoring1 = {"cheating_probability": 2.5, "proctoring_status": "CLEAN", "flags": []}

    state = orchestrator.initialize_student_state()

    # Step 1: Q1 Answer Submission
    print("[Master Agent] Student submits Q1 answer...")
    state = orchestrator.process_student_action(
        current_state=state,
        command="SUBMIT_AUDIO",
        audio_or_text_input="List is mutable with square brackets and tuple is immutable."
    )

    # Step 2: Confirm Q1 (NEXT) -> Calls Agent 2 via Agent 3
    print("[Master Agent] Confirming Q1 evaluation...")
    state = orchestrator.process_student_action(current_state=state, command="NEXT")

    # Step 3: Skip Q2
    print("[Master Agent] Student skips Q2...")
    state = orchestrator.process_student_action(current_state=state, command="SKIP")

    # Step 4: Q3 Submission & Confirmation
    print("[Master Agent] Student answers Q3...")
    state = orchestrator.process_student_action(
        current_state=state,
        command="SUBMIT_AUDIO",
        audio_or_text_input="Dictionary uses key-value pairs inside curly braces."
    )
    state = orchestrator.process_student_action(current_state=state, command="NEXT")

    # Step 5: Jump back to Q2 to re-answer
    print("[Master Agent] Student jumps back to Q2...")
    state = orchestrator.process_student_action(current_state=state, command="JUMP", jump_index=1)

    state = orchestrator.process_student_action(
        current_state=state,
        command="SUBMIT_AUDIO",
        audio_or_text_input="OOP uses classes, inheritance, and encapsulation."
    )
    state = orchestrator.process_student_action(current_state=state, command="NEXT")

    # Finalize Student 1
    ui_payload = orchestrator.complete_student_viva(student1_info, state, proctoring1)

    print("\n=== LIVE TEACHER DASHBOARD PAYLOAD GENERATED ===")
    print(f"Total Processed Students : {ui_payload['summary_metrics']['total_students']}")
    print(f"Overall Class Avg        : {ui_payload['summary_metrics']['overall_average_score']}%")

    # Generate Excel Report
    file_path = orchestrator.generate_final_batch_excel_report()
    print("\n" + "="*60)
    print(f"MASTER EXCEL EXPORT GENERATED SUCCESSFULLY: {file_path}")
    print("="*60) 