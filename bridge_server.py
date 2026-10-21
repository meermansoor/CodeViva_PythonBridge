"""
Python FastAPI Bridge Server
Wraps existing Python agents as HTTP endpoints for the Node.js backend.
Run: uvicorn bridge_server:app --host 0.0.0.0 --port 8100 --reload
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING

# Add agents directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask 

# Import agents
if TYPE_CHECKING:
    # IDE Static Analysis / IntelliSense: When IDE reads from project root
    from agents.speech_agent import transcribe_and_verify
    from agents.evaluator_agent import evaluate_answer
    from agents.viva_controller import viva_controller_app
    from agents.report_agent import process_teacher_dashboard_data, export_session_to_excel
else:
    # Runtime: When 'agents' is in sys.path
    from speech_agent import transcribe_and_verify
    from evaluator_agent import evaluate_answer
    from viva_controller import viva_controller_app
    from report_agent import process_teacher_dashboard_data, export_session_to_excel

# ----------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------
app = FastAPI(
    title="Viva Agents Bridge",
    description="HTTP bridge exposing Python ML agents to the Node.js backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------
class EvaluateRequest(BaseModel):
    transcript: str
    question_data: Dict[str, Any]
    total_viva_marks: float = 15.0
    total_questions: int = 5


class VivaActionRequest(BaseModel):
    state: Dict[str, Any]


class DashboardRequest(BaseModel):
    batch_records: List[Dict[str, Any]]


class ExcelRequest(BaseModel):
    batch_records: List[Dict[str, Any]]
    filename: str = "muet_viva_summary.xlsx"


# ----------------------------------------------------------------
# Health Check
# ----------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "viva-agents-bridge", "timestamp": time.time()}


# ----------------------------------------------------------------
# Agent 1: Speech Transcription & Authenticity
# ----------------------------------------------------------------
# NOTE: Removed 'async' from 'def transcribe' so FastAPI runs this CPU-bound function in a threadpool!
@app.post("/api/transcribe")
def transcribe(file: UploadFile = File(...)):
    """
    Accepts an audio file, runs Whisper transcription + authenticity check.
    Returns: { text, is_authentic, auth_notes }
    """
    # Save uploaded file to temp location
    ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    temp_path = os.path.join(UPLOAD_DIR, f"audio_{int(time.time() * 1000)}{ext}")

    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        # Call Agent 1
        result = transcribe_and_verify(temp_path)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file (Swallow WinError 32 if file is still locked by ffmpeg/librosa)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary audio file {temp_path}: {e}")


# ----------------------------------------------------------------
# Agent 2: Answer Evaluation
# ----------------------------------------------------------------
# NOTE: Regular 'def' prevents blocking the event loop during heavy ML tasks
@app.post("/api/evaluate")
def evaluate(payload: EvaluateRequest):
    """
    Evaluates a student transcript against question keywords.
    Returns: { score, max_marks, feedback, matched_keywords, missed_keywords, accuracy_percentage }
    """
    result = evaluate_answer(
        student_answer=payload.transcript,
        question_data=payload.question_data,
        total_questions=payload.total_questions
    )
    return result


# ----------------------------------------------------------------
# Agent 3: Viva Controller (LangGraph State Machine)
# ----------------------------------------------------------------
@app.post("/api/viva/action")
def viva_action(payload: VivaActionRequest):
    """
    Processes a single viva state transition through the LangGraph engine.
    Accepts the full VivaState dict, returns the updated state.
    """
    state_dict = payload.state
    
    # CRITICAL FIX: Convert JSON string keys in 'answers_map' back to integers 
    # to avoid '0 in {"0": ...}' failing in viva_controller.py
    if "answers_map" in state_dict and isinstance(state_dict["answers_map"], dict):
        corrected_answers_map = {}
        for k, v in state_dict["answers_map"].items():
            # JSON keys are strings; convert to int for LangGraph matching
            try:
                corrected_answers_map[int(k)] = v
            except ValueError:
                corrected_answers_map[k] = v
        state_dict["answers_map"] = corrected_answers_map

    try:
        updated_state = viva_controller_app.invoke(state_dict)
        return updated_state
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"State transition error: {str(e)}")


class IndividualReportRequest(BaseModel):
    session_data: Dict[str, Any]

# ----------------------------------------------------------------
# Agent 4: Dashboard Data Processing
# ----------------------------------------------------------------
@app.post("/api/report/dashboard")
def dashboard(payload: DashboardRequest):
    """
    Aggregates batch student data into dashboard metrics + graded student table.
    """
    result = process_teacher_dashboard_data(payload.batch_records)
    return result


@app.post("/api/report/individual")
def individual_report(payload: IndividualReportRequest):
    """
    Generates an individual student grade sheet using Agent 4 logic.
    """
    if TYPE_CHECKING:
        from agents.report_agent import generate_individual_report
    else:
        from report_agent import generate_individual_report
        
    result = generate_individual_report(payload.session_data)
    return result


def cleanup_excel_file(path: str):
    """Background task to remove the generated excel file after downloading."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            print(f"Warning: Could not remove Excel file {path}: {e}")


# ----------------------------------------------------------------
# Agent 4: Excel Export
# ----------------------------------------------------------------
@app.post("/api/report/excel")
def excel_export(payload: ExcelRequest):
    """
    Generates an Excel report file and returns it as a download.
    """
    # Create a unique filename to prevent concurrency overwrites
    unique_filename = f"{int(time.time()*1000)}_{payload.filename}"
    output_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        dashboard_data = process_teacher_dashboard_data(payload.batch_records)
        export_session_to_excel(dashboard_data, output_path=output_path)

        # Use BackgroundTask to delete file after it's served to client
        return FileResponse(
            path=output_path,
            filename=payload.filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=BackgroundTask(cleanup_excel_file, output_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel export error: {str(e)}")


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Viva Agents Bridge on http://localhost:8100")
    # FIX: Provide app as import string to enable reload safely without crashing
    uvicorn.run("bridge_server:app", host="0.0.0.0", port=8100, reload=True)
