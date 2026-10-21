import os
import pandas as pd
from typing import Dict, Any, List, Tuple

# -------------------------------------------------------------
# 1. MEHRAN UNIVERSITY (MUET) GRADING LOGIC
# -------------------------------------------------------------
def get_muet_grade_and_point(percentage: float) -> Tuple[str, float]:
    """
    Official Mehran University (MUET) Grading Scheme:
    >= 90%      : A+ (4.0)
    81% - 89%   : A  (3.5)
    73% - 80%   : B+ (3.0)
    65% - 72%   : B  (2.5)
    60% - 64%   : C+ (2.0)
    55% - 59%   : C  (1.5)
    50% - 54%   : C- (1.0)
    < 50%       : F  (0.0)
    """
    pct = round(percentage)
    
    if pct >= 90:
        return "A+", 4.0
    elif 81 <= pct <= 89:
        return "A", 3.5
    elif 73 <= pct <= 80:
        return "B+", 3.0
    elif 65 <= pct <= 72:
        return "B", 2.5
    elif 60 <= pct <= 64:
        return "C+", 2.0
    elif 55 <= pct <= 59:
        return "C", 1.5
    elif 50 <= pct <= 54:
        return "C-", 1.0
    else:
        return "F", 0.0


# -------------------------------------------------------------
# 2. PROCESS TEACHER DASHBOARD DATA
# -------------------------------------------------------------
def process_teacher_dashboard_data(all_students_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Agent 4:
    - Merges scores into single "Marks Obtained" / "Percentage" field.
    - Applies official MUET grading rules.
    - Tracks integrity flags.
    """
    if not all_students_data:
        return {
            "summary_metrics": {
                "overall_average_score": 0.0,
                "pass_rate_percentage": 0.0,
                "flagged_integrity_cases": 0
            },
            "students_table": []
        }

    students_table = []
    total_percentages = []
    passed_students = 0
    flagged_cases = 0

    for item in all_students_data:
        student = item.get("student_info", {})
        viva_state = item.get("viva_state", {})
        proctoring = item.get("proctoring_logs", {})
        answers_map = viva_state.get("answers_map", {})

        # Calculate Total Marks & Score Percentage
        max_marks = viva_state.get("total_viva_marks", 15.0)
        obtained_marks = sum(q.get("score", 0.0) for q in answers_map.values())
        score_percentage = round((obtained_marks / max_marks) * 100, 2)
        
        # Calculate MUET Grade & Grade Point
        grade, grade_point = get_muet_grade_and_point(score_percentage)
        
        # Pass/Fail (Pass if Grade is not 'F')
        if grade != "F":
            passed_students += 1

        # Cheating / Integrity Flag Detection
        cheating_prob = proctoring.get("cheating_probability", 0.0)
        if cheating_prob > 20.0 or proctoring.get("proctoring_status") == "FLAGGED":
            flagged_cases += 1

        total_percentages.append(score_percentage)

        students_table.append({
            "roll_no": student.get("roll_no", "N/A"),
            "student_name": student.get("name", "N/A"),
            "marks_obtained": f"{round(obtained_marks, 2)} / {max_marks}",
            "score_percentage": score_percentage,
            "cheating_prob": cheating_prob,
            "final_grade": grade,
            "grade_point": grade_point,
            "detailed_answers": answers_map
        })

    total_count = len(all_students_data)
    overall_avg = round(sum(total_percentages) / total_count, 2) if total_count > 0 else 0.0
    pass_rate = round((passed_students / total_count) * 100, 2) if total_count > 0 else 0.0

    return {
        "summary_metrics": {
            "overall_average_score": overall_avg,
            "pass_rate_percentage": pass_rate,
            "passed_count": passed_students,
            "total_students": total_count,
            "flagged_integrity_cases": flagged_cases
        },
        "students_table": students_table
    }


def generate_individual_report(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a grade sheet for an individual student.
    Delegates to process_teacher_dashboard_data by treating the single session as a batch of 1.
    """
    dashboard_output = process_teacher_dashboard_data([session_data])
    if dashboard_output["students_table"]:
        return dashboard_output["students_table"][0]
    return {}



# -------------------------------------------------------------
# 3. EXPORT TO EXCEL (SINGLE WRITE AT SESSION END)
# -------------------------------------------------------------
def export_session_to_excel(dashboard_processed_data: Dict[str, Any], output_path: str = "muet_viva_summary.xlsx") -> str:
    """Exports batch data to Excel file using MUET grading structure."""
    table_records = dashboard_processed_data.get("students_table", [])
    excel_rows = []

    for s in table_records:
        row = {
            "Roll No": s["roll_no"],
            "Student Name": s["student_name"],
            "Marks Obtained": s["marks_obtained"],
            "Score (%)": f"{s['score_percentage']}%",
            "Cheating Prob. (%)": f"{s['cheating_prob']}%",
            "MUET Grade": s["final_grade"],
            "Grade Point": s["grade_point"]
        }
        excel_rows.append(row)

    df = pd.DataFrame(excel_rows)
    df.to_excel(output_path, index=False)
    return os.path.abspath(output_path)


# -------------------------------------------------------------
# STANDALONE TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Agent 4 (MUET Grading System & Single Score Column) ---\n")

    # Sample Viva Session Data
    simulated_session_data = [
        {
            "student_info": {"roll_no": "20CS002", "name": "Marie Curie"},
            "viva_state": {
                "total_viva_marks": 15.0,
                "answers_map": {0: {"score": 5.0}, 1: {"score": 4.5}, 2: {"score": 4.3}}  # Total 13.8 / 15 = 92% -> A+
            },
            "proctoring_logs": {"cheating_probability": 8.0, "proctoring_status": "CLEAN"}
        },
        {
            "student_info": {"roll_no": "20CS001", "name": "Albert Einstein"},
            "viva_state": {
                "total_viva_marks": 15.0,
                "answers_map": {0: {"score": 4.5}, 1: {"score": 4.0}, 2: {"score": 4.0}}  # Total 12.5 / 15 = 83.3% -> A
            },
            "proctoring_logs": {"cheating_probability": 5.0, "proctoring_status": "CLEAN"}
        },
        {
            "student_info": {"roll_no": "20CS012", "name": "John Doe"},
            "viva_state": {
                "total_viva_marks": 15.0,
                "answers_map": {0: {"score": 3.0}, 1: {"score": 3.0}, 2: {"score": 3.0}}  # Total 9.0 / 15 = 60% -> C+
            },
            "proctoring_logs": {"cheating_probability": 25.0, "proctoring_status": "FLAGGED"}
        }
    ]

    # Process Data
    dashboard_output = process_teacher_dashboard_data(simulated_session_data)

    print("=== SUMMARY METRICS ===")
    print(f"Overall Average Score : {dashboard_output['summary_metrics']['overall_average_score']}%")
    print(f"Pass Rate             : {dashboard_output['summary_metrics']['pass_rate_percentage']}%")
    print(f"Flagged Cases         : {dashboard_output['summary_metrics']['flagged_integrity_cases']}")

    print("\n=== MUET SCORE SHEET TABLE ===")
    for row in dashboard_output["students_table"]:
        print(f"[{row['roll_no']}] {row['student_name']} | Marks: {row['marks_obtained']} ({row['score_percentage']}%) | MUET Grade: {row['final_grade']} ({row['grade_point']}) | Cheating: {row['cheating_prob']}%")

    # Generate Single Excel Sheet
    excel_path = export_session_to_excel(dashboard_output)
    print("\n" + "="*60)
    print(f"MUET EXCEL REPORT GENERATED AT: {excel_path}")
    print("="*60)