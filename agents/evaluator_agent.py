from __future__ import annotations

import re
from typing import Any

from sentence_transformers import SentenceTransformer, util


# ==========================================================
# CONFIGURATION
# ==========================================================

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

RULE_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.60

SEMANTIC_COVERAGE_THRESHOLD = 0.45
REVIEW_COVERAGE_THRESHOLD = 0.40
RULE_SEMANTIC_GAP_THRESHOLD = 0.35

CONTRADICTION_PENALTY = 0.15


# ==========================================================
# SEMANTIC MODEL
# ==========================================================

semantic_model = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


# ==========================================================
# TEXT UTILITIES
# ==========================================================

def clean_text(text: str) -> str:
    """
    Normalize text for rule-based matching.
    """

    if not text:
        return ""

    text = str(text).casefold()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> set[str]:
    """
    Return normalized word tokens.
    """

    return set(
        clean_text(text).split()
    )


# ==========================================================
# EXACT PHRASE MATCHING
# ==========================================================

def exact_phrase_match(
    answer: str,
    phrase: str,
) -> bool:
    """
    Check whether a phrase exists as a complete
    word/phrase rather than using unsafe substring matching.
    """

    answer_clean = clean_text(answer)
    phrase_clean = clean_text(phrase)

    if not answer_clean or not phrase_clean:
        return False

    pattern = (
        r"\b"
        + re.escape(phrase_clean)
        + r"\b"
    )

    return bool(
        re.search(
            pattern,
            answer_clean,
        )
    )


# ==========================================================
# TOKEN OVERLAP
# ==========================================================

def token_overlap(
    answer: str,
    reference: str,
) -> float:
    """
    Calculate token overlap between answer and reference.
    """

    answer_tokens = tokenize(answer)
    reference_tokens = tokenize(reference)

    if not answer_tokens or not reference_tokens:
        return 0.0

    intersection = (
        answer_tokens & reference_tokens
    )

    return len(intersection) / len(
        reference_tokens
    )


# ==========================================================
# RULE-BASED CONCEPT SCORE
# ==========================================================

def calculate_rule_score(
    student_answer: str,
    concept: dict[str, Any],
) -> float:
    """
    Calculate rule-based evidence for one concept.

    Evidence sources:
    1. Concept name
    2. Alternative phrases
    3. Token overlap
    """

    answer = clean_text(
        student_answer
    )

    if not answer:
        return 0.0

    concept_name = str(
        concept.get(
            "name",
            "",
        )
    ).strip()

    alternatives = concept.get(
        "alternatives",
        [],
    )

    if not isinstance(
        alternatives,
        list,
    ):
        alternatives = []

    # ------------------------------------------
    # Exact concept match
    # ------------------------------------------

    if concept_name and exact_phrase_match(
        answer,
        concept_name,
    ):
        return 1.0

    # ------------------------------------------
    # Exact alternative match
    # ------------------------------------------

    for alternative in alternatives:

        if exact_phrase_match(
            answer,
            str(alternative),
        ):
            return 1.0

    # ------------------------------------------
    # Token overlap
    # ------------------------------------------

    references = []

    if concept_name:
        references.append(
            concept_name
        )

    references.extend(
        str(item)
        for item in alternatives
        if str(item).strip()
    )

    if not references:
        return 0.0

    best_overlap = max(
        (
            token_overlap(
                answer,
                reference,
            )
            for reference in references
        ),
        default=0.0,
    )

    return min(
        best_overlap,
        1.0,
    )


# ==========================================================
# SEMANTIC SCORES
# ==========================================================

def calculate_semantic_scores(
    student_answer: str,
    references: list[str],
) -> list[float]:
    """
    Calculate semantic similarity between the student's
    answer and multiple reference explanations.

    The student answer is encoded once.
    """

    if not student_answer or not references:
        return [0.0] * len(references)

    answer_embedding = semantic_model.encode(
        student_answer,
        convert_to_tensor=True,
    )

    reference_embeddings = semantic_model.encode(
        references,
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        answer_embedding,
        reference_embeddings,
    )[0]

    return [
        max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )
        for score in similarities
    ]


# ==========================================================
# CONCEPT EVALUATION
# ==========================================================

def evaluate_concept(
    student_answer: str,
    concept: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate one concept using:
    - rule-based matching
    - semantic similarity
    - hybrid scoring
    """

    concept_name = str(
        concept.get(
            "name",
            "",
        )
    ).strip()

    try:
        weight = float(
            concept.get(
                "weight",
                1.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        weight = 1.0

    weight = max(
        weight,
        0.1,
    )

    alternatives = concept.get(
        "alternatives",
        [],
    )

    if not isinstance(
        alternatives,
        list,
    ):
        alternatives = []

    expected_explanations = concept.get(
        "expected_explanations",
        [],
    )

    if not isinstance(
        expected_explanations,
        list,
    ):
        expected_explanations = []

    # ------------------------------------------
    # Rule score
    # ------------------------------------------

    rule_score = calculate_rule_score(
        student_answer,
        concept,
    )

    # ------------------------------------------
    # Semantic references
    # ------------------------------------------

    semantic_references = []

    # Use expected explanations first because
    # they contain meaningful semantic context.

    semantic_references.extend(
        str(item).strip()
        for item in expected_explanations
        if str(item).strip()
    )

    # Alternatives are also useful semantic
    # references.

    semantic_references.extend(
        str(item).strip()
        for item in alternatives
        if str(item).strip()
    )

    # Concept name is used as a fallback.

    if concept_name:
        semantic_references.append(
            concept_name
        )

    # Remove duplicate references

    unique_references = []

    seen = set()

    for reference in semantic_references:

        key = reference.casefold()

        if key not in seen:
            seen.add(key)
            unique_references.append(
                reference
            )

    semantic_references = unique_references

    # ------------------------------------------
    # Semantic score
    # ------------------------------------------

    semantic_scores = calculate_semantic_scores(
        student_answer,
        semantic_references,
    )

    semantic_score = max(
        semantic_scores,
        default=0.0,
    )

    best_reference = None

    if semantic_scores:
        best_index = semantic_scores.index(
            semantic_score
        )

        best_reference = (
            semantic_references[
                best_index
            ]
        )

    # ------------------------------------------
    # Hybrid score
    # ------------------------------------------

    hybrid_score = (
        RULE_WEIGHT * rule_score
        +
        SEMANTIC_WEIGHT * semantic_score
    )

    # ------------------------------------------
    # Classification
    # ------------------------------------------

    if hybrid_score >= 0.70:

        classification = "covered"

    elif hybrid_score >= 0.45:

        classification = "partially_covered"

    else:

        classification = "missed"

    return {
        "name": concept_name,

        "weight": weight,

        "rule_based_score": round(
            rule_score,
            3,
        ),

        "semantic_score": round(
            semantic_score,
            3,
        ),

        "hybrid_score": round(
            hybrid_score,
            3,
        ),

        "classification": classification,

        "best_semantic_reference": (
            best_reference
        ),
    }


# ==========================================================
# CONTRADICTION DETECTION
# ==========================================================

def detect_contradictions(
    student_answer: str,
    concepts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Detect simple explicit contradictions.

    This is intentionally conservative.

    It should not be treated as a complete natural-language
    contradiction detector.
    """

    answer = clean_text(
        student_answer
    )

    contradictions = []

    # ------------------------------------------
    # Common contradiction patterns
    # ------------------------------------------

    contradiction_pairs = [
        (
            r"\bmutable\b",
            r"\bimmutable\b",
        ),
        (
            r"\bcan be changed\b",
            r"\bcannot be changed\b",
        ),
        (
            r"\bcan be modified\b",
            r"\bcannot be modified\b",
        ),
        (
            r"\bmodifiable\b",
            r"\bunmodifiable\b",
        ),
        (
            r"\btrue\b",
            r"\bfalse\b",
        ),
    ]

    for positive_pattern, negative_pattern in (
        contradiction_pairs
    ):

        positive = bool(
            re.search(
                positive_pattern,
                answer,
            )
        )

        negative = bool(
            re.search(
                negative_pattern,
                answer,
            )
        )

        if positive and negative:

            contradictions.append(
                {
                    "type": "conflicting_statements",
                    "description": (
                        "The answer contains "
                        "potentially conflicting "
                        "statements."
                    ),
                }
            )

    return contradictions


# ==========================================================
# FEEDBACK GENERATION
# ==========================================================

def generate_feedback(
    matched_concepts: list[str],
    partially_matched_concepts: list[str],
    missed_concepts: list[str],
    contradictions: list[dict[str, str]],
) -> str:
    """
    Generate concise examiner-style feedback.
    """

    feedback_parts = []

    if matched_concepts:

        feedback_parts.append(
            "Correctly covered: "
            + ", ".join(
                matched_concepts
            )
            + "."
        )

    if partially_matched_concepts:

        feedback_parts.append(
            "Partially covered: "
            + ", ".join(
                partially_matched_concepts
            )
            + "."
        )

    if missed_concepts:

        feedback_parts.append(
            "Missing: "
            + ", ".join(
                missed_concepts
            )
            + "."
        )

    if contradictions:

        feedback_parts.append(
            "The answer contains a "
            "potential contradiction and "
            "should be reviewed."
        )

    if not feedback_parts:

        return (
            "The answer did not provide "
            "sufficient evidence for the "
            "required concepts."
        )

    return " ".join(
        feedback_parts
    )


# ==========================================================
# MAIN ANSWER EVALUATOR
# ==========================================================

def evaluate_answer(
    student_answer: str,
    question_data: dict[str, Any],
    total_questions: int = 1,
) -> dict[str, Any]:
    """
    Evaluate one student's answer.

    Evaluation pipeline:

    Faster-Whisper transcript
             ↓
       Rule matching
             ↓
       Embeddings
             ↓
    Concept-level scoring
             ↓
    Contradiction check
             ↓
     Teacher review decision
    """

    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    if not isinstance(
        question_data,
        dict,
    ):
        raise ValueError(
            "question_data must be a dictionary."
        )

    student_answer = str(
        student_answer or ""
    ).strip()

    try:
        total_questions = int(
            total_questions
        )
    except (
        TypeError,
        ValueError,
    ):
        total_questions = 1

    total_questions = max(
        total_questions,
        1,
    )

    # ======================================================
    # QUESTION MARKS
    # ======================================================

    try:

        max_marks = float(
            question_data.get(
                "marks",
                1.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        max_marks = 1.0

    max_marks = max(
        max_marks,
        0.0,
    )

    # ======================================================
    # CONCEPTS
    # ======================================================

    concepts = question_data.get(
        "concepts",
        [],
    )

    if not isinstance(
        concepts,
        list,
    ):
        concepts = []

    # ======================================================
    # BACKWARD COMPATIBILITY
    # ======================================================

    # If an old question only contains expected_keywords,
    # convert them into simple concepts.

    if not concepts:

        old_keywords = question_data.get(
            "expected_keywords",
            [],
        )

        if isinstance(
            old_keywords,
            list,
        ):

            concepts = [
                {
                    "name": str(keyword),
                    "weight": 1.0,
                    "alternatives": [],
                    "expected_explanations": [
                        str(keyword)
                    ],
                }
                for keyword in old_keywords
                if str(keyword).strip()
            ]

    # ======================================================
    # EMPTY ANSWER
    # ======================================================

    if not student_answer:

        missed = [
            str(
                concept.get(
                    "name",
                    "Unknown concept",
                )
            )
            for concept in concepts
        ]

        return {
            "status": "evaluated",

            "score": 0.0,

            "max_marks": max_marks,

            "rule_based_score": 0.0,

            "semantic_score": 0.0,

            "concept_coverage_percentage": 0.0,

            "matched_concepts": [],

            "partially_matched_concepts": [],

            "missed_concepts": missed,

            "concept_results": [],

            "contradictions": [],

            "teacher_review_required": True,

            "review_reason": (
                "No answer transcript was available."
            ),

            "feedback": (
                "No answer was detected. "
                "Teacher review is required."
            ),

            "evaluation_method": (
                "hybrid_rule_based_semantic"
            ),

            "semantic_model_available": True,

            "semantic_model": SEMANTIC_MODEL_NAME,
        }

    # ======================================================
    # NO CONCEPTS
    # ======================================================

    if not concepts:

        return {
            "status": "evaluation_failed",

            "score": 0.0,

            "max_marks": max_marks,

            "rule_based_score": 0.0,

            "semantic_score": 0.0,

            "concept_coverage_percentage": 0.0,

            "matched_concepts": [],

            "partially_matched_concepts": [],

            "missed_concepts": [],

            "concept_results": [],

            "contradictions": [],

            "teacher_review_required": True,

            "review_reason": (
                "No evaluation concepts were "
                "provided for this question."
            ),

            "feedback": (
                "The question has no approved "
                "evaluation criteria."
            ),

            "evaluation_method": (
                "hybrid_rule_based_semantic"
            ),

            "semantic_model_available": True,

            "semantic_model": SEMANTIC_MODEL_NAME,
        }

    # ======================================================
    # EVALUATE CONCEPTS
    # ======================================================

    concept_results = []

    for concept in concepts:

        if not isinstance(
            concept,
            dict,
        ):
            continue

        result = evaluate_concept(
            student_answer,
            concept,
        )

        concept_results.append(
            result
        )

    # ======================================================
    # WEIGHTED SCORES
    # ======================================================

    total_weight = sum(
        result["weight"]
        for result in concept_results
    )

    if total_weight <= 0:

        total_weight = 1.0

    weighted_rule_score = sum(
        result["rule_based_score"]
        * result["weight"]
        for result in concept_results
    ) / total_weight

    weighted_semantic_score = sum(
        result["semantic_score"]
        * result["weight"]
        for result in concept_results
    ) / total_weight

    concept_coverage = sum(
        result["hybrid_score"]
        * result["weight"]
        for result in concept_results
    ) / total_weight

    # ======================================================
    # CONTRADICTIONS
    # ======================================================

    contradictions = detect_contradictions(
        student_answer,
        concepts,
    )

    # ======================================================
    # CONTRADICTION PENALTY
    # ======================================================

    final_coverage = concept_coverage

    if contradictions:

        final_coverage *= (
            1.0 - CONTRADICTION_PENALTY
        )

    final_coverage = max(
        0.0,
        min(
            1.0,
            final_coverage,
        ),
    )

    # ======================================================
    # CONCEPT CLASSIFICATION
    # ======================================================

    matched_concepts = [
        result["name"]
        for result in concept_results
        if result["classification"]
        == "covered"
    ]

    partially_matched_concepts = [
        result["name"]
        for result in concept_results
        if result["classification"]
        == "partially_covered"
    ]

    missed_concepts = [
        result["name"]
        for result in concept_results
        if result["classification"]
        == "missed"
    ]

    # ======================================================
    # SCORE
    # ======================================================

    automated_score = (
        final_coverage
        * max_marks
    )

    automated_score = round(
        automated_score,
        2,
    )

    # ======================================================
    # REVIEW CONDITIONS
    # ======================================================

    review_reasons = []

    if weighted_semantic_score < (
        SEMANTIC_COVERAGE_THRESHOLD
    ):

        review_reasons.append(
            "Low semantic evidence"
        )

    if abs(
        weighted_rule_score
        - weighted_semantic_score
    ) >= RULE_SEMANTIC_GAP_THRESHOLD:

        review_reasons.append(
            "Large difference between "
            "rule-based and semantic scores"
        )

    if final_coverage < (
        REVIEW_COVERAGE_THRESHOLD
    ):

        review_reasons.append(
            "Low concept coverage"
        )

    if contradictions:

        review_reasons.append(
            "Potential contradiction detected"
        )

    teacher_review_required = bool(
        review_reasons
    )

    # ======================================================
    # FEEDBACK
    # ======================================================

    feedback = generate_feedback(
        matched_concepts,
        partially_matched_concepts,
        missed_concepts,
        contradictions,
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {
        "status": "evaluated",

        "score": automated_score,

        "max_marks": max_marks,

        "rule_based_score": round(
            weighted_rule_score,
            3,
        ),

        "semantic_score": round(
            weighted_semantic_score,
            3,
        ),

        "concept_coverage_percentage": round(
            final_coverage * 100,
            2,
        ),

        "matched_concepts": matched_concepts,

        "partially_matched_concepts": (
            partially_matched_concepts
        ),

        "missed_concepts": missed_concepts,

        "concept_results": concept_results,

        "contradictions": contradictions,

        "teacher_review_required": (
            teacher_review_required
        ),

        "review_reason": (
            "; ".join(review_reasons)
            if review_reasons
            else None
        ),

        "feedback": feedback,

        "evaluation_method": (
            "hybrid_rule_based_semantic"
        ),

        "semantic_model_available": True,

        "semantic_model": SEMANTIC_MODEL_NAME,
    }