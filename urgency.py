from typing import Tuple, Optional

# High-risk symptoms: should trigger "seek urgent care"
_HIGH = [
    "chest pain", "trouble breathing", "difficulty breathing", "shortness of breath",
    "stroke", "face drooping", "slurred speech", "seizure",
    "suicidal", "kill myself", "self harm",
    "severe allergic", "anaphylaxis", "swelling of face", "swelling of lips", "swelling of tongue"
]

# Moderate risk: might warrant same-day care depending on severity
_MED = [
    "high fever", "fever over", "severe headache", "worst headache",
    "vomiting", "dehydration", "blood in stool", "blood in urine",
    "pregnant", "pregnancy", "newborn", "infant"
]

def detect_urgency(query: str, evidence: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns (urgency_level, reason)
    urgency_level in {"LOW","MEDIUM","HIGH"}

    We scan the user's question and (optionally) retrieved evidence text.
    """
    text = (query or "")
    if evidence:
        text = text + " " + evidence
    text = text.lower()

    for t in _HIGH:
        if t in text:
            return "HIGH", f"Detected high-risk symptom: '{t}'"

    for t in _MED:
        if t in text:
            return "MEDIUM", f"Detected moderate-risk symptom: '{t}'"

    return "LOW", "Informational medical question"
