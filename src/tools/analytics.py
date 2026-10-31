import re
from statistics import mean


def duration_to_minutes(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    # HH:MM:SS
    match = re.fullmatch(r"(\d+):(\d+):(\d+)", text)

    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))

        return hours * 60 + minutes + seconds / 60

    # "1 h 32 min"
    hours = 0
    minutes = 0

    hour_match = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours)", text)
    minute_match = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes)", text)

    if hour_match:
        hours = int(hour_match.group(1))

    if minute_match:
        minutes = int(minute_match.group(1))

    if not hour_match and not minute_match:
        return None

    return hours * 60 + minutes


def minutes_to_duration(minutes):
    if minutes is None:
        return None

    minutes = round(minutes)

    hours = minutes // 60
    mins = minutes % 60

    return f"{hours} h {mins} min"


def extract_duration_from_text(text):
    if not text:
        return None

    patterns = [
        r"recorded surgery duration:\s*(\d+:\d+:\d+)",
        r"surgery duration:\s*(\d+:\d+:\d+)",
        r"recorded duration:\s*(\d+:\d+:\d+)",

        r"recorded surgery duration:\s*(\d+\s*h\s*\d+\s*min)",
        r"surgery duration:\s*(\d+\s*h\s*\d+\s*min)",
    ]

    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE)

        if match:
            return match.group(1)

    return None


def get_case_durations(retrieved_chunks):
    """
    Get duration for each unique case.
    """

    cases = {}

    for chunk in retrieved_chunks:

        case_id = chunk.get("case_id")

        if not case_id:
            continue

        # Prefer structured metadata if available
        duration = (
            chunk.get("surgery_duration")
            or chunk.get("duration")
        )

        # Otherwise extract from text
        if not duration:
            duration = extract_duration_from_text(
                chunk.get("text", "")
            )

        if not duration:
            continue

        minutes = duration_to_minutes(duration)

        if minutes is None:
            continue

        cases[case_id] = {
            "case_id": case_id,
            "source_file": chunk.get("source_file"),
            "duration": minutes_to_duration(minutes),
            "duration_minutes": minutes,
        }

    return list(cases.values())

def select_cases(cases, limit=None, sort_by="none", sort_order="none"):

    if not cases:
        return []

    if sort_by == "surgery_started":
        cases = sorted(
            cases,
            key=lambda x: x.get("surgery_started", ""),
            reverse=(sort_order == "desc"),
        )

    if limit:
        cases = cases[:limit]

    return cases
def calculate_duration_statistics(retrieved_chunks):

    # get_case_durations() already deduplicates by case_id
    cases = get_case_durations(retrieved_chunks)

    if not cases:
        return {
            "count": 0,
            "total_duration": None,
            "total_minutes": None,
            "average_duration": None,
            "minimum_duration": None,
            "maximum_duration": None,
            "cases": [],
        }

    durations = [
        case["duration_minutes"]
        for case in cases
    ]

    total_minutes = sum(durations)

    return {
        "count": len(cases),

        "total_minutes": round(total_minutes, 2),
        "total_duration": minutes_to_duration(total_minutes),

        "average_duration": minutes_to_duration(mean(durations)),
        "minimum_duration": minutes_to_duration(min(durations)),
        "maximum_duration": minutes_to_duration(max(durations)),

        "cases": cases,
    }
