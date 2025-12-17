def backoff_seconds(attempt: int) -> float:
    # Exponential backoff capped at 30s
    return min(2 ** attempt, 30)
