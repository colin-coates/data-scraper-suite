def max_concurrency(execution_mode: str) -> int:
    if execution_mode == "DRY_RUN":
        return 1
    if execution_mode == "HIGH_RISK":
        return 2
    return 10
