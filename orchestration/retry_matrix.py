def retries_for_mode(mode: str) -> int:
    if mode == "DRY_RUN":
        return 0
    if mode == "HIGH_RISK":
        return 1
    return 3
