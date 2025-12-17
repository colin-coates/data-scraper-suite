import subprocess
import sys

result = subprocess.run(
    ["python3", "scripts/precommit_guard.py"],
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.returncode != 0:
    sys.exit(1)

sys.exit(0)
