with open("frontend/index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

import re

for i, line in enumerate(lines, 1):
    # Match S01 to S25 as distinct tokens
    m = re.findall(r"\bS(0[1-9]|1[0-9]|2[0-5])\b", line)
    if m:
        print(f"Line {i:4d}: {line.strip()[:100]} (found: {m})")
