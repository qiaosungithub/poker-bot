"""Hash complete benchmark rows without timing; safely tolerate a live gzip stream."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

rows = {}
complete = False
with gzip.open(Path(sys.argv[1]).expanduser(), "rt") as stream:
    while True:
        try:
            line = stream.readline()
        except (EOFError, OSError):
            break
        if not line:
            complete = True
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        row.pop("seconds", None)
        key = f'{row["panel"]}:{row["block"]}:{row["rotation"]}'
        rows[key] = hashlib.sha256(json.dumps(row, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
print(json.dumps({"complete_gzip": complete, "tables": len(rows), "rows": rows}, sort_keys=True))
