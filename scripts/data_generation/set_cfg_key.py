"""
set_cfg_key.py - set one key in one section of the spread datasets' configs (s2..s5).

Usage (project root):
    python scripts/data_generation/set_cfg_key.py <Section> <Key> <Value> [s2 s3 s4 s5]
Example:
    python set_cfg_key.py Multimedia Derefract-Max-Shift-Px 400

* Reads keys written as 'Key: value' or 'Key = value', in any capitalisation.
* Changes only the given section. If the key appears more than once there, the first is
  set and the others are removed (the config reader stops on duplicate keys).
* A missing key is added at the end of that section, in the separator style the section
  already uses.
* Keeps the original as multicamselfcal.cfg.bak (only if none exists yet).
"""
import re
import sys
import shutil
from pathlib import Path

if len(sys.argv) < 4:
    sys.exit(__doc__)
SECTION, KEY, VALUE = sys.argv[1], sys.argv[2], sys.argv[3]
DATASETS = sys.argv[4:] or ["s2", "s3", "s4", "s5"]
ROOT = Path(__file__).resolve().parents[2] / "TestSystem"
key_re = re.compile(r"^[ \t]*" + re.escape(KEY) + r"[ \t]*[:=][ \t]*(.*?)[ \t]*$", re.I)
head_re = re.compile(r"^[ \t]*\[([^\]]+)\][ \t]*$")

for ds in DATASETS:
    hits_ds = sorted(ROOT.rglob(f"caldata_refracted_wide400_{ds}/multicamselfcal.cfg"))  # datasets live in TestSystem/<group>/
    cfg = hits_ds[0] if hits_ds else ROOT / f"caldata_refracted_wide400_{ds}" / "multicamselfcal.cfg"
    if not cfg.exists():
        print(f"{ds}: {cfg} not found - skipped")
        continue
    lines = cfg.read_text().splitlines()
    starts = [i for i, l in enumerate(lines) if head_re.match(l)]
    sec = [i for i in starts if head_re.match(lines[i]).group(1).strip().lower() == SECTION.lower()]
    if not sec:
        print(f"{ds}: no [{SECTION}] section - skipped")
        continue
    a = sec[0]
    b = next((i for i in starts if i > a), len(lines))            # section body: lines a+1 .. b-1
    hits = [i for i in range(a + 1, b) if key_re.match(lines[i])]
    body = [l for l in lines[a + 1:b] if re.match(r"^[ \t]*[^#;\s][^:=]*[:=]", l)]
    sep = ": " if sum(":" in l.split("=")[0] for l in body) >= len(body) / 2 else " = "
    bak = cfg.with_suffix(".cfg.bak")
    if not bak.exists():
        shutil.copy2(cfg, bak)
    if hits:
        old = key_re.match(lines[hits[0]]).group(1)
        m = key_re.match(lines[hits[0]])
        lines[hits[0]] = lines[hits[0]][:m.start(1)] + VALUE + lines[hits[0]][m.end(1):]
        for i in reversed(hits[1:]):
            del lines[i]
        note = f" (removed {len(hits) - 1} duplicate line(s))" if len(hits) > 1 else ""
        print(f"{ds}: [{SECTION}] {KEY} {old} -> {VALUE}{note}")
    else:
        ins = b
        while ins - 1 > a and lines[ins - 1].strip() == "":       # before trailing blank lines
            ins -= 1
        lines.insert(ins, f"{KEY}{sep}{VALUE}")
        print(f"{ds}: [{SECTION}] {KEY} added = {VALUE}")
    cfg.write_text("\n".join(lines) + "\n")
