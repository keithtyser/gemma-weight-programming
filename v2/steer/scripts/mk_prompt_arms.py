"""Prompt-space baselines for the weight edits: harness copies whose system prompt asks in words for what an edit does.

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: "Why not just change the prompt?" control. Copies harnesses/steer_ext_w5 to steer_ext_<name> and rewrites
  prompts/system.md:
    pexp   pacing rule asks for more exploration before the first edit (the explore-more edit, in words)
    pedit  adds an edit-hygiene rule from the failed-edit forensics (re-read before a second edit, quote <= 3 lines)
    pboth  both
  Usage: python mk_prompt_arms.py HARNESS_ROOT
"""
import shutil, sys
from pathlib import Path
root = Path(sys.argv[1]); src = root / "steer_ext_w5"
PACE_OLD = "Aim for **8-16 counted tool calls**, never repeat the exact same `read_file` or `run_command`, and make your first `edit_file` edit by tool call 12."
PACE_NEW = "Aim for **12-22 counted tool calls**, never repeat the exact same `read_file` or `run_command`. Explore before you edit: read every function the issue touches and its callers first; your first `edit_file` usually belongs around tool call 8-14, not earlier."
STEP_OLD = "- Apply minimal, idiomatic `edit_file` changes across all affected sync/async or sibling codepaths."
STEP_NEW = STEP_OLD + chr(10) + "- **Edit hygiene**: in `old_string` quote at most 3 consecutive lines, copied character-for-character from your most recent `read_file` of that file. After ANY `edit_file` on a file (successful or failed), `read_file` the target lines again before the next `edit_file` on it: the text you read earlier is stale. Never retry a failed `edit_file` with the same `old_string`."
s0 = (src / "prompts/system.md").read_text()
assert PACE_OLD in s0 and STEP_OLD in s0
for name, s in (("pexp", s0.replace(PACE_OLD, PACE_NEW)), ("pedit", s0.replace(STEP_OLD, STEP_NEW)),
                ("pboth", s0.replace(PACE_OLD, PACE_NEW).replace(STEP_OLD, STEP_NEW))):
    dst = root / f"steer_ext_{name}"
    assert not dst.exists(), dst
    shutil.copytree(src, dst)
    (dst / "prompts/system.md").write_text(s)
    print(name, len(s0), "->", len(s))
