# Held-out v1: serving tunnel outage (deviation record)
- 2026-11-11, about 20:26:00 to 20:30:00 ET: the tunnel from the harness host to the serving GPUs dropped; all six arms
  got connection errors and retried with backoff. Servers and harness processes stayed up; the tunnel was reopened.
- Every rollout in flight in that window lost up to ~4 min of its 7.5 min session budget, in every arm.
- Rule (fixed before looking at any outcome of those rollouts): in every arm, each rollout whose request log has
  activity both before 20:30:00 and after 20:26:00 (i.e. overlaps the window) is discarded and rerun once after the main
  run, same server and settings; the rerun row replaces it. Report results with and without the replacement.

## Second interruption (added 21:15 ET, before any outcome was looked at)

- 20:30 ET the tunnel was reopened and runs continued. From about 20:35 ET the Tailscale daemon on the sandbox host
  deadlocked (journal: `ipnlocal watchdog ... reportDeadlock`, 20:46:09), which cut the tunnel again; the host was
  hard-rebooted at about 20:50 ET, killing all six runs.
- Rule, replacing the one above and fixed without reference to outcomes: a rollout is kept only if its request log was
  last written before 20:26:00 ET (01:26:00 UTC), i.e. it finished before the first outage. Every other (task, rollout)
  item of every arm (missing, or finished later) is run once more into `heldout_v1_<arm>_p2`, same server, same
  adapter, same settings (`v2/steer/confirm_resume.py`). Kept / rerun per arm: base 175 / 215, base2 170 / 220,
  bneg_t0.125 157 / 233, bneg_t0.25 163 / 227, bneg_t0.5 157 / 233, bnm25_s1 162 / 228.
- The tunnel now runs detached on the sandbox host. Analysis is reported for (a) kept + rerun rows (primary) and
  (b) kept rows only.
