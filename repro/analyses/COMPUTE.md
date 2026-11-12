# Compute statement: GPU time of the evaluation runs

Snapshot: 2026-11-12 14:35 ET (round 4/5 batches still running; their figures are partial lower bounds). Source data: `src/compute.json`. Method and assumptions at the end.

## Totals (GPU-hours)

| Group | GPU-hours |
|---|---:|
| (a) Main results: development batches 3-16 | 44.0 |
| (a) Main results: three-author benchmark, batches 21-28 | 17.8 |
| (a) Main results: held-out Tests 1 and 2 | 6.6 |
| **(a) Main results total** | 68.3 |
| (a) Main results total without batch 16 (incomplete, not in schedule table) | 66.8 |
| (b) Rounds 2-3, batches 29-34 (b32 never existed; b31 aborted; b33/b34 stopped) | 7.9 |
| (c) Round 4 so far, batches 35-58 (partial) | 14.4 |
| (c) Round 5 so far, batch 59 (partial, started 14:32 ET) | 0.1 |
| Other runs not in the paper (b0-b2, s1, s2, b17-b20) | 17.0 |
| **All evaluation runs, as of snapshot** | 107.7 |

## GPU-minutes per arm per 170-task run (complete batches)

| Set | Batches | Mean | Range | Mean batch wall-clock (h) |
|---|---:|---:|---|---:|
| Development 3-15 | 13 | 22.7 | 16.3-42.0 | 2.80 |
| Development 6-15 (single server) | 10 | 19.2 | 16.3-22.8 | 2.56 |
| Benchmark 21-28 | 8 | 22.3 | 20.1-24.6 | 2.23 |
| All complete 170-task batches | 26 | 23.2 | 16.3-42.0 | 2.54 |

Mean recorded agent-episode duration in complete batches: 195 s. Total recorded episode time (all runs, summed over episodes, sandbox side): 2691 h.

## Per batch

Times ET. Tasks = completed tasks per arm (min-max over arms). GPU-min/arm = GPU-hours x 60 / arms.

| Batch | Group | Arms | Start | End | Wall h | GPUs | GPU-h | Server | Tasks/arm | GPU-min/arm | Status | Note |
|---|---|---:|---|---|---:|---:|---:|---|---|---:|---|---|
| b0 | other (not in paper) | 4 | 11-10 13:16 | 11-10 15:59 | 2.71 | 1 | 2.71 | rented GPU box, 1 GPU (remote endpoint) | 170 | 40.7 | complete |  |
| b1 | other (not in paper) | 7 | 11-10 17:27 | 11-10 18:38 | 1.17 | 1 | 1.17 | rented GPU box, 1 GPU (remote endpoint) | 64 | 10.1 | complete (64-task subset design) |  |
| b2 | other (not in paper) | 7 | 11-10 18:47 | 11-10 21:13 | 2.44 | 1 | 2.44 | rented GPU box, 1 GPU (remote endpoint) | 170 | 20.9 | complete |  |
| b3 | a: main | 9 | 11-10 20:13 | 11-10 22:44 | 2.50 | 2 | 3.98 | evalhost GPU0 + evalhost GPU1 | 170 | 26.5 | complete | evalhost GPU1 4 arms 2.50 h; evalhost GPU0 5 arms 1.48 h |
| b4 | a: main | 11 | 11-10 22:42 | 11-11 03:58 | 5.26 | 2 | 7.71 | evalhost GPU0 + evalhost GPU1 | 170 | 42.0 | complete | evalhost GPU1 4 arms 5.02 h; evalhost GPU0 7 arms 2.68 h |
| b5 | a: main | 9 | 11-11 04:25 | 11-11 07:23 | 2.98 | 2 | 5.16 | evalhost GPU0 + evalhost GPU1 | 170 | 34.4 | complete | evalhost GPU1 5 arms 2.98 h; evalhost GPU0 4 arms 2.18 h |
| b6 | a: main | 8 | 11-11 06:57 | 11-11 10:00 | 3.04 | 1 | 3.04 | evalhost GPU0 | 170 | 22.8 | complete |  |
| b7 | a: main | 8 | 11-11 07:24 | 11-11 09:50 | 2.43 | 1 | 2.43 | evalhost GPU1 | 170 | 18.2 | complete |  |
| b8 | a: main | 8 | 11-11 10:02 | 11-11 12:45 | 2.72 | 1 | 2.72 | evalhost GPU0 | 170 | 20.4 | complete |  |
| b9 | a: main | 8 | 11-11 10:06 | 11-11 12:36 | 2.49 | 1 | 2.49 | evalhost GPU1 | 170 | 18.7 | complete |  |
| b10 | a: main | 8 | 11-11 12:18 | 11-11 14:42 | 2.40 | 1 | 2.40 | rented GPU box, 1 GPU (remote endpoint) | 170 | 18.0 | complete |  |
| b11 | a: main | 8 | 11-11 12:49 | 11-11 15:39 | 2.84 | 1 | 2.84 | evalhost GPU0 | 170 | 21.3 | complete |  |
| b12 | a: main | 8 | 11-11 12:51 | 11-11 15:22 | 2.53 | 1 | 2.53 | evalhost GPU1 | 170 | 19.0 | complete |  |
| b13 | a: main | 8 | 11-11 14:43 | 11-11 16:53 | 2.17 | 1 | 2.17 | rented GPU box, 1 GPU (remote endpoint) | 170 | 16.3 | complete |  |
| b14 | a: main | 8 | 11-11 15:41 | 11-11 18:22 | 2.68 | 1 | 2.68 | evalhost GPU0 | 170 | 20.1 | complete |  |
| b15 | a: main | 8 | 11-11 15:24 | 11-11 17:42 | 2.30 | 1 | 2.30 | evalhost GPU1 | 170 | 17.2 | complete |  |
| b16 | a: main | 8 | 11-11 16:55 | 11-11 18:25 | 1.50 | 1 | 1.50 | rented GPU box, 1 GPU (remote endpoint) | 118-142 | 11.3 | stopped (partial) | stopped incomplete (no arm.json); in user-specified main range but not in the paper schedule table |
| b17 | other (not in paper) | 8 | 11-11 18:24 | 11-11 20:46 | 2.37 | 1 | 2.37 | evalhost GPU0 | 106 | 17.7 | stopped (partial) | killed incomplete when evalhost was hard-rebooted ~20:50 ET (held-out v1 outage) |
| b18 | other (not in paper) | 8 | 11-11 18:24 | 11-11 20:46 | 2.36 | 1 | 2.36 | evalhost GPU1 | 160-170 | 17.7 | stopped (partial) | killed incomplete when evalhost was hard-rebooted ~20:50 ET (held-out v1 outage) |
| b19 | other (not in paper) | 6 | 11-11 21:03 | 11-11 23:22 | 2.33 | 1 | 2.33 | evalhost GPU0 | 170 | 23.3 | complete |  |
| b20 | other (not in paper) | 6 | 11-11 21:03 | 11-11 23:01 | 1.97 | 1 | 1.97 | evalhost GPU1 | 170 | 19.7 | complete |  |
| b21 | a: main | 6 | 11-11 23:04 | 11-12 01:07 | 2.06 | 1 | 2.06 | evalhost GPU1 | 170 | 20.6 | complete |  |
| b22 | a: main | 6 | 11-11 23:24 | 11-12 01:45 | 2.35 | 1 | 2.35 | evalhost GPU0 | 170 | 23.5 | complete |  |
| b23 | a: main | 6 | 11-12 01:09 | 11-12 03:10 | 2.02 | 1 | 2.02 | evalhost GPU1 | 170 | 20.2 | complete |  |
| b24 | a: main | 6 | 11-12 01:48 | 11-12 04:15 | 2.46 | 1 | 2.46 | evalhost GPU0 | 170 | 24.6 | complete |  |
| b25 | a: main | 6 | 11-12 03:12 | 11-12 05:12 | 2.01 | 1 | 2.01 | evalhost GPU1 | 170 | 20.1 | complete |  |
| b26 | a: main | 6 | 11-12 04:16 | 11-12 06:43 | 2.45 | 1 | 2.45 | evalhost GPU0 | 170 | 24.5 | complete |  |
| b27 | a: main | 6 | 11-12 05:13 | 11-12 07:18 | 2.08 | 1 | 2.08 | evalhost GPU1 | 170 | 20.8 | complete |  |
| b28 | a: main | 6 | 11-12 06:43 | 11-12 09:07 | 2.39 | 1 | 2.39 | evalhost GPU0 | 170 | 23.9 | complete |  |
| b29 | b: rounds 2-3 | 6 | 11-12 08:57 | 11-12 11:16 | 2.32 | 1 | 2.32 | evalhost GPU1 | 96-170 | 23.2 | stopped (partial) |  |
| b30 | b: rounds 2-3 | 6 | 11-12 09:08 | 11-12 11:39 | 2.51 | 1 | 2.51 | evalhost GPU0 | 170 | 25.1 | complete |  |
| b31 | b: rounds 2-3 | 6 | 11-12 11:18 | 11-12 11:26 | 0.13 | 1 | 0.13 | evalhost GPU1 | 3-18 | 1.3 | stopped (partial) | aborted after ~8 min (relaunched as b33) |
| b33 | b: rounds 2-3 | 6 | 11-12 11:29 | 11-12 13:02 | 1.56 | 1 | 1.56 | evalhost GPU1 | 56-106 | 15.6 | stopped (partial) | round 3; stopped 13:03 ET at ~52% for sandbox capacity |
| b34 | b: rounds 2-3 | 6 | 11-12 11:37 | 11-12 13:02 | 1.41 | 1 | 1.41 | evalhost GPU0 | 49-84 | 14.1 | stopped (partial) | round 3; stopped 13:03 ET at ~42% for sandbox capacity |
| b35 | c: round 4 | 6 | 11-12 12:17 | 11-12 13:38 | 1.35 | 1 | 1.35 | AMD 8-GPU node GPU0 | 42-52 | 13.5 | stopped (partial) | stopped 13:38 ET at ~27% (sandbox capacity) |
| b36 | c: round 4 | 6 | 11-12 12:18 | 11-12 14:35 | 2.29 | 1 | 2.29 | AMD 8-GPU node GPU1 | 82-94 | 22.9 | running (partial) |  |
| b37 | c: round 4 | 6 | 11-12 12:18 | 11-12 12:38 | 0.34 | 1 | 0.34 | AMD 8-GPU node GPU2 | 5-7 | 3.4 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b38 | c: round 4 | 6 | 11-12 12:18 | 11-12 12:38 | 0.33 | 1 | 0.33 | AMD 8-GPU node GPU3 | 4-8 | 3.3 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b39 | c: round 4 | 6 | 11-12 12:19 | 11-12 12:38 | 0.33 | 1 | 0.33 | AMD 8-GPU node GPU4 | 4-6 | 3.3 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b40 | c: round 4 | 6 | 11-12 12:19 | 11-12 12:38 | 0.32 | 1 | 0.32 | AMD 8-GPU node GPU5 | 4-6 | 3.2 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b41 | c: round 4 | 6 | 11-12 12:19 | 11-12 12:38 | 0.32 | 1 | 0.32 | AMD 8-GPU node GPU6 | 4-5 | 3.2 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b42 | c: round 4 | 6 | 11-12 12:20 | 11-12 12:38 | 0.31 | 1 | 0.31 | AMD 8-GPU node GPU7 | 4-7 | 3.1 | stopped (partial) | stopped ~12:38 ET (evalhost disk overload); partial dirs ignored by analysis |
| b43 | c: round 4 | 6 | 11-12 12:43 | 11-12 14:35 | 1.86 | 1 | 1.86 | AMD 8-GPU node GPU2 | 96-114 | 18.6 | running (partial) |  |
| b44 | c: round 4 | 6 | 11-12 12:45 | 11-12 14:35 | 1.83 | 1 | 1.83 | AMD 8-GPU node GPU3 | 99-170 | 18.3 | running (partial) |  |
| b45 | c: round 4 | 6 | 11-12 13:19 | 11-12 14:35 | 1.26 | 1 | 1.26 | evalhost GPU1 | 71-81 | 12.6 | running (partial) |  |
| b46 | c: round 4 | 6 | 11-12 13:20 | 11-12 14:30 | 1.16 | 1 | 1.16 | evalhost GPU0 | 61-70 | 11.6 | running (partial) | stopped ~14:30 ET to free evalhost GPU0 for round 5 (b59) |
| b47 | c: round 4 | 6 | 11-12 14:08 | 11-12 14:31 | 0.39 | 1 | 0.39 | AMD 8-GPU node GPU4 | 41-48 | 3.9 | running (partial) |  |
| b48 | c: round 4 | 6 | 11-12 14:08 | 11-12 14:31 | 0.39 | 1 | 0.39 | AMD 8-GPU node GPU5 | 42-55 | 3.9 | running (partial) |  |
| b49 | c: round 4 | 6 | 11-12 14:08 | 11-12 14:31 | 0.39 | 1 | 0.39 | AMD 8-GPU node GPU6 | 40-50 | 3.9 | running (partial) |  |
| b50 | c: round 4 | 6 | 11-12 14:08 | 11-12 14:31 | 0.39 | 1 | 0.39 | AMD 8-GPU node GPU7 | 44-101 | 3.9 | running (partial) |  |
| b51 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU0 | 7-14 | 1.4 | running (partial) |  |
| b52 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU1 | 8-16 | 1.4 | running (partial) |  |
| b53 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU2 | 7-15 | 1.4 | running (partial) |  |
| b54 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU3 | 8-27 | 1.4 | running (partial) |  |
| b55 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU4 | 7-12 | 1.4 | running (partial) |  |
| b56 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU5 | 8-15 | 1.4 | running (partial) |  |
| b57 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU6 | 6-12 | 1.4 | running (partial) |  |
| b58 | c: round 4 | 6 | 11-12 14:23 | 11-12 14:31 | 0.14 | 1 | 0.14 | rented 8-GPU CUDA box GPU7 | 12-19 | 1.4 | running (partial) |  |
| b59 | c: round 5 | 8 | 11-12 14:32 | 11-12 14:35 | 0.05 | 1 | 0.05 | evalhost GPU0 | 0-4 | 0.4 | running (partial) | round 5 (started after the round-4 brief); evalhost GPU0 |
| s1 | other (not in paper) | 8 | 11-11 10:31 | 11-11 11:25 | 0.89 | 1 | 0.89 | rented GPU box, 1 GPU (remote endpoint) | 73 | 6.7 | complete (73-task subset design) |  |
| s2 | other (not in paper) | 8 | 11-11 11:26 | 11-11 12:15 | 0.81 | 1 | 0.81 | rented GPU box, 1 GPU (remote endpoint) | 73 | 6.1 | complete (73-task subset design) |  |

## Held-out tests (results/steer_confirm)

| Run | Server | Arms | Start | End | GPU-h | Rows per arm | Note |
|---|---|---:|---|---|---:|---|---|
| Test 1 main, server A | rented 2-GPU box GPU A | 3 | 11-11 19:46 | 11-11 20:46 | 0.99 | 235, 223, 218 | ended by sandbox-host outage/reboot ~20:50 ET; last write used as end |
| Test 1 main, server B | rented 2-GPU box GPU B | 3 | 11-11 19:47 | 11-11 20:45 | 0.97 | 221, 219, 214 | ended by sandbox-host outage/reboot ~20:50 ET; last write used as end |
| Test 1 rerun (p2), server A | rented 2-GPU box GPU A | 3 | 11-11 21:02 | 11-11 22:58 | 1.93 | 215, 227, 228 | reruns of rollouts not kept from main run |
| Test 1 rerun (p2), server B | rented 2-GPU box GPU B | 3 | 11-11 21:03 | 11-11 22:09 | 1.09 | 220, 233, 233 | reruns of rollouts not kept from main run |
| Test 2 | evalhost GPU1 | 4 | 11-12 07:20 | 11-12 08:55 | 1.58 | 390, 390, 390, 390 | 4 arms x 130 tasks x 3 rollouts, complete |

## Author sessions (wsp_arena, CPU only; budget 90 min wall)

No session transcript or start/stop log is stored. Start = birth of BRIEF.md and of the three author directories (22:33 ET, 11 Nov); end = last write to that author's directory in round 1.

| Author | Start | Last write | Wall (min) | Build time |
|---|---|---|---:|---|
| Claude | 11-11 22:33 | 11-11 22:53 | 19.9 | about 30-33 s per edit (first to last write of each build log, 3 single builds); the paper's "about 40 s" comes from Claude's notes |
| GPT | 11-11 22:33 | 11-11 22:47 | 14.4 | recorded: all 5 edits in 20.8 s wall (99.7 s user CPU), `/usr/bin/time` in evidence/build.log; single rebuilds 7-11 s |
| Gemini | 11-11 22:33 | 11-11 23:17 | 44.7 | not recorded |

All three finished well within the 90-minute budget on file evidence. An author may have kept working without writing files, so these are lower bounds on session length. Claude's later build bursts (rounds 2, 3 and 5; session starts not recorded): round 2 08:46-08:51 ET (9 edits, single builds 30-52 s), round 3 11:52-11:58 ET (19 edits built in parallel, 46-125 s each), round 5 14:21-14:29 ET (9 edits, 58-142 s each, partly parallel).

## Batches that lack timing or are incomplete

- No batch records per-episode start/end timestamps; all wall-clock figures come from file birth and modification times.
- Incomplete (stopped): b16, b17, b18 (b17/b18 killed by the evalhost reboot about 20:50 ET on 11 Nov), b29 (one arm, ac2_pre_d100, stopped at 96 tasks; other arms complete), b31 (aborted after 8 min), b33, b34 (stopped 13:03 ET for sandbox capacity), b35 (stopped 13:38 ET), b37-b42 (stopped about 12:38 ET), b46 (stopped about 14:30 ET for round 5).
- Running at snapshot: b36, b43, b44, b45, b47-b58, b59.
- b1 (64 rebench tasks) and s1/s2 (73-task subsets) are complete by design but are not 170-task runs.
- GPU model unknown for the rented endpoints (see assumptions).

## Method and assumptions

Read-only scan of /path/to/gemma-v2/results/steer_eval, results/steer_confirm and wsp_arena on evalhost (stat birth/mtime, task_results.jsonl, arm.json, worker_summary.json). No process, file or server was changed.

- Batch wall-clock = earliest birth time (ext4 statx) of any arm directory of the batch to the latest end of any arm. Arm end = mtime of arm.json (written once when the arm finishes); arms without arm.json (stopped or running) use the latest mtime of any file in the arm directory (tmp_docker_contexts excluded). No result file records per-episode start/end timestamps; per-episode duration_seconds exist and are summed separately as episode-hours (sandbox-side agent time, not GPU time).
- One vLLM server = one GPU. Each batch used one server, except batches 3-5, whose arms were split over two servers (evalhost GPU0 and GPU1); there GPU-hours = sum of each server's own first-start..last-end window.
- GPU-hours count only the window in which a batch had arms running. Server start-up (about 1-2 min), idle servers between batches, and the gap between Test 1's main run and its rerun are not counted. The AMD-node servers were started at 0.30 memory utilisation but are counted as one whole GPU each.
- Server attribution comes from each arm's evaluation log (endpoint), serve_w.sh (CUDA_VISIBLE_DEVICES), the launch scripts and wsp_arena/round4_predictions.md. The remote endpoint used by b0-b2, s1, s2, b10, b13 and b16 is a rented box whose GPU model is not recorded on evalhost; the Test 1 servers are the two GPUs of a rented 2-GPU box. evalhost's GPUs are NVIDIA RTX PRO 6000 Blackwell Max-Q. GPU models of the AMD node and the rented 8-GPU CUDA box are not recorded on evalhost.
- Tasks completed per arm = competition traces written (106 max) + SWE-rebench rows (64 max); 170 = complete.
- Still-running batches are as of the snapshot time; their figures are lower bounds. A batch counts as running if any of its files was written within 10 minutes of the snapshot.
- Held-out Test 1's main run ended in the sandbox-host outage and reboot (about 20:50 ET); its last write (about 20:46) is used as its end. The six failed_launch attempts (about 1 s each, zero rows) are ignored.
- Per-arm cost = batch GPU-minutes / number of arms in the batch, averaged over complete 170-task batches. Arms of a batch share one server at concurrency 4 per arm, so this is an attribution, not a measured per-arm time.
- Batch 16 is in the requested main range (b3-b16) but stopped incomplete and is not in the paper's schedule table; the main total without it is also given.

## Paper-ready sentence

> Evaluation ran on single-GPU vLLM servers, mostly NVIDIA RTX PRO 6000 Blackwell. The runs behind the reported results (development batches 3-16, benchmark batches 21-28 and both held-out tests) took about 68 GPU-hours, or about 23 GPU-minutes per arm per 170-task run; all evaluation, including later rounds, took about 108 GPU-hours at the time of writing. Edits were built on a CPU in under a minute each, and each author session used under 45 minutes of its 90-minute CPU-only budget.

