Author: gpt.

I inspected the brief, adapter checker, MLP edit/collection tools, two stored activation sets, and all 2,210 traces in complete base runs b3–b15. Both result schemas contribute; success comes from resolved. Baseline mean was 26.31/170; 42 tasks varied between success and failure.

I counted 1892/4916 failed edit_file calls (38.5%), including 165 missing-argument errors. This broader definition and fixed run set differ from the brief's 32%. First edits followed 4.80 observed tool calls on average. Re-reading before the next edit correlated with higher edit success; it does not prove causation.

The main edit applies anti-edit steering at the start of thinking. Its heldout detector AUC is 1.0 and other sampled positions barely activate it. Its control preserves the neuron choice, gates, write norms, and size while randomizing write directions.

The second construction rotates existing MLP outputs without changing their FP32 length. The stack combines this with start steering. The fourth edit discourages submission after a command tool error: 10.9% of such errors were followed immediately by submission. Its error detector has heldout AUC 0.988–0.990 among thinking-start positions. I found too few stored direct edit-failure examples to justify that separate claim.

All edits use hand-specified formulas, train-split means, and closed-form ridge readouts. No gradient steps, task-outcome search, model server, GPU, or evaluation ran. Heldout rows only diagnose local effects; they do not choose doses or layers. Weights and stored data stayed read-only. Peak builder RSS was 4.62 GiB; numerical work used at most eight CPU threads.

Local single-layer checks do not establish rollout improvements. Stored prompts may differ from future trajectories. Extra exploration can waste calls, and command errors can reflect unrelated environment faults. Predictions therefore allow losses. Confidence rank in PROPOSALS.json differs from LoRA rank, which is 16 for every adapter.

Each standalone build reproduced identical adapter bytes. All five passed check_adapter.py. Reports, hashes, and checks are in evidence/ and each edit directory. code/build.py --all rebuilds all five; PROPOSALS.json lists separate commands.
