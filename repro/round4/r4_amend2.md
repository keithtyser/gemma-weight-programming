
## Amendment 2 (12 Nov 2026, 1:51 PM ET; written before any episode of b47-b50)

- Replicate batches b47-b50 run entirely on the AMD node: sandboxes start in a second docker daemon on the node
  (image store on its NVMe volume), and the harness runs there from a copy of tii-steer, eval_data/golden_v2 and
  eval-venv (comp image id checked against the golden_v2 manifest). evalhost runs none of their containers.
- Servers and arms are the replicate sets of the table above: b47 = GPU 4 (the b35/b39 arms), b48 = GPU 5 (the
  b36/b40 arms), b49 = GPU 6 (the b37/b41 arms), b50 = GPU 7 (the b38/b42 arms). Each has its own `base`.
- Concurrency is 8 per arm (not 4), so the batches can finish inside the lease. An edit is still compared only with
  the `base` of its own batch, which runs at the same concurrency on the same server and host.
- Predictions and the decision rule are unchanged. For the rule, the two batches of an edit are its first batch
  (b35, b36, b43 or b44) and its node batch (b47-b50); b45 and b46, if they finish, are reported as third batches.
- These batches differ from the first ones in sandbox host and concurrency. That is a batch-level difference shared
  by every arm of a batch; it is noted wherever the two batches are averaged.
