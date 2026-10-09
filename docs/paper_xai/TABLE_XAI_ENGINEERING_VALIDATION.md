# XAI Engineering Validation

Engineering validity does not imply scientific model faithfulness.

| Engineering check | Status | Evidence |
| --- | --- | --- |
| Canonical probability parity | PASS | max |Δp| = 0.0e+00 |
| Argmax agreement | PASS | 100.0% |
| Accuracy difference | PASS | 0.0e+00 |
| Macro-F1 difference | PASS | 0.0e+00 |
| Cohen κ difference | PASS | 0.0e+00 |
| 75-epoch wrapper parity | PASS | max |Δp| = 5.960e-08 |
| Cache/model identity guard | PASS | A1-zero identity and checkpoint SHA-256 unchanged |
| Fixed-graph 1,200-forward memory | PASS | current growth = 0 bytes |
| Feedback guard | PASS | enabled/λ>0/real target fail fast |
| Maintained tests before freeze | 122/122 | HC final validation |
