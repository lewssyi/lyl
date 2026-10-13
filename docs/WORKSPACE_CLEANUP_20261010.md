# Remote workspace cleanup - 2026-10-10

Scope: `/home/b502/wssyi/LIME-for-Time-Series` only.

External backup: `/home/b502/wssyi/project_backups/LIME-for-Time-Series_cleanup_20261010_021513`.

Backup archive content hashes were verified before deletion.

## Retained

All outputs, logs, models, data, dataset protocols, experiment records,
paper documents, figures and historical reports remain at their original paths.
All retained Python/YAML source hashes, run_main.py and canonical artifact
SHA256 hashes were verified unchanged. All protected file paths, sizes,
mtime, modes and symlink targets were scanned before and after deletion.

## Removed from active workspace

- ` `
- `.idea`
- `.pytest_cache`
- `.tf_tmp`
- `.trae`
- `__pycache__`
- `clean_tags.py`
- `experiments`
- `run_generalization_suite.py`
- `tmp_prepare_combinations_20260924.py`
- `src/__pycache__`
- `tests/__pycache__`
- `tests/test_sequence_context_batch_numerical_protocol.py`
- `tests/test_sequence_context_batch_parity_diagnostic.py`
- `tmp/eogsync`
- `tools/__pycache__`
- `tools/aggregate_hc_validation.py`
- `tools/aggregate_hierarchical_context_xai_feasibility.py`
- `tools/aggregate_mf_surrogate_small_validation.py`
- `tools/aggregate_sequence_lime_two_versions.py`
- `tools/generate_sequence_xai_training_targets.py`
- `tools/hc_classic_baseline.py`
- `tools/hierarchical_context_xai_feasibility.py`
- `tools/hierarchical_context_xai_guards.py`
- `tools/launch_hc_development.py`
- `tools/launch_hc_validation.py`
- `tools/mf_seqlime_surrogate_improvement.py`
- `tools/sequence_context_batch_numerical_protocol.py`
- `tools/sequence_context_batch_parity_diagnostic.py`
- `tools/sequence_lime_two_version_evaluation.py`
- `tools/xai_feedback_head_smoke.py`
- `configs/xai/abc_diagnostic.yaml`
- `configs/xai/abc_full.yaml`
- `configs/xai/abc_smoke.yaml`
- `configs/xai/hc_seqlime_smoke.yaml`
- `configs/xai/hc_seqlime_test.yaml`
- `configs/xai/hc_seqlime_validation.yaml`
- `configs/xai/mf_seqlime_test.yaml`
- `configs/xai/mf_seqlime_validation.yaml`
- `configs/xai/mf_surrogate_smoke.yaml`
- `configs/xai/mf_surrogate_test.yaml`
- `configs/xai/mf_surrogate_validation.yaml`
- `src/xai/__pycache__`

## Relocated

`tmp_prepare_combinations_20260924.py` ->
`tools/prepare_paper_combinations.py`, byte-identical.

## Validation

Static Python imports: no retained tool/test references a removed module.
Result/log preservation: PASS.
Canonical checkpoint, prediction artifact and ISRUC split SHA256: PASS.
Retained source byte identity: PASS.
Existing retained test suite: **212 passed, 1 pre-existing stale caption assertion failed** (32.40 s).
The failed test and caption are byte-identical to pre-cleanup HEAD. The assertion
expects an older literal sentence; the frozen caption now explicitly states that
human and expert validation have not been performed. No frozen paper evidence was
changed to satisfy this stale assertion. Full output: /home/b502/wssyi/project_backups/LIME-for-Time-Series_cleanup_20261010_021513/pytest.log.
Pytest 9.1.1 and its dependencies are isolated under /home/b502/wssyi/project_backups/LIME-for-Time-Series_cleanup_20261010_021513/validation_dependencies; the canonical Python environment was not modified.

## Recovery

The archive includes deleted sources/configs/tests/caches and the two
original README files. Restore selected files from the archive to their
original relative paths after reviewing any later edits. No Git history
or branch was rewritten. No sibling project or experiment worktree was changed.

## GitHub publication - 2026-10-13

The tracked cleanup was reproduced in an isolated checkout from the exact
pre-cleanup GitHub commit 382c1dde42d0391fcfe945c494a5bed447cd0112,
because the original server was temporarily unreachable. The published
change preserves all retained source bytes and relocates the paper helper
without content changes. Generated results and logs remain on the server
at the original paths and were not added to this Git commit.
