# A1-zero Mainline Manifest

## Mainline decision

- Current branch: `codex/a1-zero-mainline`.
- Frozen classifier: `A1_concept_film_zero`, ISRUC, seed 42.
- Historical clean model-code anchor: `e5fdba04c1fd1389a65b2be7acbc38b6300cd328` (`perf: A1_zero + probs + filter[clinical_canonical]`).
- Reconstruction evidence: the clean detached commit above loads the frozen checkpoint and reproduces all 13,072 `test_probs_raw` rows exactly when deterministic TensorFlow inference uses sequence-window batch 16. Maximum and mean absolute probability differences are both `0.0`.
- Audit-start working-tree anchor: `df4972ee303d00965a6a5476c3b68cffa69960e8`, with pre-existing tracked and untracked work preserved in the external backup below.
- Recovery tag: `recovery/pre-a1-consolidation-20260719-022132`.
- External backup: `/home/b502/wssyi/project_backups/LIME-for-Time-Series_20260719_022132`.

The checkpoint is therefore reconstructable from a clean historical model implementation. The evaluator did not exist in that historical commit and is versioned by this consolidation.

## Canonical artifacts

| Artifact | Path | SHA256 |
|---|---|---|
| Checkpoint | `outputs/checkpoints/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42.weights.h5` | `e86a84790fd7351c1864158eef79fbf28437a7a8a5f26662b93bcf1aef14b6ed` |
| Subject split | `dataset_protocols/ISRUC/split.json` | `8363546a86b81676fe6171f2e50405d2d7447cbef1f48dbf397cd53bf3f037d9` |
| Prediction artifact | `outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_predictions.npz` | `0b9856dcffd4ea45be891f746b00cf8e26144d9d53e0549feb2c80a106cca340` |
| Split probability/label artifact | `outputs/predictions/result_20260704_024634_isruc_A1_zero_probs_ckpt_seed42_split_predictions.npz` | `53cd6b51c008e0dd47dd187654c8eddacd5d0a69037c48aeecafe5efa9594e39` |
| Training log | `outputs/logs/result_20260704_024634_isruc_A1_zero_probs_ckpt.txt` | `fc42b372e98dc7550df6784713eb52d829dc213b877c58bd63995cbf156945d9` |
| Historical experiment record | `experiment_records/20260703_184238_a1zero_lfts_probs.json` | `2a5ef2017a364a71f87e9d485aa9768316a56683984bf5b92f20b3c76ec773df` |
| Exact float32 parity reference | `outputs/canonical_a1_zero_parity/historical_exact_batch16_seed42.npy` | `1941e10ec16c60d08e6780de362e43ae947ec0445ca262911e94645926e4940e` |

Large artifacts remain outside Git. Their paths and hashes are authoritative; copies of the parity reference and all original working-tree state are in the external backup.

## Entrypoints

- Training launcher: `tools/run_main_with_protocol_split.py`.
- Training/inference program: `run_main.py`.
- Model factory: `src/model_training.py::create_sequence_sleep_model`.
- Data loader and preprocessing: `src/data_processing.py::load_all_data` and `preprocess_multichannel_epochs`.
- Offline explanation evaluator: `tools/sequence_lime_paper_evaluation.py`.
- Frozen split: 70 train, 15 validation, and 15 test subjects/recordings; split membership is defined only by `dataset_protocols/ISRUC/split.json`.

Strict parity command (GPU selection may be changed, numerical settings may not):

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONHASHSEED=42 TF_DETERMINISTIC_OPS=1 \
TF_ENABLE_ONEDNN_OPTS=0 TF_CUDNN_WORKSPACE_LIMIT_IN_MB=1024 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/b502/app/anaconda3/envs/l_venv/bin/python3.11 \
tools/sequence_lime_paper_evaluation.py --dataset isruc --seed 42 \
--mode parity-diagnostic --window-inference-batch-size 16 \
--disable-lime-visualizations
```

## Frozen enabled settings

- Dataset preset `isruc_A1_concept_film_zero`; random seed 42.
- Two training GPUs (`4,5` in the recorded run), 16 samples per replica, global training batch 32, prediction batch 16, balanced memory profile, deterministic TensorFlow operators.
- Fixed subject split; sequence input; sequence length 75; training stride 10; prediction stride 5.
- Epoch encoder `v3_multiscale`; sequence backbone `v3_tcn`; sequence transformer enabled.
- Micro/macro staged optimization enabled: 5 micro-pretraining epochs, 4 macro-stabilization epochs, 8 macro auxiliary-ramp epochs.
- Categorical cross-entropy; balanced sequence sampling; transition-boundary weighting and auxiliary head enabled with auxiliary weight 0.020.
- MixStyle enabled (`p=0.50`, `alpha=0.30`); stage-conditioned CORAL domain alignment enabled with weight 0.005.
- Concept FiLM plumbing enabled only in `zero` mode, strength 0.10, stop-gradient enabled. The clinical vector is replaced by zero before modulation and supplies no clinical signal.
- Filter backend `scipy`; filter profile `clinical_canonical`; normalization `recording_robust_zscore_canonical` (historical environment spelling: `recording_robust_zscore`). This uses unlabeled statistics from the complete current recording and is not a causal/deployment-safe transform.
- Validation-selected decoder: temperature argmax, `T=1.10`. LIME explains raw pre-temperature, pre-HMM neural probabilities.

## Frozen disabled settings

- Explanation feedback and online LIME: disabled; `concept_prior_weight=0`; concept, attention, and bottleneck loss weights are zero.
- Sequence clinical-concept feature branch and concept language: disabled.
- Real/shuffled concept FiLM input: disabled; late concept residual and confidence gate: disabled.
- SCR/stage concept router, Epoch-UTF-Lite, channel-specialized residual, SupCon variants, N1 context contrast, MPDG/DPCM, OT-TACB, concept-derivative auxiliary, EOG residual, risk/hard-example weighting, and external initialization: disabled.
- HMM and quality-aware HMM are optional validation candidates only; neither is the selected canonical decoder.
- Test-side oracle selection is diagnostic only and is excluded from canonical metrics and paper main tables.
- XAI dashboards are disabled in the classifier run. Clinical expert validation is pending.

## Canonical test results

| Metric | Value |
|---|---:|
| Epochs | 13,072 |
| Accuracy | 0.830860 |
| Macro-F1 | 0.819873 |
| Kappa | 0.782580 |
| N1-F1 | 0.594013 |
| REM-F1 | 0.899167 |

The raw argmax and validation-selected temperature-argmax labels are identical for this run.

## Runtime

- Interpreter: `/home/b502/app/anaconda3/envs/l_venv/bin/python3.11` (Python 3.11.15).
- TensorFlow 2.15.0; Keras 2.15.0; NumPy 1.26.4; SciPy 1.17.0; scikit-learn 1.3.0; pandas 2.0.3; matplotlib 3.10.8; h5py 3.16.0.
- GPU parity reconstruction: NVIDIA GeForce RTX 3090, cuDNN 8.9.4.
- Full environment snapshot: `/home/b502/wssyi/project_backups/LIME-for-Time-Series_20260719_022132/pip_freeze.txt`.

## Acceptance guard

Every mainline change must pass checkpoint loading, argmax agreement `1.0`, zero difference in Accuracy/Macro-F1/Kappa, maximum absolute probability difference `<=1e-6`, and sequence-wrapper maximum difference `<=1e-6`. Failure blocks cleanup and training.
