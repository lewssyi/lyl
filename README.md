# Multimodal LIME for EEG Sleep Stage Classification

This project builds an interpretable sleep staging pipeline for 30-second
physiological epochs. It classifies five AASM-style sleep stages (`W`, `N1`,
`N2`, `N3`, `REM`) from three synchronized channels (`EEG`, `EOG`, `EMG`) and
then explains both individual stage decisions and stage transitions with
time-series XAI methods.

The repository started from a generic LIME time-series idea, but the current
implementation is focused on EEG sleep staging, multimodal channel evidence,
clinical concept priors, and sequence-aware postprocessing.

## Highlights

- Multichannel Sleep-EDF style input: `.npz` files with `x`, `y`, optional
  `fs`, and channel labels.
- Hybrid TensorFlow/Keras model with raw waveform input and segment-level
  physiological features.
- Dynamic channel routing for EEG/EOG/EMG attention.
- Clinical concept branch for sleep-relevant features such as EEG delta,
  theta, alpha, sigma, beta, EOG movement, and EMG tone.
- File-level train/validation/test splitting to avoid mixing epochs from the
  same recording within a split.
- Quality confidence scoring for noisy, flat, nonfinite, or artifact-heavy
  epochs.
- Temperature scaling, N1-aware thresholding, smoothing, and clinical
  HMM/Viterbi decoding.
- LIME dashboards with channel contribution scores, temporal segment evidence,
  clinical concept evidence, and prior alignment.
- Transition-aware XAI dashboards for stage changes such as `N1 -> N2`,
  `N2 -> N3`, and `N2 -> REM`.
- Lightweight tests for preprocessing, postprocessing, LIME segmentation, and
  transition explanation utilities.

## Repository Layout

```text
.
|-- run_main.py                 # End-to-end training, evaluation, XAI pipeline
|-- requirements.txt            # Core Python dependencies
|-- src/
|   |-- data_processing.py      # NPZ loading, preprocessing, quality scoring
|   |-- model_training.py       # Hybrid EEG/EOG/EMG sleep staging model
|   |-- lime_explanation.py     # Multimodal LIME and concept/DCCS logic
|   |-- visualization.py        # Diagnosis and transition XAI dashboards
|   |-- grad_cam.py             # Grad-CAM helper for the hybrid model
|   `-- eeg_utils.py            # Smaller EEG loading/preprocessing helpers
|-- tests/                      # Pytest coverage for core utilities
|-- EEG_data/
|   |-- prepare_physionet.py    # EDF-to-NPZ extraction script
|   |-- download_physionet.sh   # Sleep-EDF download helper
|   `-- eeg_eog_emg/            # Local NPZ recordings, if present
|-- notebooks/                  # Exploratory notebook and saved notebook models
|-- models/                     # Saved model artifacts
`-- outputs/                    # Logs, reports, XAI figures, and generated plots
```

## Data Format

The main pipeline expects one `.npz` file per recording. Each file should contain:

| Key | Shape / Type | Meaning |
| --- | --- | --- |
| `x` | `(epochs, samples, channels)` | 30-second epochs, usually `(N, 3000, 3)` at 100 Hz |
| `y` | `(epochs,)` | Integer labels: `0=W`, `1=N1`, `2=N2`, `3=N3`, `4=REM` |
| `fs` | scalar, optional | Sampling frequency, defaults to 100 Hz if absent |
| `ch_label` | list, optional | Channel names, expected order is EEG, EOG, EMG |

The EDF conversion helper in `EEG_data/prepare_physionet.py` extracts:

```text
EEG Fpz-Cz
EOG horizontal
EMG submental
```

from Sleep-EDF style PSG/Hypnogram pairs and saves them as `.npz` files.

## Installation

Create a virtual environment and install the core dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For EDF extraction from raw PhysioNet files, also install `mne`:

```bash
pip install mne
```

## Running the Pipeline

The main entry point is:

```bash
python run_main.py
```

Cross-dataset generalization ablations can be selected with
`LFTS_GENERALIZATION_PRESET`:

```bash
LFTS_GENERALIZATION_PRESET=baseline python run_main.py
LFTS_GENERALIZATION_PRESET=mixstyle python run_main.py
LFTS_GENERALIZATION_PRESET=mixstyle_coral python run_main.py
LFTS_GENERALIZATION_PRESET=mixstyle_coral_tent python run_main.py
```

On Windows PowerShell, set the variable first:

```powershell
$env:LFTS_GENERALIZATION_PRESET="mixstyle_coral_tent"
python run_main.py
```

To run the paired suite, use:

```bash
python run_generalization_suite.py --dry-run
python run_generalization_suite.py
```

Before running, check the data path near the top of `run_main.py`:

```python
MULTI_CHANNEL_PATH = os.path.join(PARENT_DIR, "sleepEDF", "multichannel_extracted")
```

If your local data is inside this repository, change it to:

```python
MULTI_CHANNEL_PATH = os.path.join(PROJECT_DIR, "EEG_data", "eeg_eog_emg")
```

`run_main.py` performs the full workflow:

1. Load `.npz` recordings and split them into train, validation, and test sets.
2. Preprocess EEG/EOG with band-pass filtering and channel normalization.
3. Compute epoch quality confidence scores.
4. Build dynamic semantic segments and segment-level spectral features.
5. Train the hybrid raw-waveform plus segment-feature model.
6. Select the best validation checkpoint using calibrated sequence metrics.
7. Evaluate argmax, temperature-scaled, N1-calibrated, smoothed, clinical HMM,
   and quality-aware HMM decoders.
8. Save classification reports and confusion matrices.
9. Generate representative LIME diagnosis dashboards for each sleep stage.
10. Generate transition-aware XAI dashboards for correctly predicted stage
    changes.

Logs are written to:

```text
outputs/logs/result_YYYYMMDD_HHMMSS.txt
```

## Model Overview

The current model is created by `src/model_training.py::create_hybrid_model_v2`.
It has three main components:

- Raw waveform branch: dynamic channel routing, Conv1D blocks with squeeze and
  excitation, optional MixStyle domain-statistic augmentation, and BiLSTM
  temporal aggregation.
- Segment feature branch: semantic segment statistics plus spectral band-power
  features, optional MixStyle augmentation, followed by sequence modeling.
- Clinical concept branch: concept features initialized with sleep-stage priors
  for EEG/EOG/EMG physiological evidence, with optional MixStyle augmentation
  to reduce dataset-specific feature-statistic dependence.

The model outputs:

1. `main_output`: sleep stage probabilities.
2. `channel_attention_weights`: EEG/EOG/EMG routing weights.
3. `concept_output`: concept-prior sleep stage probabilities.

## Explainability

The XAI layer is centered on multimodal LIME for time series:

- Each epoch is split into 30 dynamic physiological segments.
- Perturbations are applied per channel and per segment.
- A local ridge model estimates the effect of each channel-segment feature on
  the predicted stage.
- DCCS, or Dynamic Channel Contribution Score, summarizes channel importance.
- Concept-level DCCS connects segment evidence to clinical concepts such as
  delta power, sigma/spindle evidence, rapid eye movement, and EMG tone.
- Transition summaries aggregate several neighboring epochs around a stage
  boundary and compare observed evidence shifts with clinical transition priors.

Representative outputs include:

```text
outputs/XAI_Diagnosis_W.png
outputs/XAI_Diagnosis_N1.png
outputs/XAI_Diagnosis_N2.png
outputs/XAI_Diagnosis_N3.png
outputs/XAI_Diagnosis_REM.png
outputs/XAI_Transition_SC4021E0_N2_to_REM.png
outputs/XAI_Transition_SC4022E0_N2_to_N3.png
outputs/classification_report_visual.png
outputs/confusion_matrix.png
```

## Example Results

One complete local run recorded in
`outputs/logs/result_20260525_201404.txt` used 38 recordings and 5,970 test
epochs. The selected validation decoder was `quality_aware_hmm`.

| Decoder | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Raw argmax | 0.7915 | 0.7251 | Direct model probabilities |
| N1-protected smoothing | 0.7943 | 0.7254 | Local sequence smoothing |
| Clinical HMM/Viterbi | 0.8139 | 0.7276 | Uses transition constraints |
| Quality-aware HMM/Viterbi | 0.8121 | 0.7210 | Downweights low-quality epochs |

The logs show that N1 remains the hardest class, which is expected for this
dataset because N1 is rare and physiologically transitional.

## Tests

Run the lightweight test suite with:

```bash
pytest -q
```

Current test coverage checks:

- NPZ loading and shape formatting.
- Quality scoring without destructive amplitude clipping.
- Temperature scaling and probability blending.
- Clinical transition masks and Viterbi decoding.
- Quality-aware adaptive Viterbi behavior.
- LIME semantic segmentation and transition summary alignment.
- Clinical concept DCCS decomposition.

## Notes

- Full training is compute-heavy. A single run can take hours depending on CPU,
  GPU, and dataset size.
- `outputs/` contains generated artifacts and may become large.
- `models/` and `notebooks/` contain saved exploratory model artifacts from
  earlier experiments.
- The main script is the most current path for reproducible end-to-end runs.

## License

This project is licensed under the GNU General Public License v3.0. See
`LICENSE` for details.

## Acknowledgments

This work builds on the original LIME idea from
["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
by Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin, and uses
Sleep-EDF-style polysomnography data for sleep staging experiments.
