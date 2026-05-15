# 🧠 NeuroGuard — Epileptic Seizure Detection System

**Developed by:** Asad Ali Asim & Muhammad Daniyal Khan  
**Program:** BSAI-IV | Semester Project  
**Dataset:** [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (~42 GB, 23 patients)

---

## Overview

NeuroGuard is an end-to-end AI pipeline that detects epileptic seizures from raw scalp EEG signals. It processes the full CHB-MIT dataset, trains two complementary models (a Random Forest classifier and a 1D-CNN), and exposes predictions through an interactive clinical dashboard built with Streamlit.

The system is designed around two core principles:
- **Clinical validity** — patient-aware data splitting prevents data leakage between training and evaluation, giving honest performance estimates
- **Memory safety** — 42 GB of EEG data is processed lazily and cached in chunks so the pipeline runs on standard consumer hardware

---

## Project Structure

```
.
├── app.py                        # Streamlit clinical dashboard
├── Model_training_fixed.ipynb    # Full training pipeline (corrected)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── data_cache/                   # Auto-generated: per-file .npz cache
├── X_all.npy                     # Memory-mapped feature matrix
├── y_all.npy                     # Memory-mapped labels
│
├── scaler.pkl                    # Fitted RobustScaler
├── best_rf_tuned_model.pkl       # Tuned Random Forest (hyperparameter searched)
├── advanced_model.h5             # Trained 1D-CNN
├── model_metrics.pkl             # Saved evaluation metrics
└── training_history.pkl          # CNN training curves
```

---

## Setup

### Requirements

- Python 3.9+
- ~8 GB RAM minimum (16 GB recommended)
- CHB-MIT dataset downloaded locally

### Installation

```bash
git clone <repository-url>
cd seizure-detection

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the Models

1. Open `Model_training_fixed.ipynb` in Jupyter
2. **Cell 1:** Set `data_dir` to your CHB-MIT folder path
3. Run all cells top to bottom
4. Expected runtime: 45–90 minutes (full dataset, first run)
   - Subsequent runs are faster — already-cached files are skipped automatically

The notebook will produce:
- `scaler.pkl`, `best_rf_tuned_model.pkl`, `advanced_model.h5`
- `model_metrics.pkl`, `training_history.pkl`
- `data_cache/` with one `.npz` per EDF file

### Step 2 — Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Step 3 — Test the System

Upload any `.edf` file from CHB-MIT. Use the known seizure timestamps below for instant verification:

| File | Seizure onset |
|------|--------------|
| chb01_03.edf | ~2996s |
| chb01_04.edf | ~1467s |
| chb03_02.edf | ~1088s |
| chb03_04.edf | ~1440s |
| chb06_04.edf | ~582s  |
| chb06_09.edf | ~360s  |
| chb08_05.edf | ~1656s |
| chb08_11.edf | ~1200s |
| chb12_08.edf | ~1440s |
| chb12_09.edf | ~1200s |

---

## Models

### Random Forest

Trained on hand-crafted statistical and spectral features extracted per EEG window and per channel:

**Time-domain (9 per channel):** mean, standard deviation, variance, 25th/75th percentile, skewness, kurtosis, min, max

**Frequency-domain (5 per channel):** band power in Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–50 Hz) using Welch's method

Hyperparameters tuned via `RandomizedSearchCV` (10 iterations, 3-fold CV). Class imbalance handled with `class_weight='balanced'`.

### 1D Convolutional Neural Network

Learns temporal features directly from the raw scaled EEG signal without manual feature engineering.

Architecture:
- Conv1D(16) → BatchNorm → MaxPool
- Conv1D(32) → BatchNorm → MaxPool
- Conv1D(64) → BatchNorm → GlobalAveragePool
- Dense(64) + Dropout(0.5)
- Dense(1, sigmoid)

Regularisation: L2 on all Conv and Dense layers, EarlyStopping (patience=5), ReduceLROnPlateau.  
Class imbalance: `class_weight` passed to `model.fit()`.

---

## Key Design Decisions

### Patient-Aware Data Splitting
`GroupShuffleSplit` ensures that all EEG recordings from a single patient appear in only one of train or test. Without this, the model memorises patient-specific brain patterns and reports inflated accuracy. This is the most common methodological flaw in published seizure detection work.

### Memory-Safe Loading
The full CHB-MIT dataset is ~42 GB and cannot be held in RAM. The pipeline uses three techniques:
1. `preload=False` in MNE — EDF files are read from disk lazily
2. Per-file `.npz` caching — windows are saved to disk immediately after each file is processed, then freed from RAM
3. `np.lib.format.open_memmap` — the final dataset is assembled into disk-backed arrays that the OS pages in and out as needed

### Class Imbalance
Seizure segments represent ~1–3% of CHB-MIT. Accuracy alone is misleading (a model predicting "normal" always achieves ~97%). The pipeline uses class weights, and evaluation focuses on sensitivity (seizure recall), specificity, F1-score, and ROC-AUC.

---

## Expected Performance

| Metric | Random Forest | 1D-CNN |
|--------|:---:|:---:|
| Test Accuracy | 88–93% | 90–95% |
| Sensitivity (seizure recall) | 60–75% | 72–85% |
| Specificity | 95–98% | 96–99% |
| F1-score (seizure) | 0.55–0.72 | 0.68–0.80 |
| ROC-AUC | 0.82–0.90 | 0.88–0.95 |

> **Why accuracy looks high:** CHB-MIT is severely imbalanced. Sensitivity and AUC are the clinically meaningful metrics.

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Manual Scanner** | Analyse any specific 5-second window from an uploaded EDF |
| **Live Feed** | Simulate continuous patient monitoring — scans rolling windows every second, halts on detection |
| **History & Export** | Full prediction log with CSV download |
| **Model switcher** | Toggle between RF and CNN at runtime |
| **Session stats** | Live count of windows scanned and seizures detected |

---

## Dependencies

```
streamlit==1.41.0
mne==1.8.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.5.0
scipy>=1.10.0
joblib==1.4.0
tensorflow==2.16.0
imbalanced-learn==0.12.0
matplotlib==3.8.4
seaborn==0.13.1
```

---

## Limitations

- Inter-patient variability is the hardest challenge in EEG seizure detection. Performance on unseen patients may be lower than reported test metrics.
- The 5-second non-overlapping window may split seizure onset across two windows, slightly reducing ictal sample count.
- The dashboard simulates real-time monitoring — it is not connected to live EEG hardware.

---

## References

1. Shoeb, A. H. (2009). *Application of machine learning to epileptic seizure onset detection and treatment*. MIT PhD Thesis.
2. Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, PhysioNet. *Circulation*, 101(23).
3. Craik, A. et al. (2019). Deep learning for electroencephalogram (EEG) classification tasks. *Journal of Neural Engineering*, 16(3).
