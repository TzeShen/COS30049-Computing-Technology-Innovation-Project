# Phishing URL Detection — COS30049 Assignment 2

This project investigates offline URL classification, clustering and model explainability. It extracts signals directly from URL text, including length, IP-address usage, digits, character patterns, subdomains and top-level domains. It never opens the submitted URL, resolves its hostname or downloads its webpage.

The command-line predictor uses the saved PhiUSIIL Explainable Boosting Machine (EBM). Training and evaluation use two separate datasets; their rows are never combined.

## 1. Environment setup

Use Python 3.11. Open PowerShell in the project root, where `requirements.txt`, `src/`, `data/` and `models/` are located. With Conda installed, run:

```powershell
conda create -n phishing-a2 python=3.11 -y
conda activate phishing-a2
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The first pip command installs the recorded ML dependencies. The second makes the source package available to Python. Keep the recorded dependency versions when loading the saved models.

Package installation and downloading the original datasets require internet access. Feature extraction and prediction operate offline after setup.

## 2. Predict a URL

Run from the project root:

```powershell
python -m phishing_detector.predict_url "https://www.example.org"
```

The predictor loads these existing files:

- `models/innovation/PhiUSIIL_Phishing_URL_Dataset/ebm.joblib`
- `models/innovation/PhiUSIIL_Phishing_URL_Dataset/feature_columns.joblib`

It extracts the same 26 features, in the same order, used during training. No retraining or raw dataset is needed for prediction.

The output includes:

- **Model phishing score:** the model's predicted probability for class 1.
- **Trust score:** `100 * (1 - model phishing probability)`.
- **Classification:** higher predicted risk when the probability is at least the fixed threshold of `0.5`.
- **Model contributions:** the strongest EBM term contributions, measured in log odds. These can include feature interactions.
- **Detected URL indicators:** separate display heuristics, such as an IP address or excessive URL length.

The probability is uncalibrated. A high trust score does not establish that a website is safe. Display heuristics are separate from the model's contributions and do not necessarily explain its prediction.

## 3. Datasets and labels

| Original filename | Source | Class 0 used by this project | Class 1 used by this project |
| --- | --- | --- | --- |
| `malicious_phish.csv` | [Malicious URLs dataset, Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | Benign | Phishing, malware or defacement |
| `PhiUSIIL_Phishing_URL_Dataset.csv` | [PhiUSIIL, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset) | Legitimate | Phishing |

The original PhiUSIIL labels use `1` for legitimate and `0` for phishing. Preprocessing reverses this mapping so that class 1 consistently represents the positive class in this project.

Only the original URL and label columns are read from each source. All 26 predictive features are calculated locally from URL text. Supplied webpage features and precomputed PhiUSIIL features are excluded.

Preprocessing removes unusable records, conflicting binary-label duplicate groups and duplicate URL keys within each source. It produces 640,929 rows for `malicious_phish.csv` and 235,358 rows for `PhiUSIIL_Phishing_URL_Dataset.csv`.

## 4. Data splits and evaluation procedure

URLs are grouped by domain using an offline public-suffix snapshot. Stable hashing with seed 42 assigns approximately 70% of domain groups to training, 15% to validation and 15% to testing. Row percentages vary because domain groups differ in size.

| Dataset | Training rows | Validation rows | Test rows |
| --- | ---: | ---: | ---: |
| malicious_phish | 455,554 | 92,321 | 93,054 |
| PhiUSIIL | 165,614 | 34,850 | 34,894 |

There is no domain-group overlap between the three partitions within either dataset. The datasets retain separate training runs, models and results.

Models and preprocessing are fitted on training data. Validation results guide model selection and LightGBM early stopping. The classification threshold remains `0.5`. Final model choices were recorded before test data was opened in `evaluation/final/selection_before_test.json`. Test results are used for reporting and error analysis, not further selection or tuning.

## 5. Models and experiments

| Experiment | Purpose |
| --- | --- |
| Majority baseline | Reference for the imbalanced classification task |
| Logistic Regression | Linear classification benchmark |
| Random Forest | Nonlinear tree ensemble benchmark |
| EBM | Additive boosted model with inspectable local contributions |
| LightGBM | Gradient-boosted tree classifier |
| LightGBM without `uses_https` | Checks sensitivity to the HTTPS feature while retaining the other 25 features |
| MiniBatch K-Means | Explores patterns among class 1 training URLs |

`train_website.py` is the HTTPS feature-ablation experiment. It does not build or launch a website.

Clustering uses only class 1 training rows and excludes the label from its predictors. Candidate values of `k` range from 2 to 6. Selection uses silhouette scores on a fixed sample. The selected configurations are `k=3` for malicious_phish, with silhouette 0.2398, and `k=2` for PhiUSIIL, with silhouette 0.4080. Cluster profiles, representative URLs and feature comparisons support interpretation.

## 6. Saved final results

| Evaluation | Model selected using validation | Test F1 | Test recall |
| --- | --- | ---: | ---: |
| malicious_phish | LightGBM without `uses_https` | 0.8674 | 0.9173 |
| PhiUSIIL | EBM | 0.9941 | 0.9895 |
| Main-source model tested on the cross-source PhiUSIIL subset | LightGBM without `uses_https` | 0.5856 | 0.9861 |

The cross-source subset contains 32,321 PhiUSIIL test URLs whose domain groups are absent from the entire malicious_phish source. The main-source model classified all 18,749 legitimate URLs in that subset as malicious, producing zero true negatives. Its high recall therefore accompanies a severe false-positive problem and poor transfer to this source. The two datasets also have different positive-class definitions.

The CLI uses the PhiUSIIL EBM because its target is specifically phishing and it was selected on that dataset's validation results. Strong within-source test performance does not establish reliable performance on arbitrary real-world URLs.

`evaluation/final/` contains metrics, predictions, confusion matrices, curves, subgroup results and sampled errors. Sampled error CSVs include a `manual_review_note` column for human observations. Contribution CSVs can be joined to the examples using `test_row_number`; generating these files does not itself complete manual error analysis.

## 7. Reproduce the experiments

The saved models can be used immediately with the prediction command above. The following commands document how to reproduce the experiments; they are not needed to use the predictor.

Run these commands in order from the project root. They use the supplied frozen splits and write to a new `reproduction/` folder to preserve the submitted results:

```powershell
python -m phishing_detector.train_classification --split_dir data/splits --out_dir reproduction/models/baselines
python -m phishing_detector.train_clustering --split_dir data/splits --out_dir reproduction/clustering
python -m phishing_detector.train_innovation_model --split_dir data/splits --baseline_dir reproduction/models/baselines --out_dir reproduction/models/innovation
python -m phishing_detector.train_website --split_dir data/splits --reference_dir reproduction/models/innovation --out_dir reproduction/models/robustness
python -m phishing_detector.evaluate_models --split_dir data/splits --baseline_dir reproduction/models/baselines --innovation_dir reproduction/models/innovation --robustness_dir reproduction/models/robustness --out_dir reproduction/evaluation
```

Training commands process both datasets separately by default. They use seed 42; EBM uses one worker. Existing experiment output folders are protected against overwriting. Use a fresh destination for another reproduction run.

To regenerate processed data and splits from the original downloads, place the two original CSVs in `data/raw/` without renaming them, then run:

```powershell
python -m phishing_detector.data_prep --out_dir reproduction/data/processed
python -m phishing_detector.split_data --processed_dir reproduction/data/processed --out_dir reproduction/data/splits
```

To train using those regenerated splits, replace `--split_dir data/splits` with `--split_dir reproduction/data/splits` in every training and evaluation command, and use unused output destinations. Supply original raw CSVs to preprocessing, not previously processed files.

## 8. Important project files

| Location | Contents |
| --- | --- |
| `src/phishing_detector/` | Feature extraction, data processing, training, evaluation and CLI prediction code |
| `data/processed/` | Cleaned feature datasets, kept separate by source, and preprocessing audits |
| `data/splits/` | Frozen train, validation and test CSVs, domain assignments and split summaries |
| `models/baselines/` | Saved baseline classifiers and validation evidence |
| `models/innovation/` | Saved EBM and LightGBM models, feature lists and validation evidence |
| `models/robustness/` | Saved LightGBM models without the HTTPS feature and comparison results |
| `clustering_output/grouped/` | Clustering models, profiles, examples, selection results and figures |
| `evaluation/final/` | Frozen selections, final evaluations, cross-source results and error-review files |
| `requirements.txt` | Recorded versions of the ML dependencies |
| `pyproject.toml` | Python package configuration |

Keep model feature lists and training summaries with their corresponding saved models. Keep split summaries and domain assignments with the split CSVs because evaluation uses them to check consistency and construct the cross-source challenge.