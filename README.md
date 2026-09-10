# Visual Appraisal Recognition in TED Talks Using Facial Expressions

This bachelor's thesis project investigating whether **linguistic appraisal in TED Talk segments can be recognized from facial expressions**.

The repository contains the preprocessing pipeline, OpenFace-derived facial features, two machine learning approaches, evaluation code and visualizations used to study the relationship between linguistically annotated appraisal and facial behaviour.

## Overview

Linguistic appraisal describes evaluative content in language. In this project, linguistic appraisal annotations are used as ground-truth labels while the prediction itself relies exclusively on visual facial signals. Suitable segments were extracted from **TED Talk videos** and processed with **OpenFace**. Instead of using raw video frames directly, facial behaviour is represented through frame-level **Facial Action Unit (AU) intensities**. The resulting task is formulated as binary classification:

- `0` — no appraisal
- `1` — appraisal present

The project investigates **three main questions**:

*(i) To what extent can the presence of linguistic appraisal be recognized in TED Talk clips from the facial expressions of the speaker?* \
*(ii) Which individual Action Units and combinations of Action Units are most informative for recognizing linguistic appraisal?* \
*(iii) Does preserving local temporal information provide an advantage over clip-level feature aggregation?* \

**Two complementary modelling approaches** are implemented:

1. Aggregation-based Logistic Regression --> as an interpretable clip-level baseline
2. Attention-based Multiple Instance Learning (MIL) --> to model a clip as a sequence of local facial-expression windows and identify temporally relevant regions

## Facial Features

OpenFace is used to extract facial behaviour from the video clips. The final experiments use the intensity estimates of eight Facial Action Units:

| Feature | Facial Action Unit |
| --- | --- |
| `AU01_r` | Inner Brow Raiser |
| `AU02_r` | Outer Brow Raiser |
| `AU04_r` | Brow Lowerer |
| `AU05_r` | Upper Lid Raiser |
| `AU06_r` | Cheek Raiser |
| `AU07_r` | Lid Tightener |
| `AU09_r` | Nose Wrinkler |
| `AU12_r` | Lip Corner Puller |

Other OpenFace outputs such as facial landmarks, gaze features, head pose information and binary AU occurrence predictions are removed during preprocessing because this work focuses on AU intensity signals.

The experiments currently use ten TED Talk source IDs:

`Art6`, `Bus14`, `Edu2`, `Ent5`, `Med15`, `Phi3`, `Pol2`, `Pol9`, `Tech10` and `Tech21`.

## Dataset Structure

The repository contains the OpenFace feature files and the derived tabular datasets used by the models. The original TED Talk video material is not included.

```text
data/
├── openface/           # Frame-level OpenFace CSV files for individual clips
├── processed/
│   └── combined.csv    # Combined and cleaned frame-level dataset
├── aggregated/
│   ├── simAgg.csv      # Simple clip-level aggregation
│   └── advAgg.csv      # Advanced clip-level aggregation
├── labels.csv          # Binary labels, source IDs and linguistic appraisal annotations
└── metadata.csv        # Metadata for the TED Talk sources
```

## Methodology

### 1. Aggregation-based Logistic Regression

The first approach removes the temporal dimension by aggregating the frame-level AU intensities into one feature vector per clip.
Two aggregation methods were tested: 

**Simple aggregation**
- mean AU intensity over the complete clip

**Advanced aggregation**
- mean
- standard deviation
- 95th percentile

A `Logistic Regression` classifier is trained on the aggregated features. The implementation uses a **RobustScaler** for normalization. In addition to models containing all selected AUs, individual Action Units can be evaluated separately to estimate how much predictive information each signal provides on its own.

### 2. Attention-based Multiple Instance Learning

The second approach preserves local temporal information by treating each video clip as a bag and short temporal windows as instances.

For every clip:

1. Frame-level AU sequences are divided into overlapping windows.
2. The AU intensities inside each window are averaged into a local feature vector.
3. All windows belonging to the same clip form a variable-length bag.
4. A neural encoder transforms the window features into hidden representations.
5. An attention mechanism assigns a weight to every window.
6. The weighted representation of the complete bag is classified as appraisal or non-appraisal.

The model architecture is:

```text
Window features
      │
      ▼
Linear(input_dim → 64)
      │
     ReLU
      │
   Dropout
      │
      ▼
Hidden window representations
      │
      ▼
Linear(64 → 32)
      │
     Tanh
      │
Linear(32 → 1)
      │
   Softmax
      │
      ▼
Attention weights
      │
      ▼
Weighted sum of window representations
      │
      ▼
Linear(64 → 1)
      │
      ▼
Binary appraisal prediction
```

**Padding and masking** allow clips with different numbers of windows to be processed in the same batch. Attention weights are normalized only across valid windows.

### Final MIL Configuration

A suitable hyperparameter configuration was determined through a **Sequential Hyperparameter Search**. The fixed configuration used for the final MIL experiments is defined in `src/configuration/config.py`.

| Parameter | Value |
| --- | ---: |
| Window size | 10 frames |
| Stride | 5 frames |
| Learning rate | 0.003 |
| Weight decay | 0.001 |
| Batch size | 16 |
| Dropout | 0.2 |
| Epochs | 25 |
| Classification threshold | 0.5 |
| Base random seed | 45 |

The 10-frame windows with a stride of 5 correspond to 50% overlap. If the end of a clip would otherwise remain uncovered, the final window is shifted to include the last frames.

## Evaluation Strategy

Both modelling approaches are evaluated using **Leave-One-Source-Out (LOSO)** evaluation.

For each fold, all clips from one TED Talk source are held out for testing while the remaining sources are used for model development. This prevents clips originating from the same source video from appearing in both training and test data and reduces the risk that the models exploit source- or speaker-specific characteristics.

The main evaluation metric is **Balanced Accuracy**, which gives equal importance to both classes despite their unequal frequency. Additional metrics include:

- precision for class 0 and class 1
- recall for class 0 and class 1
- class-specific F1 scores
- Macro F1
- binary cross-entropy/log loss
- confusion matrices

For the MIL approach, a validation split inside the training data is additionally used for model selection during each LOSO fold.

## Attention Analysis

A major advantage of the MIL architecture is that it provides a temporal indication of which parts of a clip contributed most strongly to a prediction.

The repository therefore includes utilities to:

- visualize attention weights over the windows of a clip
- compare attention with local AU intensities
- inspect the highest-attention windows
- analyse evidence and contribution values for individual windows
- compare correctly and incorrectly classified examples

These visualizations are used to investigate whether the model attends to meaningful facial-expression changes instead of treating the complete clip as equally informative.

## Repository Structure

```text
TED-Talk-Appraisal-Recognition/
├── data/
│   ├── aggregated/
│   ├── openface/
│   ├── processed/
│   ├── labels.csv
│   └── metadata.csv
│
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 00_preprocessing.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 01_logistic_regression_single_signal.ipynb
│   ├── 02_mil_config_testing.ipynb
│   ├── 02_mil_combinations.ipynb
│   └── 02_mil_attention_analysis.ipynb
│
├── results/
│   ├── logistic_regression/
│   └── mil/
│
├── src/
│   ├── AggregationBasedArchitecture/
│   ├── MILArchitecture/
│   ├── configuration/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── visualization/
│
└── README.md
```

## Notebooks and Suggested Workflow

The notebooks are numbered according to the main experimental workflow.

**Exploration and preprocessing**

`00_eda.ipynb`  
Exploratory analysis of the dataset and facial signals.

`00_preprocessing.ipynb`  
Combines the OpenFace files with their labels, removes unused features and creates the processed datasets.

**Aggregation-based experiments**

`01_logistic_regression.ipynb`  
Runs the main Logistic Regression experiments on the aggregated features.

`01_logistic_regression_single_signal.ipynb`  
Evaluates the selected Action Units individually.

**Attention-based MIL experiments**

`02_mil_config_testing.ipynb`  
Evaluates candidate hyperparameter configurations.

`02_mil_combinations.ipynb`  
Runs experiments with individual AUs and combinations of facial signals.

`02_mil_attention_analysis.ipynb`  
Inspects learned attention weights and visualizes selected examples.

## Results

Generated outputs are stored in `results/`.

```text
results/
├── logistic_regression/
│   ├── cms/
│   └── tables/
└── mil/
    ├── attention_heatmaps/
    ├── cms/
    ├── hyperparameter_tuning/
    └── tables/
```

The tables contain results for the main models, individual Action Units and AU combinations. Confusion matrices provide class-specific error information while the attention heatmaps support qualitative analysis of the MIL predictions.

One recurring observation across the experiments is that **AU02 (Outer Brow Raiser)** provides a particularly useful individual visual signal. At the same time, the overall results show that facial behaviour alone does not provide an unambiguous signal for every instance of linguistic appraisal. The task therefore remains challenging and the repository should be understood as an investigation of whether facial behaviour contains complementary appraisal-related information rather than as a general-purpose appraisal detector.

## Setup

Clone the repository:

```bash
git clone https://github.com/zimmermanna/TED-Talk-Appraisal-Recognition.git
cd TED-Talk-Appraisal-Recognition
```

Create and activate a virtual environment, for example:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

The main Python dependencies used throughout the project are:

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn jupyter tensorboard
```

Then start Jupyter:

```bash
jupyter notebook
```

> **Note:** The repository currently does not contain a pinned `requirements.txt` or environment file. For exact reproducibility, adding pinned dependency versions is recommended.

## Reproducibility

The MIL pipeline explicitly seeds Python, NumPy and PyTorch. Deterministic PyTorch algorithms are enabled where supported.

The final configuration is stored centrally in:

```text
src/configuration/config.py
```

For comparable results, run the experiments with the stored seed and locked configuration without changing the source-level split strategy.

## Limitations

This project was developed for a bachelor's thesis and should be interpreted as an exploratory study. Important limitations include the relatively small number of source videos, unequal class and source distributions, the TED Talk recording setting and errors that can occur in automatically estimated OpenFace Action Units.

In addition, the attention-based MIL model can identify influential temporal regions but does not explicitly model the chronological order of those regions. Future work could therefore investigate sequence models such as recurrent neural networks or Transformers, larger and more diverse datasets and additional visual signals such as head pose or gaze.
