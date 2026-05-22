# Visual Appraisal Recognition in TED Talks Using Facial Expressions

## Overview

> This project investigates wheteher facial expression features can predict the 
presence of linguistic appraisal in public speaking scenarios using deep learning 
methods. 

**Objective:**
Modelling as a binary classification problem that reliably predicts the presence 
of appraisal.
*   0 - Absence
*   1 - Presence

**Dataset:**
The dataset contains data of 5 TED talks *(Last update: 21.05.2026)*. From each 
video about 20 clips where extracted and labeled. The labeling relies on the
previous work of the group of Prof. Lapshinova-Koltunski which already analyzed
these videos on a linguistic level.

---

## Tech Stack

### Languages
- Python

### Libraries / Frameworks
... ToDo ...

### Tools
... ToDo ...

---

## Project Structure (Provisional)

```text
TED-Talk-Appraisal_Recognition/
│
├── data/                  # Datasets and preprocessing
│   ├── raw/
│   │   └── openface/			# Original OpenFace data (.csv)
│   │
│   └── processed/
│       ├── cleaned_openface/	# Filtered, cleaned and merged with labels (.csv)
│       └── aggregated/			# Aggregated frames (.csv)
│
├── notebooks/             		# Experimental notebooks
│
├── src/                   		# Source code
│   ├── 0_preprocessing/
│   │		
│   ├── 1_models/
│   │   ├── baseline.py
│   │   └── mil.py
│   │
│   ├── 2_training/
│   │   ├── train.py
│   │   └── evaluate.py
│   │		
│   ├── 3_evaluation/
│   │
│   ├── 4_visualization/
│   │   ├── plots.py
│   │   └── confusion_matrix.py
│   │		
│   └── utils/
│       └── seed.py
│
├── configs/               	# Config files
│   ├── baseline.yaml
│   └── mil.yaml
│
├── outputs/               	# All metrics, plots, outputs
│   └── figures/
│
├── thesis/					# Most important figures for my thesis
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── README.md
└── .gitignore