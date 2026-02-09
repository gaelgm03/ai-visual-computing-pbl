"""
Evaluation module for similarity-based facial recognition systems.

Compatible with:
- Template-based enrollment
- Geometric + descriptor matcher
- Score fusion producing similarity scores in [0,1]

Metrics:
- accuracy
- precision
- recall
- f1_score
- FAR
- TAR
- confusion matrix
- similarities
- predictions

Charts:
- Similarity distribution
- Confusion matrix
- ROC curve (FAR vs TAR)
- Metrics summary
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)