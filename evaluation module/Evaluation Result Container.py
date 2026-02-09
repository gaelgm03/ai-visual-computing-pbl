@dataclass
class EvaluationResult:
    similarities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    threshold: float

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float
    tar: float
    confusion_matrix: np.ndarray