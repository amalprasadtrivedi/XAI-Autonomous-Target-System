# trust_score.py

from trustscore import TrustScore
import numpy as np
from sklearn.preprocessing import StandardScaler

class TrustScoreEvaluator:
    def __init__(self, k=10, alpha=0.0, filtering="none"):
        """
        Initialize TrustScore evaluator.
        Args:
            k: Number of neighbors to use.
            alpha: Smoothing parameter.
            filtering: "none" or "density" (removes outliers from training data).
        """
        self.trust_model = TrustScore(k=k, alpha=alpha, filtering=filtering)
        self.scaler = StandardScaler()

    def fit(self, train_features, train_labels):
        """
        Fit the TrustScore model using training features and labels.
        Args:
            train_features: numpy array (n_samples, n_features)
            train_labels: numpy array (n_samples,)
        """
        # Normalize
        scaled_train = self.scaler.fit_transform(train_features)
        self.trust_model.fit(scaled_train, train_labels)

    def compute(self, test_features, predicted_labels):
        """
        Compute trust scores for test samples.
        Args:
            test_features: numpy array (n_samples, n_features)
            predicted_labels: predicted class labels (n_samples,)
        Returns:
            trust_scores: list of trust scores for each test sample
        """
        scaled_test = self.scaler.transform(test_features)
        trust_scores = self.trust_model.get_score(scaled_test, predicted_labels)
        return trust_scores
