# -*- coding: utf-8 -*-
"""
Feature selection using a statistical t-test.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List


class TTestFeatureSelector:
    """
    Selects informative features with an independent two-sample t-test.
    Features whose class-conditional means differ significantly between
    class 0 and class 1 are retained.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: significance level (alpha). Default 0.05.
                Smaller values give a stricter selection.
        """
        self.significance_level = significance_level
        self.selected_features = None
        self.p_values = None
        self.t_statistics = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TTestFeatureSelector':
        """
        Runs a t-test for every feature and keeps the significant ones.

        Args:
            X: feature matrix (samples, features)
            y: binary labels (0 or 1)

        Returns:
            self
        """
        # Split the class-0 and class-1 samples
        class_0_mask = (y == 0)
        class_1_mask = (y == 1)

        X_class_0 = X[class_0_mask]
        X_class_1 = X[class_1_mask]

        print(f"Class 0 samples: {X_class_0.shape[0]}")
        print(f"Class 1 samples: {X_class_1.shape[0]}")

        n_features = X.shape[1]
        p_values = []
        t_stats = []

        # t-test for every feature
        for i in range(n_features):
            feature_class_0 = X_class_0[:, i]
            feature_class_1 = X_class_1[:, i]

            # Drop NaNs if present
            feature_class_0 = feature_class_0[~np.isnan(feature_class_0)]
            feature_class_1 = feature_class_1[~np.isnan(feature_class_1)]

            # Independent two-sample t-test
            t_stat, p_value = stats.ttest_ind(feature_class_0, feature_class_1)

            t_stats.append(t_stat)
            p_values.append(p_value)

        self.p_values = np.array(p_values)
        self.t_statistics = np.array(t_stats)

        # Keep features whose p-value is below the significance level
        self.selected_features = np.where(self.p_values < self.significance_level)[0]

        print(f"\nSelected {len(self.selected_features)} significant features out of {n_features}")
        print(f"   (significance level: {self.significance_level})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Filters the feature matrix, keeping only the significant features.

        Args:
            X: feature matrix

        Returns:
            filtered feature matrix
        """
        if self.selected_features is None:
            raise ValueError("fit() must be called before transform().")

        return X[:, self.selected_features]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Runs fit and transform in one step.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns the feature statistics as a DataFrame.

        Returns:
            DataFrame with columns: feature_idx, t_statistic, p_value,
            abs_t_statistic, is_significant
        """
        if self.p_values is None:
            raise ValueError("fit() must be called before get_feature_importance().")

        df = pd.DataFrame({
            'feature_idx': range(len(self.p_values)),
            't_statistic': self.t_statistics,
            'p_value': self.p_values,
            'abs_t_statistic': np.abs(self.t_statistics),
            'is_significant': self.p_values < self.significance_level
        })

        # Sort by p-value, smallest first
        df = df.sort_values('p_value')

        return df

    def print_summary(self, top_n: int = 10):
        """
        Prints a summary of the selected features.

        Args:
            top_n: number of top features to display
        """
        df = self.get_feature_importance()

        print("\n" + "=" * 70)
        print("t-test feature selection summary")
        print("=" * 70)
        print(f"Significance level (alpha): {self.significance_level}")
        print(f"Total features: {len(self.p_values)}")
        print(f"Significant features: {len(self.selected_features)}")
        print(f"Share of selected features: {100 * len(self.selected_features) / len(self.p_values):.1f}%")

        print(f"\nTop {top_n} features (smallest p-value):")
        print("-" * 70)
        print(df.head(top_n).to_string(index=False))

        if len(self.selected_features) > 0:
            print("\nStatistics of the significant features:")
            significant_df = df[df['is_significant']]
            print(f"   mean |t-statistic|: {significant_df['abs_t_statistic'].mean():.4f}")
            print(f"   mean p-value: {significant_df['p_value'].mean():.6f}")
            print(f"   minimum p-value: {significant_df['p_value'].min():.6e}")

        print("=" * 70 + "\n")


def create_binary_labels(returns: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Converts returns into binary labels.

    Args:
        returns: array of returns
        threshold: split threshold (default 0)
            return > threshold -> class 1 (positive)
            return <= threshold -> class 0 (negative / neutral)

    Returns:
        array of binary labels
    """
    return (returns > threshold).astype(int)
