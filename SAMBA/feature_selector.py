# -*- coding: utf-8 -*-
"""
Feature selection using statistical t-test
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List


class TTestFeatureSelector:
    """
    انتخاب فیچرهای مهم با استفاده از t-test
    فیچرهایی که اختلاف معنی‌داری بین کلاس 0 و 1 دارن رو نگه می‌داره
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: سطح معنی‌داری (آلفا). پیش‌فرض 0.05
                               هرچی کوچیک‌تر باشه، سخت‌گیرتره!
        """
        self.significance_level = significance_level
        self.selected_features = None
        self.p_values = None
        self.t_statistics = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TTestFeatureSelector':
        """
        محاسبه t-test برای هر فیچر و انتخاب فیچرهای معنی‌دار

        Args:
            X: ماتریس فیچرها (samples, features)
            y: برچسب‌های باینری (0 یا 1)

        Returns:
            self
        """
        # جدا کردن نمونه‌های کلاس 0 و 1
        class_0_mask = (y == 0)
        class_1_mask = (y == 1)

        X_class_0 = X[class_0_mask]
        X_class_1 = X[class_1_mask]

        print(f"📊 تعداد نمونه‌های کلاس 0: {X_class_0.shape[0]}")
        print(f"📊 تعداد نمونه‌های کلاس 1: {X_class_1.shape[0]}")

        n_features = X.shape[1]
        p_values = []
        t_stats = []

        # محاسبه t-test برای هر فیچر
        for i in range(n_features):
            feature_class_0 = X_class_0[:, i]
            feature_class_1 = X_class_1[:, i]

            # حذف NaN ها اگه وجود داشته باشه
            feature_class_0 = feature_class_0[~np.isnan(feature_class_0)]
            feature_class_1 = feature_class_1[~np.isnan(feature_class_1)]

            # t-test دو نمونه‌ای مستقل
            t_stat, p_value = stats.ttest_ind(feature_class_0, feature_class_1)

            t_stats.append(t_stat)
            p_values.append(p_value)

        self.p_values = np.array(p_values)
        self.t_statistics = np.array(t_stats)

        # انتخاب فیچرهایی که p-value کمتر از سطح معنی‌داری دارن
        self.selected_features = np.where(self.p_values < self.significance_level)[0]

        print(f"\n✅ از {n_features} فیچر، {len(self.selected_features)} فیچر معنی‌دار انتخاب شد")
        print(f"   (سطح معنی‌داری: {self.significance_level})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        فیلتر کردن فیچرها و نگه داشتن فقط فیچرهای معنی‌دار

        Args:
            X: ماتریس فیچرها

        Returns:
            ماتریس فیلتر شده
        """
        if self.selected_features is None:
            raise ValueError("ابتدا باید fit رو صدا بزنی!")

        return X[:, self.selected_features]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        fit و transform رو با هم انجام می‌ده
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        برگردوندن اهمیت فیچرها به صورت DataFrame

        Returns:
            DataFrame با ستون‌های: feature_idx, t_statistic, p_value, is_significant
        """
        if self.p_values is None:
            raise ValueError("ابتدا باید fit رو صدا بزنی!")

        df = pd.DataFrame({
            'feature_idx': range(len(self.p_values)),
            't_statistic': self.t_statistics,
            'p_value': self.p_values,
            'abs_t_statistic': np.abs(self.t_statistics),
            'is_significant': self.p_values < self.significance_level
        })

        # مرتب‌سازی بر اساس p-value (کوچک‌ترین اول)
        df = df.sort_values('p_value')

        return df

    def print_summary(self, top_n: int = 10):
        """
        چاپ خلاصه‌ای از فیچرهای مهم

        Args:
            top_n: تعداد فیچرهای برتر برای نمایش
        """
        df = self.get_feature_importance()

        print("\n" + "=" * 70)
        print(f"📈 خلاصه انتخاب فیچر با t-test")
        print("=" * 70)
        print(f"سطح معنی‌داری (α): {self.significance_level}")
        print(f"تعداد کل فیچرها: {len(self.p_values)}")
        print(f"تعداد فیچرهای معنی‌دار: {len(self.selected_features)}")
        print(f"درصد فیچرهای انتخاب شده: {100 * len(self.selected_features) / len(self.p_values):.1f}%")

        print(f"\n🏆 {top_n} فیچر برتر (با کمترین p-value):")
        print("-" * 70)
        print(df.head(top_n).to_string(index=False))

        if len(self.selected_features) > 0:
            print(f"\n📊 آمار فیچرهای معنی‌دار:")
            significant_df = df[df['is_significant']]
            print(f"   میانگین |t-statistic|: {significant_df['abs_t_statistic'].mean():.4f}")
            print(f"   میانگین p-value: {significant_df['p_value'].mean():.6f}")
            print(f"   کمترین p-value: {significant_df['p_value'].min():.6e}")

        print("=" * 70 + "\n")


def create_binary_labels(returns: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    تبدیل بازده‌ها به برچسب‌های باینری

    Args:
        returns: آرایه بازده‌ها
        threshold: آستانه برای تقسیم‌بندی (پیش‌فرض 0)
                  بازده > threshold -> کلاس 1 (مثبت)
                  بازده <= threshold -> کلاس 0 (منفی/خنثی)

    Returns:
        آرایه برچسب‌های باینری
    """
    return (returns > threshold).astype(int)
