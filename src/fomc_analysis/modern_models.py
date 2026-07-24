# -*- coding: utf-8 -*-
"""
نماذج وخوارزميات تصنيف حديثة للتنبؤ بقرار سعر الفائدة، مع تحقّق زمني سليم.

Modern classification models and algorithms for predicting the rate decision, with a
methodologically sound time-series validation scheme.

ثلاث مشكلات منهجية تعالجها هذه الوحدة — three methodological problems addressed
-----------------------------------------------------------------------------
1. **التحقّق الزمني (Walk-Forward Validation).** التحقّق المتقاطع الاعتيادي يخلط
   الترتيب الزمني فيسمح للنموذج بأن يتدرّب على المستقبل ويتنبّأ بالماضي، وهو تسرّب
   للمعلومة (Look-ahead Bias). البديل هنا نافذة متوسّعة (Expanding Window): يتدرّب
   النموذج على كل ما سبق ويُختبر على ما يليه، مرّة بعد مرّة.

2. **الترتيب الطبيعي للفئات (Ordinality).** الفئات ليست اسمية: تخفيض < تثبيت < رفع.
   الخطأ الذي يتنبّأ برفعٍ بدل تخفيضٍ أشدّ من الخطأ الذي يتنبّأ بتثبيتٍ بدل تخفيض.
   يعالج ذلك مُصنِّف رتبي (Ordinal Classifier) بتفكيك تراكمي على طريقة فرانك وهول.

3. **عدم توازن الفئات (Class Imbalance).** فئة التثبيت تفوق غيرها بكثير، فالتنبؤ
   الساذج بها وحدها يعطي دقّة مرتفعة زائفة. تُعالَج بأوزان الفئات لا بحذف بيانات.

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

#: ترتيب فئات القرار من الأدنى إلى الأعلى — the decision classes, in order
DECISION_ORDER = [-1, 0, 1]

#: أسماء الفئات بالعربية والإنجليزية — class labels
DECISION_LABELS = {
    -1: ("تخفيض", "Lower"),
    0: ("تثبيت", "Hold"),
    1: ("رفع", "Raise"),
}

#: أسماء المقاييس بالعربية — metric labels in Arabic
METRIC_LABELS_AR = {
    "accuracy": "الدقّة",
    "balanced_accuracy": "الدقّة المتوازنة",
    "f1_macro": "مقياس F1 الكلّي",
    "f1_weighted": "مقياس F1 الموزون",
    "mae_ordinal": "الخطأ المطلق الرتبي",
    "n_train": "حجم التدريب",
    "n_test": "حجم الاختبار",
}


# ======================================================================
# 1. التحقّق الزمني المتدرّج — walk-forward validation
# ======================================================================

def walk_forward_splits(n_samples, n_splits=5, min_train=None, gap=0):
    """توليد تقسيمات زمنية بنافذة متوسّعة.

    Generate expanding-window time-series splits.

    كل تقسيم يتدرّب على كل المشاهدات السابقة ويُختبر على الكتلة التالية، فلا
    يرى النموذج أبدًا معلومة من المستقبل.

    Each split trains on all earlier observations and tests on the next block, so the
    model never sees information from the future.

    المعاملات — Parameters
    ----------
    gap : int
        عدد المشاهدات المُستبعَدة بين التدريب والاختبار، لتفادي التسرّب عند وجود
        خصائص محسوبة بنوافذ متحرّكة.
        Observations skipped between train and test, to avoid leakage when features
        are built from rolling windows.

    تُعيد — Yields
    --------------
    (train_idx, test_idx) : tuple of ndarray
    """
    if min_train is None:
        min_train = max(int(n_samples * 0.4), 20)
    if min_train + gap >= n_samples:
        raise ValueError(
            f"حجم التدريب الأدنى ({min_train}) مع الفجوة ({gap}) يستهلك كل المشاهدات ({n_samples})")

    remaining = n_samples - min_train - gap
    fold = max(remaining // n_splits, 1)

    for i in range(n_splits):
        train_end = min_train + i * fold
        test_start = train_end + gap
        test_end = test_start + fold if i < n_splits - 1 else n_samples
        if test_start >= n_samples:
            break
        yield np.arange(0, train_end), np.arange(test_start, min(test_end, n_samples))


# ======================================================================
# 2. المُصنِّف الرتبي — ordinal classifier
# ======================================================================

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """مُصنِّف رتبي بتفكيك تراكمي (فرانك وهول 2001).

    Ordinal classifier using the cumulative binary decomposition of Frank & Hall (2001).

    الفكرة — the idea
    -----------------
    لفئات مرتّبة عددها K، نُدرّب K−1 مُصنِّفًا ثنائيًّا، يقدّر المُصنِّف رقم k
    الاحتمال ``P(y > k)``. ثمّ يُستخرَج احتمال كل فئة بالفروق:

        P(y = الفئة الأولى) = 1 − P(y > 1)
        P(y = الفئة k)      = P(y > k−1) − P(y > k)
        P(y = الفئة الأخيرة) = P(y > K−1)

    بهذا يستفيد النموذج من كون التخفيض والرفع طرفَين متقابلَين والتثبيت بينهما،
    وهو ما يتجاهله التصنيف الاسمي تمامًا.

    For K ordered classes, K−1 binary classifiers estimate ``P(y > k)``; the per-class
    probabilities follow from the differences. The model thereby exploits the fact that
    Lower and Raise are opposite extremes with Hold in between, which nominal
    classification ignores entirely.
    """

    def __init__(self, base_estimator=None, classes_order=None):
        self.base_estimator = base_estimator
        self.classes_order = classes_order

    def fit(self, X, y):
        self.classes_ = np.array(
            self.classes_order if self.classes_order is not None else sorted(np.unique(y)))
        if len(self.classes_) < 3:
            warnings.warn("المُصنِّف الرتبي يفترض ثلاث فئات على الأقل — fewer than 3 classes")

        base = self.base_estimator if self.base_estimator is not None else \
            LogisticRegression(max_iter=1000, class_weight="balanced")

        y = np.asarray(y)
        self.estimators_ = []
        # مُصنِّف لكل حدّ فصل بين الفئات — one classifier per cut point
        for k in range(len(self.classes_) - 1):
            threshold = self.classes_[k]
            y_bin = (y > threshold).astype(int)
            est = clone(base)
            est.fit(X, y_bin)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        # P(y > k) لكل حدّ — the cumulative probabilities
        cum = np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])
        # ضمان عدم تزايد الاحتمالات التراكمية بسبب اختلاف المُصنِّفات
        # enforce monotonically non-increasing cumulative probabilities
        cum = np.minimum.accumulate(cum, axis=1)

        n, k = cum.shape
        probs = np.zeros((n, k + 1))
        probs[:, 0] = 1.0 - cum[:, 0]
        for j in range(1, k):
            probs[:, j] = cum[:, j - 1] - cum[:, j]
        probs[:, k] = cum[:, k - 1]

        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ======================================================================
# 3. مكتبة النماذج — the model zoo
# ======================================================================

class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """غلاف يُرمِّز الفئات إلى 0..K-1 ثمّ يعيدها إلى قيمها الأصلية.

    Wrapper that encodes classes to 0..K-1 and decodes predictions back.

    تشترط بعض المكتبات مثل 'XGBoost' أن تكون الفئات أعدادًا صحيحة تبدأ من الصفر،
    فترفض التسميات ‎-1‎ و0 و1 المستخدمة هنا. يعالج هذا الغلاف ذلك دون تغيير
    التسميات في بقيّة الشيفرة.

    Libraries such as XGBoost require zero-indexed integer classes and reject the
    -1/0/1 labels used here. This wrapper handles that without changing the labels
    anywhere else.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(np.unique(y)))
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([self._to_idx[v] for v in y])
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_enc)
        return self

    def predict(self, X):
        return self.classes_[self.estimator_.predict(X).astype(int)]

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _lightgbm(**kw):
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                  min_child_samples=5, subsample=0.9, colsample_bytree=0.8,
                  class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def _catboost(**kw):
    from catboost import CatBoostClassifier
    params = dict(iterations=300, learning_rate=0.05, depth=4,
                  loss_function="MultiClass", auto_class_weights="Balanced",
                  random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False)
    params.update(kw)
    return CatBoostClassifier(**params)


def _xgboost(**kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=3,
                  subsample=0.9, colsample_bytree=0.8, random_state=RANDOM_STATE,
                  eval_metric="mlogloss", n_jobs=4)
    params.update(kw)
    return XGBClassifier(**params)


def model_zoo(include_ordinal=True):
    """بناء قاموس النماذج المتاحة للمقارنة.

    Build the dictionary of models available for comparison.

    النماذج التي تحتاج مكتبات غير مثبَّتة تُستبعَد بهدوء مع تنبيه.
    Models needing uninstalled libraries are skipped quietly with a notice.
    """
    zoo = {}

    # نماذج لا تحتاج تبعيات إضافية — no extra dependencies
    zoo["random_forest"] = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=4)
    zoo["extra_trees"] = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=4)
    zoo["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        class_weight="balanced", random_state=RANDOM_STATE)
    zoo["logistic"] = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    # نماذج التعزيز الحديثة — modern boosting libraries
    for key, factory in (("lightgbm", _lightgbm), ("catboost", _catboost)):
        try:
            zoo[key] = factory()
        except ImportError:
            print(f"تُجوهل {key}: المكتبة غير مثبَّتة — library not installed")

    # 'XGBoost' يشترط فئات تبدأ من الصفر، فنغلّفه بمُرمِّز التسميات
    # XGBoost requires zero-indexed classes, so wrap it in the label encoder
    try:
        zoo["xgboost"] = LabelEncodedClassifier(_xgboost())
    except ImportError:
        print("تُجوهل xgboost: المكتبة غير مثبَّتة — library not installed")

    if include_ordinal:
        # المُصنِّف الرتبي فوق أساسَين مختلفَين — the ordinal wrapper on two bases
        zoo["ordinal_logistic"] = OrdinalClassifier(
            base_estimator=Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]),
            classes_order=DECISION_ORDER)
        try:
            zoo["ordinal_lightgbm"] = OrdinalClassifier(
                base_estimator=_lightgbm(n_estimators=200),
                classes_order=DECISION_ORDER)
        except ImportError:
            pass

    return zoo


# ======================================================================
# 4. التقييم — evaluation
# ======================================================================

def ordinal_mae(y_true, y_pred):
    """الخطأ المطلق الرتبي: يعاقب الخطأ بقدر بُعده على سُلّم الفئات.

    Ordinal mean absolute error: penalises an error by its distance on the class scale.

    التنبؤ برفع بدل تخفيض خطؤه 2، والتنبؤ بتثبيت بدل تخفيض خطؤه 1.
    Predicting Raise instead of Lower costs 2; predicting Hold instead of Lower costs 1.
    """
    return mean_absolute_error(np.asarray(y_true, dtype=float),
                               np.asarray(y_pred, dtype=float))


def _fold_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mae_ordinal": ordinal_mae(y_true, y_pred),
    }


def evaluate_walk_forward(X, y, model, n_splits=5, min_train=None, gap=0,
                          return_predictions=False):
    """تقييم نموذج بالتحقّق الزمني المتدرّج.

    Evaluate a model with walk-forward validation.

    تُعيد — Returns
    -------
    dict
        يضمّ ``folds`` (إطار نتائج كل طيّة)، و``mean`` و``std`` للمقاييس،
        و``predictions`` إن طُلبت.
        Contains ``folds``, the per-fold results, plus ``mean``/``std`` and optionally
        the out-of-sample ``predictions``.
    """
    X = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y)
    n = len(y)

    rows, preds = [], []
    for fold, (tr, te) in enumerate(walk_forward_splits(n, n_splits, min_train, gap), 1):
        Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        Xte = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
        est = clone(model)
        est.fit(Xtr, y[tr])
        yp = est.predict(Xte)

        rec = {"fold": fold, "n_train": len(tr), "n_test": len(te)}
        rec.update(_fold_metrics(y[te], yp))
        rows.append(rec)

        if return_predictions:
            preds.append(pd.DataFrame({"fold": fold, "index": te,
                                       "y_true": y[te], "y_pred": yp}))

    folds = pd.DataFrame(rows)
    metric_cols = [c for c in folds.columns if c not in ("fold", "n_train", "n_test")]
    out = {"folds": folds,
           "mean": folds[metric_cols].mean().to_dict(),
           "std": folds[metric_cols].std().to_dict()}
    if return_predictions:
        out["predictions"] = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return out


def compare_models(X, y, models=None, n_splits=5, min_train=None, gap=0, verbose=True):
    """مقارنة عدّة نماذج بالتحقّق الزمني نفسه وإعادة جدول مرتّب.

    Compare several models under the same walk-forward scheme and return a ranked table.
    """
    zoo = models if models else model_zoo()
    rows = []
    for name, model in zoo.items():
        try:
            res = evaluate_walk_forward(X, y, model, n_splits, min_train, gap)
            rec = {"model": name}
            rec.update({k: round(v, 4) for k, v in res["mean"].items()})
            rec["f1_macro_std"] = round(res["std"]["f1_macro"], 4)
            rows.append(rec)
            if verbose:
                print(f"  {name:24} F1={rec['f1_macro']:.3f}  "
                      f"دقّة متوازنة={rec['balanced_accuracy']:.3f}  "
                      f"خطأ رتبي={rec['mae_ordinal']:.3f}")
        except Exception as err:  # noqa: BLE001
            if verbose:
                print(f"  {name:24} فشل — failed: {type(err).__name__}: {err}")
    return (pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
            .reset_index(drop=True))


def naive_baselines(y):
    """حساب أداء القواعد الساذجة، وهي الحدّ الذي يجب على النموذج تجاوزه.

    Compute naive rule performance, the bar any model must clear.

    القاعدة الأولى: تنبّأ دائمًا بالفئة الأغلب (التثبيت).
    الثانية: تنبّأ بما حدث في الاجتماع السابق (الاستمرارية).

    Rule one: always predict the majority class (Hold).
    Rule two: predict the previous meeting's decision (persistence).
    """
    y = np.asarray(y)
    majority = pd.Series(y).mode().iloc[0]
    rows = []

    y_major = np.full_like(y, majority)
    rec = {"rule": "الفئة الأغلب — majority class"}
    rec.update(_fold_metrics(y, y_major))
    rows.append(rec)

    y_persist = np.concatenate([[majority], y[:-1]])
    rec = {"rule": "الاستمرارية — persistence"}
    rec.update(_fold_metrics(y, y_persist))
    rows.append(rec)

    return pd.DataFrame(rows)


# ======================================================================
# 5. التفسير والمعايرة — interpretation and calibration
# ======================================================================

def shap_importance(model, X, feature_names=None, max_display=20, sample=None):
    """حساب أهمية الخصائص بقيم شابلي (SHAP).

    Compute feature importance with SHAP values.

    تتجاوز قيم شابلي أهميّة الخصائص التقليدية في الأشجار لأنّها تُنسِب لكلّ خصيصة
    أثرها في كلّ تنبؤ على حِدة، فتكشف الاتجاه لا الحجم فقط.

    SHAP goes beyond tree-based feature importance by attributing to each feature its
    contribution to every individual prediction, revealing direction as well as
    magnitude.

    تُعيد — Returns
    -------
    (importance_df, shap_values)
    """
    import shap

    Xd = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
        X, columns=feature_names if feature_names is not None
        else [f"f{i}" for i in range(np.asarray(X).shape[1])])
    if sample and len(Xd) > sample:
        Xd = Xd.sample(sample, random_state=RANDOM_STATE).sort_index()

    # فحص الجمعية (Additivity Check) يفشل أحيانًا في نماذج التعزيز متعدّدة الفئات
    # لأسباب عددية بحتة، فنعيد المحاولة بتعطيله بدل أن نفقد التفسير كلّه.
    # The additivity check sometimes fails on multi-class boosting models for purely
    # numerical reasons; retry with it disabled rather than losing the explanation.
    values = None
    try:
        values = shap.TreeExplainer(model)(Xd)
    except Exception as tree_err:  # noqa: BLE001
        try:
            values = shap.TreeExplainer(model)(Xd, check_additivity=False)
            print("تنبيه: عُطّل فحص الجمعية في SHAP — additivity check disabled")
        except Exception:  # noqa: BLE001 - ليس نموذجًا شجريًّا
            values = shap.Explainer(model, Xd)(Xd)
            del tree_err

    arr = np.abs(values.values)
    # للتصنيف متعدّد الفئات تكون الأبعاد (مشاهدات, خصائص, فئات)
    # for multi-class the array is (samples, features, classes)
    while arr.ndim > 2:
        arr = arr.mean(axis=-1)
    mean_abs = arr.mean(axis=0)

    imp = (pd.DataFrame({"feature": Xd.columns, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False)
           .head(max_display).reset_index(drop=True))
    return imp, values


def calibration_report(y_true, probs, classes=None, n_bins=10):
    """فحص معايرة الاحتمالات: هل احتمال 0.7 يعني فعلًا تحقّقًا في 70% من الحالات؟

    Check probability calibration: does a predicted 0.7 really occur 70% of the time?

    المعايرة مهمّة في السياسة النقدية لأنّ المستخدم لا يريد الفئة المتوقّعة فقط،
    بل درجة الثقة فيها.

    Calibration matters for monetary policy because a user wants not only the predicted
    class but a trustworthy confidence in it.
    """
    classes = classes if classes is not None else DECISION_ORDER
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rows = []
    for i, cls in enumerate(classes):
        if i >= probs.shape[1]:
            break
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        bins = min(n_bins, max(int(y_bin.sum()), 2))
        frac_pos, mean_pred = calibration_curve(y_bin, probs[:, i], n_bins=bins,
                                                strategy="quantile")
        for fp, mp in zip(frac_pos, mean_pred):
            rows.append({"class": DECISION_LABELS.get(cls, (cls, cls))[0],
                         "mean_predicted": mp, "observed_frequency": fp})
    return pd.DataFrame(rows)


def confusion_table(y_true, y_pred, classes=None):
    """مصفوفة الالتباس بأسماء الفئات العربية.

    Confusion matrix with Arabic class names.
    """
    classes = classes if classes is not None else DECISION_ORDER
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    names = [DECISION_LABELS.get(c, (str(c), str(c)))[0] for c in classes]
    return pd.DataFrame(cm,
                        index=[f"فعلي: {n}" for n in names],
                        columns=[f"متوقّع: {n}" for n in names])
