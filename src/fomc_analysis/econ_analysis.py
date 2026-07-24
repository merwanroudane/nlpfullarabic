# -*- coding: utf-8 -*-
"""
تحليلات اقتصادية وقياسية على مؤشرات النصّ ومسار السياسة النقدية.

Econometric analyses linking text-based indices to the monetary policy path.

يجيب هذا الملفّ عن أسئلة لا يجيب عنها التصنيف وحده — questions classification alone
cannot answer
------------------------------------------------------------------------------------
1. **هل يَسبق النصّ القرار أم يتبعه؟** اختبار سببية غرانجر (Granger Causality)
   يفحص ما إذا كانت القيم الماضية لمؤشّر النصّ تُحسّن التنبؤ بتغيّر السعر بعد
   ضبط ماضي السعر نفسه.
2. **هل يختلف الأداء بين الحقب؟** فترة الحدّ الأدنى الصفري (Zero Lower Bound) تُقيّد
   السعر، فتُنقل السياسة إلى أدوات كمّية ويصير التنبؤ بالسعر مضلّلًا.
3. **هل تختلف أساليب الرؤساء؟** مقارنة مؤشرات النصّ بين ولايات الرؤساء.
4. **ماذا يقول قاعدة تايلور؟** مقارنة السعر الفعلي بما تُوصي به القاعدة.

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

import numpy as np
import pandas as pd

# فترات الحدّ الأدنى الصفري في الولايات المتحدة — US zero lower bound episodes
ZLB_PERIODS = [
    ("2008-12-16", "2015-12-15"),   # بعد الأزمة المالية — post financial crisis
    ("2020-03-15", "2022-03-16"),   # جائحة كوفيد-19 — covid-19 pandemic
]

# فترات الكساد حسب المكتب الوطني للبحوث الاقتصادية — NBER recessions
NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ======================================================================
# 1. السكون والسببية — stationarity and causality
# ======================================================================

def adf_test(series, name=None, regression="c"):
    """اختبار ديكي-فولر الموسَّع لجذر الوحدة.

    Augmented Dickey-Fuller test for a unit root.

    الفرضية الصفرية: السلسلة غير ساكنة (تحتوي جذر وحدة). رفضها شرطٌ لازم قبل
    اختبار غرانجر، لأنّ السلاسل غير الساكنة تُنتج سببية زائفة (Spurious Causality).

    Null hypothesis: the series is non-stationary. Rejecting it is a precondition for
    the Granger test, since non-stationary series produce spurious causality.
    """
    from statsmodels.tsa.stattools import adfuller

    s = pd.Series(series).dropna()
    if len(s) < 12:
        return {"series": name, "n": len(s), "adf_stat": np.nan, "p_value": np.nan,
                "stationary_5pct": None, "ملاحظة": "عدد مشاهدات غير كافٍ"}
    stat, pval, used_lag, nobs, crit, _ = adfuller(s.values, regression=regression,
                                                   autolag="AIC")
    return {
        "series": name if name else getattr(series, "name", "series"),
        "n": int(nobs),
        "lags": int(used_lag),
        "adf_stat": round(float(stat), 4),
        "p_value": round(float(pval), 4),
        "crit_5pct": round(float(crit["5%"]), 4),
        "stationary_5pct": bool(pval < 0.05),
    }


def granger_causality(cause, effect, maxlag=4, difference=False, verbose=True):
    """اختبار سببية غرانجر من ``cause`` إلى ``effect``.

    Granger causality test from ``cause`` to ``effect``.

    الفرضية الصفرية: ماضي ``cause`` لا يُضيف قدرة تفسيرية على ماضي ``effect`` وحده.
    القيمة الاحتمالية الصغيرة تعني رفض الصفرية، أي أنّ المؤشّر النصّي يَسبق القرار.

    Null hypothesis: the past of ``cause`` adds no explanatory power beyond the past of
    ``effect`` itself. A small p-value rejects it, meaning the text index leads the
    decision.

    ⚠️ سببية غرانجر سببيةٌ تنبؤية لا سببية هيكلية: تقول إنّ المتغيّر يَسبق، لا إنّه
    يُسبِّب. — Granger causality is predictive, not structural: it says a variable
    leads, not that it causes.

    المعاملات — Parameters
    ----------
    difference : bool
        أخذ الفرق الأول لتحقيق السكون قبل الاختبار — first-difference for stationarity.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    df = pd.concat([pd.Series(effect).rename("effect"),
                    pd.Series(cause).rename("cause")], axis=1).dropna()
    if difference:
        df = df.diff().dropna()

    if len(df) < maxlag * 3 + 10:
        return pd.DataFrame([{"lag": np.nan, "p_value": np.nan,
                              "ملاحظة": f"عدد مشاهدات غير كافٍ ({len(df)})"}])

    res = grangercausalitytests(df[["effect", "cause"]].values, maxlag=maxlag,
                                verbose=False)
    rows = []
    for lag in range(1, maxlag + 1):
        ftest = res[lag][0]["ssr_ftest"]
        rows.append({"lag": lag,
                     "F_stat": round(float(ftest[0]), 4),
                     "p_value": round(float(ftest[1]), 4),
                     "significant_5pct": bool(ftest[1] < 0.05)})
    out = pd.DataFrame(rows)
    if verbose:
        sig = out[out["significant_5pct"]]
        if len(sig):
            print(f"  سببية غرانجر معنوية عند الإبطاء: {list(sig['lag'])} — significant at lags")
        else:
            print("  لا سببية غرانجر معنوية عند أيّ إبطاء — no significant causality")
    return out


def lead_lag_correlation(x, y, max_lag=6):
    """ارتباط متقاطع بإزاحات موجبة وسالبة لكشف أيّ السلسلتين تَسبق.

    Cross-correlation at positive and negative shifts to reveal which series leads.

    الإزاحة الموجبة تعني أنّ ``x`` يَسبق ``y``. — A positive lag means ``x`` leads ``y``.
    """
    sx = pd.Series(x).astype(float)
    sy = pd.Series(y).astype(float)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        corr = sx.shift(lag).corr(sy)
        rows.append({"lag": lag, "correlation": round(float(corr), 4) if pd.notna(corr) else np.nan,
                     "التفسير": ("x يَسبق y" if lag > 0 else ("y يَسبق x" if lag < 0 else "متزامن"))})
    return pd.DataFrame(rows)


# ======================================================================
# 2. تحليل الحقب — regime analysis
# ======================================================================

def tag_regime(dates, zlb_periods=None, recessions=None):
    """وسم كل تاريخ بحقبته: حدّ صفري، كساد، أو اعتيادي.

    Tag each date with its regime: zero lower bound, recession, or normal.
    """
    zlb_periods = zlb_periods if zlb_periods is not None else ZLB_PERIODS
    recessions = recessions if recessions is not None else NBER_RECESSIONS

    idx = pd.to_datetime(pd.Series(dates).values)
    out = pd.DataFrame({"date": idx})
    out["zlb"] = False
    out["recession"] = False

    for start, end in zlb_periods:
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
        out.loc[mask, "zlb"] = True
    for start, end in recessions:
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
        out.loc[mask, "recession"] = True

    out["regime"] = np.where(out["zlb"], "الحدّ الصفري",
                             np.where(out["recession"], "كساد", "اعتيادي"))
    return out


def performance_by_regime(pred_df, date_col="date", true_col="y_true",
                          pred_col="y_pred"):
    """تفصيل أداء النموذج بحسب الحقبة الاقتصادية.

    Break down model performance by economic regime.

    يكشف هذا التفصيل ما يُخفيه المتوسط: النموذج قد يبدو جيّدًا لأنّه يُصيب في فترة
    الحدّ الصفري حيث القرار «تثبيت» شبه دائم، ويفشل حين يكون القرار فعليًّا متغيّرًا.

    This breakdown reveals what the average hides: a model may look good because it is
    right during the zero lower bound, when the decision is almost always Hold, yet
    fail when the decision genuinely varies.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    df = pred_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    tags = tag_regime(df[date_col])
    df = df.reset_index(drop=True)
    df["regime"] = tags["regime"].values

    rows = []
    for regime, grp in df.groupby("regime"):
        if len(grp) < 3:
            continue
        rows.append({
            "الحقبة": regime,
            "n": len(grp),
            "accuracy": round(accuracy_score(grp[true_col], grp[pred_col]), 4),
            "balanced_accuracy": round(balanced_accuracy_score(grp[true_col], grp[pred_col]), 4),
            "f1_macro": round(f1_score(grp[true_col], grp[pred_col], average="macro",
                                       zero_division=0), 4),
            "نسبة التثبيت الفعلية": round(float((grp[true_col] == 0).mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ======================================================================
# 3. قاعدة تايلور — the Taylor rule
# ======================================================================

def taylor_rule(inflation, output_gap, r_star=2.0, pi_star=2.0,
                a_pi=0.5, a_y=0.5):
    """قاعدة تايلور المعيارية.

    The standard Taylor rule.

        i = r* + π + a_π (π − π*) + a_y × فجوة الناتج

    المعاملات — Parameters
    ----------
    r_star : float
        سعر الفائدة الحقيقي التوازني — the equilibrium real rate.
    pi_star : float
        هدف التضخّم — the inflation target.
    """
    pi = pd.Series(inflation).astype(float)
    gap = pd.Series(output_gap).astype(float)
    return r_star + pi + a_pi * (pi - pi_star) + a_y * gap


def balanced_approach_rule(inflation, output_gap, r_star=2.0, pi_star=2.0):
    """قاعدة المقاربة المتوازنة: تُضاعف وزن فجوة الناتج.

    The balanced-approach rule: it doubles the weight on the output gap.
    """
    return taylor_rule(inflation, output_gap, r_star, pi_star, a_pi=0.5, a_y=1.0)


def inertia_rule(prev_rate, inflation, output_gap, rho=0.85, r_star=2.0, pi_star=2.0):
    """قاعدة القصور الذاتي: تُمهّد السعر بربطه بمستواه السابق.

    The inertial rule: it smooths the rate by anchoring it to its own previous level.

    تعكس هذه القاعدة سلوك البنوك المركزية الفعلي، فهي تُحرّك السعر تدريجيًّا
    لا قفزات. — This rule reflects actual central bank behaviour: rates move gradually.
    """
    prescribed = taylor_rule(inflation, output_gap, r_star, pi_star)
    return rho * pd.Series(prev_rate).astype(float) + (1 - rho) * prescribed


def taylor_gap_table(actual_rate, inflation, output_gap, prev_rate=None):
    """جدول يقارن السعر الفعلي بما توصي به القواعد الثلاث.

    A table comparing the actual rate with what the three rules prescribe.

    الفرق الموجب يعني أنّ السياسة أكثر تساهلًا مما توصي به القاعدة.
    A positive gap means policy is more accommodative than the rule prescribes.
    """
    out = pd.DataFrame({"actual": pd.Series(actual_rate).astype(float)})
    out["taylor"] = taylor_rule(inflation, output_gap)
    out["balanced"] = balanced_approach_rule(inflation, output_gap)
    if prev_rate is not None:
        out["inertia"] = inertia_rule(prev_rate, inflation, output_gap)
    rule_cols = [c for c in out.columns if c != "actual"]
    for col in rule_cols:
        # الفرق الموجب: القاعدة توصي بسعر أعلى من الفعلي، أي أنّ السياسة أكثر تساهلًا
        # positive gap: the rule prescribes a higher rate than actual, i.e. policy is
        # more accommodative than the rule
        out[f"gap_{col}"] = out[col] - out["actual"]
    return out


# ======================================================================
# 4. المقارنة بين الرؤساء ودراسة الأحداث — chairs and event study
# ======================================================================

def compare_chairs(metrics_df, chair_col="speaker", metrics=None, min_docs=5):
    """مقارنة إحصائية لمؤشرات النصّ بين رؤساء المجلس.

    Statistical comparison of text indices across Fed chairs.

    تُعيد المتوسط والانحراف وعدد الوثائق، مع اختبار كروسكال-واليس لمعنوية الفرق.
    Returns mean, standard deviation and document count, plus a Kruskal-Wallis test of
    whether the differences are significant.
    """
    from scipy.stats import kruskal

    cols = metrics if metrics else [c for c in metrics_df.columns
                                    if metrics_df[c].dtype.kind in "fc"]
    counts = metrics_df[chair_col].value_counts()
    keep = counts[counts >= min_docs].index
    df = metrics_df[metrics_df[chair_col].isin(keep)]

    summary = df.groupby(chair_col)[cols].agg(["mean", "std", "count"]).round(3)

    tests = []
    for col in cols:
        groups = [g[col].dropna().values for _, g in df.groupby(chair_col)]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            stat, pval = kruskal(*groups)
            tests.append({"المؤشّر": col, "H_stat": round(float(stat), 4),
                          "p_value": round(float(pval), 4),
                          "فرق معنوي 5%": bool(pval < 0.05)})
        except ValueError:
            continue
    return summary, pd.DataFrame(tests)


def event_study(series, event_dates, window=5, normalise=True):
    """دراسة أحداث حول تواريخ الإعلانات.

    Event study around announcement dates.

    تحسب متوسط مسار المتغيّر في النافذة المحيطة بكل حدث، لقياس ردّ فعل السوق
    قبل الإعلان وبعده.

    Computes the average path of a variable in the window around each event, measuring
    the market reaction before and after the announcement.

    المعاملات — Parameters
    ----------
    window : int
        عدد المشاهدات قبل الحدث وبعده — observations before and after the event.
    normalise : bool
        طرح قيمة يوم الحدث ليبدأ كل مسار من الصفر — subtract the event-day value.
    """
    s = pd.Series(series).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    paths = []
    for ev in pd.to_datetime(pd.Series(event_dates).dropna().values):
        pos = s.index.searchsorted(ev)
        if pos - window < 0 or pos + window >= len(s):
            continue
        seg = s.iloc[pos - window: pos + window + 1].copy()
        vals = seg.values.astype(float)
        if normalise:
            vals = vals - vals[window]
        paths.append(vals)

    if not paths:
        return pd.DataFrame(columns=["offset", "mean", "std", "n"])

    arr = np.vstack(paths)
    offsets = np.arange(-window, window + 1)
    return pd.DataFrame({
        "offset": offsets,
        "mean": arr.mean(axis=0).round(4),
        "std": arr.std(axis=0).round(4),
        "n": arr.shape[0],
    })


def summarise_decisions(decision_series, date_index=None):
    """ملخّص توزيع القرارات وتكرار التغيير عبر الزمن.

    Summarise the distribution of decisions and how often policy changes.
    """
    s = pd.Series(decision_series).dropna().astype(int)
    if date_index is not None:
        s.index = pd.to_datetime(date_index)

    labels = {-1: "تخفيض", 0: "تثبيت", 1: "رفع"}
    counts = s.value_counts().sort_index()
    total = len(s)
    rows = [{"القرار": labels.get(k, k), "العدد": int(v),
             "النسبة %": round(100 * v / total, 2)} for k, v in counts.items()]
    rows.append({"القرار": "الإجمالي", "العدد": total, "النسبة %": 100.0})
    dist = pd.DataFrame(rows)

    changes = int((s != 0).sum())
    extra = {
        "عدد الاجتماعات": total,
        "عدد اجتماعات التغيير": changes,
        "نسبة التغيير %": round(100 * changes / total, 2),
        "أطول سلسلة تثبيت": int((s == 0).astype(int)
                                .groupby((s != 0).cumsum()).sum().max()) if total else 0,
    }
    return dist, extra
