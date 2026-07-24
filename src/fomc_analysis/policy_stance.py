# -*- coding: utf-8 -*-
"""
قياس موقف السياسة النقدية (متشدّد / متساهل / محايد) بنماذج المحوّلات على مستوى الجملة.

Measuring the monetary policy stance (hawkish / dovish / neutral) with transformer
models at sentence level.

الأساس المنهجي — methodological background
------------------------------------------
تحليل المشاعر العام (إيجابي/سلبي) لا يكفي لنصوص البنوك المركزية، لأنّ الخبر
الاقتصادي «الجيّد» قد يعني سياسة أكثر تشدّدًا. لذلك يُقاس الموقف على محور
متشدّد–متساهل (Hawkish–Dovish) لا على محور إيجابي–سلبي:

* **متشدّد (Hawkish):** ميل إلى تقييد السياسة، أي رفع سعر الفائدة، عادةً استجابةً
  لضغوط تضخّمية.
* **متساهل (Dovish):** ميل إلى تخفيف السياسة، أي تخفيض سعر الفائدة، عادةً استجابةً
  لضعف النشاط أو ارتفاع البطالة.
* **محايد (Neutral):** عبارات وصفية أو إجرائية لا تحمل إشارة سياسة.

General positive/negative sentiment is inadequate for central bank text, because
"good" economic news can imply tighter policy. The stance is therefore measured on a
hawkish–dovish axis rather than a positive–negative one.

ثمّ يُبنى **مؤشّر صافي التشدّد (Net Hawkishness)** لكلّ وثيقة أو اجتماع:

    net_hawkishness = (عدد الجُمل المتشدّدة − عدد الجُمل المتساهلة) / إجمالي الجُمل

وهو مؤشّر محصور بين ‎-1‎ و‎+1‎، ويُقارَن بمسار سعر الفائدة الفعلي.

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

import os
import re

import numpy as np
import pandas as pd

# ======================================================================
# النماذج المتاحة — available models
# ======================================================================

#: نماذج مُدرَّبة يمكن استخدامها لقياس الموقف.
#: Pre-trained models usable for stance measurement.
MODELS = {
    # نموذج مضبوط على جُمل بلاغات الاحتياطي الفدرالي نفسها، وهو الأنسب للمهمّة.
    # Fine-tuned on Federal Reserve communication sentences; the best fit here.
    "fomc": {
        "name": "gtfintechlab/FOMC-RoBERTa",
        "axis": "hawkish_dovish",
        "ar": "نموذج مضبوط على بلاغات الاحتياطي الفدرالي (متشدّد/متساهل/محايد)",
        "en": "RoBERTa fine-tuned on Federal Reserve communications",
    },
    # نموذج المشاعر المالية العامة: محوره إيجابي/سلبي لا متشدّد/متساهل.
    # General financial sentiment; its axis is positive/negative, not hawkish/dovish.
    "finbert": {
        "name": "ProsusAI/finbert",
        "axis": "positive_negative",
        "ar": "نموذج المشاعر المالية العامة (إيجابي/سلبي/محايد)",
        "en": "FinBERT, general financial sentiment",
    },
}

#: الفئات المعياريّة التي نُوحّد إليها مخارج أيّ نموذج.
#: The canonical classes to which any model's outputs are normalised.
CANONICAL = ("hawkish", "dovish", "neutral")

#: أسماء الفئات بالعربية — Arabic class labels
CLASS_LABELS_AR = {
    "hawkish": "متشدّد",
    "dovish": "متساهل",
    "neutral": "محايد",
}


def _canonical_label(raw_label, axis):
    """توحيد اسم الفئة الخارج من النموذج إلى إحدى الفئات المعياريّة.

    Normalise a model's raw label name to one of the canonical classes.

    لا نعتمد على ترتيب ثابت للفئات، بل نقرأ خريطة الفئات من إعداد النموذج
    نفسه (``config.id2label``) ثمّ نطابقها بالاسم. هذا يمنع الخطأ الشائع في
    عكس المتشدّد بالمتساهل عند تغيّر ترتيب الفئات بين إصدارات النموذج.

    We do not rely on a fixed class order; we read the label map from the model's own
    ``config.id2label`` and match by name. This prevents the common error of swapping
    hawkish and dovish when a model's class order changes between versions.
    """
    lab = str(raw_label).strip().lower()

    if axis == "hawkish_dovish":
        if "hawk" in lab:
            return "hawkish"
        if "dov" in lab:
            return "dovish"
        if "neutral" in lab or "none" in lab:
            return "neutral"
    else:  # positive_negative
        # الإيجابي في السياق المالي يُقابل ضمنًا نشاطًا أقوى، وهو ما يدفع نحو التشدّد.
        # In a financial context, positive maps loosely onto stronger activity, which
        # pushes towards tightening. This mapping is a convenience, not an identity.
        if "positive" in lab:
            return "hawkish"
        if "negative" in lab:
            return "dovish"
        if "neutral" in lab:
            return "neutral"

    return None


def split_sentences(text, min_words=3):
    """تقسيم الوثيقة إلى جُمل صالحة للتصنيف.

    Split a document into sentences suitable for classification.

    يُحسَب الحدّ الأدنى بالكلمات الأبجدية لا بالرموز، حتى لا تُستبعَد جُمل الإشارة
    القصيرة مثل «Inflation remains elevated» وهي من أقوى إشارات التشدّد، وتُستبعَد
    في الوقت نفسه شظايا الجداول مثل «See table 3».

    The minimum is counted in alphabetic words, not whitespace tokens, so that short
    signalling sentences such as "Inflation remains elevated" are kept — they are among
    the strongest hawkish signals — while table fragments such as "See table 3" are
    still dropped.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return []
    txt = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.!?])\s+", txt)
    out = []
    for p in parts:
        p = p.strip()
        if len(re.findall(r"[a-zA-Z']+", p)) >= min_words:
            out.append(p)
    return out


class StanceScorer:
    """مُصنِّف موقف السياسة النقدية على مستوى الجملة.

    Sentence-level monetary policy stance classifier.

    مثال للاستخدام — Example usage
    ------------------------------
    >>> scorer = StanceScorer("fomc")            # يُنزّل النموذج عند أول استخدام
    >>> scorer.score_sentences(["Inflation remains elevated."])
    >>> doc = scorer.score_document(statement_text)
    >>> doc["net_hawkishness"]

    ملاحظة عن الأداء — performance note
    -----------------------------------
    التصنيف على المعالج المركزي بطيء جدًّا لعدد كبير من الجُمل. يُنصح بوحدة معالجة
    رسومية، أو بتصفية النصوص بالكلمات المفتاحية قبل التصنيف.
    CPU inference is very slow for large sentence counts; a GPU is recommended, or
    filter the text by keywords beforehand.
    """

    def __init__(self, model_key="fomc", device=None, batch_size=32, max_length=256,
                 label_override=None):
        """
        المعاملات — Parameters
        ----------
        label_override : dict, optional
            خريطة يدوية من رقم الفئة إلى الفئة المعياريّة، مثل
            ``{0: "dovish", 1: "hawkish", 2: "neutral"}``. تُستخدَم حين لا يُسمّي
            النموذج فئاته (LABEL_0 وما شابه) فلا يمكن استنتاج الترتيب من الأسماء.
            A manual map from class index to canonical class, used when a model does
            not name its classes (LABEL_0 and similar) so the order cannot be inferred.
        """
        if model_key not in MODELS:
            raise ValueError(f"نموذج غير معروف: {model_key}. المتاح: {list(MODELS)}")
        self.model_key = model_key
        self.spec = MODELS[model_key]
        self.batch_size = batch_size
        self.max_length = max_length
        self.label_override = label_override
        self._device = device
        self._tokenizer = None
        self._model = None
        self._label_map = None

    # ------------------------------------------------------------------
    def _load(self):
        """تحميل النموذج والمُقطِّع عند أول حاجة إليهما (تحميل مؤجَّل).

        Lazily load the model and tokenizer on first use.
        """
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = self.spec["name"]
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSequenceClassification.from_pretrained(name)
        self._model.eval()

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

        # قراءة خريطة الفئات من إعداد النموذج بدل افتراض ترتيب ثابت
        # Read the label map from the model config instead of assuming an order
        if self.label_override:
            self._label_map = {int(k): v for k, v in self.label_override.items()}
            return

        id2label = getattr(self._model.config, "id2label", {}) or {}
        axis = self.spec["axis"]
        self._label_map = {}
        unmapped = []
        for idx, raw in id2label.items():
            canon = _canonical_label(raw, axis)
            if canon is None:
                unmapped.append(raw)
                canon = "neutral"
            self._label_map[int(idx)] = canon

        # لو لم تُطابَق أيّ فئة فالنتيجة ستكون «محايد» دائمًا، وهو فشل صامت يُفسد
        # كل التحليل. نُوقف التنفيذ بخطأ واضح بدل إنتاج مؤشّر بلا معنى.
        # If no label could be matched, every sentence would come out "neutral" — a
        # silent failure that invalidates the whole analysis. Fail loudly instead.
        if len(unmapped) == len(id2label) and id2label:
            raise RuntimeError(
                "لم يُمكن استنتاج فئات النموذج من أسمائها: "
                f"{list(id2label.values())}\n"
                "مرّر خريطة يدوية عبر label_override، مثل:\n"
                '    StanceScorer("fomc", label_override={0: "dovish", 1: "hawkish", '
                '2: "neutral"})\n'
                "وتحقّق من الترتيب الصحيح في صفحة النموذج قبل الاعتماد على النتائج.\n"
                "Could not infer the model's classes from their names; pass a manual "
                "label_override and verify the order on the model card first.")
        if unmapped:
            print("تنبيه: فئات لم تُطابَق وعُدّت محايدة — unmapped labels treated as "
                  f"neutral: {unmapped}")

    # ------------------------------------------------------------------
    @property
    def device(self):
        """جهاز الحساب المستخدَم فعليًّا — the device actually in use."""
        self._load()
        return self._device

    @property
    def label_map(self):
        """خريطة الفئات كما قُرئت من إعداد النموذج — the resolved label map."""
        self._load()
        return dict(self._label_map)

    # ------------------------------------------------------------------
    def score_sentences(self, sentences, return_probs=True):
        """تصنيف قائمة جُمل وإعادة إطار بيانات بالفئة والاحتمالات.

        Classify a list of sentences and return a DataFrame of labels and probabilities.
        """
        if not sentences:
            return pd.DataFrame(columns=["sentence", "label"] + list(CANONICAL))

        self._load()
        import torch

        records = []
        with torch.no_grad():
            for start in range(0, len(sentences), self.batch_size):
                batch = sentences[start:start + self.batch_size]
                enc = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for sent, row in zip(batch, probs):
                    # تجميع الاحتمالات على الفئات المعياريّة
                    # aggregate probabilities onto the canonical classes
                    agg = dict.fromkeys(CANONICAL, 0.0)
                    for idx, p in enumerate(row):
                        agg[self._label_map.get(idx, "neutral")] += float(p)
                    rec = {"sentence": sent,
                           "label": max(agg, key=agg.get)}
                    if return_probs:
                        rec.update({c: agg[c] for c in CANONICAL})
                    records.append(rec)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    def score_document(self, text, min_words=4):
        """تصنيف وثيقة كاملة وإعادة مؤشرات مجمّعة على مستوى الوثيقة.

        Classify a whole document and return document-level aggregate indices.
        """
        sents = split_sentences(text, min_words=min_words)
        if not sents:
            return {"n_sentences": 0, "n_hawkish": 0, "n_dovish": 0, "n_neutral": 0,
                    "net_hawkishness": np.nan, "hawkish_share": np.nan,
                    "dovish_share": np.nan, "mean_hawkish_prob": np.nan,
                    "mean_dovish_prob": np.nan, "polarisation": np.nan}

        df = self.score_sentences(sents)
        return aggregate_stance(df)

    # ------------------------------------------------------------------
    def score_frame(self, frame, text_col="text", date_col="date",
                    extra_cols=("type", "speaker"), verbose=True):
        """تصنيف كل صفوف إطار بيانات وإعادة إطار المؤشرات.

        Classify every row of a DataFrame and return a frame of indices.
        """
        rows = []
        total = len(frame)
        for i, (_, row) in enumerate(frame.iterrows(), 1):
            rec = self.score_document(row[text_col])
            if date_col in frame.columns:
                rec[date_col] = row[date_col]
            for col in extra_cols:
                if col in frame.columns:
                    rec[col] = row[col]
            rows.append(rec)
            if verbose and (i % 25 == 0 or i == total):
                print(f"  {i}/{total} وثيقة صُنّفت", flush=True)

        out = pd.DataFrame(rows)
        if date_col in out.columns:
            out = out.sort_values(date_col).reset_index(drop=True)
        return out


# ======================================================================
# التجميع وبناء المؤشرات — aggregation and index construction
# ======================================================================

def aggregate_stance(sentence_df):
    """بناء مؤشرات الوثيقة من تصنيفات جُملها.

    Build document-level indices from its sentence classifications.

    المؤشرات — indices
    ------------------
    net_hawkishness : صافي التشدّد، محصور بين ‎-1‎ و‎+1‎
    polarisation    : الاستقطاب، أي نسبة الجُمل غير المحايدة — how opinionated the text is
    """
    n = len(sentence_df)
    if n == 0:
        return {"n_sentences": 0, "net_hawkishness": np.nan}

    counts = sentence_df["label"].value_counts()
    n_h = int(counts.get("hawkish", 0))
    n_d = int(counts.get("dovish", 0))
    n_n = int(counts.get("neutral", 0))

    out = {
        "n_sentences": n,
        "n_hawkish": n_h,
        "n_dovish": n_d,
        "n_neutral": n_n,
        "hawkish_share": n_h / n,
        "dovish_share": n_d / n,
        "net_hawkishness": (n_h - n_d) / n,
        "polarisation": (n_h + n_d) / n,
    }
    for canon, col in (("hawkish", "mean_hawkish_prob"), ("dovish", "mean_dovish_prob")):
        out[col] = float(sentence_df[canon].mean()) if canon in sentence_df.columns else np.nan
    return out


def build_stance_index(stance_df, date_col="date", value_col="net_hawkishness",
                       freq=None, smooth=None):
    """بناء سلسلة زمنية لمؤشّر الموقف، مع تجميع وتمهيد اختياريَّين.

    Build a time series of the stance index, with optional resampling and smoothing.

    المعاملات — Parameters
    ----------
    freq : str, optional
        دورية التجميع بصيغة pandas مثل "QE" أو "ME" — pandas offset alias.
    smooth : int, optional
        نافذة المتوسط المتحرك — moving average window.
    """
    s = stance_df[[date_col, value_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).set_index(date_col)[value_col]
    if freq:
        s = s.resample(freq).mean()
    if smooth:
        s = s.rolling(smooth, min_periods=1).mean()
    return s


def stance_vs_decision(stance_df, decision_df, date_col="date",
                       decision_col="RateDecision", value_col="net_hawkishness"):
    """مطابقة مؤشّر الموقف بقرار سعر الفائدة لفحص قدرته التفسيرية.

    Align the stance index with the rate decision to examine its explanatory power.

    تُعيد إطارًا يضمّ المؤشّر والقرار، وجدولًا بمتوسط المؤشّر لكلّ فئة قرار.
    Returns the merged frame and a table of the mean index per decision class.
    """
    left = stance_df[[date_col, value_col]].copy()
    right = decision_df[[date_col, decision_col]].copy()
    for frame in (left, right):
        frame[date_col] = pd.to_datetime(frame[date_col])

    merged = pd.merge(left, right, on=date_col, how="inner").dropna()
    summary = merged.groupby(decision_col)[value_col].agg(["mean", "std", "count"])

    # ترتيب الفئات: تخفيض ثمّ تثبيت ثمّ رفع — order: lower, hold, raise
    label_ar = {-1: "تخفيض", 0: "تثبيت", 1: "رفع"}
    summary.index = [label_ar.get(i, i) for i in summary.index]
    return merged, summary


def save_stance(stance_df, out_dir, name="fomc_stance"):
    """حفظ مؤشرات الموقف بصيغتَي pickle و csv.

    Save the stance indices as both pickle and csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    p_pickle = os.path.join(out_dir, f"{name}.pickle")
    p_csv = os.path.join(out_dir, f"{name}.csv")
    stance_df.to_pickle(p_pickle)
    stance_df.to_csv(p_csv, index=False)
    print(f"حُفظ في:\n  {p_pickle}\n  {p_csv}")
    return p_pickle, p_csv
