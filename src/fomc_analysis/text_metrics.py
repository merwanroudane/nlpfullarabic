# -*- coding: utf-8 -*-
"""
مقاييس كمّية لخطاب البنك المركزي: قابلية القراءة، والتعقيد، وعدم اليقين، والتحوّط.

Quantitative metrics for central bank language: readability, complexity, uncertainty
and hedging.

الأساس المنهجي — methodological background
------------------------------------------
تقيس أدبيات شفافية البنوك المركزية مدى وضوح البلاغات الرسمية بمؤشرات قابلية القراءة
(Readability)، انطلاقًا من فرضية أنّ البلاغ الأصعب قراءةً يحمل غموضًا أكبر ويترك
للأسواق مجالًا أوسع للتأويل. كما تُقاس نبرة عدم اليقين (Uncertainty) بعدّ كلمات
التحوّط (Hedging Words) مثل "may" و"could" و"appears"، وهي أدوات لغوية يستخدمها
صانع السياسة ليتجنّب الالتزام الصريح.

The central bank transparency literature measures the clarity of official
communication using readability indices, on the premise that a harder-to-read
statement carries more ambiguity. Uncertainty tone is measured by counting hedging
words such as "may", "could" and "appears".

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

import re

import numpy as np
import pandas as pd

# ======================================================================
# معاجم مغلقة — closed-form lexicons
# ======================================================================

# كلمات التحوّط: تُخفّف الالتزام وتترك الباب مفتوحًا لتأويلات متعدّدة.
# Hedging words: they soften commitment and leave room for multiple readings.
HEDGING_WORDS = {
    "may", "might", "could", "would", "should", "possibly", "perhaps", "probably",
    "appears", "appear", "seems", "seem", "suggests", "suggest", "somewhat",
    "moderately", "relatively", "generally", "largely", "broadly", "roughly",
    "approximately", "about", "likely", "unlikely", "uncertain", "uncertainty",
    "tentative", "preliminary", "potential", "potentially", "presumably",
    "apparently", "arguably", "conceivably", "reportedly",
}

# كلمات الالتزام الصريح: نقيض التحوّط، وتدلّ على تعهّد أقوى.
# Strong commitment words: the opposite of hedging, signalling firmer intent.
COMMITMENT_WORDS = {
    "will", "must", "committed", "commit", "commits", "determined", "ensure",
    "ensures", "certainly", "definitely", "clearly", "strongly", "firmly",
    "decided", "decides", "resolved", "expects", "anticipates", "intends",
}

# مصطلحات السياسة النقدية المحورية — core monetary policy vocabulary
POLICY_TERMS = {
    "inflation", "unemployment", "employment", "growth", "rate", "rates",
    "target", "policy", "committee", "accommodation", "accommodative",
    "tightening", "easing", "stance", "outlook", "risks", "balance",
    "guidance", "purchases", "securities", "reserves", "liquidity",
}

# لاحقات صوتية تُعالَج بشكل خاصّ عند عدّ المقاطع
# suffixes handled specially when counting syllables
_VOWELS = "aeiouy"


# ======================================================================
# أدوات مساعدة — helpers
# ======================================================================

def _sentences(text):
    """تقسيم النصّ إلى جُمل بفاصل بسيط لا يحتاج تنزيل بيانات إضافية.

    Split text into sentences with a lightweight splitter that needs no downloads.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p for p in parts if p.strip()]


def _words(text):
    """استخراج الكلمات الأبجدية فقط بأحرف صغيرة.

    Extract alphabetic words only, lower-cased.
    """
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def count_syllables(word):
    """تقدير عدد المقاطع الصوتية في كلمة إنجليزية.

    Estimate the number of syllables in an English word.

    تقدير تقريبي يكفي لمؤشرات قابلية القراءة، ولا يحتاج معجمًا صوتيًّا.
    An approximation sufficient for readability indices, needing no phonetic
    dictionary.
    """
    word = re.sub(r"[^a-z]", "", str(word).lower())
    if not word:
        return 0
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # الحرف e الصامت في النهاية لا يُحتسَب مقطعًا — a silent trailing "e"
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def _is_complex(word):
    """الكلمة المركّبة: ثلاثة مقاطع أو أكثر، وهي أساس مؤشّر Gunning Fog.

    A complex word has three or more syllables; the basis of the Gunning Fog index.
    """
    return count_syllables(word) >= 3


# ======================================================================
# مؤشرات قابلية القراءة — readability indices
# ======================================================================

def flesch_reading_ease(text):
    """مؤشّر سهولة القراءة لـ فلِش: كلّما ارتفع كان النصّ أسهل.

    Flesch Reading Ease: the higher the score, the easier the text.

    الصيغة — formula: 206.835 - 1.015 * (كلمات/جُمل) - 84.6 * (مقاطع/كلمات)
    """
    words, sents = _words(text), _sentences(text)
    if not words or not sents:
        return np.nan
    syl = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syl / len(words))


def flesch_kincaid_grade(text):
    """مستوى فلِش-كنكيد الدراسي: يقابل عدد سنوات التعليم اللازمة لفهم النصّ.

    Flesch-Kincaid Grade Level: the years of schooling needed to understand the text.
    """
    words, sents = _words(text), _sentences(text)
    if not words or not sents:
        return np.nan
    syl = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sents)) + 11.8 * (syl / len(words)) - 15.59


def gunning_fog(text):
    """مؤشّر الضباب لـ غَنِنغ: يعتمد طول الجملة ونسبة الكلمات المركّبة.

    Gunning Fog index: based on sentence length and the share of complex words.
    """
    words, sents = _words(text), _sentences(text)
    if not words or not sents:
        return np.nan
    complex_share = sum(1 for w in words if _is_complex(w)) / len(words)
    return 0.4 * ((len(words) / len(sents)) + 100 * complex_share)


def smog_index(text):
    """مؤشّر SMOG: يقدّر سنوات التعليم اللازمة انطلاقًا من الكلمات المركّبة.

    SMOG index: estimates required schooling years from polysyllabic words.
    """
    sents = _sentences(text)
    if len(sents) < 3:
        return np.nan
    poly = sum(1 for w in _words(text) if count_syllables(w) >= 3)
    return 1.0430 * np.sqrt(poly * (30 / len(sents))) + 3.1291


# ======================================================================
# مؤشرات النبرة والتعقيد — tone and complexity indices
# ======================================================================

def lexical_diversity(text):
    """التنوّع المعجمي: نسبة الكلمات الفريدة إلى مجموع الكلمات.

    Lexical diversity: the type-token ratio, unique words over total words.
    """
    words = _words(text)
    return len(set(words)) / len(words) if words else np.nan


def hedging_ratio(text, per=100):
    """كثافة كلمات التحوّط لكلّ مئة كلمة.

    Density of hedging words per hundred words.
    """
    words = _words(text)
    if not words:
        return np.nan
    return per * sum(1 for w in words if w in HEDGING_WORDS) / len(words)


def commitment_ratio(text, per=100):
    """كثافة كلمات الالتزام الصريح لكلّ مئة كلمة.

    Density of strong commitment words per hundred words.
    """
    words = _words(text)
    if not words:
        return np.nan
    return per * sum(1 for w in words if w in COMMITMENT_WORDS) / len(words)


def policy_focus(text, per=100):
    """كثافة مصطلحات السياسة النقدية، وهي مقياس لتركيز الوثيقة على السياسة.

    Density of monetary policy vocabulary, a measure of how policy-focused a
    document is.
    """
    words = _words(text)
    if not words:
        return np.nan
    return per * sum(1 for w in words if w in POLICY_TERMS) / len(words)


def hedging_balance(text):
    """ميزان التحوّط: التحوّط ناقص الالتزام.

    Hedging balance: hedging minus commitment.

    القيمة الموجبة تعني خطابًا مُتحوّطًا يترك خياراته مفتوحة، والسالبة تعني
    خطابًا حاسمًا.
    A positive value means hedged language that keeps options open; a negative
    value means decisive language.
    """
    h, c = hedging_ratio(text), commitment_ratio(text)
    if pd.isna(h) or pd.isna(c):
        return np.nan
    return h - c


def uncertainty_from_lm(text, lm_dict, per=100):
    """كثافة كلمات عدم اليقين حسب معجم لوغران وماكدونالد.

    Density of uncertainty words according to the Loughran-McDonald dictionary.

    المعاملات — Parameters
    ----------
    lm_dict : dict or set
        مجموعة كلمات عدم اليقين بأحرف صغيرة، أو قاموس يحمل مفتاح 'Uncertainty'.
        A lower-cased set of uncertainty words, or a dict with an 'Uncertainty' key.
    """
    if isinstance(lm_dict, dict):
        vocab = {w.lower() for w in lm_dict.get("Uncertainty", lm_dict.get("uncertainty", []))}
    else:
        vocab = {w.lower() for w in lm_dict}
    words = _words(text)
    if not words or not vocab:
        return np.nan
    return per * sum(1 for w in words if w in vocab) / len(words)


# ======================================================================
# الحساب على مستوى الوثيقة وإطار البيانات — document and frame level
# ======================================================================

#: قائمة المقاييس المحسوبة لكلّ وثيقة — the metrics computed per document
METRIC_FUNCS = {
    "flesch_ease": flesch_reading_ease,
    "flesch_kincaid": flesch_kincaid_grade,
    "gunning_fog": gunning_fog,
    "smog": smog_index,
    "lexical_diversity": lexical_diversity,
    "hedging_per100": hedging_ratio,
    "commitment_per100": commitment_ratio,
    "hedging_balance": hedging_balance,
    "policy_focus_per100": policy_focus,
}

#: أسماء المقاييس بالعربية للعرض في الجداول والرسوم
#: Arabic labels for display in tables and plots
METRIC_LABELS_AR = {
    "flesch_ease": "سهولة القراءة (فلِش)",
    "flesch_kincaid": "المستوى الدراسي (فلِش-كنكيد)",
    "gunning_fog": "مؤشّر الضباب (غَنِنغ)",
    "smog": "مؤشّر SMOG",
    "lexical_diversity": "التنوّع المعجمي",
    "hedging_per100": "كلمات التحوّط لكل 100 كلمة",
    "commitment_per100": "كلمات الالتزام لكل 100 كلمة",
    "hedging_balance": "ميزان التحوّط",
    "policy_focus_per100": "تركيز مصطلحات السياسة",
    "n_words": "عدد الكلمات",
    "n_sentences": "عدد الجُمل",
    "avg_sentence_len": "متوسط طول الجملة",
    "complex_word_share": "نسبة الكلمات المركّبة",
}


def document_metrics(text):
    """حساب كل المقاييس لوثيقة واحدة وإعادتها في قاموس.

    Compute every metric for a single document and return them as a dict.
    """
    words, sents = _words(text), _sentences(text)
    out = {
        "n_words": len(words),
        "n_sentences": len(sents),
        "avg_sentence_len": (len(words) / len(sents)) if sents else np.nan,
        "complex_word_share": (sum(1 for w in words if _is_complex(w)) / len(words)) if words else np.nan,
    }
    for name, func in METRIC_FUNCS.items():
        out[name] = func(text)
    return out


def frame_metrics(df, text_col="text", date_col="date", extra_cols=("type", "speaker")):
    """حساب المقاييس لكلّ صفوف إطار البيانات وإعادة إطار جديد.

    Compute the metrics for every row of a DataFrame and return a new frame.

    المعاملات — Parameters
    ----------
    df : DataFrame
        إطار يحتوي عمود النصّ وعمود التاريخ — must contain the text and date columns.
    extra_cols : tuple
        أعمدة تُنقَل كما هي إن وُجدت، مثل نوع الوثيقة والمتحدّث.
        Columns carried over unchanged if present, such as document type and speaker.
    """
    rows = []
    for _, row in df.iterrows():
        rec = document_metrics(row[text_col])
        if date_col in df.columns:
            rec[date_col] = row[date_col]
        for col in extra_cols:
            if col in df.columns:
                rec[col] = row[col]
        rows.append(rec)
    out = pd.DataFrame(rows)
    if date_col in out.columns:
        out = out.sort_values(date_col).reset_index(drop=True)
    return out


def aggregate_by_period(metrics_df, date_col="date", freq="YE", metrics=None):
    """تجميع المقاييس زمنيًّا لرصد تطوّر خطاب البنك المركزي.

    Aggregate the metrics over time to track how central bank language evolves.

    المعاملات — Parameters
    ----------
    freq : str
        دورية التجميع بصيغة pandas: "YE" سنويًّا، "QE" فصليًّا.
        pandas offset alias: "YE" for yearly, "QE" for quarterly.
    """
    cols = metrics if metrics else [c for c in METRIC_FUNCS if c in metrics_df.columns]
    tmp = metrics_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    return tmp.set_index(date_col)[cols].resample(freq).mean()


def compare_by_group(metrics_df, group_col="speaker", metrics=None):
    """مقارنة المقاييس بين مجموعات، كمقارنة أساليب رؤساء المجلس.

    Compare metrics across groups, such as comparing the chairs' styles.
    """
    cols = metrics if metrics else [c for c in METRIC_FUNCS if c in metrics_df.columns]
    agg = metrics_df.groupby(group_col)[cols].agg(["mean", "std", "count"])
    return agg
