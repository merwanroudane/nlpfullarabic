# -*- coding: utf-8 -*-
"""
تنزيل السلاسل الاقتصادية والمالية آليًّا من قاعدة FRED دون الحاجة إلى مفتاح واجهة (API Key).

Automatic download of economic and financial series from FRED, without an API key.

يوسّع الكتالوج من 14 سلسلة إلى 32 سلسلة، ويضيف مجموعات لم تكن موجودة:
توقّعات التضخّم (Inflation Expectations)، ومنحنى العائد (Yield Curve)، وفروق
الائتمان (Credit Spreads)، ومؤشرات الأوضاع المالية (Financial Conditions)،
وميزانية الاحتياطي الفدرالي (Fed Balance Sheet)، ومؤشر الكساد الرسمي (NBER).

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

import io
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

# نقطة النهاية العامة لتنزيل السلاسل بصيغة csv — public CSV endpoint, no API key needed
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# مهلة الاتصال وعدد المحاولات — connection timeout and retry count
TIMEOUT = 60
RETRIES = 3
RETRY_WAIT = 3


# ======================================================================
# كتالوج السلاسل — series catalogue
# ======================================================================
# المفتاح: رمز السلسلة في FRED. القيم: الوصف العربي، الوصف الإنجليزي،
# الدورية، والمجموعة المنهجية التي تنتمي إليها السلسلة.
# key: FRED series id. values: Arabic label, English label, frequency, group.

SERIES = {
    # ---------------- أسعار الفائدة الرسمية — policy rates ----------------
    "DFEDTAR":   ("سعر الفائدة الفدرالي المستهدف (قبل 2008)", "Federal Funds Target Rate", "daily", "policy_rate"),
    "DFEDTARU":  ("الحدّ الأعلى للنطاق المستهدف", "Federal Funds Target Range Upper Limit", "daily", "policy_rate"),
    "DFEDTARL":  ("الحدّ الأدنى للنطاق المستهدف", "Federal Funds Target Range Lower Limit", "daily", "policy_rate"),
    "DFF":       ("سعر الفائدة الفدرالي الفعلي", "Effective Federal Funds Rate", "daily", "policy_rate"),
    "IORB":      ("الفائدة على أرصدة الاحتياطي", "Interest on Reserve Balances", "daily", "policy_rate"),

    # ---------------- النشاط الحقيقي — real activity ----------------
    "GDPC1":     ("الناتج المحلي الإجمالي الحقيقي", "Real Gross Domestic Product", "quarterly", "activity"),
    "GDPPOT":    ("الناتج المحلي الإجمالي الكامن الحقيقي", "Real Potential GDP", "quarterly", "activity"),
    "INDPRO":    ("مؤشّر الإنتاج الصناعي", "Industrial Production Index", "monthly", "activity"),
    "TCU":       ("معدّل استغلال الطاقة الإنتاجية", "Capacity Utilization", "monthly", "activity"),
    "RRSFS":     ("مبيعات التجزئة الحقيقية المُسبقة", "Advance Real Retail and Food Services Sales", "monthly", "activity"),
    "HSN1F":     ("مبيعات المنازل الجديدة", "New One Family Houses Sold", "monthly", "activity"),
    "HOUST":     ("بدايات بناء المساكن", "Housing Starts", "monthly", "activity"),

    # ---------------- التضخّم — inflation ----------------
    "PCEPILFE":  ("مؤشّر نفقات الاستهلاك الشخصي الأساسي", "Core PCE Price Index", "monthly", "inflation"),
    "PCEPI":     ("مؤشّر نفقات الاستهلاك الشخصي الكلّي", "Headline PCE Price Index", "monthly", "inflation"),
    "CPIAUCSL":  ("مؤشّر أسعار المستهلك", "Consumer Price Index for All Urban Consumers", "monthly", "inflation"),
    "CPILFESL":  ("مؤشّر أسعار المستهلك الأساسي", "Core Consumer Price Index", "monthly", "inflation"),

    # ---------------- توقّعات التضخّم — inflation expectations ----------------
    "T5YIE":     ("توقّعات التضخّم لخمس سنوات (نقطة التعادل)", "5-Year Breakeven Inflation Rate", "daily", "inflation_exp"),
    "T10YIE":    ("توقّعات التضخّم لعشر سنوات (نقطة التعادل)", "10-Year Breakeven Inflation Rate", "daily", "inflation_exp"),
    "T5YIFR":    ("توقّعات التضخّم الآجلة 5 سنوات بعد 5", "5-Year Forward Inflation Expectation Rate", "daily", "inflation_exp"),
    "MICH":      ("توقّعات التضخّم لدى المستهلكين (ميشيغان)", "University of Michigan Inflation Expectation", "monthly", "inflation_exp"),

    # ---------------- سوق العمل — labour market ----------------
    "UNRATE":    ("معدّل البطالة", "Unemployment Rate", "monthly", "labour"),
    "PAYEMS":    ("التشغيل غير الزراعي الإجمالي", "All Employees, Total Nonfarm", "monthly", "labour"),
    "CIVPART":   ("معدّل المشاركة في القوة العاملة", "Labor Force Participation Rate", "monthly", "labour"),
    "ICSA":      ("طلبات إعانة البطالة الأولية", "Initial Claims", "weekly", "labour"),
    "AHETPI":    ("متوسط الأجر بالساعة", "Average Hourly Earnings", "monthly", "labour"),

    # ---------------- منحنى العائد — yield curve ----------------
    "DGS3MO":    ("عائد الخزانة لثلاثة أشهر", "3-Month Treasury Yield", "daily", "yield_curve"),
    "DGS2":      ("عائد الخزانة لسنتين", "2-Year Treasury Yield", "daily", "yield_curve"),
    "DGS10":     ("عائد الخزانة لعشر سنوات", "10-Year Treasury Yield", "daily", "yield_curve"),
    "T10Y2Y":    ("فرق العائد بين 10 سنوات وسنتين", "10Y minus 2Y Treasury Spread", "daily", "yield_curve"),
    "T10Y3M":    ("فرق العائد بين 10 سنوات و3 أشهر", "10Y minus 3M Treasury Spread", "daily", "yield_curve"),
    "DFII10":    ("العائد الحقيقي لعشر سنوات (TIPS)", "10-Year Real Treasury Yield", "daily", "yield_curve"),

    # ---------------- المخاطر والأوضاع المالية — risk and financial conditions ----------------
    "VIXCLS":    ("مؤشّر التقلّب الضمني", "CBOE Volatility Index (VIX)", "daily", "risk"),
    "BAA10Y":    ("فرق عائد سندات Baa عن الخزانة", "Baa Corporate Bond Spread over 10Y Treasury", "daily", "risk"),
    "BAMLH0A0HYM2": ("فرق عائد السندات عالية المخاطر", "High Yield Option-Adjusted Spread", "daily", "risk"),
    "NFCI":      ("مؤشّر الأوضاع المالية الوطني", "Chicago Fed National Financial Conditions Index", "weekly", "risk"),
    "STLFSI4":   ("مؤشّر الضغط المالي (سانت لويس)", "St. Louis Fed Financial Stress Index", "weekly", "risk"),

    # ---------------- النقد وميزانية الاحتياطي — money and Fed balance sheet ----------------
    "M2SL":      ("المعروف النقدي M2", "M2 Money Stock", "monthly", "money"),
    "WALCL":     ("إجمالي أصول الاحتياطي الفدرالي", "Total Assets of the Federal Reserve", "weekly", "money"),
    "TOTRESNS":  ("إجمالي أرصدة الاحتياطي", "Total Reserve Balances", "monthly", "money"),

    # ---------------- المعنويات والحقب — sentiment and regimes ----------------
    "UMCSENT":   ("مؤشّر معنويات المستهلك", "University of Michigan Consumer Sentiment", "monthly", "sentiment"),
    "USREC":     ("مؤشّر الكساد الرسمي (NBER)", "NBER Recession Indicator", "monthly", "regime"),
}

# السلاسل التي كان المشروع يعتمدها أصلًا — the originally used subset
LEGACY_SERIES = [
    "DFEDTAR", "DFEDTARU", "DFEDTARL", "DFF", "GDPC1", "GDPPOT",
    "PCEPILFE", "CPIAUCSL", "UNRATE", "PAYEMS", "RRSFS", "HSN1F",
]

# سلاسل ISM غير متاحة على FRED بسبب قيود الترخيص، وتُجلَب من مصدر آخر.
# ISM series are not available on FRED due to licensing; they must come from elsewhere.
UNAVAILABLE_ON_FRED = {
    "ISM_MAN_PMI": "مؤشّر مديري المشتريات الصناعي — ISM Manufacturing PMI",
    "ISM_NONMAN_NMI": "المؤشّر غير الصناعي — ISM Non-Manufacturing Index",
}

# سلاسل تُنشَر عبر النقطة العامة بنافذة زمنية محدودة فقط بسبب قيود الترخيص،
# فتصل بتاريخ بداية حديث لا يمثّل عمرها الحقيقي. تُستخدَم للفترة الأخيرة فقط.
# Series that the public endpoint serves with a limited rolling window because of
# licensing restrictions; their start date is recent and does not reflect the true
# history. Use them for the recent period only.
LICENSE_LIMITED = {
    "BAMLH0A0HYM2": "مؤشرات ICE BofA تُتاح بنافذة نحو ثلاث سنوات فقط — ICE BofA indices are served with a ~3-year window",
}


def groups():
    """تُعيد قائمة المجموعات المنهجية الموجودة في الكتالوج.

    Returns the distinct methodological groups present in the catalogue.
    """
    return sorted({meta[3] for meta in SERIES.values()})


def series_in_group(group):
    """تُعيد رموز السلاسل المنتمية إلى مجموعة معيّنة.

    Returns the series ids belonging to a given group.
    """
    return [sid for sid, meta in SERIES.items() if meta[3] == group]


def catalogue_frame():
    """تُعيد الكتالوج كإطار بيانات لعرضه في الدفاتر.

    Returns the catalogue as a DataFrame for display in notebooks.
    """
    rows = []
    for sid, (ar, en, freq, grp) in SERIES.items():
        rows.append({"series_id": sid, "الوصف": ar, "description": en,
                     "الدورية": freq, "المجموعة": grp})
    return pd.DataFrame(rows).sort_values(["المجموعة", "series_id"]).reset_index(drop=True)


def _fetch(series_id):
    """تنزيل نصّ csv الخاصّ بسلسلة واحدة مع إعادة المحاولة عند الفشل.

    Download the raw CSV text for one series, retrying on transient failure.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            # لا نضبط ترويسة User-Agent مخصّصة: بعض الوكلاء الشبكية تحجب الترويسات
            # الشبيهة بالمتصفّحات فيتعلّق الطلب حتى انتهاء المهلة.
            # No custom User-Agent: some network proxies block browser-like headers,
            # causing the request to hang until it times out.
            with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            if attempt < RETRIES:
                time.sleep(RETRY_WAIT * attempt)
    raise RuntimeError(f"فشل تنزيل {series_id} بعد {RETRIES} محاولات: {last_err}")


def download_series(series_id, as_frame=True):
    """تنزيل سلسلة واحدة وإعادتها كإطار بيانات مفهرس بالتاريخ.

    Download a single series and return it as a date-indexed DataFrame.

    تعالج الدالة اختلاف اسم عمود التاريخ بين إصدارات FRED (DATE أو
    observation_date)، وتحوّل القيم المفقودة المرمّزة بنقطة "." إلى NaN.

    Handles the DATE / observation_date header difference across FRED versions and
    converts the "." missing-value code to NaN.
    """
    text = _fetch(series_id)
    df = pd.read_csv(io.StringIO(text))

    # اسم عمود التاريخ يختلف بحسب إصدار الموقع — the date column name varies
    date_col = None
    for candidate in ("observation_date", "DATE", "date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]

    value_col = [c for c in df.columns if c != date_col][0]
    df = df.rename(columns={date_col: "date", value_col: series_id})
    df["date"] = pd.to_datetime(df["date"])
    # FRED يرمّز القيم المفقودة بنقطة — FRED encodes missing values as "."
    df[series_id] = pd.to_numeric(df[series_id].replace(".", pd.NA), errors="coerce")
    df = df.set_index("date").sort_index()
    return df if as_frame else df[series_id]


def download_all(out_dir, only=None, skip_existing=True, verbose=True):
    """تنزيل كل سلاسل الكتالوج وحفظها ملفًّا لكلّ سلسلة.

    Download every catalogued series and save one CSV per series.

    المعاملات — Parameters
    ----------
    out_dir : str
        مجلّد الحفظ. تُسمّى الملفات ``FRED_<ID>.csv`` بما يوافق ما تتوقّعه
        الدفاتر من 1 إلى 8.
        Output directory; files are named ``FRED_<ID>.csv`` to match what
        notebooks 1 to 8 expect.
    only : list, optional
        قائمة رموز محدّدة بدل الكتالوج كاملًا — a subset of series ids.
    skip_existing : bool
        تجاوز الملفات المنزّلة سابقًا — skip files already on disk.

    تُعيد — Returns
    -------
    dict
        قاموس بالسلاسل الناجحة والفاشلة — a report of succeeded and failed ids.
    """
    os.makedirs(out_dir, exist_ok=True)
    targets = only if only else list(SERIES.keys())
    ok, failed, skipped = [], {}, []

    for i, sid in enumerate(targets, 1):
        path = os.path.join(out_dir, f"FRED_{sid}.csv")
        if skip_existing and os.path.exists(path):
            skipped.append(sid)
            if verbose:
                print(f"[{i:2}/{len(targets)}] {sid:14} موجود مسبقًا — skipped")
            continue
        try:
            df = download_series(sid)
            df.to_csv(path)
            ok.append(sid)
            if verbose:
                span = f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}"
                print(f"[{i:2}/{len(targets)}] {sid:14} {len(df):6} صفًّا  {span}")
        except Exception as err:  # noqa: BLE001 - نريد تقريرًا لا توقّفًا
            failed[sid] = str(err)
            if verbose:
                print(f"[{i:2}/{len(targets)}] {sid:14} فشل — FAILED: {err}")

    if verbose:
        print(f"\nنجح: {len(ok)} | تُجوهل: {len(skipped)} | فشل: {len(failed)}")
        if failed:
            print("السلاسل الفاشلة — failed series:")
            for sid, msg in failed.items():
                print(f"  {sid}: {msg}")
        if UNAVAILABLE_ON_FRED:
            print("\nغير متاح على FRED ويجب جلبه من مصدر آخر — not available on FRED:")
            for sid, label in UNAVAILABLE_ON_FRED.items():
                print(f"  {sid}: {label}")

    return {"ok": ok, "skipped": skipped, "failed": failed}


def load_panel(data_dir, series=None, how="outer"):
    """تجميع السلاسل المنزّلة في لوحة واحدة مفهرسة بالتاريخ.

    Combine the downloaded series into a single date-indexed panel.

    مفيدة للتحليل الاستكشافي وحساب الارتباطات بين المؤشرات كلّها معًا.
    Useful for exploratory analysis and cross-indicator correlations.
    """
    targets = series if series else list(SERIES.keys())
    frames = []
    missing = []
    for sid in targets:
        path = os.path.join(data_dir, f"FRED_{sid}.csv")
        if not os.path.exists(path):
            missing.append(sid)
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frames.append(df[[sid]])
    if not frames:
        raise FileNotFoundError(f"لا توجد أيّ سلسلة منزّلة في {data_dir}")
    panel = pd.concat(frames, axis=1, join=how).sort_index()
    if missing:
        print(f"سلاسل غير منزّلة ({len(missing)}): {', '.join(missing)}")
    return panel


def _usage(pg_name):
    print("Usage:", pg_name, "<out_dir> [group|all|legacy]")
    print()
    print("  out_dir : مجلّد الحفظ — output directory")
    print("  الوسيط الثاني — second argument:")
    print("     all      : كل السلاسل (الافتراضي) —", len(SERIES), "series")
    print("     legacy   : السلاسل الأصلية فقط —", len(LEGACY_SERIES), "series")
    print("     <group>  : مجموعة واحدة من:", ", ".join(groups()))


if __name__ == "__main__":
    pg_name = sys.argv[0]
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _usage(pg_name)
        sys.exit(0 if args else 1)

    out_dir = args[0]
    selector = args[1].lower() if len(args) > 1 else "all"

    if selector == "all":
        subset = None
    elif selector == "legacy":
        subset = LEGACY_SERIES
    elif selector in groups():
        subset = series_in_group(selector)
    else:
        print("وسيط غير معروف:", selector)
        _usage(pg_name)
        sys.exit(1)

    print(f"التنزيل إلى: {out_dir}")
    print(f"عدد السلاسل المطلوبة: {len(subset) if subset else len(SERIES)}\n")
    report = download_all(out_dir, only=subset)
    sys.exit(0 if not report["failed"] else 1)
