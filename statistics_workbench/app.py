"""Beginner-friendly statistical analysis workbench."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stat_engine import (
    assumption_checks, categorical_association, correlation, descriptive_table,
    factorial_anova, independent_ttest, linear_regression, logistic_regression,
    nonparametric_test, one_way_anova, paired_posthoc, paired_ttest,
    pairwise_ttests, repeated_measures_anova, tukey_hsd,
)

st.set_page_config(page_title="Statistics Workbench", page_icon="📊", layout="wide")
st.title("📊 Statistics Workbench｜統計分析工作台")
st.caption("上傳資料 → 選擇問題 → 檢查假設 → 看結果與圖 → 展開公式核對計算")
st.page_link("pages/1_Statistics_Tutorial.py", label="📘 Statistics Tutorial｜統計檢定教學")


@st.cache_data
def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    subjects = np.repeat(np.arange(1, 16), 3)
    time = np.tile(["day0", "day7", "day14"], 15)
    subject_group = np.tile(np.repeat(["Control", "BCI", "Training"], 5), 3).reshape(3, 15).T.reshape(-1)
    subject_baseline = rng.normal(72, 5, 15)
    baseline = np.repeat(subject_baseline, 3)
    time_gain = np.tile([0, 3, 6], 15)
    treatment_gain = np.select([subject_group == "Control", subject_group == "BCI", subject_group == "Training"], [0, 5, 2])
    ages = np.repeat(rng.integers(20, 66, 15), 3)
    sexes = np.repeat(np.tile(["F", "M"], 8)[:15], 3)
    return pd.DataFrame({"subject": subjects, "group": subject_group, "time": time,
                         "baseline": baseline.round(1),
                         "accuracy": (baseline + time_gain + treatment_gain + rng.normal(0, 2.5, 45)).round(1),
                         "age": ages, "sex": sexes})


def read_file(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    if upload.name.lower().endswith(".csv"):
        for encoding in ("utf-8-sig", "utf-8", "big5"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("無法辨識 CSV 編碼；請另存為 UTF-8。")
    return pd.read_excel(io.BytesIO(raw))


def show_result(result: dict, alpha: float, statistic_name: str = "Statistic"):
    c1, c2, c3 = st.columns(3)
    c1.metric(statistic_name, f"{result['statistic']:.4g}")
    c2.metric("p-value", f"{result['p_value']:.4g}")
    c3.metric("判定", "顯著" if result["p_value"] < alpha else "未達顯著")
    st.info(f"在 α = {alpha:g} 下，p {'<' if result['p_value'] < alpha else '≥'} α；"
            f"因此{'拒絕' if result['p_value'] < alpha else '沒有足夠證據拒絕'}虛無假設。這不是效果大小或實務重要性的判定。")


def download_table(table: pd.DataFrame, name: str):
    st.download_button("下載結果 CSV", table.to_csv(index=False).encode("utf-8-sig"), name, "text/csv")


with st.sidebar:
    st.header("1｜載入資料")
    uploaded = st.file_uploader("CSV 或 Excel", type=["csv", "xlsx", "xls"])
    use_demo = st.checkbox("使用示範資料", value=uploaded is None)
    alpha = st.number_input("顯著水準 α", min_value=.001, max_value=.20, value=.05, step=.01, format="%.3f")
    st.divider()
    st.markdown("**建議的長格式**：每列是一筆觀察，欄位如 `subject`、`group`、`time`、`outcome`。配對 t-test 則選兩個數值欄。")

try:
    data = read_file(uploaded) if uploaded else demo_data() if use_demo else None
except Exception as exc:
    st.error(f"讀取失敗：{exc}")
    st.stop()
if data is None or data.empty:
    st.warning("請上傳資料，或勾選示範資料。")
    st.stop()

st.subheader("資料預覽")
c1, c2, c3 = st.columns(3)
c1.metric("列數", len(data)); c2.metric("欄數", len(data.columns)); c3.metric("缺失值", int(data.isna().sum().sum()))
st.dataframe(data.head(50), width="stretch")
with st.expander("欄位型態與缺失值"):
    st.dataframe(pd.DataFrame({"欄位": data.columns, "型態": data.dtypes.astype(str).values,
                               "缺失數": data.isna().sum().values,
                               "唯一值數": data.nunique(dropna=True).values}), width="stretch")

numeric = list(data.select_dtypes(include=np.number).columns)
all_cols = list(data.columns)
if not numeric:
    st.error("資料中找不到數值欄。請確認結果欄不是含有文字或符號的字串。"); st.stop()
numeric_options = sorted(numeric, key=lambda c: (
    not any(token in c.lower() for token in ("outcome", "accuracy", "score", "performance", "value")),
    any(token in c.lower() for token in ("subject", "patient", "participant", "_id", "id_")), c))


def group_options(exclude: str | None = None) -> list[str]:
    """Put likely categorical columns first so the beginner defaults are sensible."""
    candidates = [c for c in all_cols if c != exclude and data[c].nunique(dropna=True) >= 2]
    return sorted(candidates, key=lambda c: (not any(token in c.lower() for token in ("group", "condition", "treatment", "arm")),
                                              data[c].nunique(dropna=True) > 12,
                                              pd.api.types.is_numeric_dtype(data[c]),
                                              data[c].nunique(dropna=True)))


analysis = st.selectbox("2｜你想回答什麼問題？", [
    "描述資料與分布", "比較兩個獨立組別（t-test）", "比較同一批人的兩次測量（paired t-test）",
    "比較三組以上（ANOVA + post-hoc）", "不符合常態的組別比較（非參數）",
    "同一批人的三次以上測量（repeated-measures ANOVA）", "兩個數值是否相關",
    "兩個類別變數是否相關（Chi-square / Fisher）", "用多個變數預測數值結果（線性迴歸）",
    "預測二元結果（logistic regression）", "多因子 ANOVA（含交互作用）",
])

if analysis == "描述資料與分布":
    value = st.selectbox("要摘要的數值欄", numeric_options)
    group_choice = st.selectbox("分組欄（可不選）", ["不分組"] + group_options(value))
    group = None if group_choice == "不分組" else group_choice
    table = descriptive_table(data, value, group)
    st.dataframe(table, width="stretch"); download_table(table, "descriptive_statistics.csv")
    fig = px.histogram(data, x=value, color=group, marginal="box", opacity=.65, barmode="overlay",
                       title=f"{value} 的分布")
    st.plotly_chart(fig, width="stretch")
    with st.expander("這些數字怎麼算？"):
        st.latex(r"\bar{x}=\frac{\sum x_i}{n},\quad s=\sqrt{\frac{\sum(x_i-\bar{x})^2}{n-1}},\quad SE=\frac{s}{\sqrt n}")
        st.latex(r"95\%\ CI=\bar{x}\pm t_{0.975,n-1}\times SE")

elif analysis == "比較兩個獨立組別（t-test）":
    value = st.selectbox("結果數值欄", numeric_options); group = st.selectbox("兩組的分組欄", group_options(value))
    levels = list(data[group].dropna().unique())
    if len(levels) < 2: st.error("分組欄至少需要兩組。"); st.stop()
    a = st.selectbox("組別 A", levels); b = st.selectbox("組別 B", [x for x in levels if x != a])
    equal = st.checkbox("假設兩組變異數相等（Student t；不確定時不要勾）")
    result = independent_ttest(data.loc[data[group] == a, value], data.loc[data[group] == b, value], equal)
    show_result(result, alpha, "t"); st.write(pd.DataFrame([{k: v for k, v in result.items() if np.isscalar(v)}]))
    fig = px.box(data[data[group].isin([a, b])], x=group, y=value, points="all", color=group, title=f"{a} vs {b}")
    st.plotly_chart(fig, width="stretch")
    with st.expander("逐步計算與公式"):
        st.write(f"平均差 = {result['mean1']:.4g} − {result['mean2']:.4g} = {result['mean_difference']:.4g}")
        st.latex(r"t=\frac{\bar{x}_1-\bar{x}_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}")
        st.write(f"標準誤 = {result['standard_error']:.6g}；df = {result['df']:.4g}；Cohen's d = {result['cohens_d']:.4g}")

elif analysis == "比較同一批人的兩次測量（paired t-test）":
    before = st.selectbox("測量 1", numeric_options); after = st.selectbox("測量 2", [c for c in numeric_options if c != before])
    result = paired_ttest(data[after], data[before]); show_result(result, alpha, "t")
    pairs = data[[before, after]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    long = pairs.reset_index(names="pair").melt(id_vars="pair", var_name="time", value_name="value")
    fig = px.line(long, x="time", y="value", color="pair", markers=True, title="每一對測量的變化")
    fig.update_layout(showlegend=False); st.plotly_chart(fig, width="stretch")
    with st.expander("逐步計算與公式"):
        st.latex(r"d_i=x_{i,2}-x_{i,1},\qquad t=\frac{\bar d}{s_d/\sqrt n},\qquad df=n-1")
        st.write(f"完整配對 n = {result['n_pairs']}；差值平均 = {result['mean_difference']:.5g}；差值 SD = {result['sd_difference']:.5g}；Cohen's dz = {result['cohens_dz']:.4g}")

elif analysis == "比較三組以上（ANOVA + post-hoc）":
    value = st.selectbox("結果數值欄", numeric_options); group = st.selectbox("分組欄", group_options(value))
    welch = st.checkbox("使用 Welch ANOVA（變異數不齊時建議）")
    normal, levene = assumption_checks(data, value, group)
    with st.expander("先看假設檢查", expanded=True):
        st.write("Shapiro–Wilk（各組常態性；p < α 表示常態假設有疑慮）")
        st.dataframe(normal, width="stretch")
        st.write(f"Levene 變異數同質性：statistic={levene['statistic']:.4g}, p={levene['p_value']:.4g}")
        st.caption("假設檢定對樣本數敏感，也請一起看 Q–Q plot、分布、離群值與研究設計。")
    result = one_way_anova(data, value, group, welch); show_result(result, alpha, "F")
    fig = px.box(data, x=group, y=value, color=group, points="all", title=f"{value} across {group}")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Post-hoc 逐對比較")
    method = st.selectbox("方法／p 值校正", ["逐對 Welch t-test｜不校正（預設）", "逐對 Welch t-test｜Holm", "逐對 Welch t-test｜FDR (BH)", "逐對 Welch t-test｜Bonferroni", "Tukey HSD"])
    if method == "Tukey HSD":
        post = tukey_hsd(data, value, group, alpha)
        st.warning("Tukey HSD 自己會控制 family-wise error；它不是『未校正』比較。")
    else:
        correction = {"逐對 Welch t-test｜不校正（預設）": "none", "逐對 Welch t-test｜Holm": "holm",
                      "逐對 Welch t-test｜FDR (BH)": "fdr_bh", "逐對 Welch t-test｜Bonferroni": "bonferroni"}[method]
        post = pairwise_ttests(data, value, group, correction)
        if correction == "none": st.warning("未校正的多次檢定會提高至少一次偽陽性的機率；請在報告中明確註明。")
    st.dataframe(post, width="stretch"); download_table(post, "post_hoc.csv")
    with st.expander("ANOVA 是怎麼算的？"):
        st.latex(r"SS_B=\sum_j n_j(\bar x_j-\bar x)^2,\quad SS_W=\sum_j\sum_i(x_{ij}-\bar x_j)^2")
        st.latex(r"F=MS_B/MS_W=(SS_B/df_B)/(SS_W/df_W)")
        if not welch: st.json({k: round(float(result[k]), 6) for k in ["ss_between", "ss_within", "ms_between", "ms_within", "eta_squared"]})

elif analysis == "不符合常態的組別比較（非參數）":
    value = st.selectbox("結果數值欄", numeric_options); group = st.selectbox("分組欄", group_options(value))
    result = nonparametric_test(data, value, group); show_result(result, alpha, "U / H")
    st.plotly_chart(px.box(data, x=group, y=value, color=group, points="all", title=result["test"]), width="stretch")
    with st.expander("怎麼算？"):
        st.write("先把所有觀察值合併並轉成名次，再比較各組名次總和。兩組使用 Mann–Whitney U；三組以上使用 Kruskal–Wallis H。它們檢定的是分布／名次，不應一概描述成平均數差異。")

elif analysis == "同一批人的三次以上測量（repeated-measures ANOVA）":
    value = st.selectbox("結果數值欄", numeric_options)
    subject_choices = sorted([c for c in all_cols if c != value], key=lambda c: not any(t in c.lower() for t in ("subject", "patient", "participant", "id")))
    subject = st.selectbox("受試者 ID 欄", subject_choices)
    time_choices = sorted([c for c in group_options(value) if c != subject], key=lambda c: not any(t in c.lower() for t in ("time", "day", "session", "visit", "condition")))
    time = st.selectbox("時間／條件欄", time_choices)
    try:
        table, complete = repeated_measures_anova(data, value, time, subject)
    except ValueError as exc:
        st.error(str(exc)); st.caption("此分析需要長格式：同一個 subject 在每個 time 各有一列。"); st.stop()
    st.dataframe(table, width="stretch")
    p = float(table.iloc[0]["Pr > F"]); f = float(table.iloc[0]["F Value"])
    show_result({"statistic": f, "p_value": p}, alpha, "F")
    st.plotly_chart(px.line(complete, x=time, y=value, color=subject, markers=True,
                            title="每位受試者在各時間／條件的變化"), width="stretch")
    correction_label = st.selectbox("事後配對 t-test 校正", ["none", "holm", "fdr_bh", "bonferroni"])
    post = paired_posthoc(complete, value, time, subject, correction_label)
    if correction_label == "none": st.warning("未校正的多次檢定會提高偽陽性風險。")
    st.dataframe(post, width="stretch"); download_table(post, "repeated_measures_posthoc.csv")
    with st.expander("怎麼算與要注意什麼？"):
        st.write("Repeated-measures ANOVA 把受試者間差異從誤差中分離，再比較時間／條件的平均數。這裡只納入每個時間點都有資料的完整受試者。")
        st.write("三個以上水準還需要球形性假設；本工具目前不計算 Mauchly 檢定或 Greenhouse–Geisser 修正。若球形性不合理，建議使用線性混合模型。")

elif analysis == "兩個數值是否相關":
    x = st.selectbox("X", numeric_options); y = st.selectbox("Y", [c for c in numeric_options if c != x]); method = st.radio("方法", ["pearson", "spearman"], horizontal=True)
    result = correlation(data, x, y, method); show_result(result, alpha, "r / ρ")
    st.plotly_chart(px.scatter(data, x=x, y=y, trendline="ols" if method == "pearson" else None, title=result["test"]), width="stretch")
    with st.expander("公式與解讀"):
        st.latex(r"r=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}")
        st.write("Spearman 使用資料名次後再計算相關。相關不代表因果。")

elif analysis == "兩個類別變數是否相關（Chi-square / Fisher）":
    row = st.selectbox("列變數", group_options())
    column = st.selectbox("欄變數", [c for c in group_options() if c != row])
    result = categorical_association(data, row, column)
    show_result(result, alpha, "χ²")
    st.write(f"df = {result['df']}；Cramér's V = {result['cramers_v']:.4g}")
    if "fisher_p_value" in result:
        st.write(f"2×2 Fisher exact p = {result['fisher_p_value']:.4g}；odds ratio = {result['odds_ratio']:.4g}")
    c1, c2 = st.columns(2)
    c1.write("觀察次數"); c1.dataframe(result["observed"], width="stretch")
    c2.write("虛無假設下的期望次數"); c2.dataframe(result["expected"], width="stretch")
    proportions = result["observed"].div(result["observed"].sum(axis=1), axis=0).reset_index().melt(id_vars=row, var_name=column, value_name="proportion")
    st.plotly_chart(px.bar(proportions, x=row, y="proportion", color=column, barmode="stack", title="各列的類別比例"), width="stretch")
    with st.expander("公式與選擇 Fisher 的時機"):
        st.latex(r"E_{ij}=\frac{(row\ total_i)(column\ total_j)}{n},\qquad \chi^2=\sum\frac{(O_{ij}-E_{ij})^2}{E_{ij}}")
        st.write("若 2×2 表的期望次數很小，優先參考 Fisher exact p-value。Cramér's V 是 0–1 的關聯強度，但不代表因果。")

elif analysis == "用多個變數預測數值結果（線性迴歸）":
    outcome = st.selectbox("結果 Y", numeric_options); predictors = st.multiselect("預測變數 X（可多選）", [c for c in numeric_options if c != outcome])
    if not predictors: st.info("請至少選一個預測變數。"); st.stop()
    result = linear_regression(data, outcome, predictors)
    c1, c2, c3 = st.columns(3); c1.metric("n", result["n"]); c2.metric("R²", f"{result['r_squared']:.4g}"); c3.metric("Adjusted R²", f"{result['adjusted_r_squared']:.4g}")
    st.dataframe(result["coefficients"], width="stretch"); download_table(result["coefficients"], "regression_coefficients.csv")
    diag = pd.DataFrame({"fitted": result["fitted"], "residual": result["residuals"]})
    st.plotly_chart(px.scatter(diag, x="fitted", y="residual", title="Residuals vs fitted（應無明顯曲線或漏斗）"), width="stretch")
    with st.expander("模型如何估計？"):
        st.latex(r"Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\epsilon,\qquad \hat\beta=(X^TX)^{-1}X^Ty")
        st.write("最小平方法選擇使殘差平方和 Σ(y−ŷ)² 最小的係數。係數是在其他納入變數固定時，X 增加 1 單位對 Y 的預期改變。")

elif analysis == "預測二元結果（logistic regression）":
    outcome = st.selectbox("二元結果欄", group_options())
    levels = list(data[outcome].dropna().unique())
    if len(levels) != 2: st.error("結果欄必須剛好有兩個類別。"); st.stop()
    positive = st.selectbox("哪一類定義為事件（Y=1）", levels)
    predictors = st.multiselect("數值預測變數", [c for c in numeric if c != outcome])
    if not predictors: st.info("請至少選一個預測變數。"); st.stop()
    try:
        result = logistic_regression(data, outcome, positive, predictors)
    except Exception as exc:
        st.error(f"模型無法估計：{exc}"); st.caption("常見原因是樣本太少、完全分離或預測變數高度重複。"); st.stop()
    c1, c2, c3 = st.columns(3); c1.metric("n", result["n"]); c2.metric("McFadden pseudo-R²", f"{result['pseudo_r_squared']:.4g}"); c3.metric("AIC", f"{result['aic']:.4g}")
    st.dataframe(result["coefficients"], width="stretch"); download_table(result["coefficients"], "logistic_coefficients.csv")
    coef = result["coefficients"].query("term != 'const'")
    fig = go.Figure(go.Scatter(x=coef["odds_ratio"], y=coef["term"], mode="markers",
                               error_x=dict(type="data", symmetric=False,
                                            array=coef["or_ci95_high"] - coef["odds_ratio"],
                                            arrayminus=coef["odds_ratio"] - coef["or_ci95_low"])))
    fig.add_vline(x=1, line_dash="dash"); fig.update_layout(title="Odds ratios 與 95% CI", xaxis_type="log", xaxis_title="Odds ratio（log scale）", yaxis_title="Predictor")
    st.plotly_chart(fig, width="stretch")
    with st.expander("公式與解讀"):
        st.latex(r"\log\frac{p}{1-p}=\beta_0+\beta_1X_1+\cdots+\beta_pX_p,\qquad OR=e^{\beta}")
        st.write("OR > 1 表示事件勝算隨 X 增加而上升；OR < 1 表示下降。這是勝算，不是機率的直接倍數。")

else:
    outcome = st.selectbox("結果數值欄", numeric_options); factors = st.multiselect("類別因子（1–3 個）", group_options(outcome), max_selections=3)
    if not factors: st.info("請選 1–3 個因子。"); st.stop()
    model, table, formula = factorial_anova(data, outcome, factors)
    st.dataframe(table, width="stretch"); download_table(table, "factorial_anova.csv")
    st.caption(f"內部模型式：{formula}；使用 Type II sums of squares。")
    if len(factors) >= 2:
        means = data.groupby(factors, observed=True)[outcome].mean().reset_index()
        st.plotly_chart(px.line(means, x=factors[0], y=outcome, color=factors[1], markers=True,
                                title="交互作用圖：線條不平行可能表示交互作用"), width="stretch")
    with st.expander("怎麼解讀？"):
        st.write("主效應描述某因子平均而言是否相關；交互作用表示一個因子的效果會隨另一因子的水準而改變。若交互作用顯著，優先解讀簡單效應，而不是單獨解讀主效應。")

st.divider()
st.caption("本工具協助計算與教學，不取代研究設計、領域判斷或專業統計諮詢。請報告樣本數、效果大小、信賴區間、精確 p 值、缺失值處理及事先規劃的分析。")
