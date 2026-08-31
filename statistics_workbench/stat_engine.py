"""Pure statistical functions used by the Streamlit interface.

The functions deliberately return intermediate quantities.  The UI can therefore
show where every test statistic came from instead of presenting a black-box p-value.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans


def clean_numeric(values: Iterable) -> np.ndarray:
    """Convert values to finite floats, silently removing missing/non-numeric rows."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def descriptive_table(df: pd.DataFrame, value: str, group: str | None = None) -> pd.DataFrame:
    work = df[[value] + ([group] if group else [])].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    if group:
        grouped = work.dropna().groupby(group, observed=True)[value]
    else:
        work["All data"] = "All data"
        group = "All data"
        grouped = work.dropna().groupby(group, observed=True)[value]
    out = grouped.agg(n="count", mean="mean", median="median", sd="std", minimum="min", maximum="max")
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    out["ci95_low"] = out["mean"] - stats.t.ppf(.975, out["n"] - 1) * out["sem"]
    out["ci95_high"] = out["mean"] + stats.t.ppf(.975, out["n"] - 1) * out["sem"]
    return out.reset_index()


def independent_ttest(a: Iterable, b: Iterable, equal_var: bool = False) -> dict:
    a, b = clean_numeric(a), clean_numeric(b)
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group needs at least 2 valid observations.")
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
        d_denominator = np.sqrt(pooled_var)
    else:
        pooled_var = np.nan
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        d_denominator = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    t_value = (m1 - m2) / se
    p_value = 2 * stats.t.sf(abs(t_value), df)
    critical = stats.t.ppf(.975, df)
    diff = m1 - m2
    return {
        "test": "Student independent t-test" if equal_var else "Welch independent t-test",
        "n1": n1, "n2": n2, "mean1": m1, "mean2": m2, "variance1": v1, "variance2": v2,
        "mean_difference": diff, "standard_error": se, "statistic": t_value, "df": df,
        "p_value": p_value, "ci_low": diff - critical * se, "ci_high": diff + critical * se,
        "cohens_d": diff / d_denominator if d_denominator else np.nan, "pooled_variance": pooled_var,
    }


def paired_ttest(a: Iterable, b: Iterable) -> dict:
    pair = pd.DataFrame({"a": pd.to_numeric(pd.Series(a), errors="coerce"),
                         "b": pd.to_numeric(pd.Series(b), errors="coerce")}).dropna()
    diff = (pair["a"] - pair["b"]).to_numpy()
    if len(diff) < 2:
        raise ValueError("At least 2 complete pairs are required.")
    n, md, sd = len(diff), diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n)
    t_value, df = md / se, n - 1
    critical = stats.t.ppf(.975, df)
    return {"test": "Paired t-test", "n_pairs": n, "mean_difference": md, "sd_difference": sd,
            "standard_error": se, "statistic": t_value, "df": df,
            "p_value": 2 * stats.t.sf(abs(t_value), df),
            "ci_low": md - critical * se, "ci_high": md + critical * se,
            "cohens_dz": md / sd if sd else np.nan, "differences": diff}


def one_way_anova(df: pd.DataFrame, value: str, group: str, welch: bool = False) -> dict:
    work = df[[value, group]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    arrays = [g[value].to_numpy() for _, g in work.groupby(group, observed=True)]
    names = [str(name) for name, _ in work.groupby(group, observed=True)]
    if len(arrays) < 2 or any(len(x) < 2 for x in arrays):
        raise ValueError("At least 2 groups with 2 observations each are required.")
    if welch:
        result = anova_oneway(arrays, use_var="unequal", welch_correction=True)
        return {"test": "Welch one-way ANOVA", "statistic": float(result.statistic),
                "df_between": float(result.df_num), "df_within": float(result.df_denom),
                "p_value": float(result.pvalue), "groups": names}
    grand = work[value].mean()
    ns = np.array([len(x) for x in arrays])
    means = np.array([x.mean() for x in arrays])
    ss_between = float(np.sum(ns * (means - grand) ** 2))
    ss_within = float(sum(np.sum((x - x.mean()) ** 2) for x in arrays))
    ss_total = ss_between + ss_within
    df_between, df_within = len(arrays) - 1, len(work) - len(arrays)
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    f_value = ms_between / ms_within
    return {"test": "One-way ANOVA", "statistic": f_value, "df_between": df_between,
            "df_within": df_within, "p_value": stats.f.sf(f_value, df_between, df_within),
            "ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total,
            "ms_between": ms_between, "ms_within": ms_within,
            "eta_squared": ss_between / ss_total if ss_total else np.nan, "groups": names}


def pairwise_ttests(df: pd.DataFrame, value: str, group: str, correction: str = "none") -> pd.DataFrame:
    work = df[[value, group]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    rows = []
    for a, b in combinations(work[group].unique(), 2):
        result = independent_ttest(work.loc[work[group] == a, value], work.loc[work[group] == b, value])
        rows.append({"group_1": a, "group_2": b, "mean_difference": result["mean_difference"],
                     "t": result["statistic"], "df": result["df"], "p_raw": result["p_value"],
                     "cohens_d": result["cohens_d"]})
    out = pd.DataFrame(rows)
    method_map = {"holm": "holm", "fdr_bh": "fdr_bh", "bonferroni": "bonferroni"}
    if correction in method_map and not out.empty:
        reject, p_adj, _, _ = multipletests(out["p_raw"], method=method_map[correction])
        out["p_reported"], out["reject"] = p_adj, reject
    else:
        out["p_reported"], out["reject"] = out["p_raw"], out["p_raw"] < .05
    out["correction"] = correction
    return out


def paired_posthoc(df: pd.DataFrame, value: str, time: str, subject: str,
                   correction: str = "none") -> pd.DataFrame:
    """Pairwise paired t-tests after reshaping complete subject pairs."""
    work = df[[subject, time, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    pivot = work.dropna().pivot_table(index=subject, columns=time, values=value, aggfunc="mean")
    rows = []
    for a, b in combinations(pivot.columns, 2):
        complete = pivot[[a, b]].dropna()
        result = paired_ttest(complete[a], complete[b])
        rows.append({"time_1": a, "time_2": b, "n_pairs": result["n_pairs"],
                     "mean_difference": result["mean_difference"], "t": result["statistic"],
                     "df": result["df"], "p_raw": result["p_value"], "cohens_dz": result["cohens_dz"]})
    out = pd.DataFrame(rows)
    method_map = {"holm": "holm", "fdr_bh": "fdr_bh", "bonferroni": "bonferroni"}
    if correction in method_map and not out.empty:
        reject, p_adj, _, _ = multipletests(out["p_raw"], method=method_map[correction])
        out["p_reported"], out["reject"] = p_adj, reject
    else:
        out["p_reported"], out["reject"] = out["p_raw"], out["p_raw"] < .05
    out["correction"] = correction
    return out


def repeated_measures_anova(df: pd.DataFrame, value: str, time: str, subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-factor repeated-measures ANOVA on subjects complete at every time."""
    work = df[[subject, time, value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    if work.duplicated([subject, time]).any():
        work = work.groupby([subject, time], observed=True, as_index=False)[value].mean()
    wide = work.pivot(index=subject, columns=time, values=value).dropna()
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        raise ValueError("Need at least 2 complete subjects measured at 2 or more time points.")
    complete = wide.reset_index().melt(id_vars=subject, var_name=time, value_name=value)
    result = AnovaRM(complete, depvar=value, subject=subject, within=[time]).fit()
    table = result.anova_table.reset_index().rename(columns={"index": "effect"})
    return table, complete


def tukey_hsd(df: pd.DataFrame, value: str, group: str, alpha: float = .05) -> pd.DataFrame:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    work = df[[value, group]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    result = pairwise_tukeyhsd(work[value], work[group].astype(str), alpha=alpha)
    table = result.summary().data
    return pd.DataFrame(table[1:], columns=table[0])


def assumption_checks(df: pd.DataFrame, value: str, group: str) -> tuple[pd.DataFrame, dict]:
    work = df[[value, group]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    rows, arrays = [], []
    for name, g in work.groupby(group, observed=True):
        x = g[value].to_numpy()
        arrays.append(x)
        if 3 <= len(x) <= 5000:
            statistic, p = stats.shapiro(x)
            rows.append({"group": name, "n": len(x), "Shapiro_W": statistic, "p_value": p})
        else:
            rows.append({"group": name, "n": len(x), "Shapiro_W": np.nan, "p_value": np.nan})
    if len(arrays) >= 2 and all(len(x) >= 2 for x in arrays):
        lev = stats.levene(*arrays, center="median")
        levene = {"statistic": lev.statistic, "p_value": lev.pvalue}
    else:
        levene = {"statistic": np.nan, "p_value": np.nan}
    return pd.DataFrame(rows), levene


def nonparametric_test(df: pd.DataFrame, value: str, group: str) -> dict:
    work = df[[value, group]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna()
    arrays = [g[value].to_numpy() for _, g in work.groupby(group, observed=True)]
    if len(arrays) == 2:
        r = stats.mannwhitneyu(*arrays, alternative="two-sided")
        return {"test": "Mann–Whitney U", "statistic": r.statistic, "p_value": r.pvalue}
    if len(arrays) > 2:
        r = stats.kruskal(*arrays)
        return {"test": "Kruskal–Wallis H", "statistic": r.statistic, "p_value": r.pvalue,
                "df": len(arrays) - 1}
    raise ValueError("At least 2 groups are required.")


def correlation(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> dict:
    work = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        raise ValueError("At least 3 complete observations are required.")
    result = stats.pearsonr(work[x], work[y]) if method == "pearson" else stats.spearmanr(work[x], work[y])
    return {"test": f"{method.title()} correlation", "n": len(work), "statistic": result.statistic,
            "p_value": result.pvalue, "x": work[x].to_numpy(), "y": work[y].to_numpy()}


def categorical_association(df: pd.DataFrame, row: str, column: str) -> dict:
    table = pd.crosstab(df[row], df[column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Each variable needs at least 2 observed categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    cramers_v = np.sqrt(phi2 / min(table.shape[0] - 1, table.shape[1] - 1))
    result = {"test": "Pearson chi-square", "statistic": chi2, "p_value": p, "df": dof,
              "cramers_v": cramers_v, "observed": table,
              "expected": pd.DataFrame(expected, index=table.index, columns=table.columns)}
    if table.shape == (2, 2):
        odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
        result.update({"fisher_p_value": fisher_p, "odds_ratio": odds_ratio})
    return result


def linear_regression(df: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    work = df[[outcome] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) <= len(predictors) + 1:
        raise ValueError("More complete rows than model parameters are required.")
    X = sm.add_constant(work[predictors], has_constant="add")
    model = sm.OLS(work[outcome], X).fit()
    ci = model.conf_int()
    coefficients = pd.DataFrame({"term": model.params.index, "estimate": model.params.values,
                                 "std_error": model.bse.values, "t": model.tvalues.values,
                                 "p_value": model.pvalues.values,
                                 "ci95_low": ci[0].values, "ci95_high": ci[1].values})
    return {"model": model, "coefficients": coefficients, "n": int(model.nobs), "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj, "f_statistic": model.fvalue, "f_p_value": model.f_pvalue,
            "observed": work[outcome].to_numpy(), "fitted": model.fittedvalues.to_numpy(),
            "residuals": model.resid.to_numpy()}


def logistic_regression(df: pd.DataFrame, outcome: str, positive, predictors: list[str]) -> dict:
    work = df[[outcome] + predictors].copy()
    work["__binary_outcome"] = (work[outcome] == positive).astype(float)
    work[predictors] = work[predictors].apply(pd.to_numeric, errors="coerce")
    work = work.dropna(subset=predictors + [outcome])
    if work["__binary_outcome"].nunique() != 2:
        raise ValueError("The selected data must contain both outcome classes.")
    X = sm.add_constant(work[predictors], has_constant="add")
    model = sm.Logit(work["__binary_outcome"], X).fit(disp=False)
    ci = model.conf_int()
    coefficients = pd.DataFrame({"term": model.params.index, "log_odds": model.params.values,
                                 "std_error": model.bse.values, "z": model.tvalues.values,
                                 "p_value": model.pvalues.values, "odds_ratio": np.exp(model.params.values),
                                 "or_ci95_low": np.exp(ci[0].values), "or_ci95_high": np.exp(ci[1].values)})
    return {"model": model, "coefficients": coefficients, "n": int(model.nobs),
            "pseudo_r_squared": model.prsquared, "aic": model.aic,
            "observed": work["__binary_outcome"].to_numpy(), "predicted": model.predict(X).to_numpy()}


def factorial_anova(df: pd.DataFrame, outcome: str, factors: list[str]) -> tuple[object, pd.DataFrame, str]:
    if not 1 <= len(factors) <= 3:
        raise ValueError("Choose 1 to 3 factors.")
    safe = df[[outcome] + factors].copy().dropna()
    safe[outcome] = pd.to_numeric(safe[outcome], errors="coerce")
    safe = safe.dropna()
    renamed = {outcome: "outcome", **{factor: f"factor_{i}" for i, factor in enumerate(factors)}}
    safe = safe.rename(columns=renamed)
    terms = " * ".join(f"C({renamed[f]})" for f in factors)
    formula = f"outcome ~ {terms}"
    model = smf.ols(formula, data=safe).fit()
    table = anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    return model, table, formula
