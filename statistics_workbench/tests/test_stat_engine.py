import numpy as np
import pandas as pd

from stat_engine import (
    categorical_association, independent_ttest, logistic_regression, one_way_anova,
    paired_ttest, pairwise_ttests, repeated_measures_anova,
)


def test_independent_t_matches_known_manual_value():
    result = independent_ttest([1, 2, 3], [4, 5, 6], equal_var=True)
    assert np.isclose(result["statistic"], -3.674234614)
    assert result["df"] == 4
    assert np.isclose(result["mean_difference"], -3)


def test_paired_t_uses_complete_pairs():
    result = paired_ttest([2, 4, np.nan, 8], [1, 2, 3, 4])
    assert result["n_pairs"] == 3
    assert np.isclose(result["mean_difference"], 7 / 3)


def test_anova_sums_of_squares_add_up():
    df = pd.DataFrame({"g": list("AAABBBCCC"), "y": [1, 2, 3, 2, 3, 4, 7, 8, 9]})
    result = one_way_anova(df, "y", "g")
    assert np.isclose(result["ss_total"], result["ss_between"] + result["ss_within"])
    assert 0 <= result["eta_squared"] <= 1


def test_uncorrected_posthoc_reports_raw_p():
    df = pd.DataFrame({"g": list("AAABBBCCC"), "y": [1, 2, 3, 2, 3, 4, 7, 8, 9]})
    result = pairwise_ttests(df, "y", "g", "none")
    assert len(result) == 3
    assert np.allclose(result["p_raw"], result["p_reported"])


def test_repeated_measures_keeps_complete_subjects():
    df = pd.DataFrame({"id": np.repeat([1, 2, 3], 3), "time": list("ABC") * 3,
                       "y": [1, 2, 4, 2, 4, 7, 3, 5, 8]})
    table, complete = repeated_measures_anova(df, "y", "time", "id")
    assert len(complete) == 9
    assert table.loc[0, "F Value"] > 0


def test_chi_square_returns_expected_counts_and_effect_size():
    df = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 10,
                       "b": ["yes"] * 8 + ["no"] * 2 + ["yes"] * 2 + ["no"] * 8})
    result = categorical_association(df, "a", "b")
    assert result["expected"].to_numpy().sum() == 20
    assert 0 <= result["cramers_v"] <= 1
    assert "fisher_p_value" in result


def test_logistic_regression_reports_odds_ratios():
    x = np.arange(-5, 6, dtype=float)
    df = pd.DataFrame({"event": (x > 0).astype(int), "x": x + np.linspace(-.2, .2, len(x))})
    # Add overlap so the maximum-likelihood estimate is finite.
    df.loc[[4, 7], "event"] = [1, 0]
    result = logistic_regression(df, "event", 1, ["x"])
    assert result["n"] == len(df)
    assert (result["coefficients"]["odds_ratio"] > 0).all()
