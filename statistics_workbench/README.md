# Statistics Workbench｜統計分析工作台

給統計初學者使用的互動式 Python 工具。它能讀取 CSV/Excel，用中文引導選欄位，產生統計結果、效果大小、假設檢查、互動圖表，以及可核對的公式與中間值。

## 安裝與啟動

```bash
cd statistics_workbench
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

瀏覽器會開啟 `http://localhost:8501`。沒有資料也可直接使用內建示範資料。

macOS/Linux 也可以直接執行 `./run.sh`（第一次會自動安裝套件）。建議使用 Python 3.10 以上。

macOS 使用者也可以直接雙擊 `Start Statistics Workbench.command`。啟動後請保持該 Terminal 視窗開啟；關閉視窗或按 `Control+C` 會停止網站。

## 建議資料格式

一般分析建議使用「長格式」，每一列是一筆觀察：

```csv
subject,group,time,accuracy,age
1,BCI,day0,72,25
1,BCI,day7,84,25
2,Control,day0,73,31
2,Control,day7,75,31
```

目前包含：描述統計、獨立與配對 t-test、one-way/Welch ANOVA、repeated-measures ANOVA、post-hoc（None/Holm/FDR/Bonferroni/Tukey）、Mann–Whitney、Kruskal–Wallis、Pearson/Spearman、Chi-square/Fisher exact、複線性迴歸、logistic regression，以及多因子 ANOVA 與交互作用。

## 重要統計原則

- 「未達顯著」不等於「證明沒有差異」。
- 未校正 post-hoc 是刻意提供的選項，但比較愈多，偽陽性風險愈高。
- Tukey HSD 本身是多重比較程序，不能稱為完全未校正。
- 先根據研究設計決定檢定；不要只靠常態性 p 值挑方法。
- 圖、效果大小、信賴區間與資料品質，和 p 值一樣重要。
