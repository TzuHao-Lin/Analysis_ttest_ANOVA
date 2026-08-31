"""Teaching page for every analysis exposed by the workbench."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="統計檢定教學", page_icon="📘", layout="wide")
st.title("📘 統計檢定教學")
st.caption("先從研究問題選方法，再看資料格式與假設；不要只根據 p-value 決定一切。")
st.markdown("← 使用左側導覽列可回到 **app｜統計分析工作台**")


WORKED_EXAMPLES = {
    "Independent t-test｜兩個獨立組別": {
        "story": "六位不同受試者完成同一項任務。三人使用 BCI，另外三人使用 Control。我們想知道 BCI 的平均正確率是否較高。",
        "data": {"participant": ["B1", "B2", "B3", "C1", "C2", "C3"],
                 "group": ["BCI"] * 3 + ["Control"] * 3,
                 "accuracy": [82, 84, 86, 72, 74, 76]},
        "chart": "strip", "x": "group", "y": "accuracy", "color": "group",
        "look": "先只看資料：BCI 的三個點都比 Control 高。BCI 平均 84，Control 平均 74，看到的平均差是 10 分。",
        "calculation": "檢定不是只問『差幾分』，還問『每組內部有多亂』。這裡兩組內都只上下波動約 2 分，所以 10 分差異相對於組內雜訊很大。t 值就是把平均差除以平均差的標準誤，也就是一種「訊號 ÷ 雜訊」。",
        "bridge": "如果 Control 改成 60、74、88，平均仍是 74，但組內非常分散；同樣的 10 分平均差就不再那麼有說服力。這就是 t-test 為什麼同時需要平均數、SD 與樣本數。",
        "plain": "人話結論：這份小資料顯示 BCI 組整體較高；正式結論還要看 95% CI、p-value、效果大小與研究設計。",
    },
    "One-way ANOVA｜三個以上獨立組別": {
        "story": "我們比較 Control、Training、BCI 三組不同受試者的正確率。若分別做三次 t-test，錯把隨機波動判成差異的機會會一直增加。",
        "data": {"group": ["Control"] * 3 + ["Training"] * 3 + ["BCI"] * 3,
                 "accuracy": [70, 72, 74, 76, 78, 80, 84, 86, 88]},
        "chart": "strip", "x": "group", "y": "accuracy", "color": "group",
        "look": "三組平均分別是 72、78、86。組與組之間相隔明顯，而每一組內部只波動約 2 分。",
        "calculation": "ANOVA 把總變動拆成兩部分：組別平均造成的『組間變動』，以及同組受試者仍不一樣的『組內變動』。F = 組間變動 ÷ 組內變動。F 很大表示組別訊號大於背景雜訊。",
        "bridge": "若三組平均仍是 72、78、86，但每組資料散到 40–110，組內雜訊就會蓋過組間差異，F 會下降。ANOVA 因此不是只比較三個平均數。",
        "plain": "ANOVA 顯著只能說至少一組不同；接著才用 post-hoc 找出差異在哪裡。",
    },
    "Factorial ANOVA｜多因子與交互作用": {
        "story": "現在同時考慮治療組別與時間。我們想知道 BCI 是否有效、大家是否隨時間進步，以及 BCI 組是不是進步得特別快。",
        "data": {"group": ["Control", "Control", "BCI", "BCI"],
                 "time": ["day0", "day14", "day0", "day14"],
                 "mean_accuracy": [70, 72, 70, 85]},
        "chart": "line", "x": "time", "y": "mean_accuracy", "color": "group",
        "look": "day0 兩組都是 70；14 天後 Control 只到 72，BCI 到 85。圖上兩條線不平行。",
        "calculation": "Group 主效應比較跨時間平均：Control 71、BCI 77.5。Time 主效應比較跨組平均：day0 70、day14 78.5。交互作用比較『改變量的差』：BCI 增加 15，Control 增加 2，所以 difference-in-differences = 15 − 2 = 13。",
        "bridge": "這個 13 分就是交互作用的直覺：時間的效果取決於使用哪一組系統。若兩組都增加 5 分，兩條線平行，便沒有 Group × Time 交互作用。",
        "plain": "角色對照：accuracy 是 outcome；group、time 是 factors；BCI、Control、day0、day14 是 levels；subject 是 ID。若同一人跨時間重測，需 mixed ANOVA／混合模型，不能用普通 factorial ANOVA。",
    },
    "Paired t-test｜同一批人的兩次測量": {
        "story": "四位受試者在介入前後各測一次。每個 after 必須和同一人的 before 配在一起。",
        "data": {"subject": ["P1", "P2", "P3", "P4"], "before": [60, 70, 80, 90],
                 "after": [65, 75, 85, 95], "difference": [5, 5, 5, 5]},
        "chart": "paired", "x": "subject", "y": "score", "color": "time",
        "look": "每個人的起點差很多，但每個人都增加 5 分。重點不是 before 與 after 各自有多分散，而是四個差值有多一致。",
        "calculation": "先逐人算 d = after − before，得到 5、5、5、5；再檢定平均差是否為 0。個體固定的高低被配對消除，所以常比獨立 t-test 更有力。",
        "bridge": "若忽略配對，只看兩欄 SD，60–95 的個體差異會被當成雜訊；paired t-test 正是利用『同一個人和自己比較』。",
        "plain": "人話結論：每位受試者都改善 5 分。實際資料通常不會如此整齊，因此仍要看差值 SD、CI 與異常差值。",
    },
    "Repeated-measures ANOVA｜同一批人的三次以上測量": {
        "story": "三位受試者在 day0、day7、day14 測量。P3 一直比 P1 高，但我們真正關心的是每個人是否隨時間改變。",
        "data": {"subject": ["P1"] * 3 + ["P2"] * 3 + ["P3"] * 3,
                 "time": ["day0", "day7", "day14"] * 3,
                 "accuracy": [60, 66, 72, 70, 75, 81, 80, 86, 91]},
        "chart": "line", "x": "time", "y": "accuracy", "color": "subject",
        "look": "三條線的高度不同，代表個體基準不同；但三條線都往上，代表共同的時間趨勢。",
        "calculation": "Repeated-measures ANOVA 先辨認『P3 本來就高』這類受試者差異，再用剩餘誤差評估時間效果。因此 subject 不是普通組別，而是重複測量的配對單位。",
        "bridge": "普通 ANOVA 會把九個點當成九個互不相關的人，破壞資料結構。重複測量模型則知道同一條線上的三個點屬於同一人。",
        "plain": "三個以上時間點還涉及球形性；若資料缺失、不等間隔或結構更複雜，線性混合模型通常更合適。",
    },
    "Mann–Whitney U｜兩個獨立組別的名次比較": {
        "story": "兩組疼痛分數非常偏態，還有極端值。與其讓極端數字主導平均數，我們先把所有分數從小到大排成名次。",
        "data": {"score": [1, 2, 3, 6, 8, 30], "group": ["Treatment"] * 3 + ["Control"] * 3,
                 "rank": [1, 2, 3, 4, 5, 6]},
        "chart": "strip", "x": "group", "y": "score", "color": "group",
        "look": "Treatment 佔據較低的名次 1、2、3；Control 佔據 4、5、6。30 很大，但在名次中只是最高的第 6 名。",
        "calculation": "Mann–Whitney 比較兩組名次總和，也可理解成隨機抽一位 Treatment 和一位 Control 時，哪一組較常出現較大的值。",
        "bridge": "因為它丟掉實際距離，只保留順序，所以較不受 30 這種極端值影響；代價是它回答分布／排序問題，不是直接比較平均數。",
        "plain": "只有兩組分布形狀相近時，才較適合簡化成『中位數位置不同』。",
    },
    "Kruskal–Wallis H｜三組以上的名次比較": {
        "story": "把 Mann–Whitney 的名次想法擴展到 Control、Training、BCI 三組。",
        "data": {"group": ["Control"] * 3 + ["Training"] * 3 + ["BCI"] * 3,
                 "score": [1, 3, 4, 5, 6, 7, 8, 9, 12]},
        "chart": "strip", "x": "group", "y": "score", "color": "group",
        "look": "Control 多位於較低名次，BCI 多位於較高名次，Training 居中。",
        "calculation": "先把九個值一起轉成名次，再比較各組平均名次相距多遠。H 越大，代表組別名次分布越難用隨機波動解釋。",
        "bridge": "它像是『對名次做 one-way ANOVA』的直覺版本，但顯著後仍需適當的名次 post-hoc。",
        "plain": "H 顯著不代表三組彼此都不同，也不自動表示中位數不同。",
    },
    "Pearson correlation｜線性相關": {
        "story": "觀察訓練時數與正確率是否一起上升。",
        "data": {"training_hours": [1, 2, 3, 4, 5], "accuracy": [55, 60, 62, 72, 78]},
        "chart": "scatter", "x": "training_hours", "y": "accuracy",
        "look": "點大致沿著右上方向排列：訓練時數高的人，正確率通常也高。",
        "calculation": "Pearson r 同時查看每個 X 與平均 X 的距離，以及 Y 與平均 Y 的距離。兩者常同方向偏離時，乘積多為正，r 就接近 +1。",
        "bridge": "若點沿直線但向右下，r 接近 −1；若像一團雲，正負乘積互相抵銷，r 接近 0。r 是線性排列程度，不是斜率。",
        "plain": "即使 r 很高，也可能是能力、年齡等第三變數同時影響訓練時數與正確率；相關不證明因果。",
    },
    "Spearman correlation｜單調名次相關": {
        "story": "五位受試者的訓練名次和表現名次幾乎一致，但實際分數差距不是直線。",
        "data": {"participant": ["A", "B", "C", "D", "E"], "training_rank": [1, 2, 3, 4, 5],
                 "performance_rank": [1, 2, 4, 3, 5]},
        "chart": "scatter", "x": "training_rank", "y": "performance_rank",
        "look": "除了 C、D 小幅交換，排序大致一致。",
        "calculation": "Spearman 先把原始數值各自轉成名次，再對兩組名次算 Pearson correlation。因此它問的是『排序是否一起變動』。",
        "bridge": "只要 X 增加時 Y 大致持續增加，關係即使彎曲也能有高 Spearman ρ；但 U 型關係先降後升，不是單調關係。",
        "plain": "它適合序位或偏態資料，但仍應看散點圖，避免把非單調形狀濃縮成一個數字。",
    },
    "Chi-square / Fisher exact｜兩個類別變數": {
        "story": "比較 BCI 與 Control 組的成功／失敗人數。",
        "data": {"group": ["BCI", "Control"], "success": [8, 2], "failure": [2, 8]},
        "chart": "bar-wide", "x": "group", "y": "count", "color": "outcome",
        "look": "BCI 是 8 成成功，Control 是 2 成成功，兩組比例看起來不同。",
        "calculation": "H₀ 說 group 與 outcome 無關。若無關，兩組都應接近整體成功率 50%，也就是每組期望 5 成功、5 失敗。χ² 把觀察到的 8/2 與期望的 5/5 差距標準化後加總。",
        "bridge": "觀察次數 O 和無關情況下的期望次數 E 差越大，χ² 越大。小型 2×2 表因近似可能不穩，Fisher exact 直接計算所有同樣邊際總數的表。",
        "plain": "人話結論：成功比例與組別有關聯；是否為因果仍取決於是否隨機分派等研究設計。",
    },
    "Multiple linear regression｜預測連續結果": {
        "story": "我們想用訓練時數預測正確率，同時控制 age。",
        "data": {"training_hours": [1, 2, 3, 4, 5], "age": [30, 30, 30, 30, 30],
                 "accuracy": [53, 56, 59, 62, 65]},
        "chart": "scatter", "x": "training_hours", "y": "accuracy",
        "look": "age 固定為 30 時，每多訓練 1 小時，accuracy 約增加 3 分。",
        "calculation": "模型嘗試畫出一個能讓所有預測誤差平方和最小的平面。training 的 β≈3，意思是 age 相同時，訓練多 1 小時的預期差異是 3 分。",
        "bridge": "『控制 age』是在模型中比較 age 相同但 training 不同的人；它不是按一個按鈕就消除所有混雜，未測量變數仍可能造成偏誤。",
        "plain": "截距是所有 X=0 時的預測；若 0 不在合理範圍，截距通常只有計算用途。",
    },
    "Logistic regression｜預測二元結果": {
        "story": "Outcome 只有成功／失敗。我們想知道訓練時數增加時，成功的可能性如何改變。",
        "data": {"training_hours": [1, 2, 3, 4, 5], "successes_out_of_10": [1, 2, 4, 6, 8]},
        "chart": "scatter", "x": "training_hours", "y": "successes_out_of_10",
        "look": "成功比例從 10% 上升到 80%，但機率被限制在 0–100%，不能用一條無限延伸的普通直線。",
        "calculation": "Logistic regression 先把機率轉成 odds=p/(1−p)，再取 log，讓模型可以用線性組合預測。e^β 是 X 每增加 1 單位時 odds 乘上的倍數。",
        "bridge": "機率 20% 的 odds 是 .2/.8=.25；機率 50% 的 odds 是 1。odds 變 4 倍讓 .25 變 1，也就是機率從 20% 變 50%，不是變成 80%。",
        "plain": "因此 OR=2 表示勝算乘 2，不表示成功機率直接乘 2。",
    },
}


def render_worked_example(title: str) -> None:
    ex = WORKED_EXAMPLES.get(title)
    if not ex:
        return
    st.markdown("#### 先從一個具體例子開始")
    st.write(ex["story"])
    frame = pd.DataFrame(ex["data"])
    st.dataframe(frame, hide_index=True, width="stretch")
    chart_frame = frame.copy()
    if ex["chart"] == "paired":
        chart_frame = frame.melt(id_vars="subject", value_vars=["before", "after"], var_name="time", value_name="score")
        fig = px.line(chart_frame, x="time", y="score", color="subject", markers=True)
    elif ex["chart"] == "bar-wide":
        chart_frame = frame.melt(id_vars="group", value_vars=["success", "failure"], var_name="outcome", value_name="count")
        fig = px.bar(chart_frame, x="group", y="count", color="outcome", barmode="stack")
    elif ex["chart"] == "line":
        fig = px.line(chart_frame, x=ex["x"], y=ex["y"], color=ex["color"], markers=True)
    elif ex["chart"] == "scatter":
        fig = px.scatter(chart_frame, x=ex["x"], y=ex["y"], trendline="ols")
    else:
        fig = px.strip(chart_frame, x=ex["x"], y=ex["y"], color=ex["color"])
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=20), showlegend=True)
    st.plotly_chart(fig, width="stretch", key=f"example_{title}")
    st.markdown("**① 先用眼睛觀察**  " + ex["look"])
    st.markdown("**② 再看它怎麼計算**  " + ex["calculation"])
    st.markdown("**③ 把例子連回原理**  " + ex["bridge"])
    st.success(ex["plain"])


def lesson(
    title: str,
    question: str,
    use_when: str,
    data_shape: str,
    hypotheses: tuple[str, str],
    assumptions: list[str],
    formula: str | None,
    effect_size: str,
    report: str,
    caution: str,
) -> None:
    """Render one compact but complete beginner lesson."""
    with st.expander(title):
        render_worked_example(title)
        st.divider()
        st.markdown("#### 現在再看正式定義")
        st.markdown("**它回答的問題**  " + question)
        st.markdown("**什麼時候用**  " + use_when)
        st.markdown("**資料應該長什麼樣**  " + data_shape)
        c1, c2 = st.columns(2)
        c1.markdown("**H₀（虛無假設）**  " + hypotheses[0])
        c2.markdown("**H₁（對立假設）**  " + hypotheses[1])
        st.markdown("**主要假設／條件**")
        for item in assumptions:
            st.markdown(f"- {item}")
        if formula:
            st.markdown("**核心計算**")
            st.latex(formula)
        st.markdown("**效果大小**  " + effect_size)
        st.markdown("**報告範例**  " + report)
        st.warning(f"常見陷阱：{caution}")


st.subheader("30 秒選擇指南")
st.markdown(
    """
1. **結果是數值**：比較組別時考慮 t-test／ANOVA；預測數值時用線性迴歸。
2. **結果是二元類別**：使用 logistic regression。
3. **兩個變數都是類別**：使用 Chi-square；2×2 且期望次數很小時參考 Fisher exact。
4. **同一個人被重複測量**：兩次用 paired t-test；三次以上用 repeated-measures ANOVA。
5. **資料明顯偏態、離群且樣本小**：考慮 Mann–Whitney 或 Kruskal–Wallis，但先確認它回答的問題符合研究目的。
"""
)
st.info("p < α 表示：如果 H₀ 與模型假設成立，目前資料或更極端資料出現的機率很小。它不等於『H₀ 為真的機率』，也不代表效果一定重要。")
with st.expander("建議怎麼讀每一堂教學？", expanded=True):
    st.markdown(
        """
不要一開始背公式。每堂請照這個順序：

1. **遮住正式定義，只看小資料和圖**，先猜哪一組較高、趨勢是否不同。
2. 找到這個方法的**訊號**是什麼：平均差、名次差、共同變動、比例差，還是模型係數？
3. 找到它的**雜訊**是什麼：組內差異、差值變動、殘差，或抽樣波動？
4. 再看公式，將公式中的每個部分指回例子。
5. 最後用一句不含公式的人話說結論，再比較正式報告寫法。

幾乎所有推論統計都可以先理解成：**觀察到的訊號，相對於資料本身的雜訊有多大？**
"""
    )

mean_tab, repeated_tab, nonparametric_tab, association_tab, model_tab, posthoc_tab = st.tabs([
    "平均數比較", "重複測量", "非參數檢定", "關聯與類別資料", "迴歸與大型模型", "Post-hoc",
])

with mean_tab:
    lesson(
        "Independent t-test｜兩個獨立組別",
        "兩群互不重複的人，其母體平均數是否不同？",
        "例如 BCI 組與 Control 組是不同受試者，結果是連續數值。Welch t-test 通常是較安全的預設。",
        "長格式：每列一人，至少有 group 與 outcome 兩欄。",
        ("μ₁ − μ₂ = 0", "μ₁ − μ₂ ≠ 0（雙尾）"),
        ["兩組觀察值彼此獨立。", "每組分布沒有嚴重到足以扭曲平均數的離群值。", "Student t-test 另要求等變異；Welch 不要求。"],
        r"t=\frac{\bar{x}_1-\bar{x}_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}",
        "Cohen's d 表示標準化平均差；也應報告原單位平均差與 95% CI。",
        "BCI 組高於 Control 組 7.2 分，Welch t(17.4)=2.63, p=.017, d=1.12, 95% CI [1.4, 13.0]。",
        "同一個人若出現在兩組，資料並不獨立，不能使用 independent t-test。",
    )
    lesson(
        "One-way ANOVA｜三個以上獨立組別",
        "三個以上獨立群組的母體平均數是否至少有一組不同？",
        "一個類別因子、三個以上組別、一個連續結果。一般 ANOVA 假設等變異；Welch ANOVA 不要求。",
        "長格式：每列一人；group 欄標示組別，outcome 欄放數值。",
        ("μ₁ = μ₂ = ⋯ = μₖ", "至少有一個母體平均數不同"),
        ["觀察值彼此獨立。", "各組殘差大致常態，且沒有支配結果的極端離群值。", "一般 ANOVA 要求變異數同質；不合理時使用 Welch ANOVA。"],
        r"F=\frac{MS_{between}}{MS_{within}}=\frac{SS_B/df_B}{SS_W/df_W}",
        "η² 是組別解釋的總變異比例；也可報告 ω²。",
        "三組平均數不同，F(2,27)=6.41, p=.005, η²=.32；接著以事先選定的 post-hoc 找出哪些組不同。",
        "ANOVA 顯著只表示『至少一組不同』，不能直接說每一組都彼此不同。",
    )
    lesson(
        "Factorial ANOVA｜多因子與交互作用",
        "兩個或更多類別因子是否影響數值結果？其中一個因子的效果會不會隨另一因子而改變？",
        "例如 treatment × sex、group × time；每個因子有兩個以上水準。",
        "長格式：每列一筆獨立觀察，數個 factor 欄加一個 outcome 欄。",
        ("各主效應／交互作用的對應係數為 0", "至少一個對應效應不為 0"),
        ["觀察值獨立。", "模型殘差大致常態且變異數合理。", "不平衡設計需事先決定 Type II/III sums of squares 與對比編碼。"],
        r"Y=\beta_0+\beta_A A+\beta_B B+\beta_{AB}(A\times B)+\epsilon",
        "各效應可報 partial η²；同時呈現各 cell mean 與信賴區間。",
        "Group × Time 交互作用顯著，F(2,84)=4.52, p=.014, partial η²=.10。",
        "交互作用顯著時，單獨解讀主效應常會誤導；應查看簡單效應與交互作用圖。",
    )

with repeated_tab:
    lesson(
        "Paired t-test｜同一批人的兩次測量",
        "同一個人的兩次測量，其平均差是否為 0？",
        "Before/After、左右手，或一對一配對樣本；每個數值必須能和另一個數值正確配對。",
        "寬格式最直觀：每列一人，before 與 after 各一欄。",
        ("平均差 μd = 0", "平均差 μd ≠ 0"),
        ["每一對與其他對彼此獨立。", "差值 d，而不是兩個原始欄位，各自大致常態。", "沒有支配平均差的極端差值。"],
        r"d_i=x_{i,2}-x_{i,1},\qquad t=\frac{\bar d}{s_d/\sqrt n}",
        "Cohen's dz = 平均差／差值 SD；另報平均差與 95% CI。",
        "介入後平均增加 5.1 分，t(19)=3.24, p=.004, dz=.72, 95% CI [1.8, 8.4]。",
        "不能把有缺失的 before 與另一人的 after 錯誤配成一對；必須用 ID 對齊。",
    )
    lesson(
        "Repeated-measures ANOVA｜同一批人的三次以上測量",
        "同一批人在多個時間／條件下的平均數是否相同？",
        "同一個 subject 至少被測量三次，而且結果是連續數值。",
        "長格式：subject、time/condition、outcome；每個 subject × time 一列。",
        ("所有時間／條件的母體平均數相同", "至少一個時間／條件平均數不同"),
        ["不同受試者彼此獨立。", "模型殘差大致常態。", "三個以上水準涉及球形性；違反時需 Greenhouse–Geisser 等修正或混合模型。"],
        r"F=\frac{MS_{condition}}{MS_{condition\times subject}}",
        "可報 partial η²；事後配對比較則報平均差、95% CI 與 dz。",
        "時間效果顯著，F(2,28)=12.7, p<.001, partial η²=.48。",
        "直接使用一般 ANOVA 會把同一人的觀察誤當獨立，通常低估誤差結構。",
    )

with nonparametric_tab:
    lesson(
        "Mann–Whitney U｜兩個獨立組別的名次比較",
        "兩個獨立組別的分布／隨機取值傾向是否不同？",
        "兩組獨立資料為序位，或連續資料有強烈偏態與離群值。",
        "長格式：group 與 outcome；兩組受試者不可重複。",
        ("兩組分布相同", "兩組分布不同"),
        ["觀察值獨立。", "資料至少可排序。", "若要解釋為中位數差異，兩組分布形狀應相近。"],
        r"U_1=n_1n_2+\frac{n_1(n_1+1)}{2}-R_1",
        "可報 rank-biserial correlation 或 Cliff's delta。",
        "兩組名次分布不同，U=28, p=.021；BCI 組整體數值較高。",
        "它不是自動的『中位數檢定』；分布形狀不同時，結果可能反映形狀而非位置。",
    )
    lesson(
        "Kruskal–Wallis H｜三組以上的名次比較",
        "三個以上獨立組別的分布是否相同？",
        "One-way ANOVA 的名次型替代方案；資料可排序且組別獨立。",
        "長格式：一個 group 欄與一個 outcome 欄。",
        ("所有組的分布相同", "至少一組分布不同"),
        ["觀察值獨立。", "資料至少可排序。", "要解讀位置差異時，各組分布形狀應相近。"],
        r"H=\frac{12}{N(N+1)}\sum_j\frac{R_j^2}{n_j}-3(N+1)",
        "可報 ε²；顯著後使用適當的名次 post-hoc。",
        "組別分布不同，H(2)=8.71, p=.013, ε²=.24。",
        "顯著結果仍未指出是哪幾組不同，也不能直接改跑許多未校正 Mann–Whitney 而不說明。",
    )

with association_tab:
    lesson(
        "Pearson correlation｜線性相關",
        "兩個連續變數是否呈線性共同變化？",
        "兩個數值變數，散點圖顯示關係大致是直線而非曲線。",
        "每列一個獨立觀察，至少有 X 與 Y 兩欄。",
        ("母體線性相關 ρ = 0", "ρ ≠ 0"),
        ["觀察值獨立。", "關係大致線性。", "沒有主導結果的離群點；傳統推論通常假設雙變量常態。"],
        r"r=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}}",
        "r 本身就是標準化效果大小，介於 −1 與 1；同時報 95% CI。",
        "Accuracy 與 training hours 呈正相關，r(28)=.46, p=.010。",
        "相關不代表因果；r 接近 0 也可能仍存在強烈曲線關係。",
    )
    lesson(
        "Spearman correlation｜單調名次相關",
        "X 增加時，Y 是否整體傾向持續增加或持續減少？",
        "序位資料、偏態資料，或關係是單調但不一定線性。",
        "每列一個獨立觀察，包含 X 與 Y。",
        ("母體名次相關 ρs = 0", "ρs ≠ 0"),
        ["觀察值獨立。", "變數至少可排序。", "關係若存在，應大致單調。"],
        r"\rho_s=cor(rank(X),rank(Y))",
        "ρs 本身為效果大小；應搭配原始資料散點圖。",
        "兩變數呈正向單調相關，ρs=.52, p=.003。",
        "它不是『沒有任何假設』，也不能偵測 U 型等非單調關係。",
    )
    lesson(
        "Chi-square / Fisher exact｜兩個類別變數",
        "兩個類別變數是否獨立？",
        "例如 treatment group × recovered yes/no；資料是人數，不是平均數。",
        "每列一個人與兩個類別欄；或可建立列聯表。",
        ("兩個類別變數互相獨立", "兩者有關聯"),
        ["每個人只貢獻一次且觀察值獨立。", "類別互斥。", "Chi-square 的期望次數不能普遍過小；2×2 小樣本參考 Fisher exact。"],
        r"E_{ij}=\frac{row_i\times column_j}{n},\qquad \chi^2=\sum\frac{(O-E)^2}{E}",
        "Cramér's V 表示關聯強度；2×2 可另報 odds ratio。",
        "Group 與 recovery 有關聯，χ²(2)=7.83, p=.020, Cramér's V=.31。",
        "百分比方向要說清楚分母；統計關聯也不自動表示因果。",
    )

with model_tab:
    lesson(
        "Multiple linear regression｜預測連續結果",
        "多個預測變數在彼此控制後，如何共同預測一個連續結果？",
        "Y 是連續數值；X 可有多個。本工具目前的簡易介面接受數值型 X。",
        "每列一個獨立觀察，含 outcome 與所有 predictors。",
        ("指定係數 βj = 0", "βj ≠ 0"),
        ["觀察值獨立。", "Y 與各 X 的條件關係大致線性。", "殘差變異大致一致且推論時近似常態。", "沒有嚴重多重共線性或高影響觀察。"],
        r"Y=\beta_0+\beta_1X_1+\cdots+\beta_pX_p+\epsilon",
        "R²/adjusted R² 描述解釋比例；各 β、標準化係數或 partial R² 描述個別效果。",
        "控制 age 後，training 每增加 1 小時，accuracy 平均增加 1.8 分，β=1.8, 95% CI [0.7,2.9], p=.002。",
        "『控制其他變數』不是自動消除混雜；錯誤變數選擇仍可能造成偏誤。",
    )
    lesson(
        "Logistic regression｜預測二元結果",
        "多個變數如何預測事件發生的機率／勝算？",
        "Y 只有兩類，例如 success/failure；X 可有多個。本工具目前接受數值型 X。",
        "每列一個獨立觀察，含二元 outcome 與 predictors。",
        ("指定係數 βj = 0，也就是 OR = 1", "βj ≠ 0，也就是 OR ≠ 1"),
        ["觀察值獨立。", "連續 X 與 log-odds 大致呈線性。", "樣本與事件數足夠，沒有完全分離。", "沒有嚴重多重共線性或高影響觀察。"],
        r"\log\frac{p}{1-p}=\beta_0+\sum_j\beta_jX_j,\qquad OR=e^{\beta_j}",
        "Odds ratio 與 95% CI；整體模型可報 AIC、pseudo-R²、校準與辨識表現。",
        "Age 每增加 10 年，成功勝算為原來的 1.42 倍，OR=1.42, 95% CI [1.08,1.88], p=.012。",
        "OR=2 不代表機率直接變成兩倍；基準機率不同時，機率差會不同。",
    )

with posthoc_tab:
    st.subheader("為什麼 ANOVA 後還需要 Post-hoc？")
    st.write("ANOVA 只回答『是否至少有一組不同』。Post-hoc 才逐對找出差異位置。但每增加一次比較，就增加至少一次偽陽性的機會。")
    st.markdown("#### 從一個具體例子開始")
    st.write("假設 ANOVA 比較 Control、Training、BCI 後顯著。現在共有三個逐對問題，而不是一個問題。")
    posthoc_example = pd.DataFrame({
        "comparison": ["Control vs Training", "Control vs BCI", "Training vs BCI"],
        "mean_difference": [6, 14, 8],
        "raw_p": [.040, .003, .032],
    })
    st.dataframe(posthoc_example, hide_index=True, width="stretch")
    st.markdown("**① 先用眼睛觀察**  三個 raw p 都小於 .05，看起來三組都不同。")
    st.markdown(r"**② 連回原理**  每次檢定各自容許 5% 偽陽性，不代表整組三次比較仍只有 5%。若三次檢定彼此獨立且 H₀ 都成立，至少誤判一次的機率約為 $1-(1-.05)^3=14.3\%$。十次比較時約為 40.1%。")
    st.markdown("**③ 校正在做什麼**  Bonferroni、Holm、Tukey 會提高判定門檻或調整 p-value，以控制整個比較家族的錯誤；FDR 則控制被宣布為發現的結果中，預期錯誤所占比例。")
    st.success("人話結論：未校正結果仍可計算，但必須說明比較數量與較高的偽陽性風險；不能把三個 raw p 當作只做了一次檢定。")
    st.markdown(
        """
| 方法 | 何時考慮 | p-value 的意義 |
|---|---|---|
| 不校正 pairwise Welch t-test | 少數、事先規劃且願意承擔較高偽陽性風險的比較 | 原始 p；必須明確註明未校正 |
| Holm | 想控制 family-wise error，通常比 Bonferroni 有力 | 逐步校正後的 p |
| FDR (Benjamini–Hochberg) | 探索性、大量比較，重點是控制發現中的錯誤比例 | FDR 校正後的 p |
| Bonferroni | 非常保守、比較數少 | 原始 p × 比較數，上限為 1 |
| Tukey HSD | 所有組別都要兩兩比較，且傳統 ANOVA 假設合理 | Tukey family-wise-adjusted p |
"""
    )
    st.error("Tukey HSD 本身已經是多重比較程序，不能稱為『without correction』。如果要完全未校正，應選 pairwise test 的 None，並在論文中說明。")
    st.markdown("**建議報告內容**：比較的兩組、平均差、95% CI、效果大小、raw p、是否校正、校正方法，以及總比較數。")

st.divider()
st.caption("教學內容用來協助選擇與理解方法；正式研究仍需依研究設計、抽樣方式、缺失資料與領域知識決定分析。")
