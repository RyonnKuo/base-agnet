import glob
import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

# ==================== 1. 設定 4 組 CROSS 實驗資料夾路徑對照表 ====================
# 請依據您實際輸出的資料夾名稱進行調整（亦支援萬用字元如 wildcard 搜尋）
CROSS_EXPERIMENT_CONFIGS = [
    {
        "folder": (
            "social-simulate-v3-agent-cross-authority-auth_deepseek-r1_8b-mem-result"
        ),
        "condition": "DeepSeek-R1:8B 權威",
        "auth_model": "deepseek-r1:8b",
        "has_authority": True,
    },
    {
        "folder": (
            "social-simulate-v3-agent-cross-authority-auth_llama3_8b-mem-result"
        ),
        "condition": "LLaMA3:8B 權威",
        "auth_model": "llama3:8b",
        "has_authority": True,
    },
    {
        "folder": (
            "social-simulate-v3-agent-cross-authority-auth_qwen3_8b-mem-result"
        ),
        "condition": "Qwen3:8B 權威",
        "auth_model": "qwen3:8b",
        "has_authority": True,
    },
    {
        "folder": "social-simulate-v3-agent-cross-control-mem-result",
        "condition": "控制組 (無權威平權)",
        "auth_model": "None",
        "has_authority": False,
    },
]

MODEL_NAME_MAP = {
    "llama3:8b": "LLaMA3:8B",
    "qwen3:8b": "Qwen3:8B",
    "deepseek-r1:8b": "DeepSeek-R1:8B",
}


def clean_model_name(raw_name):
    """清理並標準化模型名稱"""
    raw_lower = str(raw_name).lower()
    for key, label in MODEL_NAME_MAP.items():
        if key in raw_lower:
            return label
    return raw_name


def parse_micro_and_alignment(df, has_authority):
    """解析每一場 Run 的 agent_details_json，計算『權威引導成功率』與『微觀 Agent 紀錄』"""
    micro_records = []
    run_alignment_rates = []

    for run_idx, row in df.iterrows():
        json_str = row.get("agent_details_json", "{}")
        try:
            details = json.loads(json_str)
        except Exception:
            run_alignment_rates.append(np.nan)
            continue

        auth_name = None
        auth_init_stance = None
        auth_model = None

        # 1. 識別權威角色
        for name, info in details.items():
            if info.get("is_authority", False):
                auth_name = name
                auth_init_stance = info.get("initial_stance", "")
                auth_model = clean_model_name(
                    info.get("model_name", "Unknown")
                )
                break

        aligned_count = 0
        non_auth_count = 0

        # 2. 計算受試鄉民反應
        for name, info in details.items():
            is_auth = info.get("is_authority", False)
            model_name = clean_model_name(info.get("model_name", "Unknown"))
            init_s = float(info.get("initial_score", 5.0))
            final_s = float(info.get("final_score", 5.0))
            shift = final_s - init_s
            shift_type = info.get("shift_type", "堅持")

            swayed_towards_auth = False
            if has_authority and auth_name and not is_auth:
                non_auth_count += 1
                # 權威引導成功判定邏輯 (Stance Alignment)
                if auth_init_stance == "Support":
                    if final_s >= 6.0 or shift > 0:
                        swayed_towards_auth = True
                        aligned_count += 1
                elif auth_init_stance == "Oppose":
                    if final_s <= 4.0 or shift < 0:
                        swayed_towards_auth = True
                        aligned_count += 1

            micro_records.append({
                "run_id": row.get("run_id", run_idx + 1),
                "agent_name": name,
                "agent_model": model_name,
                "is_authority": is_auth,
                "initial_score": init_s,
                "final_score": final_s,
                "stance_shift": shift,
                "shift_type": shift_type,
                "swayed_towards_auth": (
                    swayed_towards_auth if not is_auth else np.nan
                ),
            })

        if has_authority and non_auth_count > 0:
            run_alignment_rates.append((aligned_count / non_auth_count) * 100)
        else:
            run_alignment_rates.append(np.nan)

    df["stance_alignment_rate"] = run_alignment_rates
    return pd.DataFrame(micro_records), df


def process_cross_folder(config):
    """讀取並計算單一實驗組別的 7 大巨觀與微觀指標"""
    folder_path = config["folder"]
    condition = config["condition"]
    has_auth = config["has_authority"]

    csv_pattern = os.path.join(folder_path, "*_metrics.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(
            f"⚠️ 找不到 CSV 檔案：'{folder_path}'，嘗試搜尋模糊匹配名稱..."
        )
        # 彈性匹配路徑
        alt_pattern = os.path.join(f"*{folder_path}*", "*_metrics.csv")
        csv_files = glob.glob(alt_pattern)
        if not csv_files:
            print(f"❌ 錯誤：無法在 {folder_path} 找到任何 metrics 檔案。")
            return None, None, None

    target_csv = sorted(csv_files)[-1]
    df = pd.read_csv(target_csv)

    micro_df, df_with_align = parse_micro_and_alignment(df, has_auth)

    # 計算 7 大巨觀指標（包含 Mean 與 SD，便於繪圖與寫表格）
    macro_summary = {
        "實驗組別": condition,
        "總重複次數": len(df),
        "平均立場位移 (Mean Shift)": (
            f"{df['mean_shift'].mean():.3f} ± {df['mean_shift'].std():.3f}"
        ),
        "群體極化指數 (Polarization)": (
            f"{df['polarization_index'].mean():.3f} ± {df['polarization_index'].std():.3f}"
        ),
        "群體分裂指數 (Fragmentation)": (
            f"{df['fragmentation_index'].mean():.3f} ± {df['fragmentation_index'].std():.3f}"
        ),
        "從眾率 (%) (Conformity Rate)": (
            f"{df['conformity_rate'].mean()*100:.2f}% ± {df['conformity_rate'].std()*100:.2f}%"
        ),
        "權威被提及平均次數 (In-degree)": (
            f"{df['authority_in_degree'].mean():.2f} ± {df['authority_in_degree'].std():.2f}"
            if has_auth
            else "N/A"
        ),
        "級聯擴散平均深度 (Cascade Depth)": (
            f"{df['cascade_depth'].mean():.2f} ± {df['cascade_depth'].std():.2f}"
            if has_auth
            else "N/A"
        ),
        "權威引導成功率 (%) (Stance Alignment)": (
            f"{df_with_align['stance_alignment_rate'].dropna().mean():.2f}% ± {df_with_align['stance_alignment_rate'].dropna().std():.2f}%"
            if has_auth
            else "N/A"
        ),
    }

    return macro_summary, micro_df, df_with_align


def perform_inferential_anova(all_runs_data):
    """執行單因子變異數分析 (One-way ANOVA)，為論文提供統計顯著性檢定 (F, p-value)"""
    print("\n" + "=" * 90)
    print(" 📈 三組權威實驗之單因子變異數分析 (One-Way ANOVA 推論統計檢定)")
    print("=" * 90)

    # 僅篩選有權威的三組進行 ANOVA 檢定
    auth_groups = {
        k: v for k, v in all_runs_data.items() if "控制組" not in k
    }

    metrics_to_test = [
        ("mean_shift", "平均立場位移"),
        ("polarization_index", "群體極化指數"),
        ("fragmentation_index", "群體分裂指數"),
        ("conformity_rate", "從眾率"),
        ("authority_in_degree", "權威 In-degree"),
        ("cascade_depth", "級聯擴散深度"),
        ("stance_alignment_rate", "權威引導成功率"),
    ]

    anova_results = []

    for metric_col, metric_label in metrics_to_test:
        group_samples = []
        for cond_name, df in auth_groups.items():
            if metric_col in df.columns:
                valid_data = df[metric_col].dropna()
                if not valid_data.empty:
                    group_samples.append(valid_data.values)

        if len(group_samples) >= 2:
            f_val, p_val = stats.f_oneway(*group_samples)
            sig_star = (
                "***"
                if p_val < 0.001
                else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            )

            anova_results.append({
                "檢定指標": metric_label,
                "F-Statistic": round(f_val, 3),
                "p-Value": (
                    f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
                ),
                "統計顯著性": sig_star,
            })
            print(
                f"  ├─ [{metric_label}]: F = {f_val:.3f}, p = {p_val:.4f} ({sig_star})"
            )

    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(
        "cross_anova_significance_test.csv", index=False, encoding="utf-8-sig"
    )
    print("\n✅ ANOVA 推論統計報告已匯出至：cross_anova_significance_test.csv")


def main():
    print("=" * 90)
    print(
        " 🚀 開始執行 CROSS 跨家族多模型 (3 權威組 + 1 控制組) 完整數據與統計分析"
    )
    print("=" * 90 + "\n")

    macro_results = []
    micro_all = []
    runs_df_dict = {}

    for config in CROSS_EXPERIMENT_CONFIGS:
        cond = config["condition"]
        print(f"🔍 正在處理：[{cond}]...")

        macro_sum, micro_df, df_runs = process_cross_folder(config)

        if macro_sum:
            macro_results.append(macro_sum)
            runs_df_dict[cond] = df_runs

        if micro_df is not None and not micro_df.empty:
            micro_df["Authority_Condition"] = cond
            micro_all.append(micro_df)

    # ------------------ 輸出 1：7 大巨觀指標矩陣 ------------------
    if macro_results:
        macro_df = pd.DataFrame(macro_results)
        print("\n" + "=" * 100)
        print("📊 表 1：CROSS 模式 7 大巨觀指標比較表 (Mean ± SD)")
        print("=" * 100)
        print(macro_df.to_string(index=False))
        macro_df.to_csv(
            "cross_macro_summary_7_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ------------------ 輸出 2：微觀跨家族 Agent 反應拆解表 ------------------
    if micro_all:
        combined_micro = pd.concat(micro_all, ignore_index=True)
        micro_summary_list = []

        # 嚴格過濾一般 Agent，排除權威自身
        non_auth_micro = combined_micro[combined_micro["is_authority"] == False]

        for cond_name in combined_micro["Authority_Condition"].unique():
            sub_cond = non_auth_micro[
                non_auth_micro["Authority_Condition"] == cond_name
            ]
            for model_name in ["LLaMA3:8B", "Qwen3:8B", "DeepSeek-R1:8B"]:
                agent_sub = sub_cond[sub_cond["agent_model"] == model_name]
                if not agent_sub.empty:
                    avg_shift = agent_sub["stance_shift"].mean()
                    persist_rate = (
                        agent_sub["shift_type"] == "堅持"
                    ).mean() * 100
                    sway_rate = agent_sub["swayed_towards_auth"].mean() * 100

                    micro_summary_list.append({
                        "權威條件": cond_name,
                        "受試 Agent 模型家族": model_name,
                        "平均立場位移": round(avg_shift, 3),
                        "立場堅持率 (%)": round(persist_rate, 2),
                        "被權威同化率 (%)": (
                            round(sway_rate, 2)
                            if not np.isnan(sway_rate)
                            else "N/A"
                        ),
                    })

        micro_summary_df = pd.DataFrame(micro_summary_list)
        print("\n" + "=" * 100)
        print("🔬 表 2：微觀受試 Agent 跨家族同化與堅持反應對比表")
        print("=" * 100)
        print(micro_summary_df.to_string(index=False))
        micro_summary_df.to_csv(
            "cross_micro_agent_breakdown.csv", index=False, encoding="utf-8-sig"
        )

    # ------------------ 輸出 3：執行 ANOVA 統計顯著性檢定 ------------------
    if runs_df_dict:
        perform_inferential_anova(runs_df_dict)


if __name__ == "__main__":
    main()