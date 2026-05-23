import os
import csv
import json
import re
from datetime import datetime
from typing import List, Dict
import numpy as np

# === 3. 社會影響指標計算 (量化公式) ===
def calculate_metrics(initial_scores: List[float], final_scores: List[float]) -> dict:
    """
    依據論文第 3.4 節量化公式計算社會影響動態指標
    """
    i_scores = np.array(initial_scores, dtype=float)
    f_scores = np.array(final_scores, dtype=float)

    # 1. 平均立場位移 (Mean Shift)
    mean_shift = np.mean(f_scores - i_scores)

    # 2. 群體極化指數 (Group Polarization Index)
    # 衡量群體平均而言是遠離還是靠近中立點
    neutral_point = 5

    init_dist = np.mean(np.abs(i_scores - neutral_point))
    final_dist = np.mean(np.abs(f_scores - neutral_point))
    polarization_index = final_dist - init_dist

    # 3. 群體分裂/碎片化指數 (Group Fragmentation Index)
    # 論文中用來衡量意見的分散與不一致程度，這裡使用最終分數的標準差(Standard Deviation)作為客觀指標
    # 標準差越高，代表群體意見分裂越嚴重；標準差趨近於0，代表高度從眾與共識
    fragmentation_index = np.std(f_scores)

    # 4. 微觀從眾率估算 (Micro-Conformity Rate)
    # 統計有多少 Agent 向群體平均值或中立點靠攏
    # 如果最終分數比初始分數更接近初始的群體平均值，則視為從眾行為
    init_group_mean = np.mean(i_scores)
    conformity_count = sum(1 for i, f in zip(i_scores, f_scores) if np.abs(f - init_group_mean) < np.abs(i - init_group_mean))
    conformity_rate = conformity_count / len(initial_scores)

    return {
        "mean_shift": round(float(mean_shift), 2),
        "polarization_index": round(float(polarization_index), 2),
        "fragmentation_index": round(float(fragmentation_index), 2),
        "conformity_rate": round(float(conformity_rate), 2)
    }


# === Data Logging Service (Service Component) ===
class SimulationLoggerService:
    """
    Service responsible for persisting experimental simulation data.
    Creates a designated directory and file using explicit script names and timestamps.
    """

    def __init__(self, topic: str, model_name: str, script_name: str = "social-simulate-base-ver2"):
        """
        Args:
            topic: The discussion topic.
            model_name: The name of the LLM model.
            script_name: The name of the calling file/script. Defaults to "social-simulate-base-ver2".
        """
        self.topic = topic
        self.model_name = model_name
        self.script_name = script_name

        # 1. 取得當前日期的時間戳記 (格式：YYYYMMDD，例如 20260521)
        date_suffix = datetime.now().strftime("%Y%m%d")

        # 2. 設定目標資料夾名稱為 {script_name}-result
        self.output_dir = f"{self.script_name}-result"
        os.makedirs(self.output_dir, exist_ok=True)

        # 3. 定義 CSV 檔案路徑：{script_name}_{date_suffix}.csv (例如 social-simulate-base-ver2_20260521.csv)
        csv_filename = f"{self.script_name}_{date_suffix}.csv"
        self.csv_path = os.path.join(self.output_dir, csv_filename)

        # 4. 如果檔案不存在，初始化並寫入欄位標頭 (Header)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", 
                    "model_name", 
                    "topic", 
                    "run_id",
                    "mean_shift", 
                    "polarization_index",
                    "agent_details_json", 
                    "full_conversation_text"
                ])

    def get_next_run_id(self) -> int:
        """ Reads the CSV file to dynamically determine the next experimental run ID """
        if not os.path.exists(self.csv_path):
            return 1
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:  # Only header exists
                    return 1
                # Retrieve the run_id from the last row and increment it
                last_row = rows[-1]
                return int(last_row[3]) + 1
        except Exception:
            return 1

    def save_experiment_result(self, metrics: dict, agent_details: dict, conversation_history: List[str]):
        """ Appends the final simulation metrics and logs into the CSV file """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = self.get_next_run_id()

        # Serialize complex dictionaries into strings for single CSV cell storage
        agent_details_json = json.dumps(agent_details, ensure_ascii=False)
        full_conversation_text = "\n".join(conversation_history)

        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.model_name,
                self.topic,
                run_id,
                metrics["mean_shift"],
                metrics["polarization_index"],
                agent_details_json,
                full_conversation_text
            ])
        print(f"\n[Service] Data successfully saved to {self.csv_path} (Run ID: #{run_id})")