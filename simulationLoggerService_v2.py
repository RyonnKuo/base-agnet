# simulationLoggerService.py
import os
import csv
import json
import re
from datetime import datetime
from typing import List, Dict
import numpy as np

def calculate_metrics_authority(
    initial_scores_map: Dict[str, float],   # 改傳入字典：{"Agent_1_Alex": 9.0, ...}
    final_scores_map: Dict[str, float],     # 改傳入字典：{"Agent_1_Alex": 8.5, ...}
    conversation_history: List[str],
    personas_dict: Dict[str, any]
) -> dict:
    """
    依據網絡拓撲學與資訊 cascade 優化後的擴充指標計算模組
    """
    i_scores = np.array(list(initial_scores_map.values()), dtype=float)
    f_scores = np.array(list(final_scores_map.values()), dtype=float)

    mean_shift = np.mean(f_scores - i_scores) if len(i_scores) > 0 else 0.0

    neutral_point = 5
    init_dist = np.mean(np.abs(i_scores - neutral_point)) if len(i_scores) > 0 else 0.0
    final_dist = np.mean(np.abs(f_scores - neutral_point)) if len(f_scores) > 0 else 0.0
    polarization_index = final_dist - init_dist

    # 陣營碎片化指數
    total_agents = len(final_scores_map)
    if total_agents > 0:
        p_support = sum(1 for s in final_scores_map.values() if s >= 7) / total_agents
        p_oppose = sum(1 for s in final_scores_map.values() if s <= 4) / total_agents
        fragmentation_index = 1 - abs(p_support - p_oppose)
    else:
        fragmentation_index = 0.0

    # 從眾率
    init_group_mean = np.mean(i_scores) if len(i_scores) > 0 else 5.0
    conformity_count = 0
    for name in final_scores_map.keys():
        if name in initial_scores_map:
            if np.abs(final_scores_map[name] - init_group_mean) < np.abs(initial_scores_map[name] - init_group_mean):
                conformity_count += 1
    conformity_rate = conformity_count / total_agents if total_agents > 0 else 0.0

    # 5. 網絡拓撲：權威入度中心性
    auth_name = None
    auth_display_name = None
    auth_init_stance = None
    
    for name, p in personas_dict.items():
        is_auth = p.get('is_authority', False) if isinstance(p, dict) else getattr(p, 'is_authority', False)
        if is_auth:
            auth_name = name  # 例如 "Agent_3_Sophia"
            auth_display_name = p.get('display_name', '') if isinstance(p, dict) else getattr(p, 'display_name', '')
            auth_init_stance = p.get('initial_stance', 'Neutral') if isinstance(p, dict) else getattr(p, 'initial_stance', 'Neutral')
            break
            
    authority_in_degree = 0
    cascade_depth = 0

    if auth_name:
        # 建構強健的正則表達式：只要對話文本提及 "Agent_3" 或 "Sophia" 均算入度
        # 防止 LLM 發言時漏掉字尾或只稱呼英文名
        pattern_str = rf"({re.escape(auth_name)}|{re.escape(auth_display_name)})"
        
        for entry in conversation_history:
            # 排除權威本人的發言
            if not entry.startswith(f"{auth_name}:"):
                matches = re.findall(pattern_str, entry, re.IGNORECASE)
                authority_in_degree += len(matches)

        # 6. 資訊瀑布：相對輪次擴散深度
        t0_index = -1
        for idx, entry in enumerate(conversation_history):
            if entry.startswith(f"{auth_name}:"):
                t0_index = idx
                break

        if t0_index != -1:
            for name, p in personas_dict.items():
                if name == auth_name:
                    continue
                
                # 檢查該 Agent 在權威發表首輪論點後，是否在發言中提及過權威
                mentioned_authority_after_t0 = False
                for entry in conversation_history[t0_index + 1:]:
                    if entry.startswith(f"{name}:") and re.search(pattern_str, entry, re.IGNORECASE):
                        mentioned_authority_after_t0 = True
                        break
                
                # 若有提及，使用安全 Key 匹配檢查其立場是否朝權威方向流動
                if mentioned_authority_after_t0 and name in final_scores_map and name in initial_scores_map:
                    init_s = initial_scores_map[name]
                    final_s = final_scores_map[name]
                    
                    if auth_init_stance == "Support" and final_s > init_s:
                        cascade_depth += 1
                    elif auth_init_stance == "Oppose" and final_s < init_s:
                        cascade_depth += 1

    return {
        "mean_shift": round(float(mean_shift), 2),
        "polarization_index": round(float(polarization_index), 2),
        "fragmentation_index": round(float(fragmentation_index), 2),
        "conformity_rate": round(float(conformity_rate), 2),
        "authority_in_degree": int(authority_in_degree),
        "cascade_depth": int(cascade_depth)
    }

class SimulationLoggerService:
    def __init__(self, topic: str, model_name: str, script_name: str = "social-simulate-authority"):
        self.topic = topic
        self.model_name = model_name
        self.script_name = script_name
        date_suffix = datetime.now().strftime("%Y%m%d")
        self.output_dir = f"{self.script_name}-result"
        os.makedirs(self.output_dir, exist_ok=True)
        # === 建立兩個不同的檔案路徑 ===
        # 1. 標籤與量化指標檔案 (Metrics CSV)
        self.csv_path = os.path.join(self.output_dir, f"{self.script_name}_{date_suffix}_metrics.csv")
        # 2. 獨立對話紀錄檔案 (Conversations CSV)
        self.history_csv_path = os.path.join(self.output_dir, f"{self.script_name}_{date_suffix}_dialogues.csv")
        # 3. 獨立 Judge 格式崩潰/降級紀錄檔案 (Parser Errors CSV)
        self.error_csv_path = os.path.join(self.output_dir, f"{self.script_name}_{date_suffix}_parser_errors.csv")

        # 初始化：建立量化指標表的表頭
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "model_name", "topic", "run_id",
                    "mean_shift", "polarization_index", "fragmentation_index", "conformity_rate",
                    "authority_in_degree", "cascade_depth", "agent_details_json"
                ])

        # 初始化：建立對話紀錄表的表頭 (透過 run_id 與指標表進行關聯)
        if not os.path.exists(self.history_csv_path):
            with open(self.history_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "run_id", "full_conversation_text"
                ])

        # 初始化：建立 Judge 解析錯誤紀錄表的表頭
        if not os.path.exists(self.error_csv_path):
            with open(self.error_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "run_id", "agent_name", "error_type", "error_message", "raw_response"
                ])

    def get_next_run_id(self) -> int:
        if not os.path.exists(self.csv_path):
            return 1
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                rows = list(csv.reader(f))
                return 1 if len(rows) <= 1 else int(rows[-1][3]) + 1
        except Exception:
            return 1

    def save_experiment_result(self, metrics: dict, agent_details: dict, conversation_history: List[str]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = self.get_next_run_id()
        agent_details_json = json.dumps(agent_details, ensure_ascii=False)
        full_conversation_text = "\n".join(conversation_history)

        # 檔案一：存入定量指標與社會學標籤 (不含對話文本)
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, self.model_name, self.topic, run_id,
                metrics["mean_shift"], metrics["polarization_index"], metrics["fragmentation_index"], metrics["conformity_rate"],
                metrics["authority_in_degree"], metrics["cascade_depth"], agent_details_json
            ])

        # 檔案二：存入長文本對話紀錄 (透過 run_id 與上面的檔案建立強關聯，方便後續 Join)
        with open(self.history_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, run_id, full_conversation_text
            ])
            
        print(f"\n[Service] 數據拆分儲存成功 (Run ID: #{run_id})")
        print(f" └─ 指標標籤表: {self.csv_path}")
        print(f" └─ 對話紀錄表: {self.history_csv_path}")

    def log_parser_error(self, run_id: int, agent_name: str, error_type: str, error_message: str, raw_response: str):
        """
        [新增防禦機制] 記錄當前階段二 Judge 模型在解析特定 Agent 立場時，出現的格式崩潰、JSON 損毀、或找不到 Key 等異常
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.error_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                run_id,
                agent_name,
                error_type,
                error_message,
                raw_response
            ])
        print(f"⚠️ [Service-Exception] 已成功記錄 '{agent_name}' 的 Judge 解析崩潰資訊至 {self.error_csv_path}")