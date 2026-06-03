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

# === 社會影響與網絡拓撲指標計算 (量化公式) ===
def calculate_metrics_authority(
    initial_scores: List[float], 
    final_scores: List[float], 
    conversation_history: List[str],
    personas_dict: Dict[str, any]
) -> dict:
    """
    依據論文與網絡拓撲學擴充計算指標。
    新增：權威入度中心性 (In-degree Centrality)、資訊瀑布擴散深度 (Cascade Depth)
    """
    i_scores = np.array(initial_scores, dtype=float)
    f_scores = np.array(final_scores, dtype=float)

    # 1. 平均立場位移 (Mean Shift)
    mean_shift = np.mean(f_scores - i_scores)

    # 2. 群體極化指數 (Group Polarization Index)
    neutral_point = 5
    init_dist = np.mean(np.abs(i_scores - neutral_point))
    final_dist = np.mean(np.abs(f_scores - neutral_point))
    polarization_index = final_dist - init_dist

    # 3. 群體分裂/碎片化指數 (Group Fragmentation Index)
    # [學術優化] 仿照論文定義：將 1-10分 轉換為 陣營比例
    # 1-4分 定義為反對派 (Oppose), 7-10分 定義為支持派 (Support)
    total_agents = len(final_scores)
    if total_agents > 0:
        p_support = sum(1 for s in final_scores if s >= 7) / total_agents
        p_oppose = sum(1 for s in final_scores if s <= 4) / total_agents
        # 兩極對立愈平衡，該值愈趨近於 1；若單方壓倒性獲勝或全員中立，則趨近於 0
        fragmentation_index = 1 - abs(p_support - p_oppose)
    else:
        fragmentation_index = 0.0

    # 4. 微觀從眾率估算 (Micro-Conformity Rate)
    init_group_mean = np.mean(i_scores)
    conformity_count = sum(1 for i, f in zip(i_scores, f_scores) if np.abs(f - init_group_mean) < np.abs(i - init_group_mean))
    conformity_rate = conformity_count / len(initial_scores) if len(initial_scores) > 0 else 0.0

    # 5. [新增] 網絡拓撲：權威入度中心性 (Authority In-degree Centrality)
    # 找出誰是權威
    # [Bug 修正] 針對 JSON 讀取出來的 dict 結構，全面改用 .get() 確保安全讀取
    auth_name = None
    auth_init_stance = None
    for name, p in personas_dict.items():
        # 自動相容物件與字典
        is_auth = p.get('is_authority', False) if isinstance(p, dict) else getattr(p, 'is_authority', False)
        if is_auth:
            auth_name = name
            auth_init_stance = p.get('initial_stance', 'Neutral') if isinstance(p, dict) else getattr(p, 'initial_stance', 'Neutral')
            break
            
    authority_in_degree = 0
    cascade_depth = 0

    if auth_name:
        # 掃描非權威本人發表的言論中，提及權威名字的次數 (顯性文本提及)
        for entry in conversation_history:
            if not entry.startswith(f"{auth_name}:"):
                # 使用正則表達式不區分大小寫匹配權威名字 (如 Agent_3)
                matches = re.findall(rf"\b{re.escape(auth_name)}\b", entry, re.IGNORECASE)
                authority_in_degree += len(matches)

        # 6. 資訊瀑布：相對輪次擴散深度 (Authority Cascade Depth)
        # [Bug 修正] 尋找權威在對話歷史中第一次發言的索引位置 (時間起點 T_0)
        t0_index = -1
        for idx, entry in enumerate(conversation_history):
            if entry.startswith(f"{auth_name}:"):
                t0_index = idx
                break

        # 如果找到了權威發言起點，觀察在 T_0 之後發言的一般 Agent 是否向權威立場靠攏
        if t0_index != -1:
            for name, p in personas_dict.items():
                if name == auth_name:
                    continue
                
                # A. 檢查 T_0 之後該 Agent 是否發言，且文本中是否提及了權威
                mentioned_authority_after_t0 = False
                for entry in conversation_history[t0_index + 1:]:
                    if entry.startswith(f"{name}:") and re.search(rf"\b{re.escape(auth_name)}\b", entry, re.IGNORECASE):
                        mentioned_authority_after_t0 = True
                        break
                
                # B. 如果有互動，進一步比對其立場分數是否受到權威方向的感應
                if mentioned_authority_after_t0:
                    agent_idx = list(personas_dict.keys()).index(name)
                    init_s = initial_scores[agent_idx]
                    final_s = final_scores[agent_idx]
                    
                    # 權威支持(>5分)，且一般人分數上升 -> 認定為受權威瀑布渲染
                    if auth_init_stance == "Support" and final_s > init_s:
                        cascade_depth += 1
                    # 權威反對(<5分)，且一般人分數下降 -> 認定為受權威瀑布渲染
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


# === Data Logging Service (Service Component) ===
class SimulationLoggerService:
    """
    Service responsible for persisting experimental simulation data.
    """

    def __init__(self, topic: str, model_name: str, script_name: str = "social-simulate-authority"):
        self.topic = topic
        self.model_name = model_name
        self.script_name = script_name

        date_suffix = datetime.now().strftime("%Y%m%d")
        self.output_dir = f"{self.script_name}-result"
        os.makedirs(self.output_dir, exist_ok=True)

        csv_filename = f"{self.script_name}_{date_suffix}.csv"
        self.csv_path = os.path.join(self.output_dir, csv_filename)

        # 🌟 修改：在 Header 中加入新指標：authority_in_degree, cascade_depth
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
                    "fragmentation_index",  # 補上原本漏掉的
                    "conformity_rate",      # 補上原本漏掉的
                    "authority_in_degree",  # [新拓撲指標]
                    "cascade_depth",        # [新瀑布指標]
                    "agent_details_json", 
                    "full_conversation_text"
                ])

    def get_next_run_id(self) -> int:
        if not os.path.exists(self.csv_path):
            return 1
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:
                    return 1
                last_row = rows[-1]
                return int(last_row[3]) + 1
        except Exception:
            return 1

    def save_experiment_result(self, metrics: dict, agent_details: dict, conversation_history: List[str]):
        """ Appends the final simulation metrics and logs into the CSV file """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = self.get_next_run_id()

        agent_details_json = json.dumps(agent_details, ensure_ascii=False)
        full_conversation_text = "\n".join(conversation_history)

        # 🌟 修改：寫入對應的新增欄位
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.model_name,
                self.topic,
                run_id,
                metrics["mean_shift"],
                metrics["polarization_index"],
                metrics["fragmentation_index"],
                metrics["conformity_rate"],
                metrics["authority_in_degree"],
                metrics["cascade_depth"],
                agent_details_json,
                full_conversation_text
            ])
        print(f"\n[Service] Data successfully saved to {self.csv_path} (Run ID: #{run_id})")