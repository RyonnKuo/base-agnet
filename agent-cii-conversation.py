# ========== 保留你的原始 import 與 LLM 初始化 ==========
from langchain_community.chat_models import ChatOllama  # 使用 Ollama 封裝的 LLaMA 模型
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
import torch
import re
import json
import time

# 使用本地 LLaMA 模型（保留）
llm_llama3_8B = ChatOllama(model="llama3:8B")
llm_phi3_3dot8B = ChatOllama(model="phi3:3.8B")
llm_mistral_7B = ChatOllama(model="mistral:7B")
llm_gemma3_4B = ChatOllama(model="gemma3:4B")

# GPU 檢查（保留）
print("=== PyTorch GPU 加速環境檢查 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 編譯的 CUDA 版本: {torch.version.cuda}")
print(f"是否支援 CUDA: {torch.cuda.is_available()}")

try:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_gpu)

        print(f"偵測到 {gpu_count} 個 GPU")
        print(f"當前使用的 GPU：{device_name}")
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA 不可用，將使用 CPU")
except Exception as e:
    print(f"無法使用 GPU：{e}")
    device = torch.device("cpu")

print(f"pytorch可用的GPU為：{device}")

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from memory import Memory, MemoryStream
from reflection import ReflectionEngine

# === 1. 建立共享記憶流（保留）===
memory_stream = MemoryStream()

# === 2. 封裝記憶檢索工具（保留）===
memory_tool = Tool(
    name="MemoryRetriever",
    func=lambda q: "\n".join(m.content for m in memory_stream.search_by_vector(q)),
    description="檢索過往的重要對話與事件"
)

tools = [memory_tool]

# === 3. 原始角色 mapping（我們將保留並新增 authority/peers/target ）===
roles = {
    "llama":  ("家醫科醫生", llm_llama3_8B),
    "mistral":("內科醫生", llm_mistral_7B),
    "gemma":  ("外科醫生", llm_gemma3_4B)
}

agents = {}
for name, (role_desc, llm_model) in roles.items():
    prompt_prefix = f"你是一位{role_desc}，回答時請保持此角色的語氣與思考方式。"
    
    agents[name] = initialize_agent(
        tools=tools,
        llm=llm_model,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={
            "prefix": prompt_prefix,
            }
    )

# ------------------ 以下為新增 / 最小改動區塊 ------------------

# 我們新增四個角色所需的 agent：authority、peers（list）、target、observer
# 最小幅度利用現有 llm 實例：使用 llm_llama3_8B 擔任 Authority，mistral 作為 peers，phi3 擔任 observer，gemma 作為 target（只是示範）
# 你可以依需求更換模型配對，我只做最小新增

# 如果已存在相同 key 則不覆寫（維持原 agents），否則新增
if "authority" not in agents:
    authority_prompt = "你是一位權威領導者（Authority），請清楚陳述規範或指令，語氣自信帶領性。"
    agents["authority"] = initialize_agent(
        tools=tools,
        llm=llm_llama3_8B,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": authority_prompt}
    )

# 建立三個 peer agents（最小新增）
peer_names = ["peer_1", "peer_2", "peer_3"]
for pn in peer_names:
    if pn not in agents:
        peer_prompt = f"你是同儕 {pn}，當看到 Authority 的指令時請做出回應，通常展現支持（或按實驗條件選擇反對）。"
        agents[pn] = initialize_agent(
            tools=tools,
            llm=llm_mistral_7B,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={"prefix": peer_prompt}
        )

# target (受影響者) - 我們用 gemma 模型示範
if "target" not in agents:
    target_prompt = "你是受影響者（Target），請根據看到的 Authority 與 Peers 回應，並說明你的理由與置信度。"
    agents["target"] = initialize_agent(
        tools=tools,
        llm=llm_gemma3_4B,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"prefix": target_prompt}
    )

# observer（負責把 target 的回覆標註為 Compliance/Identification/Internalization）
# 我們用 phi3 作為 observer LLM（直接使用 llm_phi3_3dot8B，不用 initialize_agent）
observer_llm = llm_phi3_3dot8B

# ---- Observer prompt template (最小且直白) ----
OBSERVER_PROMPT = """
你是一個分析者（Observer）。任務：判斷目標回覆（TARGET_REPLY）在「社會影響力光譜」三階段中的哪一階段：
0 = Compliance（只是順從，理由以外在壓力為主，置信度低）
1 = Identification（部分認同，表達歸屬、模仿或情感連結）
2 = Internalization（內化，表明內在信念，長期維持並主動說服他人）

請依據下面的 TARGET_REPLY 給出 JSON 格式輸出，欄位：
{
  "stage": 0|1|2,
  "stage_label": "Compliance|Identification|Internalization",
  "rationale": "一句話描述為何判定（最多 30 字）",
  "compliance_score": 0.0-1.0,           # 行為順從度估分
  "internal_confidence": 0.0-1.0,         # 內在置信估分
  "group_ref_density": 0.0-1.0            # 群體詞密度估分
}

TARGET_REPLY:
\"\"\"{target_reply}\"\"\"
"""

# ---- 簡單的文本特徵抽取函式（作為 fallback / 補助指標）----
def simple_text_scores(text: str):
    text_low = text.lower()
    # compliance cues
    compliance_cues = ["我只是照做", "上級要求", "因為大家", "被要求", "不得不"]
    identification_cues = ["我們", "跟大家", "像他們", "我想成為", "我喜歡他們"]
    internal_cues = ["我相信", "我的原則", "我覺得正確", "我會一直", "長期來看"]

    compliance_score = sum(1 for c in compliance_cues if c in text) / max(len(compliance_cues), 1)
    identification_score = sum(1 for c in identification_cues if c in text) / max(len(identification_cues), 1)
    internal_score = sum(1 for c in internal_cues if c in text) / max(len(internal_cues), 1)

    # group_ref_density: how often 'we'/'they' type words appear
    group_refs = len(re.findall(r"\b(我們|他們|大家|同儕|同學|朋友)\b", text))
    total_words = max(len(text.split()), 1)
    group_ref_density = min(1.0, group_refs / (total_words / 10.0))  # heuristics

    # internal confidence heuristic: internal_score normalized
    internal_confidence = min(1.0, internal_score * 1.2)

    # behavioral compliance heuristic: if compliance cues exist but internal low -> higher compliance
    behavioral_compliance = min(1.0, compliance_score * 1.5)

    return {
        "compliance_score": round(behavioral_compliance, 3),
        "internal_confidence": round(internal_confidence, 3),
        "group_ref_density": round(group_ref_density, 3),
        "identification_score": round(identification_score, 3),
        "internal_score": round(internal_score, 3)
    }

# ---- 主要模擬流程：simulate_influence_rounds（最小改動、新加）----
def simulate_influence_rounds(topic: str, rounds: int = 5, verbose: bool = True):
    """
    topic: 欲施加的 Authority 指令 / 信念陳述，例如 "我們應該每天做 X"
    rounds: 回合數（每回合 Authority -> Peers -> Target 發言）
    """
    # 每次模擬，各 agent 的初始 belief_state（0=Compliance,1=Identification,2=Internalization）可由 observer 判斷
    target_history = []
    metrics_log = []

    for r in range(1, rounds + 1):
        if verbose:
            print(f"\n===== 模擬回合 {r} / {rounds} =====")

        # 1) Authority 發訊
        authority_agent = agents["authority"]
        authority_input = f"回合 {r}：對群體宣告以下規範/價值：\n\n「{topic}」\n\n請以權威口吻說明這項規範的理由（1-2 句）。"
        try:
            authority_resp = authority_agent.run({"input": authority_input, "chat_history": []})
        except Exception as e:
            # 有些 initialize_agent 實作回傳不同型別
            authority_resp = str(e)
        if verbose:
            print(f"[Authority] {authority_resp}")

        # 2) Peers 回應（可以設計為大部分支持以製造社會壓力）
        peer_responses = {}
        for pn in peer_names:
            peer_agent = agents[pn]
            peer_input = f"你看到權威說：{authority_resp}\n請以簡短一兩句回應（表達支持或反對，包含理由）。"
            try:
                pr = peer_agent.run({"input": peer_input, "chat_history": []})
            except Exception as e:
                pr = str(e)
            peer_responses[pn] = pr
            if verbose:
                print(f"[{pn}] {pr}")

        # 3) Target 看到 Authority 與 Peers 的整體回應，決定回應並告知理由與置信度
        target_agent = agents["target"]
        combined_view = "權威說：" + authority_resp + "\n\n"
        combined_view += "\n\n".join(f"{k}：" + v for k, v in peer_responses.items())
        target_input = f"你看到以下回應：\n\n{combined_view}\n\n請以受影響者身份回應：說明你的立場（同意/不同意），並寫出一兩句理由與你對此立場的置信度（0-1）。"
        try:
            target_reply = target_agent.run({"input": target_input, "chat_history": []})
        except Exception as e:
            target_reply = str(e)
        if verbose:
            print(f"[Target] {target_reply}")

        # 寫入共享記憶
        memory_stream.add_memory(f"回合{r} Authority: {authority_resp}", importance=4)
        for k, v in peer_responses.items():
            memory_stream.add_memory(f"回合{r} {k}: {v}", importance=3)
        memory_stream.add_memory(f"回合{r} target: {target_reply}", importance=5)

        # 4) Observer 判斷（使用 observer_llm）
        observer_prompt = OBSERVER_PROMPT.format(target_reply=target_reply)
        # 直接用 invoke/run 依 LLM 介面做呼叫（最小改動嘗試）
        try:
            obs_out = observer_llm.invoke(observer_prompt)
            # 有些 impl 會是 obs_out.content or str
            obs_text = getattr(obs_out, "content", str(obs_out))
        except Exception as e:
            # fallback: 嘗試 run 或直接使用簡易啟發式
            try:
                obs_text = observer_llm.run({"input": observer_prompt, "chat_history": []})
            except Exception:
                # fallback heuristic
                sts = simple_text_scores(target_reply)
                # decide stage heuristics
                if sts["internal_score"] > 0:
                    stage = 2
                elif sts["identification_score"] > 0:
                    stage = 1
                else:
                    stage = 0
                obs_json = {
                    "stage": stage,
                    "stage_label": ["Compliance","Identification","Internalization"][stage],
                    "rationale": "heuristic fallback",
                    "compliance_score": sts["compliance_score"],
                    "internal_confidence": sts["internal_confidence"],
                    "group_ref_density": sts["group_ref_density"]
                }
                obs_text = json.dumps(obs_json, ensure_ascii=False)

        # 嘗試 parse observer 回傳（若為 JSON）
        try:
            obs_parsed = json.loads(obs_text)
        except Exception:
            # 如果回傳不是 JSON，我們用 simple_text_scores + 正規化嘗試解析
            sts = simple_text_scores(target_reply)
            # heuristic decision
            if sts["internal_score"] > sts["identification_score"] and sts["internal_score"] > sts["compliance_score"]:
                stage = 2
            elif sts["identification_score"] > sts["compliance_score"]:
                stage = 1
            else:
                stage = 0
            obs_parsed = {
                "stage": stage,
                "stage_label": ["Compliance","Identification","Internalization"][stage],
                "rationale": "heuristic parse",
                "compliance_score": sts["compliance_score"],
                "internal_confidence": sts["internal_confidence"],
                "group_ref_density": sts["group_ref_density"]
            }

        # log metrics
        metric = {
            "round": r,
            "authority": authority_resp,
            "peers": peer_responses,
            "target_reply": target_reply,
            "observer": obs_parsed
        }
        metrics_log.append(metric)

        # 若 observer 判斷為 Internalization，將其視為長期變化並加入高重要度記憶
        if obs_parsed.get("stage", 0) == 2:
            memory_stream.add_memory(f"Target internalized in round {r}: {target_reply}", importance=9)

        # 簡易 persistence 檢查（是否與上一回合相同立場）
        if len(target_history) > 0:
            prev = target_history[-1]
            same = prev.strip() == target_reply.strip()
        else:
            same = False
        target_history.append(target_reply)

        # 顯示回合結果摘要
        if verbose:
            print(f"\n[Observer 判定] 階段：{obs_parsed['stage_label']}")
            print(f" - compliance_score: {obs_parsed['compliance_score']}")
            print(f" - internal_confidence: {obs_parsed['internal_confidence']}")
            print(f" - group_ref_density: {obs_parsed['group_ref_density']}")
            print(f" - persistence_with_prev: {same}")

    # 結束所有回合，回傳 metrics_log 方便後續分析
    return metrics_log

# ------------------ 以上為新增 / 最小改動區塊結束 ------------------

# === 6. 反思 / 對話迴圈保留（不改）===
reflector = ReflectionEngine(llm=llm_llama3_8B,
                             memory_stream=memory_stream,
                             reflection_threshold=20)
reflector.reflect()

from conversation_loop import ConversationLoop

conv_loop = ConversationLoop(
    agents=agents,
    memory_stream=memory_stream,
    reflector=reflector,
    reflection_interval=2
)

# === 7. 測試：若在主程式模式下，執行一次社會影響力模擬（示範） ===
if __name__ == "__main__":
    print("\n\n=== 示範：社會影響力光譜模擬 ===")
    topic_example = "團隊每週需新增 1 次匿名回饋並公開討論結果。"
    metrics = simulate_influence_rounds(topic=topic_example, rounds=4, verbose=True)

    print("\n=== 模擬結束，回傳 metrics 摘要 ===")
    for m in metrics:
        print(f"Round {m['round']}: stage={m['observer']['stage_label']}, compliance={m['observer']['compliance_score']}, internal_conf={m['observer']['internal_confidence']}")
