from langchain_community.chat_models import ChatOllama  # 使用 Ollama 封裝的 LLaMA 模型
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
import torch


# 使用本地 LLaMA 模型
llm_llama3_8B = ChatOllama(model="llama3:8B")
llm_phi3_3dot8B = ChatOllama(model="phi3:3.8B")
llm_mistral_7B = ChatOllama(model="mistral:7B")
llm_gemma3_4B = ChatOllama(model="gemma3:4B")

# GPU 加速
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

# 所有 Agent 共用同一個 MemoryStream，彼此回覆會被寫回 FAISS，後續查詢時能互相參考。
# === 1. 建立共享記憶流 ===
memory_stream = MemoryStream()

# === 2. 封裝記憶檢索工具 ===
memory_tool = Tool(
    name="MemoryRetriever",
    func=lambda q: "\n".join(m.content for m in memory_stream.search_by_vector(q)),
    description="檢索過往的重要對話與事件"
)

# initialize_agent 的 tools 參數必須是 List[Tool]
tools = [memory_tool]

# === 3. 定義多個角色與對應 LLM ===
# 新增角色，加入新的 (角色描述, llm) 配對
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


# === 4. 對話流程示例 ===
def multi_agent_round(user_input: str):
    print(f"\n=== 使用者輸入 ===\n{user_input}\n")
    # 對每個 agent 輪詢
    for name, agent in agents.items():
        response = agent.run({
            "input": user_input,
            "chat_history": []
        })
        print(f"[{name}] {response}\n")
        # 將回覆寫入共享記憶
        memory_stream.add_memory(
            f"{name} 回覆: {response}",
            importance=5
        )

def diagnose_round(medical_record: dict):
    patient_name = medical_record["patient_name"]
    notes = medical_record["notes"]

    print(f"\n=== 病歷分析 ===")
    print(f"病人姓名：{patient_name}")
    print(f"主訴與觀察紀錄：{notes}\n")

    diagnoses = {}

    # === 各專科醫師（agent）給出初步意見 ===
    for name, agent in agents.items():
        role = roles[name][0]
        prompt = f"""
        病人資料如下：
        姓名：{patient_name}
        病徵描述：{notes}

        請以一位{role}的身份，
        說明此病人可能的疾病原因，並建議該掛哪一科。
        """
        response = agent.run({
            "input": prompt,
            "chat_history": []
        })
        print(f"[{role}] 的診斷：{response}\n")

        diagnoses[name] = response

        # 寫入記憶
        memory_stream.add_memory(
            f"{role} 判斷：{response}",
            importance=5
        )

    # === 整合階段 ===
    integration_prompt = f"""
    以下是三位專科醫師的診斷意見：
    {diagnoses}

    請你根據這些診斷，
    給出最合理的「最終建議科別」，並簡述理由（請用繁體中文）。
    回答格式：
    建議科別：XXX
    理由：...
    """

    final_decision = llm_llama3_8B.invoke(integration_prompt)
    print(f"\n=== 最終判斷結果 ===\n{final_decision.content}\n")

    # 寫入記憶
    memory_stream.add_memory(
        f"最終判斷：{final_decision.content}",
        importance=8
    )



# === 5. 測試對話 ===
# 如果是 .py 就要加判斷式
# if __name__ == "__main__":
#     multi_agent_round("請大家一起討論氣候變遷對農業的影響。")
#     multi_agent_round("根據剛剛的討論，再給我一段結論。")
medical_records = [
    {"patient_name": "陳怡君", "notes": "病人右前臂出現紅疹並有輕微滲液"},
    {"patient_name": "王建國", "notes": "左膝關節疼痛兩週，走路時加劇"},
    {"patient_name": "李芳", "notes": "喉嚨刺痛並伴隨輕微發燒"},
]


# multi_agent_round("請大家一起討論氣候變遷對農業的影響。")
# multi_agent_round("根據剛剛的討論，再給我一段結論。")

for record in medical_records:
        diagnose_round(record)


# === 6. 反思與長期記憶整理 ===
reflector = ReflectionEngine(llm=llm_llama3_8B,
                             memory_stream=memory_stream,
                             reflection_threshold=20)
reflector.reflect()

from conversation_loop import ConversationLoop

# 建立對話迴圈物件
conv_loop = ConversationLoop(
    agents=agents,
    memory_stream=memory_stream,
    reflector=reflector,
    reflection_interval=2   # 每兩輪觸發一次反思
)

# 啟動對話流程
# conv_loop.run_round("請大家一起討論氣候變遷對農業的影響。")
# conv_loop.run_round("根據剛剛的討論，再給我一段結論。")
# conv_loop.run_round("請提出可行的調適策略。")  # 這一輪會觸發反思

conv_loop.run_medical_case(medical_records)