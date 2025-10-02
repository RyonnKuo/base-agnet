from reflection import ReflectionEngine
from memory import Memory, MemoryStream
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
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


# 所有 Agent 共用同一個 MemoryStream，彼此回覆會被寫回 FAISS，後續查詢時能互相參考。
# === 1. 建立共享記憶流 ===
memory_stream = MemoryStream()

# === 2. 封裝記憶檢索工具 ===
memory_tool = Tool(
    name="MemoryRetriever",
    func=lambda q: "\n".join(
        m.content for m in memory_stream.search_by_vector(q)),
    description="檢索過往的重要對話與事件"
)

# initialize_agent 的 tools 參數必須是 List[Tool]
tools = [memory_tool]

# === 3. 定義多個角色與對應 LLM ===
# 新增角色，加入新的 (角色描述, llm) 配對
roles = {
    "llama":  ("嚴謹的事實檢驗者", llm_llama3_8B),
    "mistral": ("理性分析師",       llm_mistral_7B),
    "gemma":  ("富有想像力的創意家", llm_gemma3_4B)
}

agents = {}
for name, (role_desc, llm_model) in roles.items():
    prompt_prefix = f"你是一位{role_desc}，回答時請保持此角色的語氣與思考方式。"
    # ZERO_SHOT_REACT_DESCRIPTION 會把工具描述與 prompt 自動組合
    agents[name] = initialize_agent(
        tools=tools,
        llm=llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": prompt_prefix}
    )


# === 4. 對話流程示例 ===
def multi_agent_round(user_input: str):
    print(f"\n=== 使用者輸入 ===\n{user_input}\n")
    # 對每個 agent 輪詢
    for name, agent in agents.items():
        response = agent.run(user_input)
        print(f"[{name}] {response}\n")
        # 將回覆寫入共享記憶
        memory_stream.add_memory(
            f"{name} 回覆: {response}",
            importance=5
        )


# === 5. 測試對話 ===
if __name__ == "__main__":
    multi_agent_round("請大家一起討論氣候變遷對農業的影響。")
    multi_agent_round("根據剛剛的討論，再給我一段結論。")

# === 6. 可選：反思與長期記憶整理 ===
reflector = ReflectionEngine(llm=llm_llama3_8B,
                             memory_stream=memory_stream,
                             reflection_threshold=20)
reflector.reflect()
