# from langchain.schema import SystemMessage, HumanMessage
# from typing import List


# class SocialAgent:
#     def __init__(self, name: str, llm, memory: MemoryStream):
#         self.name = name
#         self.llm = llm
#         self.memory = memory

#     def perceive(self, message: str):
#         # 將外部輸入存入記憶
#         self.memory.add_memory(f"{self.name} 接收到: {message}", importance=5)

#     def act(self, context: str) -> str:
#         # 生成回應並存入記憶
#         response = self.llm.invoke([SystemMessage(content=f"你是{self.name}"),
#                                     HumanMessage(content=context)])
#         self.memory.add_memory(f"{self.name} 回應: {response.content}", importance=5)
#         return response.content
#     def sec_act(self, context: str) -> str:
#         related = self.memory.search_by_vector(context, k=5)
#         memory_summary = "\n".join(m.content for m in related)
#         prompt = f"根據以下背景資訊：{memory_summary}\n請回答：{context}"
#         response = self.llm.invoke(prompt)
#         self.memory.add_memory(f"{self.name} 回應: {response.content}", importance=5)
#         return response.content
    
#     def run_simulation(agents: List[SocialAgent], rounds: int = 5):
#         context = "大家聊聊今天的天氣"
#         for i in range(rounds):
#             print(f"\n=== 第 {i+1} 輪 ===")
#             for agent in agents:
#                 reply = agent.act(context)
#                 print(f"{agent.name}: {reply}")
#                 # 其他代理人感知
#                 for other in agents:
#                     if other != agent:
#                         other.perceive(f"{agent.name} 說：{reply}")
