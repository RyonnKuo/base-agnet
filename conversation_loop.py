# conversation_loop.py
from datetime import datetime, timedelta
from typing import List, Dict

class ConversationLoop:
    def __init__(self, agents, memory_stream, reflector, reflection_interval=2):
        """
        agents: dict[str, AgentExecutor]
        memory_stream: MemoryStream
        reflector: ReflectionEngine
        reflection_interval: 每幾輪執行一次反思
        """
        self.agents = agents
        self.memory_stream = memory_stream
        self.reflector = reflector
        self.reflection_interval = reflection_interval
        self.round_counter = 0

    def run_round(self, user_input: str):
        print(f"\n=== 使用者輸入 ===\n{user_input}\n")
        for name, agent in self.agents.items():
            response = agent.run({
                "input": user_input,
                "chat_history": []
            })
            print(f"[{name}] {response}\n")
            self.memory_stream.add_memory(
                f"{name} 回覆: {response}",
                importance=5
            )
        self.round_counter += 1

        # 每隔幾輪觸發一次反思
        if self.round_counter % self.reflection_interval == 0:
            print("\n=== 進行反思階段 ===")
            self.reflector.reflect()

    def run_medical_case(self, medical_records: List[Dict[str, str]]):
        """
        讓多個 agent 討論一筆或多筆病歷，並在數輪後觸發反思。
        medical_records: List[{"patient_name": str, "notes": str}]
        """
        for record in medical_records:
            patient_name = record["patient_name"]
            notes = record["notes"]

            print(f"\n=== 病歷討論開始：{patient_name} ===")
            print(f"病歷摘要：{notes}\n")

            user_input = f"病人 {patient_name} 的描述如下：「{notes}」。請討論應該轉介到哪一個醫學科別，並說明原因。"

            for name, agent in self.agents.items():
                response = agent.run({
                    "input": user_input,
                    "chat_history": []
                })
                print(f"[{name}] {response}\n")

                # 寫入共享記憶
                self.memory_stream.add_memory(
                    f"{name} 對 {patient_name} 的判斷: {response}",
                    importance=6
                )

            self.round_counter += 1

            # 每隔幾輪觸發一次反思
            if self.round_counter % self.reflection_interval == 0:
                print("\n=== 🧠 進行反思階段 ===")
                self.reflector.reflect()

        print("\n=== 所有病歷討論結束 ===")