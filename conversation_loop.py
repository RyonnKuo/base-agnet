# conversation_loop.py
from datetime import datetime, timedelta

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
