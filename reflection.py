# 反思
class ReflectionEngine:
    def __init__(self, llm, memory_stream, reflection_threshold=50):
        self.llm = llm  # LangChain ChatModel 或任何 LLM
        self.memory_stream = memory_stream
        self.reflection_threshold = reflection_threshold

    def should_reflect(self):
        return sum(m.importance for m in self.memory_stream.memories) >= self.reflection_threshold

    def get_reflection_prompt(self, top_k=10):
        # 取出重要性高的記憶
        top_memories = self.memory_stream.get_top_memories_by_importance(n=top_k)
        memory_text = "\n".join(f"- {m.content}" for m in top_memories)

        prompt = (
            "根據以下觀察記憶，請總結出 3 個高層次的洞察（以句子表示）：\n"
            f"{memory_text}\n"
            "請使用清楚的自然語言列出三點觀察。"
        )
        return prompt

    def reflect(self):
        if not self.should_reflect():
            print("目前尚未達到反思門檻，不執行反思。")
            return []

        prompt = self.get_reflection_prompt()
        response = self.llm.invoke(prompt)
        # 假設回傳為以數字開頭的多行文字
        reflections = [line.strip("123. ").strip() for line in response.content.strip().split("\n") if line.strip()]
        
        # 寫回反思記憶（importance 可設為高）
        for insight in reflections:
            self.memory_stream.add_memory(insight, importance=8)

        print("已完成反思，新增記憶：")
        for r in reflections:
            print(f" - {r}")
        return reflections