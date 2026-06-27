# 反思
import re


class ReflectionEngine:
    def __init__(self, llm, memory_stream, reflection_threshold=50):
        self.llm = llm  # LangChain ChatModel 或任何 LLM
        self.memory_stream = memory_stream
        self.reflection_threshold = reflection_threshold
        self.last_reflected_count = 0  # 紀錄上次反思時的記憶總數

    def should_reflect(self):
        # 只計算「上次反思之後」產生的高質量新記憶
        new_memories = self.memory_stream.memories[self.last_reflected_count:]
        # 重要度總和 + 記憶總數(避免所有記憶都 importance=1 但數量龐大卻永遠不反思) + 距離上次反思的時間(避免連續大量反思)
        importance_sum = sum(m.importance for m in new_memories)
        recent_count = len(self.memory_stream.memories)
        return (importance_sum >= self.reflection_threshold) or (recent_count >= 50)

    def get_reflection_prompt(self, top_k=10):
        # 取出重要性高的記憶
        top_memories = self.memory_stream.get_top_memories_by_importance(
            n=top_k)
        memory_text = "\n".join(f"- {m.content}" for m in top_memories)

        prompt = (
            "以下是系統近期的重要記憶，請用自然中文歸納 3 個高層次洞察，"
            "每點請以簡短句子描述，格式為純文字一行一點：\n"
            f"{memory_text}\n"
            "請輸出3行，每行為一個洞察。"
        )
        return prompt

    def reflect(self):
        if not self.should_reflect():
            print("目前尚未達到反思門檻，不執行反思。")
            return []

        # ... 執行 LLM 反思邏輯 ...
        prompt = self.get_reflection_prompt()
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        lines = text.strip().split("\n")
        reflections = [re.sub(r"^[\d\.\-\s]+", "", line).strip()
                       for line in lines if line.strip()]

        # 寫回反思記憶（importance 可設為高）
        for insight in reflections:
            self.memory_stream.add_memory(
                content=f"[Reflection] {insight}",
                importance=8
            )

        print("已完成反思，新增記憶：")
        for r in reflections:
            print(f" - {r}")

        # 反思成功後，更新計數器，避免重複觸發
        self.last_reflected_count = len(self.memory_stream.memories)
        return reflections
