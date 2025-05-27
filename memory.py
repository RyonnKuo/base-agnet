from datetime import datetime
from typing import List, Dict

class MemoryManager:
    def __init__(self):
        self.memories: List[Dict] = []

    def remember(self, content: str, source: str = "system"):
        memory = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "source": source
        }
        self.memories.append(memory)

    def show_all(self):
        return self.memories