from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import uuid
import json

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    importance: int = 1
    embedding: Optional[List[float]] = None  # 語意向量

# 記憶流
class MemoryStream:
    def __init__(self):
        self.memories: list[Memory] = []

    def add_memory(self, content: str, importance: int = 1):
        memory = Memory(content=content, importance=importance)
        self.memories.append(memory)
        return memory

    def add_memories(self, memory_list: list[str], importance: int = 1):
        return [self.add_memory(content, importance) for content in memory_list]

    def get_recent_memories(self, n=5):
        return sorted(self.memories, key=lambda m: m.created_at, reverse=True)[:n]

    def update_access_time(self, memory_id: str):
        for mem in self.memories:
            if mem.id == memory_id:
                mem.last_accessed_at = datetime.now()
                return mem
        return None

    def all(self):
        return self.memories
    
    def get_top_memories_by_importance(self, n=5):
        return sorted(self.memories, key=lambda m: m.importance, reverse=True)[:n]

    def search_by_keyword(self, keyword: str, n=5):
        return [m for m in self.memories if keyword in m.content][:n]
    
    def save_to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in self.memories], f, ensure_ascii=False, indent=2)

    def load_from_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                self.memories = [Memory(**m) for m in raw_data]
        except Exception as e:
            print(f"載入記憶失敗：{e}")
            
