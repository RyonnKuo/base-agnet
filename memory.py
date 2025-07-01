from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    importance: int = 1
    
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