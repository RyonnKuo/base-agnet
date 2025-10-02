from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
import uuid
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import faiss


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    importance: int = 1
    embedding: Optional[List[float]] = None


class MemoryStream:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.memories: list[Memory] = []
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # 取得 embedding 維度
        sample_embed = self.embeddings.embed_query("Hello world")
        dim = len(sample_embed)

        # 建立空 FAISS index
        index = faiss.IndexFlatL2(dim)

        # 初始化 LangChain FAISS wrapper
        self.vector_store = FAISS(
            embedding_function = self.embeddings,
            index = index,
            docstore = InMemoryDocstore(),
            index_to_docstore_id = {}
        )

    def add_memory(self, content: str, importance: int = 1):
        # 建立 Memory
        embedding = self.embeddings.embed_query(content)
        memory = Memory(content=content, importance=importance, embedding=embedding)
        self.memories.append(memory)

        # 加入 FAISS
        # self.vector_store.add_texts([content], metadatas=[{"id": memory.id}])
        # return memory
    
        # 用 add_embeddings 代替 add_texts，避免重算
        self.vector_store.add_embeddings(
            text_embeddings=[(content, embedding)],
            metadatas=[{"id": memory.id}],
            ids=[memory.id]        # 可選，若不給會自動產生
        )
        return memory

    def add_memories(self, memory_list: list[str], importance: int = 1):
        return [self.add_memory(content, importance) for content in memory_list]

    def search_by_vector(self, query: str, k=5):
        docs = self.vector_store.similarity_search(query, k=k)
        results = []
        for d in docs:
            mem_id = d.metadata.get("id")
            mem = next((m for m in self.memories if m.id == mem_id), None)
            if mem:
                results.append(mem)
        return results

    def get_recent_memories(self, n=5):
        return sorted(self.memories, key=lambda m: m.created_at, reverse=True)[:n]

    def get_top_memories_by_importance(self, n=5):
        return sorted(self.memories, key=lambda m: m.importance, reverse=True)[:n]

    def save_to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in self.memories], f, ensure_ascii=False, indent=2)

    def load_from_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                self.memories = [Memory(**m) for m in raw_data]

            texts  = [m.content for m in self.memories]
            embeds = [m.embedding for m in self.memories if m.embedding]
            metas  = [{"id": m.id} for m in self.memories]

            # 使用已保存的 embedding 避免重算
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeds)),
                embedding=self.embeddings,
                metadatas=metas,
                ids=[m.id for m in self.memories]
            )
        except Exception as e:
            print(f"載入記憶失敗：{e}")
    
    def as_retriever(self, search_kwargs: dict = {"k": 5}):
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    

from langchain.embeddings import HuggingFaceEmbeddings
class VectorMemory:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or HuggingFaceEmbeddings()
        self.store = None  # 之後用 FAISS 存向量

    def add_text(self, text: str):
        if self.store is None:
            self.store = FAISS.from_texts([text], self.embedding_model)
        else:
            self.store.add_texts([text])

    def search(self, query: str, k: int = 3):
        return self.store.similarity_search(query, k=k)
