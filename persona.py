# persona.py
"""
Persona Generation Function Library

用途：
- 提供可重複使用的角色（Persona）生成工具
- 適用於 Generative Agents / Multi-Agent Simulation
- 可搭配 LangChain / AutoGen / 自建 Agent Framework 使用

設計目標：
- 統一 Persona 結構
- 可快速建立 agent 初始人格設定
- 支援：
    1. 單一角色生成
    2. 多角色批次生成
    3. 客製化角色屬性
    4. Prompt-ready 格式輸出
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import random


# =========================================================
# Persona Data Structure
# =========================================================

@dataclass
class Persona:
    name: str
    age: int
    gender: str
    occupation: str
    personality_traits: List[str]
    communication_style: str
    core_belief: str
    initial_stance: str
    background_story: str
    emotional_tendency: str
    stubbornness: int  # 1~10
    openness: int      # 1~10
    trust_in_authority: int  # 1~10

    def to_dict(self):
        return asdict(self)

    def to_prompt(self) -> str:
        """
        轉換成適合 LLM 使用的 Prompt 格式
        """

        traits = ", ".join(self.personality_traits)

        return f"""
You are playing the role of a human participant in a social discussion.

Persona Profile:
- Name: {self.name}
- Age: {self.age}
- Gender: {self.gender}
- Occupation: {self.occupation}
- Personality Traits: {traits}
- Communication Style: {self.communication_style}
- Core Belief: {self.core_belief}
- Initial Stance: {self.initial_stance}
- Emotional Tendency: {self.emotional_tendency}
- Stubbornness Level: {self.stubbornness}/10
- Openness to New Ideas: {self.openness}/10
- Trust in Authority: {self.trust_in_authority}/10

Background:
{self.background_story}

Behavior Rules:
- Stay consistent with your persona
- Respond naturally like a real human
- You may change your opinion only if strongly persuaded
- Your decisions should reflect your personality
- Do not act like an AI assistant
"""


# =========================================================
# Default Persona Pools
# =========================================================

DEFAULT_NAMES = [
    "Alex", "Sophia", "Daniel", "Emma",
    "Michael", "Olivia", "James", "Isabella"
]

DEFAULT_OCCUPATIONS = [
    "Software Engineer",
    "Teacher",
    "Journalist",
    "Doctor",
    "University Student",
    "Business Analyst",
    "Government Employee",
    "Researcher"
]

DEFAULT_PERSONALITY_TRAITS = [
    ["logical", "calm", "analytical"],
    ["emotional", "passionate", "expressive"],
    ["skeptical", "critical", "independent"],
    ["optimistic", "open-minded", "idealistic"],
    ["conservative", "cautious", "practical"],
    ["stubborn", "confident", "assertive"]
]

DEFAULT_COMMUNICATION_STYLES = [
    "polite and rational",
    "direct and aggressive",
    "friendly and open",
    "formal and careful",
    "sarcastic and skeptical"
]

DEFAULT_STANCES = [
    "Support",
    "Oppose",
    "Neutral"
]

DEFAULT_BELIEFS = [
    "Scientific evidence should guide decisions.",
    "Authority figures are usually trustworthy.",
    "People should think independently.",
    "Social harmony is more important than conflict.",
    "Emotional intuition is often more reliable than logic."
]

DEFAULT_EMOTIONS = [
    "emotionally stable",
    "easily anxious",
    "easily angered",
    "highly empathetic",
    "socially sensitive"
]


# =========================================================
# Core Function
# =========================================================

def generate_persona(
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    occupation: Optional[str] = None,
    initial_stance: Optional[str] = None
) -> Persona:
    """
    產生單一 Persona
    """

    name = name or random.choice(DEFAULT_NAMES)
    age = age or random.randint(22, 55)
    gender = gender or random.choice(["Male", "Female"])
    occupation = occupation or random.choice(DEFAULT_OCCUPATIONS)

    traits = random.choice(DEFAULT_PERSONALITY_TRAITS)
    communication_style = random.choice(DEFAULT_COMMUNICATION_STYLES)
    core_belief = random.choice(DEFAULT_BELIEFS)
    initial_stance = initial_stance or random.choice(DEFAULT_STANCES)
    emotional_tendency = random.choice(DEFAULT_EMOTIONS)

    stubbornness = random.randint(3, 9)
    openness = random.randint(3, 9)
    trust_in_authority = random.randint(2, 9)

    background_story = (
        f"{name} works as a {occupation} and often participates "
        f"in discussions about social issues. "
        f"They value {core_belief.lower()}"
    )

    return Persona(
        name=name,
        age=age,
        gender=gender,
        occupation=occupation,
        personality_traits=traits,
        communication_style=communication_style,
        core_belief=core_belief,
        initial_stance=initial_stance,
        background_story=background_story,
        emotional_tendency=emotional_tendency,
        stubbornness=stubbornness,
        openness=openness,
        trust_in_authority=trust_in_authority
    )


def generate_persona_group(
    num_agents: int = 6
) -> List[Persona]:
    """
    批次產生多個 Persona
    """

    personas = []

    used_names = set()

    while len(personas) < num_agents:
        persona = generate_persona()

        if persona.name not in used_names:
            used_names.add(persona.name)
            personas.append(persona)

    return personas


def export_prompt_list(
    personas: List[Persona]
) -> List[str]:
    """
    匯出所有 persona prompt
    """

    return [p.to_prompt() for p in personas]
