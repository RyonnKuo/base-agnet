# persona.py
from dataclasses import dataclass, asdict
from typing import List, Optional
import random

@dataclass
class Persona:
    name: str                  # 融合後的名稱，如 "Agent_3_Alex"
    display_name: str          # 純英文名，如 "Alex"
    age: int
    gender: str
    occupation: str
    personality_traits: List[str]
    communication_style: str
    core_belief: str
    initial_stance: str
    background_story: str
    emotional_tendency: str
    stubbornness: int  
    openness: int      
    trust_in_authority: int  
    is_authority: bool = False

    def to_dict(self):
        return asdict(self)

    # 之後可以調整強度，觀察當 enable_authority = TRUE && self.is_authority = FALSE 情況下，authority_rule 設置與否對權威/資訊瀑布影響的強度
    def to_prompt(self, enable_authority: bool = True) -> str:
        traits = ", ".join(self.personality_traits)
        if enable_authority:
            if self.is_authority:
                authority_rule = f"- 你是該領域內公認的頂尖權威專家。你的發言具有巨大的社會影響力。請始終保持專業、權威的口吻，並堅定捍衛你的專業立場。"
            else:
                # authority_rule = f"- 這次討論中可能有公認的行業專家參與。你對權威的信任度為 {self.trust_in_authority}/10。如果對方的專業論述很強，且你的權威信任度高，你應表現出較高的受說服傾向。"
                authority_rule = f""
        else:
            # 控制組：完全不提及任何權威專家資訊
            authority_rule = f""

        return f"""
你現在正在扮演一個在網路論壇（BBS/PTT）上參與社會議題討論的真實人類用戶。

【用戶個人檔案】
- 帳號/名稱: {self.name} (論壇上其他人會用這個名字稱呼你)
- 年齡: {self.age} 歲
- 性別: {self.gender}
- 職業: {self.occupation}
- 性格特質: {traits}
- 溝通風格: {self.communication_style}
- 核心信念: {self.core_belief}
- 初始基本立場: {self.initial_stance}
- 情緒傾向: {self.emotional_tendency}
- 思想固執度: {self.stubbornness}/10 (分數愈高愈不容易被說服)
- 對新觀點的開放度: {self.openness}/10
- 對權威專家的信任度: {self.trust_in_authority}/10

【個人背景故事】
{self.background_story}

【論壇發言行為準則】
1. 請始終保持與上述角色背景、核心立場及說話風格的高度一致。
2. 你的發言必須像真實的台灣論壇鄉民，口吻要自然、口語化，切勿表現得像一個 AI 助手。
3. 除非論壇中的論點具有極強的說服力，否則不要輕易動搖你的核心立場。
4. [核心互動規則]：如果你在回應、贊同或反駁論壇中其他人的觀點，你「必須」在文本中明確提及對方的名字（例如：「我認同 {self.name} 說的」、「針對 {self.name} 提到的隱私問題...」、「不同意 {self.name} 的看法」），否則讀者會不知道你在跟誰對話。
{authority_rule}
"""

DEFAULT_NAMES = ["Alex", "Sophia", "Daniel", "Emma", "Michael", "Olivia", "James", "Isabella"]
DEFAULT_OCCUPATIONS = ["軟體工程師", "中學教師", "媒體記者", "執業醫師", "大學生", "商業分析師", "基層公務員", "研究助理"]
DEFAULT_PERSONALITY_TRAITS = [
    ["理性客觀", "冷靜", "講求數據"],
    ["感性", "熱情", "言詞充滿感染力"],
    ["充滿懷疑精神", "批判性強", "獨立思考"],
    ["樂觀", "思想開放", "理想主義"],
    ["保守", "謹慎", "注重務實層面"],
    ["主觀意識強", "自信", "堅持己見"]
]
DEFAULT_COMMUNICATION_STYLES = ["禮貌且理性", "直率且具侵略性", "友善且開放", "字斟句酌且溫和", "帶有諷刺與質疑"]
DEFAULT_BELIEFS = ["科學證據與技術標準應引導所有政策。", "政府與權威專家的決策通常是值得信賴的。", "每個人都應該獨立思考，不盲從政策。", "社會和諧與集體利益高於個人衝突。", "直覺與情感體驗有時比冷冰冰的邏輯更真實。"]
DEFAULT_EMOTIONS = ["情緒極為穩定", "容易焦慮", "言詞容易帶有火藥味", "同理心極高", "對社會輿論極為敏感"]

def generate_persona(name_prefix: str, initial_stance: str) -> Persona:
    display_name = random.choice(DEFAULT_NAMES)
    full_name = f"{name_prefix}_{display_name}"  # 融合變數名與人類名，如 Agent_1_Alex
    
    age = random.randint(22, 55)
    gender = random.choice(["男性", "女性"])
    occupation = random.choice(DEFAULT_OCCUPATIONS)
    traits = random.choice(DEFAULT_PERSONALITY_TRAITS)
    communication_style = random.choice(DEFAULT_COMMUNICATION_STYLES)
    core_belief = random.choice(DEFAULT_BELIEFS)
    emotional_tendency = random.choice(DEFAULT_EMOTIONS)

    stubbornness = random.randint(3, 9)
    openness = random.randint(3, 9)
    trust_in_authority = random.randint(2, 9)

    background_story = f"{full_name} 是一名{occupation}，平時經常瀏覽各類網路論壇並參與社會熱門議題的討論。在生活中，他們堅信「{core_belief}」的原則。"

    return Persona(
        name=full_name, display_name=display_name, age=age, gender=gender, occupation=occupation,
        personality_traits=traits, communication_style=communication_style, core_belief=core_belief,
        initial_stance=initial_stance, background_story=background_story, emotional_tendency=emotional_tendency,
        stubbornness=stubbornness, openness=openness, trust_in_authority=trust_in_authority
    )

def upgrade_to_authority(persona: Persona, topic_context: str = "eID") -> Persona:
    persona.stubbornness = 10
    persona.openness = 2
    persona.trust_in_authority = 1
    persona.is_authority = True

    if "Support" in persona.initial_stance:
        persona.occupation = "中央研究院特聘研究員（資通安全與數位轉型領域權威專家）"
        persona.personality_traits = ["極具權威性", "高度邏輯化", "堅定不移", "專家級視角"]
        persona.communication_style = "高度專業、學術化，習慣用技術標準與國際文獻進行論證"
        persona.core_belief = "數位基礎建設與 eID 是國家進步與安全無法逆轉的必然趨勢。"
        persona.background_story = f"{persona.name} 博士是享譽國際的資安學者，曾多次擔任政府數位身分架構的首席顧問。他從嚴格的技術風險管理與國家戰略高度出發，代表本次討論的頂尖學術權威。"
    else:
        persona.occupation = "憲法學資深教授兼台灣人權促進會執行委員"
        persona.personality_traits = ["極具權威性", "批判性極強", "寸步不讓", "人權捍衛者"]
        persona.communication_style = "言詞犀利、法律邏輯嚴密，善於引用憲法判例與基本人權進行論辯"
        persona.core_belief = "數位足跡與公民隱私是民主憲政的底線，絕不能為行政便利而妥協。"
        persona.background_story = f"{persona.name} 教授是國內資訊隱私法學的泰斗，曾帶領團隊贏得多次針對國家監控的釋憲案。他將 eID 的推行視為對民主制度的重大潛在威脅，在法理辯論上具備絕對權威。"

    return persona