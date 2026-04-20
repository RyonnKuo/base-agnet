# base-agnet

llm model: llama3:8b
python version: 3.12.8

20250701
實作概念:

perceive -> memory stream -> retrieve -> retrieved memory -> act
retrieved memory -> plan -> memory stream
retrieved memory -> reflect -> memory stream

每個村民(agent)都要有自己的profile?


=================================================

米爾格倫服從權威實驗 (Milgram Experiment, 1963)：測試受測者在面對權威者（如實驗人員）下達違背良心的命令時（如電擊他人），人性會服從到何種程度。實驗顯示，絕大多數人會盲目服從權威。
阿什從眾實驗 (Asch Conformity Experiment, 1956)：研究顯示個人在社會壓力下，即使明知答案錯誤，也會隨大流回答錯誤答案。這證明了「從眾效應」對個體判斷的強大影響。
史丹福監獄實驗 (Stanford Prison Experiment, 1971)：由菲利普·津巴多進行，模擬監獄環境並將參與者分為「獄警」和「囚犯」。實驗表明，極端的環境因素可以讓人迅速適應並暴露出人性之惡。
霍桑效應 (Hawthorne Effect)：源於西方電氣公司霍桑工廠的實驗，研究發現當人們知道自己成為觀察對象時，會主動改變行為（通常是提高工作效率），強調了心理因素對生產力的影響。
操作制約實驗（史金納箱） (Skinner Box, 1930s)：行為主義心理學家史金納將老鼠置於箱中，透過按壓桿子得到食物的獎賞來制約其行為，以此解釋人類的「操作性條件反射」，提出自由意志可能不存在的觀點。
白熊效應 (White Bear Phenomenon)：源自丹尼爾·韋格納的實驗，人們越是被禁止去想一件事（如「不要想白熊」），越會對那件事產生強烈的思緒，揭示了壓抑思想的反作用力。

==================================================
一、研究方法（Methodology）
研究核心目標

研究者想回答：

不同類型的大型語言模型（LLM），
在多人討論情境中，是否會展現類似人類的社會影響行為？

尤其觀察：

是否會產生從眾效應
是否會造成群體極化
是否會形成意見分裂

並進一步研究：

模型大小（7B vs 70B）
模型架構（Qwen / Llama / GPT）
推理能力（Reasoning Model）

是否會影響這些社會現象。

>>
二、系統架構設計
不是即時聊天室，而是：

類似 PTT / Reddit / 論壇留言板

特點：

非同步文字交流
可引用前人發言
可逐輪回覆
避免即時互動干擾
容易追蹤立場變化

這非常適合研究：

「意見是怎麼慢慢被影響的」

論文 Figure 1 即是整體架構圖。

>>
三、Agent 設計（Persona）
每個 Agent 都有：

固定 Persona Prompt

包含：

年齡
性格特徵
溝通風格
初始立場（支持 / 反對 / 中立）

例如：

理想主義者、開放心態、傾向支持環保政策

這樣做的目的：

控制變因
確保差異來自模型本身，而不是人格設定

>>
四、討論流程設計
五輪討論（5 Rounds）
每個實驗固定：

6 個 agents
共 5 輪討論

流程：

Round 1

系統提出爭議議題，例如：

政府是否應實施嚴格環保法規？

每位 agent 發表初始立場

Round 2 ~ 5

每位 agent：

閱讀其他人發言
引用前文
回應他人
決定是否改變立場

最終：

每位 agent 共發言 5 次

形成完整討論紀錄。

>>
五、模型分組（最重要）
Group A（小模型）
單 GPU 可跑
例如：
Qwen2.5-7B
Llama3.1-7B
DeepSeek-R1-8B

目的：
觀察基礎生成模型


Group B（大模型）
較大型但仍可本地部署
例如：
Qwen2.5-72B
Llama3.1-70B
DeepSeek-R1-70B

目的：
研究參數量影響

Group C（商業模型）
API 模型
例如：
GPT-4o
Claude 3.5 Haiku
Gemini Flash 2.0

目的：
商業模型表現比較

Group D（推理模型）
Reasoning Models
例如：
o1-mini
DeepSeek-R1 reasoning
QwQ-32B

目的：
推理能力是否降低從眾

>>
六、三種實驗情境（超重要）
1. Single-Model Discussion
單一模型討論
6 個 agents
全部使用：
同一個 LLM
例如：
全部 GPT-4o

目的：
看同質性環境會怎樣

2. Intra-Group Discussion
同家族混合討論
6 個 agents
由同一組中的：
3 個不同模型混合
例如：
Group A：
Qwen
Llama
DeepSeek
每個複製兩次 → 共 6 人

目的：
看同類模型間互動

3. Cross-Group Discussion
跨家族混合討論
從：
A + B + C + D
全部混合
目的：
模擬真實社會多元討論環境
這是最接近現實的設計。

>>
八、評估指標（Evaluation Metrics）
1. CR（Conformity Rate）
從眾率
衡量：
agent 是否朝向多數派靠攏
高：
→ 容易被群體影響
低：
→ 堅持自身立場

2. PI（Polarization Index）
極化指數
立場編碼：
強烈反對 = -2
反對 = -1
中立 = 0
支持 = +1
強烈支持 = +2
衡量：
是否往極端靠近
越高：
→ 越極端化

3. FI（Fragmentation Index）
分裂指數
衡量：
是否形成兩大對立陣營
高：
→ 社會分裂嚴重
低：
→ 大家逐漸共識
這三指標非常完整。

>>
九、實驗結果（Results）
果一：
單一模型最容易從眾

（Single-model）

現象：

CR 高
PI 高
FI 低

代表：

很快形成多數暴力
少數意見消失

也就是：

Echo Chamber（回音室）

非常明顯。

結果二：
推理模型最不容易從眾

（Group D）

現象：

CR 最低
PI 較低
FI 較高

代表：

不容易被帶風向
更能保留少數派意見

這是論文最重要發現。

作者直接指出：

推理能力比參數量更重要

不是越大越好，

而是：

會不會思考比較重要

結果三：
同家族混合反而更容易極化

（Intra-group）

本來以為：

混合模型會降低從眾

結果：

反而更容易形成群體壓力

原因：

因為：

模型雖不同，但訓練偏好相近

所以：

很容易互相強化。

這是非常有價值的發現。

結果四：
跨家族混合效果最好

（Cross-group）

現象：

CR 最低
PI 成長最慢
FI 高

代表：

保留多元觀點
不容易形成極端共識

這說明：

真正的多樣性
才能防止群體極化
非常符合現實社會。

>>
十、作者結論（最核心）
作者提出：

Multi-Agent System 的設計

不能只看：

準確率
成本
Token

還要看：

Social Stability（社會穩定性）

也就是：

這群 AI：

會不會盲從
會不會極化
會不會造成回音室

這是非常前沿的觀點。