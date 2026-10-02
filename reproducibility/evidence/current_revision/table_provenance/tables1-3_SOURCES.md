# Table 1-3 来源与数据口径（核验版 v5）

## 本次确定修改

1. SEED-X：原作者 [v2 Table3](https://arxiv.org/html/2404.14396v2#S4.T3) 整行GenEval；方法部分明确Llama2-chat-13B。17B对照表口径不用于基础LLM列。
2. Show-o：原作者 [v5 Table1](https://arxiv.org/html/2408.12528v5#S4.T1) 普通配置MMMU26.7；27.4属于连续CLIP增强配置，不能与1097.2/58.0混用。GenEval用同论文Table3修订结果0.68；DPG用TokenFlow发布检查点评测，跨表不是同一实验配置。
3. Emu3-DPO：质量分类来自 [Emu3 Table8](https://arxiv.org/html/2409.18869v1#A2.T8)；延迟124.8秒来自 [Flash-Unified Table3](https://arxiv.org/html/2603.15271v1#S4.T3)。后者同时列Overall81.60，但不能据此推断双方采用同一个图像集合。caption明确质量/延迟可来自不同来源，跨来源时间不可直接比较。
4. Emu3 MME-P1243.8：来自 [TokenFlow Table3](https://arxiv.org/html/2412.03069#S4.T3)，原表标*说明由作者使用lmms-eval测发布权重，不是Emu3原作者自报。

## 其他文献行

- LWM GenEval：[SEED-X Table3](https://arxiv.org/html/2404.14396v2#S4.T3) 与 [JanusFlow Table2](https://arxiv.org/html/2411.07975v1#S4.T2) 一致。
- ILLUME：[原文](https://arxiv.org/html/2412.06673) Table3理解、Table4 GenEval。MMBench75.1沿用原文，不取其他对照表的65.1。
- JanusFlow：[原文](https://arxiv.org/html/2411.07975v1) Table2/3/5；384配置、base LLM1.3B。未使用与当前80.09不匹配的DPG77.57/1.09s组合。
- Janus-Pro7B：[原文](https://arxiv.org/html/2501.17811v1) Table3/4/5。Corvid的MMStar46.5/AI2D68.1为不同评测来源，为避免配置混用本版不填。
- Show-o DPG：[TokenFlow Table10](https://arxiv.org/html/2412.03069#A2.T10)，复测发布检查点。
- VILA-U：两张表中采用 [Flash-Unified Table1/2](https://arxiv.org/html/2603.15271v1) 的评测行，不能将其当作VILA-U原论文384配置。
- Show-o2灰色理解质量：[原文Table3](https://arxiv.org/html/2506.15564v1#S4.T3)；1.71s来自Flash-Unified。不能用这个公开时间除以本地0.865/0.491声称实测加速。

## 不完整数据和实验解释

横线仅表示本表采用的来源/设置下未报告。另一配置、硬件或评分器上的数值即使真实，也不自动填入。参数列统一为基础LLM规模，不代表含视觉模块的全部系统参数。保留23个空格。

本地所有数值保持v4。GenEval六类别宏平均；DPG总分由逐图依赖校正得分聚合，并非五类简单平均；MME-P仅感知；MMBench-EN用circular指标；本地MMStar使用任务实现的宏平均。Exact对应选定剪枝策略参考，不是原生Show-o2；Stable改变策略及输出。完整组合系统包含FlashU参考的剪枝/缓存/采样策略，新依赖重写的独立贡献由固定策略消融隔离，不能把全部累计加速归因于新重写。

T1使用553个改写prompt×4seed、432×432；本地计时每seed排除首张warmup。T2为1065prompt、seed42；计时每run1064张，native一轮、system三轮。T3两个政策参考跑次不同，Stable的MMBench/MMStar为fresh cap128，其余任务cap16。显示下降值全部保留。

本次校验保留记录、数值显示、原论文来源和算术；未重新运行GPU推理或图像评分器。
