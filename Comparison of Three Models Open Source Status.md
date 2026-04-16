# TabPFN vs TabICL vs TARTE开源状态对比

## 结论

| | 推理代码 | 预训练权重 | Prior（生成训练数据） | 训练代码 |
|---|---|---|---|---|
| **TabPFN (v1/v2/v2.6)** | ✅ 开源 | ✅ 可下载 (42M ckpt) | ❌ 不开源 | ❌ 不开源 |
| **TARTE (NeurIPS 2024)** | ✅ 开源 | ✅ 可下载 | ✅ 开源 | ✅ 开源 |
| **TabICL (INRIA, ICML 2025)** | ✅ 开源 | ✅ 可下载 | ✅ 开源 | ✅ 开源 |
| **nanotabicl** | ✅ 开源 (170行极简版) | — | ✅ 开源 | ✅ 开源 |

> **nanotabicl 是啥呢？**根据网上查到的, TabICL 完整代码有几千行，读起来费劲。nanotabicl 是 INRIA 团队自己写的极简版，把核心逻辑压缩到只有 170 行 Python，去掉了所有工程细节（日志、分布式训练、配置文件等），只保留最核心的：怎么生成合成数据（prior）、怎么训练、怎么推理。想理解 TabICL 怎么工作，看这 170 行就够了，后面改 prior 也是在这个基础上改最方便。

## 对我们项目的可能影响

Knowledge-Guided TabPFN 的核心任务是改 prior（把医学知识注入到生成训练数据的过程中）。但 TabPFN 的 prior 代码不开源，没法改。

所以策略是：**用 TabICL 或 TARTE 的开源代码作为基础来改**。两者都完全开源，prior 和训练代码都能拿到。TabICL 还有 nanotabicl（170行极简版），适合快速理解和修改。


## 相关链接

- TabPFN: https://github.com/PriorLabs/TabPFN
- TARTE: https://github.com/SalesforceAIResearch/TARTE
- TabICL: https://github.com/soda-inria/tabicl
- nanotabicl: TabICL 仓库内的170行极简实现
