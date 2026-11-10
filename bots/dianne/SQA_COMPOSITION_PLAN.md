# SQA 最新模块组合实验

源版本：`c62cdaccb66cec9f0b49adcaa43327b88d2419b8`。
本计划在两个新候选开始跑分前固定。只做以下两个组合，不按中途成绩调参。

| 新候选 | 翻牌前 | 翻牌后、拍卖、记忆 |
|---|---|---|
| `sqa_prior_classic_v1` | `practice_prior_full_v1` | `classic_hybrid_v1` 原样保留 |
| `sqa_prior_fragile_v1` | `practice_prior_full_v1` | `fragile_both_v1` 原样保留 |

每个目标仅覆盖 `independent_preflop.py` 并新增 `distilled_prior.py`，
通过逐文件 SHA256 验证其余代码保持不变。构造脚本为
`tools/build_sqa_compositions.py`。所有输入只使用当前可见状态和 SDK 桌内记忆。

统一使用冻结 `auction-flop-v1`，每个候选完整 4,096 桌、512 个种子块。
不改变对手、规则、种子或分数定义。Mac 上只使用已记录的 CPU 亲和性适配器，
明确标记平台；评分不是 Linux SDK 时限资格。

主要比较两个新候选相对已发布 `practice_prior_full_v1` 的变化；另比较
各自目标原版，以判断替换翻牌前阶段的效果。四个比较统一给出普通配对
95% 区间和 Bonferroni 同时区间。所有结果均保存，零失败且完整审计通过
才报告分数。固定公开种子的反复开发存在过拟合风险，不能据此声称对实战
所有对手都更强。

验证真实引擎轨迹中的动作合法性、SDK 记忆、非翻牌前阶段与目标原版一致、
翻牌前与 prior 原版一致，以及 prior 不适用时的行为保持。
比赛上传仍由 SQA 操作；本分支只推代码、测试和评测结果。
