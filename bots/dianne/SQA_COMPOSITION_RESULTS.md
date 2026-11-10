# 基于 SQA 最新模块继续优化

推荐交给 SQA 复核的候选：**`sqa_prior_classic_v1`**，统一 benchmark 完整
Mac 复现为 **19.43359** 分。包为 `sqa_prior_classic_v1.zip`，已通过 Linux
SDK 运行检查。本次仅推送分支，未上传比赛网站。

源码取自 main `c62cdac`；完成前又同步到 `c02b6ba`，审计了其新公布的
`hybrid_both` 结果。旧的 16.56250 分 `preflop_loose` 已不再是最新参照。

## 同一 auction-flop-v1 结果

| 候选 | 每桌积分 | 95% 区间 | 平台与来源 |
|---|---:|---|---|
| 新 prior + classic (`sqa_prior_classic_v1`) | **19.43359** | [18.78336, 20.08383] | 本分支完整 Mac 复现 |
| 新 prior + fragile (`sqa_prior_fragile_v1`) | 17.90283 | [17.21333, 18.59234] | 本分支完整 Mac 复现 |
| main `hybrid_both` | 17.73438 | [17.11284, 18.35591] | c02b6ba 已发布 Linux 结果 |
| main `practice_prior_full_v1` | 17.65625 | 见原始结果 | c62cdac 已发布 Linux 结果 |
| main `classic_hybrid_v1` | 17.48047 | 见原始结果 | c62cdac 已发布 Linux 结果 |

每个版本均为四个固定对手面板、128 个种子、八座位轮换，共 4,096 桌。
两个新候选分别有 648,853 和 646,011 次决策，全部零失败。完整计划、
每桌积分、冻结源码、原始记录哈希及实际候选文件均已审计。

两个新候选在 Mac 上使用 CPU 亲和性接口适配器；游戏、对手、规则、
种子和评分源码完全未修改。平台说明和适配器哈希随结果保存。
**尚未对这两个新候选做 Linux 4,096 桌复验**；下述 Linux SDK 检查
是独立的运行测试，不能冒充完整 Linux 强度评测。

## 配对比较

以下四项在开跑前固定，按完整种子块、面板分层计算；区间为四项
Bonferroni 同时 95% 正态近似区间。

| 比较 | 分差 | 同时区间 |
|---|---:|---|
| prior + classic − full prior | +1.77734 | [+0.78206, +2.77263] |
| prior + fragile − full prior | +0.24658 | [+0.09275, +0.40042] |
| prior + classic − classic 原版 | +1.95313 | [+1.12335, +2.78290] |
| prior + fragile − fragile 原版 | +1.09619 | [+0.25801, +1.93437] |

后来新增 main `hybrid_both` 的探索性比较：prior + classic 高 **1.69922**，
未校正配对 95% 区间为 **[1.04535, 2.35309]**。它不属于上述预声明四项。
这些结果只说明冻结 v1 场景下的表现；公开固定种子的多轮开发可能过拟合，
尚未运行新的隐藏种子测试，不能断言对实战全部对手都占优。

## 组合内容与验证

两个组合各自只覆盖 `independent_preflop.py`、新增 `distilled_prior.py`。
拍卖、翻牌后和 memory 管理保留目标原版；classic 的对手画像也保留。
prior 来自 Practice 4 公开行动频率，不使用隐藏对手逐牌出价。

- 18 条真实引擎轨迹，4,373 个状态，覆盖 2/3/8 人、5/15/100BB、全部阶段。
- 每个组合 2,147 个非翻牌前状态的动作与 memory 和原版完全相同；
  2,226 个翻牌前动作与 full prior 相同，1,765 个不适用状态保持原版。
- Linux 官方 SDK：64 桌，两个新候选共 19,929 次决策，零失败。
  使用 1 ms 决策额、零时间储备、SDK 自带 2 ms 传输宽限。
- 两个 ZIP 通过官方静态检查，包内模块与评测、Linux 验证源码逐文件匹配。

## 复现与文件

构造计划见 `SQA_COMPOSITION_PLAN.md`，所有模块哈希见
`sqa_compositions_manifest.json`。结果与配对区间位于
`results/sqa-compositions-v1/`；两个完整原始运行分别位于
`results/sqa-prior-classic-v1-macos/` 和 `results/sqa-prior-fragile-v1-macos/`。
`tools/build_sqa_compositions.py`、`tools/validate_sqa_composition.py` 和
`tools/analyze_sqa_compositions.py` 提供构造、行为验证和独立审计入口。

推荐候选 ZIP SHA256：
`5b329f05fe6b59e7ffaa1c48f9c3dcb2829b7454d7af05028891e92baed742d3`。
旧 `selected/agent.py` 和 `agent.zip` 保留为 V7 历史产物，不是本轮推荐包。
