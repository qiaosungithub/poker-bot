# 统一评分：SQA auction-flop-v1

跨分支排名、选型和对外报告统一使用 `auction-flop-v1`。
此前本目录自有对手阵容的结果只作为开发和回归证据，不与这里的分数比较。

冻结 benchmark 来自 main 的 `481d1f0`，验证 SHA256：
`dc6c9f772eabdd0db0b31d011e548d40b8119e872773d58ecd9c077eaaa5eba6`。
main 后续 `c5ec82a` 公布的 `preflop_loose` 成绩为 16.56250；冻结
`hybrid_tournament_auction` 参考为 15.08789。

V7 候选源文件 SHA256：
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。

## 同标结果

| 策略 | 平均分 | 95% 区间 | 证据 |
|---|---:|---|---|
| main `preflop_loose` | 16.56250 | [15.97532, 17.14968] | main c5ec82a 已发布 Linux 结果 |
| 冻结 hybrid 参考 | 15.08789 | 见 main 保留结果 | main v1 校准 |
| 本分支 V7 | 12.84668 | [12.28263, 13.41073] | 完整 4,096 桌 Mac 同标复现，Linux 复核进行中 |

V7 共 587,991 次自己的决策，零失败，完整轮换计划、源码锁和积分审计通过。
目前结论是保留 main 的 `preflop_loose`；V7 的局部修正仍可作为后续组合
实验的素材，但不能依据另一套对手的分数替换统一 benchmark 的领先版本。

leader 原始来源为
[candidate-preflop_loose/summary.json](https://github.com/qiaosungithub/poker-bot/blob/c5ec82ae4b1ba0abc35395cfc2127996b25be69b/auction_flop/evidence/candidate-preflop_loose/summary.json)。
V7 的复现记录、适配器代码及哈希在 `results/main-benchmark-defend-v7-macos`。

四个固定对手面板，每个 128 个种子、8 次座位轮换，共 4,096 桌。
不更改冻结策略、规则、种子和源码。评分为平均每桌名次积分；统计按
面板分层和完整种子块计算。直接调用测强度，完整 SDK 时限另行验证。

当前原版 benchmark 正在团队 Linux 构建机的独立目录运行：
`~/work/dianne-main-benchmark/result-defend-v7`，日志为同目录上级的
`benchmark.log`。完成后需要 `audit` 和 `compare`。

Mac 复现只适配缺失的 CPU 亲和性接口；冻结策略、引擎、规则、种子和
评分源码完全未修改，PYTHONHASHSEED 固定为 0。它不能冒充 Linux 运行。
已完成的 2,120 桌 Linux 记录与 Mac 逐条完全一致，仅排除耗时字段。
两边的结果都保留明确平台说明；Linux 全量结束后再替换当前复核状态。
