# 统一评分：SQA auction-flop-v1

跨分支排名、选型和对外报告统一使用 `auction-flop-v1`。
此前本目录自有对手阵容的结果只作为开发和回归证据，不与这里的分数比较。

冻结 benchmark 来自 main 的 `481d1f0`，验证 SHA256：
`dc6c9f772eabdd0db0b31d011e548d40b8119e872773d58ecd9c077eaaa5eba6`。
main 后续 `c5ec82a` 公布的 `preflop_loose` 成绩为 16.56250；冻结
`hybrid_tournament_auction` 参考为 15.08789。

V7 候选源文件 SHA256：
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。

四个固定对手面板，每个 128 个种子、8 次座位轮换，共 4,096 桌。
不更改冻结策略、规则、种子和源码。评分为平均每桌名次积分；统计按
面板分层和完整种子块计算。直接调用测强度，完整 SDK 时限另行验证。

当前原版 benchmark 正在团队 Linux 构建机的独立目录运行：
`~/work/dianne-main-benchmark/result-defend-v7`，日志为同目录上级的
`benchmark.log`。完成后需要 `audit` 和 `compare`。

Mac 复现仅为加速参考，适配缺失的 CPU 亲和性接口；它不能冒充 Linux
运行验证。若保留该结果，必须附上适配说明，并与 Linux 重叠记录核对。
