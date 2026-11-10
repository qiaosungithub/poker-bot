# SQA 提交参考

**统一评分结论：保留 main 的 `preflop_loose`，不要用当前 V7 替换它。**
SQA `auction-flop-v1` 原版 Linux 全量评测：V7 12.84668 分，4,096 桌零失败；
main 已发布的 `preflop_loose` 为 16.56250 分。Linux 完整审计和配对比较通过，
分数与 Mac 复现一致；原始证据见 `results/main-benchmark-defend-v7-linux`，
完整说明见 `SQA_BENCHMARK.md`。

候选文件：`bots/dianne/icm_defend_v7.zip`，包内只有 `agent.py`。
`bots/dianne/agent.zip` 与它完全相同。单文件源码在 `icm_defend_v7/agent.py`
和 `selected/agent.py`，两者 SHA256 均为
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。

此次工作只发布仓库分支和本地包，比赛提交由 SQA 操作。

最终组合保留 ICM 生存逻辑，修正便宜跟注、公共牌面牌力和死牌判断，
在默认翻牌已经有利但可能被他人改选时增加有限防守出价。使用公开余额，
不假设能看见对手逐牌出价。

统一排名使用 sqa 的 `auction-flop-v1`；V7 的原版 Linux 全量评测已完成。
此前本地对照结果保存在 `OPTIMIZATION.md`，不用于跨分支排名。
Linux 官方 SDK 零时间储备测试：64 桌、9,017 次 V7 决策，零失败。

三个阶段可以单独替换，最终版本按以下命令重建：

```sh
python bots/dianne/tools/build_modular.py \
  --common bots/dianne/modular_v4/common_texture_v7.py \
  --preflop bots/dianne/modular_v4/preflop.py \
  --auction bots/dianne/modular_v4/auction_defend_v6.py \
  --postflop bots/dianne/modular_v4/postflop.py \
  --dispatcher bots/dianne/modular_v4/dispatcher.py \
  --output .runtime/rebuilt-defend-v7/agent.py
poker-agent check .runtime/rebuilt-defend-v7/agent.py
```

详细比较见 `OPTIMIZATION.md`，原始结果及配置在 `results/`。
