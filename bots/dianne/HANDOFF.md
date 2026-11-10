# SQA 提交参考

推荐文件：`bots/dianne/icm_defend_v7.zip`，包内只有 `agent.py`。
`bots/dianne/agent.zip` 与它完全相同。单文件源码在 `icm_defend_v7/agent.py`
和 `selected/agent.py`，两者 SHA256 均为
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。

此次工作只发布仓库分支和本地包，比赛提交由 SQA 操作。

最终组合保留 ICM 生存逻辑，修正便宜跟注、公共牌面牌力和死牌判断，
在默认翻牌已经有利但可能被他人改选时增加有限防守出价。使用公开余额，
不假设能看见对手逐牌出价。

独立 4,096 桌、相同固定对手和完整座位轮换：V7 18.088 分，ICM v2
15.728 分，配对差 +2.361，普通 95% 区间 [1.828, 2.894]。这些分数仅
适用于该对手阵容，不能与其他 benchmark 的绝对分数直接比较。
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
