# ICM 模块优化记录

本文保留本地开发和回归实验；跨分支排名统一使用 sqa 的
`auction-flop-v1`，见 [SQA_BENCHMARK.md](SQA_BENCHMARK.md)。

原版 `icm_v2` 和所有已测候选保持冻结。三阶段入口为
`preflop_decision(state, memory)`、`auction_decision(state, memory)`、
`postflop_decision(state, memory)`，统一返回 `(Action, bytes)`。
使用 `tools/build_modular.py` 将选定模块组合为可上传的单文件 `agent.py`。

## 主要修正

- 翻牌前对极便宜的跟注保留价格判断；已加注后的便宜续跟仍有 ICM 约束。
- 牌面计算区分公共四条的踢脚、最佳五张里的两对、公共对子上的底对，
  并排除无私人改善的公共顺子听牌和已知拍卖死牌。
- 已保证共享最大牌时保留跟注价值。
- 实验性单挑模块只拓宽小额大盲防守，保留全押和多人决策。
- 实验性拍卖保护模块只修改原本零出价的机会，保护可能被其他玩家改变的
  有利默认翻牌。其他正出价及原预算节奏保持一致。

## 固定对手的开发实验

每个版本分别对同一组七个对手；每个版本 1,024 桌，128 个完整座位轮换块。
所有版本的对手座位相同。完整原始记录在 `icm-controlled-development`
和 `icm-modular-development`，后者重用并核对了相同基准运行。

| 版本 | 每桌积分 | 相对 ICM v2 | 普通 95% 配对区间 |
|---|---:|---:|---|
| ICM v2 | 16.410 | 0 | — |
| Price v4 | 16.475 | +0.064 | [-0.026, 0.155] |
| Texture v4，拍卖和后续下注共同采用新牌面计算 | 17.061 | +0.650 | [-0.041, 1.342] |
| 仅后续下注采用 Texture v4 | 16.615 | +0.205 | [-0.294, 0.704] |
| 温和提高拍卖花费 Spend v3 | 15.971 | -0.439 | [-0.862, -0.017] |
| 更短机会预算 Spend v4 | 16.420 | +0.010 | [-0.699, 0.719] |
| Price + Spend v3 + 后续 Texture | 16.631 | +0.221 | [-0.379, 0.820] |

牌面计算同时服务拍卖和后续下注，替换共享函数会同时改变两个阶段。
`modular_v4/common.py` 特意分开 `hand_quality` 与 `postflop_hand_quality`，
使单阶段比较可以保持其他阶段动作不变。`common_texture.py` 则显式让
两个阶段都采用新计算。组合策略仍以整桌名次积分评价。

## 独立确认

`icm-practice-informed-confirmation` 使用新种子，每个版本 2,048 桌。
核心组合 `icm_texture_price_v5` 为便宜跟注修正 + 新牌面计算，保留原预算。
它平均 16.655 分，基准 15.981 分，配对差 +0.674，普通 95% 区间
[0.179, 1.169]。

同组默认翻牌保护版 `icm_defend_v6` 平均 18.345 分，相对基准提升
2.363 分，区间 [1.629, 3.098]。单挑防守版仅比核心组合多约 0.020 分，
没有明显额外收益，最终组合保留原单挑模块。

V7 随后补充了一个窄范围回归修复：公共四条且踢脚完全来自公共牌时，
正确计算只能平分或输的情况，避免被通用“使用公共牌”评分覆盖。
精确枚举回归和不受影响状态的一致性检查通过。

最终使用另一组全新种子，每个版本 4,096 桌、512 个完整轮换块：

| 版本 | 每桌积分 | 相对原版 | 普通 95% 配对区间 |
|---|---:|---:|---|
| ICM v2 | 15.728 | 0 | — |
| 核心组合 V5 | 16.504 | +0.776 | [0.423, 1.130] |
| 防守拍卖 V7 | 18.088 | +2.361 | [1.828, 2.894] |

最终选择 `icm_defend_v7`。源码 SHA256 为
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。
原始逐桌记录、源码哈希、配置分别保存在 `icm-final-holdout` 和
`icm-defense-final-holdout`，共同基准逐桌记录完全相同。
V7 的 565,178 次直接决策零失败；Linux 官方进程测试另外完成 64 桌、
9,017 次 V7 决策，零时间储备下零失败。

这些区间以整个轮换块为独立单位计算，没有多重比较校正；不同对手阵容
的绝对分数不能直接比较。直接引擎调用用于强度评测，耗时验证走独立
Linux 官方 SDK 进程路径。策略通过公开信息决策，未读取隐藏对手出价。

## 重建组合

例如核心组合可以在仓库根目录重新生成：

```sh
python bots/dianne/tools/build_modular.py \
  --common bots/dianne/modular_v4/common_texture.py \
  --preflop bots/dianne/modular_v4/preflop.py \
  --auction bots/dianne/modular_v4/auction_texture.py \
  --postflop bots/dianne/modular_v4/postflop.py \
  --output .runtime/rebuilt-v5/agent.py
```

单挑版将 `--preflop` 改为 `preflop_hu_v6.py`；默认翻牌保护版将
`--auction` 改为 `auction_defend_v6.py`。原始模块在 `modular/`，不会随
当前推荐候选更换而改变。

重建最终 V7 时，另将 `--common` 改为 `common_texture_v7.py`，并选择
`auction_defend_v6.py`。最终组合的完整命令见 [HANDOFF.md](HANDOFF.md)。
