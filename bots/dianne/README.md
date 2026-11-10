# Dianne — Boston 比赛策略

2026-11-09 开赛后实现。分支 `dianne/boston-strategy`。

## 使用入口

**当前提交建议：保留 main 的 `preflop_loose`。** SQA 原版 Linux 全量评测中，
本分支 `icm_defend_v7` 为 12.84668 分，低于 `preflop_loose` 的 16.56250 分；
不建议用 V7 替换该版本。4,096 桌零失败，完整审计通过。

[`selected/agent.py`](selected/agent.py) 是此前本地选型留下的 V7 候选别名，
不代表统一 benchmark 的优胜版本。
它组合了价格敏感的翻牌前跟注、修正后的牌面计算，以及保护有利默认翻牌的拍卖。
**跨分支排名统一使用 sqa 的 `auction-flop-v1` 分数**，进度和结果见
[SQA_BENCHMARK.md](SQA_BENCHMARK.md)。此前本地固定对手实验作为诊断和回归证据保存。
具体模块、候选取舍和验证见 [OPTIMIZATION.md](OPTIMIZATION.md)。
三阶段拆分与固定下注的拍卖对照见 [PHASE_ANALYSIS.md](PHASE_ANALYSIS.md)。
三个阶段的独立源码及单文件生成方法见 [modular/README.md](modular/README.md)，
114,785 个官方引擎状态中与原 ICM v2 行为完全一致。
保留的候选包为 [`icm_defend_v7.zip`](icm_defend_v7.zip)，与 [`agent.zip`](agent.zip)
内容一致，包内只有 `agent.py`，大小 5,914 字节。
策略运行依赖标准库、官方 SDK 和 `eval7==0.1.11`。

Python 文件 SHA256 为
`30ba58c895fd2857b6f0b927f746285e5bc06e764a6924a64b52d60c5f729b04`。
ZIP SHA256 为 `5183c0fc774da2b7e8f5a98bed66a7f24b75cb14d7ecce534e7ede08a8fb26bf`。

在安装 Boston starter kit 的 Python 3.12 环境中，从仓库根目录运行

```sh
poker-agent check bots/dianne/agent.zip
poker-agent play bots/dianne/selected \
  /path/to/starter-kit/examples/check_or_fold \
  /path/to/starter-kit/examples/min_raiser --tables 200 --no-history
python -m unittest discover -s bots/dianne/tools -p 'test_*.py'
```

这台笔记本的完整环境位于相邻目录 `../starter-kit/.venv`，可以先运行
`source ../starter-kit/.venv/bin/activate`。测试时另建了只包含 SDK、引擎和
eval7 的 `.runtime/envs/boston-test`，用于等待完整依赖下载时并行验证。

## 八个方向

| 目录 | 方法 |
|---|---|
| `value_heuristic` | 牌力、牌面结构、位置和有限拍卖预算 |
| `equity_sampling` | 小样本多玩家胜率，排除拍卖淘汰的两张死牌 |
| `auction_optimizer` | 枚举十种翻牌组合，按关键牌边际贡献分配出价 |
| `icm_survival` | 读取实际 payouts，考虑泡沫、筹码覆盖和大额跟注的 ICM |
| `opponent_adaptive` | 172 字节桌内记忆，跟踪激进程度、弃牌和公开拍卖支出 |
| `push_fold` | 短筹码推／弃，深筹码小额开池，低 SPR 价值全押 |
| `auction_shadow_price` | 将翻牌改善与剩余拍卖预算的未来价值比较 |
| `range_pressure` | 范围加权胜率、位置压力和确定性混合下注 |

八个方向由八个子代理分批完成，共用当天新写的部分牌力计算逻辑。
原始候选保持冻结，以便根据结果中的源文件哈希复现比较。

## 原版筛选记录

本节的 `selected` 指此前冻结的 ICM v2，源文件保存在 `icm_v2/agent.py`。

`icm_survival` 在原始八方案的 1,024 桌独立确认中平均 16.816 分，
`push_fold` 为 13.721 分。按 128 个完整座位轮换块计算，配对差为
3.096 分，正态近似 95% 区间为 [1.572, 4.619]。

交叉检查后新增三个修正版。`icm_v2` 修正短筹码面对超额全押时可争夺的
底池和 ICM 退款计算，并细分四同花牌面的低同花、公共三条牌面的低葫芦。
修正版在新的 1,024 桌场次中均分 13.926，原 ICM 为 13.027，配对差
0.898，95% 区间 [-0.814, 2.610]。这项差异仍有较大不确定性，选择修正版
的依据还包括已验证的赔率计算修复。

选中版本随后完成另一组 4,096 桌验证，512 个完整座位轮换块。
以下均为每桌名次积分，八人均分参照为 12.5 分。

| 策略 | 平均分 | block 标准误 |
|---|---:|---:|
| `selected` / `icm_v2` | 14.580 | 0.284 |
| `icm_survival` | 13.772 | 0.273 |
| `push_fold` | 12.827 | 0.246 |
| `push_fold_v2` | 12.798 | 0.237 |
| `opponent_adaptive` | 12.664 | 0.307 |
| `value_heuristic` | 11.304 | 0.279 |
| `shadow_price_v2` | 11.074 | 0.278 |
| `auction_shadow_price` | 10.981 | 0.269 |

选中版本与原 ICM 的配对差为 +0.808 分，普通正态近似 95% 区间
[0.002, 1.614]，与第二名差异仍接近误差边缘。结果与未经多重比较校正的
配对区间见 [`final-validation-4096`](results/final-validation-4096/)。
每次比较都有不同的对手阵容，分数应在对应场次内解读。

## 运行验证

最终 V7 在 Linux 官方 SDK 进程路径完成 64 桌、9,017 次自己的决策，
零漏答／失败。整组 66,342 次决策均零失败，设置为零时间储备，SDK 自带
2 ms 传输宽限。证据见 `results/linux-sdk-v4`。单文件 ZIP 静态检查通过。
以下是早期 ICM v2 和原始候选的验证记录。

- 全部原始策略通过静态上传检查和随机拍卖合法性检查。
- 256 桌初筛、1,024 桌确认和 1,024 桌修正版比较，共 2,132,205 次直接决策，
  非法动作与异常均为零。直接模式记录耗时，评分时不强制超时。
- 最后 4,096 桌验证再完成 3,989,250 次直接决策，非法动作与异常均为零。
- 团队 Linux 机器先完成原始八方案 32 桌，再完成修正版 64 桌官方 SDK
  进程对局，全部零失败。后一轮选中策略有 7,993 次成功决策。
- Linux 独立源码计时中，选中策略 7,078 次决策没有超过 1 ms。
  最慢阶段为拍卖，P99 0.130 ms，最大 0.149 ms，计时范围是 `Agent.act`。
- SDK 测试设 `time_bank_ms=0`。SDK 自带 2 ms 传输宽限，主决策额度为
  1 ms。源码计时与完整进程测试分别保存。
- 双人、20 手上限、零时间储备的仿验收测试完成 20 桌，选中策略
  200 次决策全部成功。
- 对五个最小加注 bot 和两个 check/fold bot 的 200 桌 SDK 压力测试中，
  选中策略平均 29.95 分，零失败。这组基础对手用于覆盖激进下注场景。
- 回归测试覆盖超额全押、多人当前街边池、正常跟注、同花大小、葫芦大小，
  以及终局 ICM 三种结果的筹码守恒。

## 结果与复现

`results/` 保存每次运行的配置、源码哈希、逐桌记录和汇总。
`linux-sdk-v1`、`linux-sdk-v2`、`linux-source-timing` 来自团队 build box。
`acceptance-like` 是 20 手上限、对 check/fold、零时间储备的双人 SDK 检查。

复现修正版比较可运行

```sh
python bots/dianne/tools/benchmark.py \
  bots/dianne/selected bots/dianne/push_fold_v2 \
  bots/dianne/shadow_price_v2 bots/dianne/icm_survival \
  bots/dianne/push_fold bots/dianne/auction_shadow_price \
  bots/dianne/value_heuristic bots/dianne/opponent_adaptive \
  --blocks 128 --workers 4 --seed my-new-boston-run \
  --out .runtime/boston/my-new-run
```

每个 block 使用完整循环座位轮换，误差估计以 block 为独立单位。
加 `--sdk --time-bank 0` 可改用官方进程路径。输出目录须为新目录。
引擎入口复用了仓库现有 `BayAreaEnv`，增加可选 variant 参数，默认值仍为
原 Bay Area 变体。本次明确传入 `flop_auction_holdem`。

## 思路来源

sqa 的规则、运行时、ICM 和 Bay Area 评测笔记提供了准备依据。
MIT 2024 Auction Hold'em 和 GTO Wizard 公开方案提供了拍卖机会价值的研究
思路。Boston 的五张公共牌竞价、只付入选牌和整桌有限预算按本次 kit
独立实现。参考资料见 [MIT 2024 课程仓库](https://github.com/mitpokerbots/class-resources-2024)
和 [GTO Wizard 2024 项目](https://github.com/gtowizard-ai/mitpoker-2024)。

第 30 分钟与 sqa 对齐时，可先比较双方提交入口、拍卖预算策略、同场均分、
失败决策数和完整运行路径的耗时，再选择下一轮需要合入的改动。
