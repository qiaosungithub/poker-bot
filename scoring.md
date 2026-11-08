# 计分与 ICM

2026-11-08 登录后Format页已给出默认积分 $p_1,\ldots,p_8=[50,30,20,0,0,0,0,0]$。实际可能按轮次或牌桌改变，应读取 `state.config.payouts`；starter kit 会提供规则和 ICM 计算器。

已淘汰的 bot 获得其名次对应的积分。达到手数上限后，存活者用 ICM（Independent Chip Model，独立筹码模型）分配剩余积分。公开规则写200手，Format页默认250手；实际读取 `state.config.hand_cap`。

1. 获得第一名的概率 = 自己的筹码 / 存活者总筹码。
2. 假定某人获得第一名，将其及其筹码移除；剩余玩家按筹码比例计算第二名的概率，依次递推。
3. 汇总所有可能排名，得分为 $\sum_{k=1}^{m} P(\text{第 }k\text{名})\,p_k$，其中 $m$ 为存活人数。

依据：[Format](https://pokerbot.sequoiacap.com/tournament/competitions/abb5539f-60ce-48b6-8925-c0c56a3102e3/format)；[官方规则 §3.3、§4.5](https://pokerbot.sequoiacap.com/rules.html)；[ICM 概率模型](https://arxiv.org/html/2506.00180v2#S2)。规则差异见 [rules.md](docs/competition/rules.md)。
