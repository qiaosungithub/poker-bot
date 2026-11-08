规则：

2026-11-08 更新：详细规则及官方资料间的冲突见 [docs/competition/rules.md](docs/competition/rules.md)，环境用法见 [docs/competition/infrastructure.md](docs/competition/infrastructure.md)。

前60分钟开发阶段。之后6轮，每隔20分钟比一次，期间可以继续修改和提交bot。积分 $T1+2T2+3T3+4T4+5T5+6T6$，T是该轮的平均每桌积分。

比赛为和其他选手的bot进行。每桌按名次计分，达到手数上限仍有多人存活时，用ICM分配剩余积分，见 [scoring.md](scoring.md)。公开规则写200手，登录后Format页默认250手；实际读取 `state.config.hand_cap`。

bot速度限制：默认单步1毫秒，正式对局默认另有每桌100毫秒时间储备；上传验收没有时间储备。详见 [runtime.md](docs/competition/runtime.md)。
