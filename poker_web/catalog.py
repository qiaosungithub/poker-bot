"""Explicit policy allowlist; browser input never selects an arbitrary file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASSIC = [
    ("txnujk_32", "txnujk · 32", "手写 + 胜率模拟", "根据牌力、位置和对手风格决策，最多模拟 32 次。"),
    ("fullhouse_reproduced", "FullHouse · 复现版", "学习策略", "我们复现的第 210 次迭代，使用真实模型权重。"),
    ("fullhouse_released", "FullHouse · 发布版", "学习策略", "原作者发布的模型，使用真实模型权重。"),
    ("tournament_specialist", "Tournament · 原版", "手写规则", "保留 benchmark 原版；有已知的三条、四条分类缺陷。"),
    ("solver_like", "Solver-like", "手写规则", "固定范围、牌型分类和混合行动；名称不代表实时求解。"),
    ("cfr_lite_32", "CFR-Lite · 32", "手写 + 胜率模拟", "按手写收益公式选择行动，没有加载训练策略。"),
    ("hydra_32", "Hydra · 32", "手写 + 对手统计", "根据对手倾向和抽样胜率调整策略。"),
    ("station", "Calling Station", "宽松跟注", "适合练习价值下注，倾向于跟注。"),
    ("maniac", "Maniac", "激进进攻", "喜欢加注，适合练习面对压力时的决策。"),
]
BAY = [
    ("nc_codex_nc_v7_adapted", "NC · v7 adapted", "规则 + 对手统计", "三牌起手评估、弃牌亮牌选择和对手倾向调整。"),
    ("den_v7", "Den · v7", "模拟 + 局部策略表", "结合对手明牌与行动估计牌力，部分短码局面查表。"),
    ("james_tree", "James · Tree", "学习树 + 模拟", "翻牌前使用训练后的决策树，其余阶段使用基础策略。"),
    ("nc_gto_final_reference", "NC · Final reference", "手写策略", "参赛项目中的最终参考版本。"),
    ("nc_codex_gto_refined", "NC · Refined", "手写策略", "参赛项目中的规则改进版本。"),
    ("nc_gto_supplied", "NC · Supplied", "手写策略", "参赛项目中的基准版本。"),
    ("james_jamtop3", "James · Jam Top 3", "全押倾向", "偏向强牌全押的比赛策略。"),
    ("james_jam_tuned", "James · Jam Tuned", "全押倾向", "调整过阈值的全押策略。"),
]
MODES = {
    "classic": {"title": "经典德扑", "subtitle": "两张底牌，五张公共牌", "max_players": 6,
                "starting_stack": 10000, "blinds": [50, 100], "hand_cap": 200,
                "payouts": [100, 60, 40, 25, 10, 0],
                "rules": ["标准 52 张牌，每人两张底牌。", "用底牌与公共牌中的最佳五张比大小。", "初始 10,000 筹码，盲注固定为 50 / 100。", "最多 200 手；淘汰按名次计分，剩余玩家按 ICM 分配积分。"],
                "defaults": ["txnujk_32", "fullhouse_reproduced", "solver_like", "tournament_specialist", "cfr_lite_32"]},
    "bay_area": {"title": "Bay Area 改版", "subtitle": "三张底牌 · 翻牌后弃一亮一", "max_players": 8,
                 "starting_stack": 2000, "blinds": [10, 20], "hand_cap": 250,
                 "payouts": [50, 30, 20, 0, 0, 0, 0, 0],
                 "rules": ["已复现的 Bay Area 比赛规则，非尚未确认的其他场次规则。", "每人三张底牌；翻牌后，弃掉一张，再亮出保留的另一张。", "所有玩家锁定选择后才同时亮牌；全押者也要选牌。", "弃牌保持私密且不再进入牌堆；摊牌时只使用两张保留牌。", "初始 2,000 筹码，盲注 10 / 20 起，每 20 手升级，最多 250 手。"],
                 "defaults": ["nc_codex_nc_v7_adapted", "den_v7", "james_tree", "nc_gto_final_reference", "nc_codex_gto_refined", "james_jamtop3", "nc_gto_supplied"]},
}


def catalog():
    result = {}
    for mode, config in MODES.items():
        bots = []
        for key, label, kind, description in (CLASSIC if mode == "classic" else BAY):
            available = True
            if key.startswith("fullhouse_"):
                model = "latency-survey/fullhouse_fp32.onnx" if key.endswith("released") else "fullhouse-repro/eval/iter_00210.onnx"
                available = (ROOT / model).is_file()
            bots.append(dict(id=key, name=label, kind=kind, description=description, available=available))
        result[mode] = dict(config, policies=bots)
    return result


def validate_config(data):
    mode = data.get("mode")
    if mode not in MODES:
        raise ValueError("请选择有效的游戏模式。")
    lineup = data.get("bots")
    if not isinstance(lineup, list) or not 1 <= len(lineup) < MODES[mode]["max_players"]:
        raise ValueError("玩家人数不在支持范围内。")
    allowed = {b["id"] for b in catalog()[mode]["policies"] if b["available"]}
    if any(not isinstance(b, str) or b not in allowed for b in lineup):
        raise ValueError("所选策略不可用或不支持该模式。")
    name = data.get("name", "你")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 20:
        raise ValueError("昵称需要 1–20 个字符。")
    return dict(mode=mode, bots=list(lineup), name=name.strip())


def labels(config):
    names = {b[0]: b[1] for b in CLASSIC + BAY}
    return [config["name"]] + [names[b] for b in config["bots"]]
