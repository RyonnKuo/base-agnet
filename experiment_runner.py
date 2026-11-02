"""Reusable experiment runner for the authority experiments.

The runner keeps model/scenario changes in one place and emits progress messages
so long-running Ollama experiments can be monitored from a notebook or terminal.
"""
from __future__ import annotations

import csv
import itertools
import json
import random
import requests
import time
from pathlib import Path
from typing import Any

from langchain_community.chat_models import ChatOllama

from persona_v2 import generate_persona, upgrade_to_authority

TOPIC = "關於數位身分證(eID)推行的隱私與便利性爭議"
AGENTS = 6
ROUNDS = 5
REPETITIONS = 20
MAX_HISTORY_MESSAGES = 6
MAX_PUBLIC_POST_CHARS = 120
STANCE_VALUES = {-2, -1, 0, 1, 2}
STANCE_SCORE = {"Support": 1, "Neutral": 0, "Oppose": -1}
SMALL_MODELS = ["llama3:8b", "qwen3:8b", "deepseek-r1:8b"]
FAMILY_MODELS = ["qwen3:8b", "qwen3:14b", "qwen3:30b"]
SINGLE_MODELS = list(dict.fromkeys(SMALL_MODELS + FAMILY_MODELS))
MODEL_FAMILY = {
    "llama3:8b": "small-heterogeneous",
    "qwen3:8b": "qwen3-family",
    "deepseek-r1:8b": "small-heterogeneous",
    "qwen3:14b": "qwen3-family",
    "qwen3:30b": "qwen3-family",
}

SCENARIOS = {
    "balanced": ["Support"] * 2 + ["Neutral"] * 2 + ["Oppose"] * 2,
    "support_majority": ["Support"] * 4 + ["Oppose"] * 2,
    "oppose_majority": ["Support"] * 2 + ["Oppose"] * 4,
}

SINGLE_SMALL_CASES = [
    ("1", "balanced", None, 3, 1),
    ("2", "support_majority", None, 3, 2),
    ("3", "oppose_majority", None, 3, 3),
    ("4-1", "balanced", "Support", 5, 1),
    ("4-2", "balanced", "Neutral", 5, 1),
    ("4-3", "balanced", "Oppose", 5, 1),
    ("5-1", "support_majority", "Support", 5, 2),
    ("6-1", "oppose_majority", "Oppose", 5, 3),
    ("7", "support_majority", "Oppose", 2, 2),
    ("8", "oppose_majority", "Support", 2, 3),
]

SINGLE_SMALL_GROUPS = {
    "single-small-control": {"1", "2", "3"},
    "single-small-authority": {"4-1", "4-2", "4-3", "5-1", "6-1"},
    "single-small-crowd": {"7", "8"},
}


def progress(message: str) -> None:
    print(f"[experiment] {message}", flush=True)


def unload_ollama_model(model_name: str) -> None:
    """Ask Ollama to release a model after the complete experiment finishes."""
    payload = {"model": model_name, "keep_alive": 0}
    try:
        requests.post("http://localhost:11434/api/generate", json=payload, timeout=5)
        requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model_name, "messages": [], "keep_alive": 0},
            timeout=5,
        )
        progress(f"unloaded model={model_name}")
    except Exception as exc:
        progress(f"unload failed model={model_name}: {type(exc).__name__}: {exc}")


def make_prompt(
    persona: Any,
    history: list[dict[str, Any]],
    authority_name: str | None,
    score: int,
    memory_context: str = "",
) -> str:
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    public_history = "\n".join(
        f"Round {item['round']} | {item['agent']}: {item['public_post']}"
        for item in recent_history
    ) or "（這是第一回合，尚無前回合留言。）"
    authority_note = (
        f"本場沒有權威角色，所有代理人的社會地位相同。"
        if authority_name is None
        else f"本場的領域權威是 {authority_name}；請評估其論點，但仍須根據證據自行判斷。"
    )
    if memory_context:
        public_history += f"\n\n代理人可用的長期記憶摘要：\n{memory_context}"
    return f"""{persona.to_prompt(enable_authority=authority_name is not None)}
你正在參與 BBS 社會議題討論。
議題：{TOPIC}
{authority_note}
目前立場分數：{score}（-2 強烈反對，-1 反對，0 中立，+1 支持，+2 強烈支持）
請只輸出 JSON，不要輸出 Markdown：
{{"stance_score": -2, "rationale": "簡短的立場判斷摘要", "public_post": "給其他代理人看的公開留言"}}
    公開留言應自然回應目前可見的留言，但不要聲稱自己讀到不存在的資訊。
    公開留言請控制在 {MAX_PUBLIC_POST_CHARS} 字以內。
    目前可見的完整留言：
{public_history}
"""


def invoke_agent(
    llm: ChatOllama,
    persona: Any,
    history: list[dict[str, Any]],
    authority_name: str | None,
    score: int,
) -> dict[str, Any]:
    try:
        raw = str(llm.invoke(make_prompt(persona, history, authority_name, score)).content).strip()
    except Exception as exc:
        return {
            "stance_score": score,
            "rationale": "model invocation fallback",
            "public_post": "",
            "parse_error": f"{type(exc).__name__}: {exc}",
        }
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("model response did not contain JSON")
        payload = json.loads(raw[start:end])
        new_score = int(payload["stance_score"])
        if new_score not in STANCE_VALUES:
            raise ValueError("stance_score must be between -2 and 2")
        return {
            "stance_score": new_score,
            "rationale": str(payload.get("rationale", "")),
            "public_post": str(payload.get("public_post", ""))[:MAX_PUBLIC_POST_CHARS],
            "parse_error": "",
        }
    except Exception as exc:
        return {
            "stance_score": score,
            "rationale": "parser fallback",
            "public_post": raw[:MAX_PUBLIC_POST_CHARS],
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def polarization(scores: list[int]) -> float:
    return sum(abs(score) for score in scores) / len(scores)


def fragmentation(scores: list[int]) -> float:
    support = sum(score > 0 for score in scores)
    oppose = sum(score < 0 for score in scores)
    return 0.0 if support + oppose == 0 else 1 - abs(support - oppose) / (support + oppose)


def metrics(records: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    by_round = [
        [int(item["stance_score"]) for item in records if item["round"] == round_id]
        for round_id in range(1, rounds + 1)
    ]
    p = [polarization(scores) for scores in by_round]
    f = [fragmentation(scores) for scores in by_round]
    aligned = 0
    opportunities = 0
    for index in range(rounds - 1):
        previous_round, next_round = by_round[index], by_round[index + 1]
        majority = 1 if sum(previous_round) > 0 else -1 if sum(previous_round) < 0 else 0
        if majority == 0:
            continue
        previous = {item["agent"]: int(item["stance_score"]) for item in records if item["round"] == index + 1}
        for item in (entry for entry in records if entry["round"] == index + 2):
            opportunities += 1
            if abs(int(item["stance_score"]) - 2 * majority) < abs(previous[item["agent"]] - 2 * majority):
                aligned += 1
    return {
        "CR": aligned / opportunities if opportunities else 0.0,
        "CR_opportunities": opportunities,
        "P": p,
        "delta_P": p[-1] - p[0],
        "F": f,
        "F5": f[-1],
    }


def _seed_personas(stances: list[str], seed: int) -> dict[str, Any]:
    state = random.getstate()
    random.seed(seed)
    try:
        return {
            f"Agent_{index + 1}": generate_persona(f"Agent_{index + 1}", stance)
            for index, stance in enumerate(stances)
        }
    finally:
        random.setstate(state)


def _authority_target(
    personas: dict[str, Any],
    target: str | None,
    authority_model: str | None,
    model_assignment: dict[str, str],
) -> str | None:
    candidates = [
        name for name, persona in personas.items()
        if (target is None or persona.initial_stance == target)
        and (authority_model is None or model_assignment[name] == authority_model)
    ]
    return candidates[0] if candidates else None


def run_condition(
    run_id: int,
    condition: str,
    scenario_id: str,
    seed: int,
    models: list[str],
    authority_stance: str | None,
    authority_model: str | None = None,
    rounds: int = ROUNDS,
    fixed_stances: list[str] | None = None,
    shuffle_models: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    stances = (fixed_stances[:] if fixed_stances is not None else SCENARIOS[scenario_id][:])
    if fixed_stances is None:
        rng.shuffle(stances)
    personas = _seed_personas(stances, seed)
    assigned_models = models[:]
    if shuffle_models:
        rng.shuffle(assigned_models)
    model_assignment = dict(zip(personas, assigned_models))
    authority_name = (
        _authority_target(personas, authority_stance, authority_model, model_assignment)
        if condition == "authority" else None
    )
    if authority_name:
        personas[authority_name] = upgrade_to_authority(personas[authority_name], TOPIC)
    agents = {
        name: ChatOllama(model=model_assignment[name], temperature=0.7)
        for index, name in enumerate(personas)
    }
    current = {name: STANCE_SCORE[persona.initial_stance] for name, persona in personas.items()}
    records: list[dict[str, Any]] = []
    previous_round: list[dict[str, Any]] = []
    for round_id in range(1, rounds + 1):
        progress(f"run={run_id} scenario={scenario_id} condition={condition} round={round_id}/{rounds}")
        current_round = []
        for name, persona in personas.items():
            visible_history = previous_round + current_round
            result = invoke_agent(agents[name], persona, visible_history, authority_name, current[name])
            current[name] = result["stance_score"]
            current_round.append({
                "run": run_id,
                "scenario": scenario_id,
                "condition": condition,
                "round": round_id,
                "agent": name,
                "model": model_assignment[name],
                "model_family": MODEL_FAMILY.get(model_assignment[name], "unknown"),
                "initial_stance": persona.initial_stance,
                "authority_stance": authority_stance or "",
                "authority_model": authority_model or "",
                "is_authority": name == authority_name,
                **result,
            })
        records.extend(current_round)
        previous_round = current_round
    summary = {
        "run": run_id,
        "scenario": scenario_id,
        "condition": condition,
        "seed": seed,
        "authority": authority_name or "",
        "authority_stance": authority_stance or "",
        "authority_model": authority_model or "",
        "model_assignment": json.dumps(model_assignment, ensure_ascii=False),
        **metrics(records, rounds),
    }
    return records, summary


def _config(kind: str) -> list[dict[str, Any]]:
    if kind == "single-small" or kind in SINGLE_SMALL_GROUPS:
        configs = []
        selected_cases = SINGLE_SMALL_GROUPS.get(kind)
        cases = [
            case for case in SINGLE_SMALL_CASES
            if selected_cases is None or case[0] in selected_cases
        ]
        for model_index, model in enumerate(SMALL_MODELS, start=1):
            for experiment_id, scenario, authority, rounds, pair_group in cases:
                configs.append({
                    "name": f"{model}-{experiment_id}",
                    "experiment": experiment_id,
                    "pair_id": model_index * 10 + pair_group,
                    "models": [model] * AGENTS,
                    "scenario": scenario,
                    "authority": authority,
                    "rounds": rounds,
                })
        return configs
    if kind == "single":
        configs = []
        for model_index, model in enumerate(SINGLE_MODELS, start=1):
            cases = [
                ("1", "balanced", None, 3, 1),
                ("2", "support_majority", None, 3, 2),
                ("3", "oppose_majority", None, 3, 3),
                ("4-1", "balanced", "Support", 5, 1),
                ("4-2", "balanced", "Neutral", 5, 1),
                ("4-3", "balanced", "Oppose", 5, 1),
                ("5-1", "support_majority", "Support", 5, 2),
                ("6-1", "oppose_majority", "Oppose", 5, 3),
                ("7", "support_majority", "Oppose", 2, 2),
                ("8", "oppose_majority", "Support", 2, 3),
            ]
            for experiment_id, scenario, authority, rounds, pair_group in cases:
                configs.append({
                    "name": f"{model}-{experiment_id}",
                    "experiment": experiment_id,
                    "pair_id": model_index * 10 + pair_group,
                    "models": [model] * AGENTS,
                    "scenario": scenario,
                    "authority": authority,
                    "rounds": rounds,
                })
        return configs
    if kind == "intra":
        configs = []
        pair_id = 100
        for permutation in itertools.permutations(FAMILY_MODELS):
            models = [model for model in permutation for _ in range(2)]
            for authority in (None, "Support", "Neutral", "Oppose"):
                configs.append({
                    "name": f"intra-{len(configs) // 4 + 1}-{authority or 'control'}",
                    "experiment": "9-12",
                    "pair_id": pair_id,
                    "models": models,
                    "scenario": "balanced",
                    "authority": authority,
                    "rounds": 5,
                    "fixed_stances": SCENARIOS["balanced"],
                    "shuffle_models": False,
                })
            pair_id += 1
        return configs
    if kind == "cross":
        configs = []
        for experiment_id, models, authority_models in (
            ("13", [model for model in SMALL_MODELS for _ in range(2)], SMALL_MODELS),
            ("14", [model for model in FAMILY_MODELS for _ in range(2)], FAMILY_MODELS),
        ):
            pair_id = 200 if experiment_id == "13" else 300
            configs.append({
                "name": f"{experiment_id}-1",
                "experiment": f"{experiment_id}-1",
                "pair_id": pair_id,
                "models": models,
                "scenario": "balanced",
                "authority": None,
                "authority_model": None,
                "rounds": 5,
                "fixed_stances": SCENARIOS["balanced"],
                "shuffle_models": False,
            })
            for index, authority_model in enumerate(authority_models, start=1):
                configs.append({
                    "name": f"{experiment_id}-2-{authority_model}",
                    "experiment": f"{experiment_id}-2",
                    "pair_id": pair_id,
                    "models": models,
                    "scenario": "balanced",
                    "authority": None,
                    "authority_model": authority_model,
                    "rounds": 5,
                    "fixed_stances": SCENARIOS["balanced"],
                    "shuffle_models": False,
                })
        return configs
    raise ValueError(f"unknown experiment kind: {kind}")


def run_experiment(kind: str, output_dir: str | Path | None = None, repetitions: int = REPETITIONS) -> Path:
    configs = _config(kind)
    destination = Path(output_dir or f"experiment-{kind}-authority-result")
    destination.mkdir(exist_ok=True)
    progress(f"start kind={kind} configurations={len(configs)} repetitions={repetitions} output={destination}")
    summaries: list[dict[str, Any]] = []
    dialogues: list[dict[str, Any]] = []
    started = time.perf_counter()
    models_used = {
        model
        for config in configs
        for model in config["models"]
    }
    try:
        for config_index, config in enumerate(configs, start=1):
            authority_label = config.get("authority") or config.get("authority_model") or "none"
            progress(f"configuration {config_index}/{len(configs)}: {config['name']} authority={authority_label}")
            for run_id in range(1, repetitions + 1):
                progress(f"replication {run_id}/{repetitions}")
                seed = 100000 * config["pair_id"] + run_id
                is_authority = bool(config["authority"] or config.get("authority_model"))
                records, summary = run_condition(
                    run_id, "authority" if is_authority else "control", config["scenario"],
                    seed, config["models"], config["authority"], config.get("authority_model"),
                    config["rounds"], config.get("fixed_stances"), config.get("shuffle_models", True),
                )
                summary["configuration"] = config["name"]
                summary["experiment"] = config["experiment"]
                summaries.append(summary)
                dialogues.extend(records)
            progress(f"configuration complete {config_index}/{len(configs)}")
        fields = [
            "configuration", "experiment", "run", "scenario", "condition", "seed", "authority", "authority_stance",
            "authority_model",
            "model_assignment", "CR", "CR_opportunities", "delta_P", "F5", "P", "F",
        ]
        with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
        with (destination / "dialogues.jsonl").open("w", encoding="utf-8") as handle:
            for row in dialogues:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress(f"complete summaries={len(summaries)} messages={len(dialogues)} elapsed={time.perf_counter() - started:.1f}s")
    finally:
        for model in sorted(models_used):
            unload_ollama_model(model)
    return destination
