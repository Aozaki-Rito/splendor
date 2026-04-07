#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import time
import random
import shutil
from contextlib import nullcontext
from threading import Thread
from typing import List, Dict, Any
from pathlib import Path

import openai
from rich.console import Console

from game.game import Game
from game.player import Player
from agents.base_agent import BaseAgent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from ui.renderer import GameRenderer
from evaluation.evaluator import Evaluator
from utils.config_loader import load_config, get_model_config, get_game_settings, get_evaluation_settings, get_available_models
from utils.llm_factory import create_llm_client
from ui.pygame_ui import PygameUI

try:
    from agents.langgraph_agent import LanggraphAgent
    LANGGRAPH_IMPORT_ERROR = None
except Exception as exc:
    LanggraphAgent = None
    LANGGRAPH_IMPORT_ERROR = exc

try:
    from agents.rl_ppo_agent import RLPPOAgent
    RL_PPO_IMPORT_ERROR = None
except Exception as exc:
    RLPPOAgent = None
    RL_PPO_IMPORT_ERROR = exc


def resolve_model_api_key(model_config: Dict[str, Any]):
    """解析模型配置对应的 API Key。"""
    env_key = model_config.get("api_key_env") or f"{model_config.get('type', '').upper()}_API_KEY"
    api_key = model_config.get("api_key") or os.environ.get(env_key)
    return api_key, env_key


def create_agent_from_model_config(
    model_config: Dict[str, Any],
    player_id: str,
    run_id: str,
    temperature_override: Any,
    use_langgraph: int,
):
    """根据模型配置创建代理，并返回 (agent, mode_label)。"""
    prompt_strategy = model_config.get("prompt_strategy", "legacy")
    temperature = temperature_override if temperature_override is not None else model_config.get("temperature", 0.5)
    model_type = model_config.get("type", "openai")

    if model_type == "rl_ppo":
        if use_langgraph == 1:
            raise ValueError("rl_ppo 模型不能与 --use_langgraph 1 同时使用。")
        if RLPPOAgent is None:
            raise ImportError(f"RL PPO 依赖未安装或导入失败: {RL_PPO_IMPORT_ERROR}")
        agent = RLPPOAgent(
            player_id=player_id,
            name=f"{model_config.get('name')} 代理",
            model_path=model_config.get("model_path") or model_config.get("model_name"),
            deterministic=model_config.get("deterministic", True),
            device=model_config.get("device", "auto"),
            run_id=run_id,
        )
        return agent, "rl_ppo"

    if use_langgraph == 1:
        if prompt_strategy == "rank_v2_auto":
            raise ValueError("--use_langgraph 1 不能与 prompt_strategy=rank_v2_auto 同时使用。")
        if LanggraphAgent is None:
            raise ImportError(f"LangGraph 依赖未安装或导入失败: {LANGGRAPH_IMPORT_ERROR}")
        api_key, _ = resolve_model_api_key(model_config)
        if not api_key:
            raise ValueError("langgraph 模式需要提供 API Key。")
        agent = LanggraphAgent(
            player_id=player_id,
            name=f"{model_config.get('name')} 代理",
            api_key=api_key,
            model_name=model_config.get("model_name"),
            temperature=temperature,
            max_tokens=model_config.get("max_tokens", 500),
            base_url=model_config.get("base_url"),
            model_type=model_config.get("type", "openai_compatible"),
            api_version=model_config.get("api_version"),
            deployment_name=model_config.get("deployment_name"),
            run_id=run_id,
        )
        return agent, "langgraph"

    if prompt_strategy == "rank_v2_auto":
        agent = RuleBasedAgent(
            player_id=player_id,
            name=f"{model_config.get('name')} 代理",
            candidate_action_limit=model_config.get("candidate_action_limit", 6),
            target_limit=model_config.get("target_limit", 4),
            noble_limit=model_config.get("noble_limit", 3),
            run_id=run_id,
        )
        return agent, "rule_based"

    api_key, _ = resolve_model_api_key(model_config)
    if not api_key:
        raise ValueError("legacy 模式需要提供 API Key。")

    llm_client = create_llm_client(model_config)
    agent = LLMAgent(
        player_id=player_id,
        name=f"{model_config.get('name')} 代理",
        llm_client=llm_client,
        temperature=temperature,
        max_tokens=model_config.get("max_tokens", 500),
        prompt_strategy=prompt_strategy,
        run_id=run_id,
    )
    return agent, "legacy"


def create_run_dir(run_id: str, mode: str, args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """为当前运行创建独立的产物目录，并保存参数快照。"""
    run_dir = Path("results") / "runs" / f"{run_id}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(Path.cwd()),
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(run_dir / "cli_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if args.config and os.path.exists(args.config):
        shutil.copy2(args.config, run_dir / "config_source.json")

    return run_dir


def bind_agents_to_game(game: Game, agents: List[BaseAgent]) -> None:
    """将代理绑定到当前游戏实例。"""
    game.agent_map = {agent.player_id: agent for agent in agents}
    for agent in agents:
        agent._game = game


def save_game_history_artifact(game: Game, run_dir: Path, file_name: str = "game_history.json") -> Path:
    """将当前游戏历史保存到运行目录。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / file_name
    game.save_game_history(str(history_path))
    return history_path


def run_game_with_render(args):
    """运行单个游戏"""
    console = Console()
    if getattr(args, "human_players", 0):
        console.print("[bold red]终端渲染模式暂不支持人类玩家，请使用 --use_pygame 1。[/bold red]")
        return
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    run_dir = create_run_dir(run_id, "game", args, config)
    console.print(f"[blue]运行产物目录: {run_dir}[/blue]")
    
    # 获取游戏设置
    game_settings = get_game_settings(config)
    
    # 合并命令行参数和配置文件设置
    num_players = args.num_players or game_settings.get("num_players", 2)
    seed = args.seed or game_settings.get("seed")
    delay = args.delay or game_settings.get("delay", 0.5)
    save_history = args.save_history or game_settings.get("save_history", False)
    
    # 创建代理
    agents = []
    
    # 获取要使用的模型列表
    model_names = []
    
    # 首先检查是否指定了多个模型
    for i in range(1, args.num_llm_agents + 1):
        model_arg = getattr(args, f"model{i}", None)
        if model_arg:
            model_names.append(model_arg)
        elif i == 1 and args.model:
            # 如果只指定了--model参数，将其用作第一个模型
            model_names.append(args.model)
            
    # 如果没有指定足够的模型，使用默认模型或第一个模型重复填充
    default_model = args.model or get_available_models(config)[0] if get_available_models(config) else None
    while len(model_names) < args.num_llm_agents and default_model:
        model_names.append(default_model)
    
    # 根据模型名称创建LLM代理
    for i, model_name in enumerate(model_names):
        model_config = get_model_config(config, model_name)
        
        if model_config:
            try:
                console.print(f"[cyan]正在创建{model_name}代理...[/cyan]")
                prompt_strategy = model_config.get("prompt_strategy", "legacy")
                model_type = model_config.get("type", "openai")
                if model_type == "rl_ppo":
                    console.print(f"[cyan]创建 RL PPO 代理: 模型文件={model_config.get('model_path') or model_config.get('model_name')}[/cyan]")
                elif args.use_langgraph == 1:
                    api_key, env_key = resolve_model_api_key(model_config)
                    if not api_key:
                        console.print(f"[bold red]错误: {model_name}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                        continue
                    console.print(f"[cyan]创建LangGraph代理: 模型={model_config.get('model_name')}[/cyan]")
                elif prompt_strategy == "rank_v2_auto":
                    console.print("[cyan]创建纯规则代理: rank_v2_auto[/cyan]")
                else:
                    api_key, env_key = resolve_model_api_key(model_config)
                    if not api_key:
                        console.print(f"[bold red]错误: {model_name}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                        continue
                    console.print(f"[cyan]创建LLM客户端: 类型={model_config.get('type')}, 模型={model_config.get('model_name')}[/cyan]")

                agent, _ = create_agent_from_model_config(
                    model_config=model_config,
                    player_id=f"llm_agent_{i+1}",
                    run_id=run_id,
                    temperature_override=args.temperature,
                    use_langgraph=args.use_langgraph,
                )
                console.print(f"[blue]代理日志文件: {agent.log_file_path}[/blue]")

                agents.append(agent)
                console.print(f"[green]已成功创建LLM代理: {model_config.get('name')}[/green]")
            except Exception as e:
                console.print(f"[bold red]创建LLM代理失败 ({model_name}): {e}[/bold red]")
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
        else:
            console.print(f"[bold red]错误: 未找到模型'{model_name}'的配置[/bold red]")
    
    # 补充随机代理，确保总共有足够的代理
    num_random_agents = num_players - len(agents)
    for i in range(num_random_agents):
        agent = RandomAgent(
            player_id=f"random_agent_{i+1}",
            name=f"随机代理 {i+1}"
        )
        agents.append(agent)
    
    # 创建玩家
    players = []
    for agent in agents:
        player = Player(agent.player_id, agent.name)
        players.append(player)
        
        # 保存玩家和代理的映射关系
        agent._player = player
    
    # 创建游戏
    game = Game(players, seed=seed)
    bind_agents_to_game(game, agents)
    
    # 创建渲染器
    renderer = GameRenderer(game)
    
    # 游戏开始
    console.print("[bold cyan]========== 璀璨宝石 LLM 代理对战 ==========[/bold cyan]")
    console.print(f"玩家: {', '.join([player.name for player in players])}")
    if seed:
        console.print(f"种子: {seed}\n")
    
    # 游戏开始时的回调
    game_state = game.get_game_state()
    for agent in agents:
        agent.on_game_start(game_state)

    # 初始渲染
    renderer.render()
    executed_turns = 0
    
    # 运行游戏直到结束
    while not game.game_over:
        if args.max_turns is not None and executed_turns >= args.max_turns:
            console.print(f"[yellow]已达到测试回合上限: {args.max_turns}，提前停止。[/yellow]")
            break
        current_player = game.get_current_player()
        current_agent = next((a for a in agents if a._player == current_player), None)
        
        if current_agent:
            console.print(f"\n[bold green]{current_player.name}[/bold green] 的回合:")
            
            # 回合开始
            game_state = game.get_game_state()
            current_agent.on_turn_start(game_state)
            
            # 获取有效动作
            valid_actions = game.get_valid_actions()
            
            if valid_actions:
                # 让代理选择动作
                start_time = time.time()
                selected_action = current_agent.select_action(game_state, valid_actions)
                end_time = time.time()
                
                # 记录决策时间
                decision_time = end_time - start_time
                console.print(f"决策时间: {decision_time:.2f}秒")
                
                # 显示选择的动作
                renderer.render_action(current_player, str(selected_action))
                
                # 执行动作
                success = game.execute_action(selected_action)
                
                # 回合结束
                game_state = game.get_game_state()
                current_agent.on_turn_end(game_state, selected_action, success)
                executed_turns += 1
            else:
                console.print("没有有效动作，跳过")
        
        # 进入下一个玩家
        if not game.game_over:
            game.next_player()
        
        # 更新渲染
        renderer.render()
        
        # 如果命令行参数中指定了延迟，则等待
        if delay > 0:
            time.sleep(delay)
    
    # 游戏结束
    renderer.render_game_over()
    
    # 游戏结束回调
    game_state = game.get_game_state()
    winner_ids = [player.player_id for player in game.winner] if game.winner else []
    
    for agent in agents:
        agent.on_game_end(game_state, winner_ids)
    
    # 保存游戏历史
    if save_history:
        history_file = save_game_history_artifact(game, run_dir)
        console.print(f"\n游戏历史已保存到: {history_file}")

def run_game_logic(
    game: Game,
    agents: List[BaseAgent],
    delay: float,
    save_history: bool = False,
    seed: int = None,
    players: List[Player] = None,
    run_dir: Path = None,
    state_lock=None,
):
    """游戏逻辑线程入口"""
    # 运行游戏直到结束
    console = Console()
    # 游戏开始
    console.print("[bold cyan]========== 璀璨宝石 LLM 代理对战 ==========[/bold cyan]")
    console.print(f"玩家: {', '.join([player.name for player in players])}")
    if seed:
        console.print(f"种子: {seed}\n")
    
    # 游戏开始时的回调
    game_state = game.get_game_state()
    for agent in agents:
        agent.on_game_start(game_state)
    max_turns = getattr(game, "max_turns", None)
    executed_turns = 0

    while not game.game_over:
        lock_ctx = state_lock if state_lock is not None else nullcontext()
        if max_turns is not None and executed_turns >= max_turns:
            console.print(f"[yellow]已达到测试回合上限: {max_turns}，提前停止逻辑线程。[/yellow]")
            game.game_over = True
            break

        with lock_ctx:
            current_player = game.get_current_player()
            current_agent = next((a for a in agents if a._player == current_player), None)
            game_state = game.get_game_state()
            valid_actions = game.get_valid_actions()

        if current_agent:
            # 回合开始
            console.print(f"\n[bold green]{current_player.name}[/bold green] 的回合:")
            current_agent.on_turn_start(game_state)

            if valid_actions:
                # 让代理选择动作
                start_time = time.time()
                selected_action = current_agent.select_action(game_state, valid_actions)
                end_time = time.time()
                decision_time = end_time - start_time
                console.print(f"决策时间: {decision_time:.2f}秒")

                with lock_ctx:
                    success = game.execute_action(selected_action)
                    game_state = game.get_game_state()
                    current_agent.on_turn_end(game_state, selected_action, success)
                    if not game.game_over:
                        game.next_player()
                executed_turns += 1
            else:
                with lock_ctx:
                    if not game.game_over:
                        game.next_player()
        
        # 如果命令行参数中指定了延迟，则等待
        if delay > 0:
            time.sleep(delay)
    
    # 游戏结束回调
    game_state = game.get_game_state()
    winner_ids = [player.player_id for player in game.winner] if game.winner else []
    
    for agent in agents:
        agent.on_game_end(game_state, winner_ids)
    
    # 保存游戏历史
    if save_history and run_dir is not None:
        save_game_history_artifact(game, run_dir)

def run_game_with_pygame(args):
    """运行单个游戏"""
    console = Console()
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"

    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    run_dir = create_run_dir(run_id, "pygame_game", args, config)
    console.print(f"[blue]运行产物目录: {run_dir}[/blue]")

    # 获取游戏设置
    game_settings = get_game_settings(config)
    
    # 合并命令行参数和配置文件设置
    num_players = args.num_players or game_settings.get("num_players", 2)
    seed = args.seed or game_settings.get("seed")
    delay = args.delay or game_settings.get("delay", 0.5)
    save_history = args.save_history or game_settings.get("save_history", False)

    if args.human_players not in (0, 1):
        console.print("[bold red]pygame 模式当前只支持 0 或 1 个真人玩家。[/bold red]")
        return
    if args.human_players == 1 and num_players != 2:
        console.print("[bold red]真人交互模式 V1 当前只支持 2 人对局。[/bold red]")
        return
    if args.human_players == 1 and not (1 <= args.human_seat <= num_players):
        console.print(f"[bold red]human seat 超出范围: {args.human_seat}[/bold red]")
        return

    agents: List[BaseAgent] = []
    
    # 获取要使用的模型列表
    model_names = []
    
    # 首先检查是否指定了多个模型
    for i in range(1, args.num_llm_agents + 1):
        model_arg = getattr(args, f"model{i}", None)
        if model_arg:
            model_names.append(model_arg)
        elif i == 1 and args.model:
            # 如果只指定了--model参数，将其用作第一个模型
            model_names.append(args.model)
            
    # 如果没有指定足够的模型，使用默认模型或第一个模型重复填充
    default_model = args.model or get_available_models(config)[0] if get_available_models(config) else None
    while len(model_names) < args.num_llm_agents and default_model:
        model_names.append(default_model)
    
    ai_agents: List[BaseAgent] = []

    for i, model_name in enumerate(model_names):
        model_config = get_model_config(config, model_name)
        
        if model_config:
            try:
                console.print(f"[cyan]正在创建{model_name}代理...[/cyan]")
                prompt_strategy = model_config.get("prompt_strategy", "legacy")
                model_type = model_config.get("type", "openai")
                if model_type == "rl_ppo":
                    console.print(f"[cyan]创建 RL PPO 代理: 模型文件={model_config.get('model_path') or model_config.get('model_name')}[/cyan]")
                elif args.use_langgraph == 1:
                    api_key, env_key = resolve_model_api_key(model_config)
                    if not api_key:
                        console.print(f"[bold red]错误: {model_name}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                        continue
                    console.print(f"[cyan]创建LangGraph代理: 模型={model_config.get('model_name')}[/cyan]")
                elif prompt_strategy == "rank_v2_auto":
                    console.print("[cyan]创建纯规则代理: rank_v2_auto[/cyan]")
                else:
                    api_key, env_key = resolve_model_api_key(model_config)
                    if not api_key:
                        console.print(f"[bold red]错误: {model_name}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                        continue
                    console.print(f"[cyan]创建LLM客户端: 类型={model_config.get('type')}, 模型={model_config.get('model_name')}[/cyan]")

                agent, _ = create_agent_from_model_config(
                    model_config=model_config,
                    player_id=f"llm_agent_{i+1}",
                    run_id=run_id,
                    temperature_override=args.temperature,
                    use_langgraph=args.use_langgraph,
                )
                console.print(f"[blue]代理日志文件: {agent.log_file_path}[/blue]")

                ai_agents.append(agent)
                console.print(f"[green]已成功创建LLM代理: {model_config.get('name')}[/green]")
            except Exception as e:
                console.print(f"[bold red]创建LLM代理失败 ({model_name}): {e}[/bold red]")
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
        else:
            console.print(f"[bold red]错误: 未找到模型'{model_name}'的配置[/bold red]")

    available_ai_seats = num_players - args.human_players
    if len(ai_agents) > available_ai_seats:
        console.print("[bold red]模型代理数量超过可用 AI 座位数，请减少 --num-llm-agents 或调整座位。[/bold red]")
        return

    # 补充随机代理，确保总共有足够的代理
    num_random_agents = available_ai_seats - len(ai_agents)
    for i in range(num_random_agents):
        agent = RandomAgent(
            player_id=f"random_agent_{i+1}",
            name=f"随机代理 {i+1}"
        )
        ai_agents.append(agent)

    seat_agents: List[BaseAgent] = [None] * num_players
    if args.human_players == 1:
        human_agent = HumanAgent(
            player_id=f"human_agent_{args.human_seat}",
            name="人类玩家",
        )
        seat_agents[args.human_seat - 1] = human_agent

    ai_iter = iter(ai_agents)
    for idx in range(num_players):
        if seat_agents[idx] is None:
            seat_agents[idx] = next(ai_iter)

    agents = [agent for agent in seat_agents if agent is not None]

    # 创建玩家
    players = []
    for agent in seat_agents:
        player = Player(agent.player_id, agent.name)
        players.append(player)
        
        # 保存玩家和代理的映射关系
        agent._player = player
    console.print(f"[bold green]已创建玩家: {', '.join([player.name for player in players])}[/bold green]")
    
    # 创建游戏
    game = Game(players, seed=seed)
    bind_agents_to_game(game, agents)
    game.max_turns = args.max_turns
    pygame_ui = PygameUI(game, agents=agents, fullscreen=bool(args.fullscreen))

    # 启动游戏逻辑线程
    logic_thread = Thread(
        target=run_game_logic,
        args=(game, agents, delay, save_history, seed, players, run_dir),
        kwargs={"state_lock": pygame_ui.lock},
        daemon=True,
    )
    logic_thread.start()

    # 主线程负责渲染
    pygame_ui.run_loop()


def run_evaluation(args):
    """运行评估"""
    console = Console()
    console.print("[bold cyan]========== 璀璨宝石 LLM 代理评估 ==========[/bold cyan]")
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    run_dir = create_run_dir(run_id, "evaluation", args, config)
    console.print(f"[blue]运行产物目录: {run_dir}[/blue]")
    
    # 获取评估设置
    eval_settings = get_evaluation_settings(config)
    
    # 合并命令行参数和配置文件设置
    num_games = args.num_games or eval_settings.get("num_games", 10)
    seed = args.seed or eval_settings.get("seed")
    
    # 创建代理
    agents = []
    
    # 获取模型配置
    model_config = get_model_config(config, args.model)
    
    if model_config:
        try:
            prompt_strategy = model_config.get("prompt_strategy", "legacy")
            model_type = model_config.get("type", "openai")
            if model_type == "rl_ppo":
                console.print(f"[cyan]创建 RL PPO 代理: 模型文件={model_config.get('model_path') or model_config.get('model_name')}[/cyan]")
            elif prompt_strategy == "rank_v2_auto":
                console.print("[cyan]创建纯规则代理: rank_v2_auto[/cyan]")
            else:
                api_key, env_key = resolve_model_api_key(model_config)
                if not api_key:
                    console.print(f"[bold red]错误: {args.model}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                    return
                console.print(f"[cyan]创建LLM客户端: 类型={model_config.get('type')}, 模型={model_config.get('model_name')}[/cyan]")

            agent, _ = create_agent_from_model_config(
                model_config=model_config,
                player_id="llm_agent",
                run_id=run_id,
                temperature_override=args.temperature,
                use_langgraph=0,
            )
            agents.append(agent)
            console.print(f"[blue]代理日志文件: {agent.log_file_path}[/blue]")
        except Exception as e:
            console.print(f"[bold red]创建LLM代理失败：{e}[/bold red]")
    
    # 添加随机代理
    agent = RandomAgent(
        player_id="random_agent",
        name="随机代理"
    )
    agents.append(agent)
    
    # 创建评估器
    evaluator = Evaluator(agents, num_games=num_games, seed=seed)
    
    # 运行评估
    results = evaluator.run_evaluation(output_dir=str(run_dir))
    
    # 显示结果摘要
    console.print("\n[bold yellow]评估结果摘要:[/bold yellow]")
    
    for agent_type, data in results["summary"]["agent_performance"].items():
        console.print(f"[bold]{data['name']}[/bold]:")
        console.print(f"  胜利: {data['wins']}/{num_games} (胜率: {data['win_rate']*100:.1f}%)")
        console.print(f"  平均分数: {data['average_score']:.2f}")
        console.print(f"  平均排名: {data['average_rank']:.2f}")
        console.print("")


def list_models(args):
    """列出可用的模型"""
    console = Console()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    
    # 获取可用模型
    models = config.get("models", [])
    
    if not models:
        console.print("[yellow]配置文件中未找到模型[/yellow]")
        return
    
    # 创建表格显示模型信息
    from rich.table import Table
    table = Table(title="可用模型")
    
    table.add_column("名称", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("模型标识", style="blue")
    table.add_column("API可用", style="yellow")
    
    for model in models:
        name = model.get("name", "未命名")
        model_type = model.get("type", "未知")
        model_id = model.get("model_name") or model.get("model_path", "未知")
        
        # 检查API密钥是否可用
        env_key = model.get("api_key_env") or f"{model_type.upper()}_API_KEY"
        api_key = model.get("api_key") or os.environ.get(env_key)
        if model_type == "rl_ppo":
            api_status = "[cyan]不需要[/cyan]"
        else:
            api_status = "[green]是[/green]" if api_key else "[red]否[/red]"
        
        table.add_row(name, model_type, model_id, api_status)
    
    console.print(table)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="璀璨宝石 LLM 代理对战")
    
    # 基本配置
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 游戏模式
    game_parser = subparsers.add_parser("game", help="运行单场游戏")
    game_parser.add_argument("--num-players", type=int, help="玩家总数；未显式创建的玩家会自动补成随机代理")
    game_parser.add_argument("--seed", type=int, help="随机种子，用于复现实验")
    game_parser.add_argument("--delay", type=float, help="相邻回合之间的显示延迟(秒)")
    game_parser.add_argument("--model", type=str, help="config.json 中的模型名称；用于所有非随机代理")
    game_parser.add_argument("--temperature", type=float, help="仅对 legacy / LangGraph 有效的温度覆盖值")
    game_parser.add_argument("--num-llm-agents", type=int, default=1, help="使用配置模型的代理数量；其余玩家自动补随机代理")
    game_parser.add_argument("--save-history", action="store_true", help="保存游戏历史到运行目录")
    game_parser.add_argument("--use_pygame", type=int, choices=[0,1], default=1, help="1=pygame 图形界面，0=终端渲染")
    game_parser.add_argument("--use_langgraph", type=int, choices=[0,1], default=0, help="1=强制使用 LangGraph；不能与 rl_ppo 或 rank_v2_auto 同时使用")
    game_parser.add_argument("--max-turns", type=int, help="调试用；达到该动作数后提前停止")
    game_parser.add_argument("--human-players", type=int, default=0, help="pygame 模式下的人类玩家数量，V1 仅支持 0 或 1")
    game_parser.add_argument("--human-seat", type=int, default=1, help="pygame 模式下人类玩家座位（1-based）")
    game_parser.add_argument("--fullscreen", type=int, choices=[0,1], default=0, help="pygame 模式下是否以全屏启动；进入游戏后也可按 F11 切换")
    

    # 为每个可能的LLM代理添加特定的模型参数
    for i in range(1, 5):  # 支持最多4个LLM代理
        game_parser.add_argument(f"--model{i}", type=str, help=f"第{i}个LLM代理使用的模型名称")
    
    # 评估模式
    eval_parser = subparsers.add_parser("eval", help="评估LLM代理性能")
    eval_parser.add_argument("--num-games", type=int, help="评估时运行的游戏数量")
    eval_parser.add_argument("--seed", type=int, help="随机种子")
    eval_parser.add_argument("--model", type=str, help="使用的LLM模型名称")
    eval_parser.add_argument("--temperature", type=float, help="LLM温度参数")
    
    # 列出模型
    list_parser = subparsers.add_parser("list-models", help="列出可用的模型")
    
    args = parser.parse_args()
    
    # 默认命令
    if not args.command:
        args.command = "game"
    
    if args.command == "game" and getattr(args, "human_players", 0) and args.use_pygame == 0:
        print("真人交互模式需要 use_pygame=1")
        sys.exit(1)

    if args.command == "game" and args.use_pygame == 0:
        print("不使用pygame图形界面:", args.use_pygame)
        run_game_with_render(args)
    elif args.command == "game" and args.use_pygame == 1:
        print("使用pygame图形界面:", args.use_pygame)
        run_game_with_pygame(args)
    elif args.command == "eval":
        run_evaluation(args)
    elif args.command == "list-models":
        list_models(args)
    else:
        print(f"未知命令: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
