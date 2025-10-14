#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import time
import random
from typing import List, Dict, Any

import openai
from rich.console import Console

from game.game import Game
from game.player import Player
from agents.base_agent import BaseAgent
from agents.llm_agent import LLMAgent
from ui.renderer import GameRenderer
from evaluation.evaluator import Evaluator
from utils.config_loader import load_config, get_model_config, get_game_settings, get_evaluation_settings, get_available_models
from utils.llm_factory import create_llm_client


class RandomAgent(BaseAgent):
    """随机代理，随机选择动作"""
    
    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Any]) -> Any:
        """随机选择一个动作"""
        return random.choice(valid_actions) if valid_actions else None
    
    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """随机选择要丢弃的宝石"""
        result = {}
        colors = [color for color, count in gems.items() if count > 0]
        
        for _ in range(num_to_discard):
            if not colors:
                break
            color = random.choice(colors)
            result[color] = result.get(color, 0) + 1
            gems[color] -= 1
            if gems[color] <= 0:
                colors.remove(color)
        
        return result
    
    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        """随机选择一个贵族"""
        return random.choice(available_nobles)["id"] if available_nobles else None


def run_game(args):
    """运行单个游戏"""
    console = Console()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    
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
                
                # 检查API密钥
                api_key = model_config.get("api_key")
                if not api_key:
                    env_key = f"{model_config.get('type', '').upper()}_API_KEY"
                    api_key = os.environ.get(env_key)
                    
                if not api_key:
                    console.print(f"[bold red]错误: {model_name}没有提供API密钥。请在config.json中设置api_key或通过{env_key}环境变量提供[/bold red]")
                    continue
                
                # 创建LLM客户端
                console.print(f"[cyan]创建LLM客户端: 类型={model_config.get('type')}, 模型={model_config.get('model_name')}[/cyan]")
                llm_client = create_llm_client(model_config)
                
                # 使用配置的温度参数，如果命令行指定则优先使用命令行参数
                temperature = args.temperature if args.temperature is not None else model_config.get("temperature", 0.5)
                
                agent = LLMAgent(
                    player_id=f"llm_agent_{i+1}",
                    name=f"{model_config.get('name')} 代理",
                    llm_client=llm_client,
                    temperature=temperature
                )
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
    
    # 运行游戏直到结束
    while not game.game_over:
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
            else:
                console.print("没有有效动作，跳过")
        
        # 进入下一个玩家
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
        os.makedirs("results", exist_ok=True)
        timestamp = int(time.time())
        history_file = os.path.join("results", f"game_history_{timestamp}.json")
        game.save_game_history(history_file)
        console.print(f"\n游戏历史已保存到: {history_file}")


def run_evaluation(args):
    """运行评估"""
    console = Console()
    console.print("[bold cyan]========== 璀璨宝石 LLM 代理评估 ==========[/bold cyan]")
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[bold red]错误：{e}[/bold red]")
        return
    
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
            # 创建LLM客户端
            llm_client = create_llm_client(model_config)
            
            # 使用配置的温度参数，如果命令行指定则优先使用命令行参数
            temperature = args.temperature if args.temperature is not None else model_config.get("temperature", 0.5)
            
            agent = LLMAgent(
                player_id="llm_agent",
                name=f"{model_config.get('name')}",
                llm_client=llm_client,
                temperature=temperature
            )
            agents.append(agent)
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
    results = evaluator.run_evaluation()
    
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
        model_id = model.get("model_name", "未知")
        
        # 检查API密钥是否可用
        api_key = model.get("api_key") or os.environ.get(f"{model_type.upper()}_API_KEY")
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
    game_parser.add_argument("--num-players", type=int, help="玩家数量")
    game_parser.add_argument("--seed", type=int, help="随机种子")
    game_parser.add_argument("--delay", type=float, help="回合之间的延迟时间(秒)")
    game_parser.add_argument("--model", type=str, help="使用的LLM模型名称(用于所有LLM代理)")
    game_parser.add_argument("--temperature", type=float, help="LLM温度参数")
    game_parser.add_argument("--num-llm-agents", type=int, default=1, help="LLM代理数量")
    game_parser.add_argument("--save-history", action="store_true", help="保存游戏历史")
    
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
    
    if args.command == "game":
        run_game(args)
    elif args.command == "eval":
        run_evaluation(args)
    elif args.command == "list-models":
        list_models(args)
    else:
        print(f"未知命令: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main() 