import json
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.box import ROUNDED

from game.game import Game
from game.player import Player
from game.card import GemColor


class GameRenderer:
    """游戏状态可视化渲染器"""
    
    # 颜色映射
    COLOR_MAP = {
        GemColor.WHITE.value: "bright_white",
        GemColor.BLUE.value: "blue",
        GemColor.GREEN.value: "green",
        GemColor.RED.value: "red",
        GemColor.BLACK.value: "bright_black",
        GemColor.GOLD.value: "yellow"
    }
    
    def __init__(self, game: Game):
        """初始化渲染器
        
        Args:
            game: 游戏实例
        """
        self.game = game
        self.console = Console()
    
    def render(self):
        """渲染当前游戏状态"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(name="board", ratio=2),
            Layout(name="players", ratio=1)
        )
        
        # 渲染标题
        layout["header"].update(self._render_header())
        
        # 渲染游戏板
        layout["board"].update(self._render_board())
        
        # 渲染玩家信息
        layout["players"].update(self._render_players())
        
        # 输出到控制台
        self.console.print(layout)
    
    def _render_header(self) -> Panel:
        """渲染游戏标题和回合信息"""
        game_state = self.game.get_game_state()
        current_player = self.game.get_current_player()
        
        if self.game.game_over:
            winner_names = [player.name for player in self.game.winner]
            title = f"[bold cyan]璀璨宝石[/bold cyan] - [yellow]游戏结束! 胜利者: {', '.join(winner_names)}[/yellow]"
        else:
            title = f"[bold cyan]璀璨宝石[/bold cyan] - 回合: {game_state['round']} - 当前玩家: [green]{current_player.name}[/green]"
        
        if self.game.last_round:
            title += " [red](最后一轮)[/red]"
        
        return Panel(title, border_style="cyan")
    
    def _render_board(self) -> Panel:
        """渲染游戏板状态"""
        board = self.game.board
        
        # 创建一个布局
        layout = Layout()
        layout.split(
            Layout(name="gems", size=5),
            Layout(name="cards"),
            Layout(name="nobles", size=10)
        )
        
        # 渲染宝石
        gems_table = Table(title="宝石代币", box=ROUNDED, border_style="cyan")
        gems_table.add_column("白色", style=self.COLOR_MAP[GemColor.WHITE.value])
        gems_table.add_column("蓝色", style=self.COLOR_MAP[GemColor.BLUE.value])
        gems_table.add_column("绿色", style=self.COLOR_MAP[GemColor.GREEN.value])
        gems_table.add_column("红色", style=self.COLOR_MAP[GemColor.RED.value])
        gems_table.add_column("黑色", style=self.COLOR_MAP[GemColor.BLACK.value])
        gems_table.add_column("黄金", style=self.COLOR_MAP[GemColor.GOLD.value])
        
        gems_table.add_row(
            str(board.gems.get(GemColor.WHITE, 0)),
            str(board.gems.get(GemColor.BLUE, 0)),
            str(board.gems.get(GemColor.GREEN, 0)),
            str(board.gems.get(GemColor.RED, 0)),
            str(board.gems.get(GemColor.BLACK, 0)),
            str(board.gems.get(GemColor.GOLD, 0))
        )
        
        layout["gems"].update(gems_table)
        
        # 渲染卡牌
        cards_layout = Layout()
        cards_layout.split_row(
            Layout(name="level1"),
            Layout(name="level2"),
            Layout(name="level3")
        )
        
        for level in [1, 2, 3]:
            cards_table = Table(title=f"{level}级卡牌 (剩余: {len(board.card_decks[level])})", box=ROUNDED, border_style="cyan")
            cards_table.add_column("ID")
            cards_table.add_column("点数")
            cards_table.add_column("颜色")
            cards_table.add_column("成本")
            
            for card in board.displayed_cards.get(level, []):
                # 格式化成本
                cost_str = ""
                for color, count in card.cost.items():
                    if count > 0:
                        color_name = color.value
                        cost_str += f"[{self.COLOR_MAP.get(color_name, 'white')}]{color_name}: {count}[/] "
                
                cards_table.add_row(
                    card.card_id,
                    str(card.points),
                    f"[{self.COLOR_MAP.get(card.gem_color.value, 'white')}]{card.gem_color.value}[/]",
                    cost_str
                )
            
            cards_layout[f"level{level}"].update(cards_table)
        
        layout["cards"].update(cards_layout)
        
        # 渲染贵族
        nobles_table = Table(title="贵族", box=ROUNDED, border_style="cyan")
        nobles_table.add_column("ID")
        nobles_table.add_column("点数")
        nobles_table.add_column("要求")
        
        for noble in board.nobles:
            # 格式化要求
            req_str = ""
            for color, count in noble.requirements.items():
                if count > 0:
                    color_name = color.value
                    req_str += f"[{self.COLOR_MAP.get(color_name, 'white')}]{color_name}: {count}[/] "
            
            nobles_table.add_row(
                noble.noble_id,
                str(noble.points),
                req_str
            )
        
        layout["nobles"].update(nobles_table)
        
        return Panel(layout, title="游戏板", border_style="blue")
    
    def _render_players(self) -> Panel:
        """渲染玩家信息"""
        players = self.game.players
        current_player_index = self.game.current_player_index
        
        layout = Layout()
        
        # 为每个玩家创建一个部分
        for i, player in enumerate(players):
            if i == current_player_index and not self.game.game_over:
                name = f"[bold green]{player.name} (当前行动)[/bold green]"
                style = "green"
            else:
                name = player.name
                style = "blue"
            
            # 创建玩家信息表格
            player_table = Table(box=ROUNDED, border_style=style, show_header=False)
            player_table.add_column("属性")
            player_table.add_column("值")
            
            # 添加基本信息
            player_table.add_row("分数", f"[bold]{player.get_score()}[/bold]")
            
            # 添加宝石信息
            gems_str = ""
            for color, count in player.gems.items():
                if count > 0:
                    color_name = color.value
                    gems_str += f"[{self.COLOR_MAP.get(color_name, 'white')}]{color_name}: {count}[/] "
            player_table.add_row("宝石", gems_str)
            
            # 添加卡牌折扣信息
            discounts = player.get_card_discounts()
            discount_str = ""
            for color, count in discounts.items():
                if count > 0:
                    color_name = color.value
                    discount_str += f"[{self.COLOR_MAP.get(color_name, 'white')}]{color_name}: {count}[/] "
            player_table.add_row("卡牌折扣", discount_str)
            
            # 添加卡牌信息
            cards_str = f"共 {len(player.cards)} 张"
            player_table.add_row("拥有卡牌", cards_str)
            
            # 添加预留卡牌信息
            reserved_str = ""
            for card in player.reserved_cards:
                reserved_str += f"{card.card_id} ({card.points}分) "
            player_table.add_row("预留卡牌", reserved_str if reserved_str else "无")
            
            # 添加贵族信息
            nobles_str = ", ".join([noble.noble_id for noble in player.nobles])
            player_table.add_row("贵族", nobles_str if nobles_str else "无")
            
            layout.add_split(
                Layout(Panel(player_table, title=name, border_style=style))
            )
        
        return Panel(layout, title="玩家信息", border_style="blue")
    
    def render_action(self, player: Player, action_str: str, success: bool = True):
        """渲染玩家执行的动作
        
        Args:
            player: 执行动作的玩家
            action_str: 动作描述
            success: 动作是否成功
        """
        if success:
            self.console.print(f"[bold green]{player.name}[/bold green] 执行: {action_str}")
        else:
            self.console.print(f"[bold red]{player.name}[/bold red] 尝试执行: {action_str} [红色](失败)[/红色]")
    
    def render_game_over(self):
        """渲染游戏结束信息"""
        if not self.game.winner:
            self.console.print("[bold red]游戏结束，没有胜利者[/bold red]")
            return
        
        winner_names = [player.name for player in self.game.winner]
        winner_scores = [player.get_score() for player in self.game.winner]
        
        if len(self.game.winner) == 1:
            self.console.print(f"[bold yellow]游戏结束! [green]{winner_names[0]}[/green] 获胜，得分: {winner_scores[0]}[/bold yellow]")
        else:
            self.console.print(f"[bold yellow]游戏结束! 平局: {', '.join([f'[green]{name}[/green]' for name in winner_names])}, 得分: {winner_scores[0]}[/bold yellow]")
        
        # 显示所有玩家的最终得分
        scores_table = Table(title="最终得分", box=ROUNDED, border_style="yellow")
        scores_table.add_column("玩家")
        scores_table.add_column("分数")
        scores_table.add_column("卡牌数")
        scores_table.add_column("贵族数")
        
        for player in sorted(self.game.players, key=lambda p: p.get_score(), reverse=True):
            is_winner = player in self.game.winner
            name = f"[green]{player.name}[/green]" if is_winner else player.name
            scores_table.add_row(
                name,
                str(player.get_score()),
                str(len(player.cards)),
                str(len(player.nobles))
            )
        
        self.console.print(scores_table) 