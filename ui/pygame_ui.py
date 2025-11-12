# ui/pygame_ui.py
import pygame
import threading
import time
import os
from pygame.locals import *
from game.card import Card
from game.noble import Noble
from game.player import Player

FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf")

# ==================== 基础颜色 ====================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (80, 80, 80)
LIGHT_GRAY = (200, 200, 200)
BLUE = (150, 180, 255)
GOLD = (255, 215, 0)

# ==================== 主UI类 ====================
class PygameUI:
    def __init__(self, game, fps: int = 30):
        """初始化 Pygame 界面"""
        self.game = game
        self.fps = fps
        self.running = True
        self.lock = threading.Lock()

        pygame.init()
        self.screen = pygame.display.set_mode((1280, 800))
        pygame.display.set_caption("Splendor - 宝石商人")

        self.clock = pygame.time.Clock()

    # ==================== 渲染线程入口 ====================
    def run_loop(self):
        """独立线程循环：以固定帧率刷新画面"""
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    pygame.quit()
                    return

            with self.lock:
                self.render()

            self.clock.tick(self.fps)

    # ==================== 渲染主函数 ====================
    def render(self):
        """主渲染逻辑"""
        self.screen.fill((240, 245, 255))

        # 1️⃣ 游戏信息
        try:
            current_player = self.game.get_current_player()
            round_text = f"第 {self.game.round_number} 回合 | 当前玩家: {current_player.name}"
        except Exception:
            current_player = None
            round_text = "初始化中..."
        self._draw_text(round_text, (40, 20), (30, 60, 120), 28)

        # 2️⃣ 绘制卡牌展示区
        y_offset = 100
        for level, cards in self.game.board.displayed_cards.items():
            self._draw_text(f"Level {level} 卡牌:", (60, y_offset - 30), BLACK, 22)
            x_offset = 60
            for card in cards:
                self._draw_card(card, (x_offset, y_offset))
                x_offset += 140
            y_offset += 210

        # 3️⃣ 绘制贵族
        self._draw_text("贵族:", (950, 100), BLACK, 22)
        for i, noble in enumerate(self.game.board.nobles):
            self._draw_noble(noble, (950, 140 + i * 120))

        # 4️⃣ 绘制玩家
        base_y = 600
        for i, player in enumerate(self.game.players):
            color = (220, 240, 255) if current_player and player.player_id == current_player.player_id else (230, 230, 230)
            self._draw_player(player, (60 + i * 440, base_y), color)

        pygame.display.flip()

    # ==================== 绘制函数 ====================
    def _draw_text(self, text, pos, color=BLACK, size=22):
        font = pygame.font.SysFont("notosanscjksc", size)
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, pos)

    def _draw_card(self, card: Card, pos):
        rect = pygame.Rect(pos[0], pos[1], 120, 180)
        pygame.draw.rect(self.screen, LIGHT_GRAY, rect, border_radius=8)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        self._draw_text(f"{card.points}分", (pos[0] + 8, pos[1] + 8))
        self._draw_text(card.gem_color.value, (pos[0] + 8, pos[1] + 35))
        y = pos[1] + 60
        for color, cost in card.cost.items():
            self._draw_text(f"{color.value}:{cost}", (pos[0] + 10, y))
            y += 20

    def _draw_noble(self, noble: Noble, pos):
        rect = pygame.Rect(pos[0], pos[1], 100, 100)
        pygame.draw.rect(self.screen, (255, 240, 180), rect, border_radius=8)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        self._draw_text(f"{noble.points}分", (pos[0] + 25, pos[1] + 5))
        y = pos[1] + 35
        for color, need in noble.requirements.items():
            if need > 0:
                self._draw_text(f"{color.value}:{need}", (pos[0] + 10, y))
                y += 18

    def _draw_player(self, player: Player, pos, bg_color):
        rect = pygame.Rect(pos[0], pos[1], 400, 120)
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        self._draw_text(f"{player.name} | 分数:{player.get_score()}", (pos[0] + 10, pos[1] + 10))
        y = pos[1] + 40
        for color, count in player.gems.items():
            if count > 0:
                self._draw_text(f"{color.value}:{count}", (pos[0] + 10, y))
                y += 20

    # ==================== 公共方法 ====================
    def stop(self):
        """结束渲染线程"""
        self.running = False

    def safe_render_once(self):
        """主线程可调用的单次刷新（带锁）"""
        with self.lock:
            self.render()
