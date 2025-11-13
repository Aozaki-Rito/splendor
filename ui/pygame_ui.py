# ui/pygame_ui.py
import pygame
import threading
import os
from pygame.locals import *
from game.card import Card
from game.noble import Noble
from game.player import Player

# ======================== 路径设定 ========================
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJK-Regular.otf")

def load_image(path, scale=None):
    """加载图片，可选缩放"""
    if not os.path.exists(path):
        print(f"[警告] 贴图不存在: {path}")
        return None
    
    img = pygame.image.load(path).convert_alpha()
    if scale:
        img = pygame.transform.smoothscale(img, scale)
    return img


# ======================== 主 UI 类 ========================
class PygameUI:
    def __init__(self, game, fps=30):
        pygame.init()
        self.game = game
        self.fps = fps
        self.running = True
        self.lock = threading.Lock()

        self.screen = pygame.display.set_mode((1280, 800))
        pygame.display.set_caption("Splendor - 宝石商人")
        self.clock = pygame.time.Clock()

        # 背景加载
        bg_path = os.path.join(ASSETS_DIR, "background.jpg")
        self.background = load_image(bg_path, (1280, 800))

        # 加载所有贴图资源
        self._load_card_images()
        self._load_noble_images()
        self._load_gem_images()


    # ======================== 贴图加载 ========================
    def _load_card_images(self):
        self.card_images = {}
        card_dir = os.path.join(ASSETS_DIR, "cards")

        if not os.path.exists(card_dir):
            print("[错误] 缺少 assets/cards 文件夹！")
            return

        for fname in os.listdir(card_dir):
            if fname.endswith(".png"):
                cid = int(fname.replace(".png", ""))
                path = os.path.join(card_dir, fname)
                self.card_images[cid] = load_image(path, (120, 180))


    def _load_noble_images(self):
        self.noble_images = {}
        noble_dir = os.path.join(ASSETS_DIR, "nobles")

        for fname in os.listdir(noble_dir):
            if fname.endswith(".png"):
                nid = int(fname.replace(".png", ""))
                path = os.path.join(noble_dir, fname)
                self.noble_images[nid] = load_image(path, (100, 100))


    def _load_gem_images(self):
        """宝石按 GemColor.value 命名"""
        self.gem_images = {}
        gem_dir = os.path.join(ASSETS_DIR, "gems")

        gem_names = ["white", "blue", "green", "red", "black", "gold"]

        for g in gem_names:
            path = os.path.join(gem_dir, f"{g}.png")
            self.gem_images[g] = load_image(path, (40, 40))


    # ======================== UI 主循环 ========================
    def run_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    pygame.quit()
                    return

            with self.lock:
                self.render()

            self.clock.tick(self.fps)


    # ======================== 渲染 ========================
    def render(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((230, 235, 245))

        # 当前玩家
        current_player = self.game.get_current_player()
        round_text = f"第 {self.game.round_number} 回合 | 当前玩家: {current_player.name}"
        self._draw_text(round_text, (40, 20), (0, 0, 0), 28)

        # 卡牌
        y = 100
        for level, cards in self.game.board.displayed_cards.items():
            self._draw_text(f"等级 {level} 卡牌:", (60, y - 30))
            x = 60
            for card in cards:
                self._draw_card(card, (x, y))
                x += 140
            y += 210

        # 贵族
        self._draw_text("贵族:", (950, 100))
        for i, noble in enumerate(self.game.board.nobles):
            self._draw_noble(noble, (950, 140 + i * 120))

        # 玩家栏
        base_y = 600
        for idx, player in enumerate(self.game.players):
            highlight = (player.player_id == current_player.player_id)
            self._draw_player_info(player, (60 + idx * 440, base_y), highlight)

        pygame.display.flip()


    # ======================== 渲染辅助函数 ========================
    def _draw_text(self, text, pos, color=(0,0,0), size=22):
        font = pygame.font.Font(FONT_PATH, size)
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, pos)


    def _draw_card(self, card: Card, pos):
        img = self.card_images.get(card.card_id)
        if img:
            self.screen.blit(img, pos)
        else:
            pygame.draw.rect(self.screen, (220, 220, 220), (*pos,120,180))
            self._draw_text(f"ID:{card.card_id}", (pos[0]+10, pos[1]+10))


    def _draw_noble(self, noble: Noble, pos):
        img = self.noble_images.get(noble.noble_id)
        if img:
            self.screen.blit(img, pos)
        else:
            pygame.draw.rect(self.screen, (255,230,200), (*pos,100,100))
            self._draw_text(f"Noble {noble.noble_id}", (pos[0]+5, pos[1]+5))


    def _draw_player_info(self, player: Player, pos, highlight=False):
        bg = (180,230,255) if highlight else (230,230,230)
        pygame.draw.rect(self.screen, bg, (*pos,400,120), border_radius=10)

        self._draw_text(f"{player.name}  |  分数: {player.get_score()}", (pos[0]+10, pos[1]+10))

        y = pos[1] + 40

        # 宝石显示
        for color, count in player.gems.items():
            gem_img = self.gem_images.get(color.value)
            if gem_img:
                self.screen.blit(gem_img, (pos[0]+10, y))
            self._draw_text(str(count), (pos[0] + 60, y + 10))
            y += 45
