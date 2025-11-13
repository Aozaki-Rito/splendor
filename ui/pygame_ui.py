# ui/pygame_ui.py
import pygame
import threading
import os
from pygame.locals import *
from game.card import Card
from game.noble import Noble
from game.player import Player, GemColor

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

        self.screen = pygame.display.set_mode((1920, 1080))
        pygame.display.set_caption("Splendor - 宝石商人")
        self.clock = pygame.time.Clock()

        # 背景加载
        bg_path = os.path.join(ASSETS_DIR, "splendor.png")
        self.background = load_image(bg_path, (1920, 1080))

        # 加载所有贴图资源
        self._load_card_images()
        self._load_noble_images()
        self._load_gem_images()

        self.show_reserved_mode = False


    # ======================== 贴图加载 ========================
    def _load_card_images(self):
        self.card_images = {}
        card_dir = os.path.join(ASSETS_DIR, "cards")

        if not os.path.exists(card_dir):
            print("[错误] 缺少 assets/cards 文件夹！")
            return

        for fname in os.listdir(card_dir):
            if fname.endswith(".png"):
                cid = fname[:-4] 
                path = os.path.join(card_dir, fname)
                self.card_images[cid] = load_image(path, (175, 255))

    def _load_noble_images(self):
        self.noble_images = {}
        noble_dir = os.path.join(ASSETS_DIR, "nobles")

        for fname in os.listdir(noble_dir):
            if fname.endswith(".png"):
                noble_id = fname[:-4]   # 去掉 ".png"，即 "N1" → "N1"
                path = os.path.join(noble_dir, fname)
                self.noble_images[noble_id] = load_image(path, (170, 170))

    def _load_gem_images(self):
        """宝石按 GemColor.value 命名"""
        self.gem_images = {}
        gem_dir = os.path.join(ASSETS_DIR, "gems")

        gem_names = ["white", "blue", "green", "red", "black", "gold"]

        for g in gem_names:
            path = os.path.join(gem_dir, f"{g}.png")
            self.gem_images[g] = load_image(path, (60, 60))


    # ======================== UI 主循环 ========================
    def run_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    pygame.quit()
                    return

                 # 按钮点击检测
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    right = self.screen.get_width() - 60
                    w = 150
                    h = 40
                    if right - w < mx < right and 50 < my < 90:
                        # 点击切换模式按钮
                        self.show_reserved_mode = not self.show_reserved_mode

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
        self._draw_text(round_text, (40, 20), (255, 255, 255), 28)

        # ===== 切换按钮 =====
        right = self.screen.get_width() - 60
        w = 150
        h = 40
        btn_rect = pygame.Rect(right - w, 50, w, h)
        pygame.draw.rect(self.screen, (200, 200, 220), btn_rect, border_radius=8)

        btn_text = "显示预购" if not self.show_reserved_mode else "显示属性"
        self._draw_text(btn_text, (right - w + 32, 57), size=22)


        # ===== 卡牌区域 =====
        y = 100
        for level, cards in self.game.board.displayed_cards.items():

            # 牌堆背面 (Lx_C)
            deck_exists = len(self.game.board.card_decks[level]) > 0
            if deck_exists:
                self._draw_deck_back(level, (60, y))

            # 牌
            # self._draw_text(f"等级 {level} 卡牌:", (60, y - 30))

            x = 260  # 卡牌区域右移为给背面让位置
            for card in cards:
                self._draw_card(card, (x, y))
                x += 195

            y += 270

        card_area_right = x

        # ===== 宝石池横条 =====
        gems_x = 60
        gems_length = 800
        gems_y = self.screen.get_height() - 170
        self._draw_gem_pool((gems_x, gems_y), gems_length)

        # ===== 贵族区域，放在最底部 & 横向排列 =====
        noble_x = gems_x + gems_length + 40
        # self._draw_text("贵族：", (60, noble_y - 30))

        x = card_area_right + 50
        noble_y = 100
        for noble in self.game.board.nobles:
            self._draw_noble(noble, (x, noble_y))
            noble_y += 190


        # ===== 右侧代理栏（纵向堆叠）=====
        w = 550
        h = 200
        offset_y = 20
        base_x = self.screen.get_width() - 60 - w
        base_y = 100
        for idx, player in enumerate(self.game.players):
            highlight = (player.player_id == current_player.player_id)
            self._draw_player_info(player, base_x, base_y + idx * (offset_y + h), w, h, highlight)

        pygame.display.flip()



    # ======================== 渲染辅助函数 ========================
    def _draw_text(self, text, pos, color=(0,0,0), size=22):
        font = pygame.font.Font(FONT_PATH, size)
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, pos)

    def _draw_card(self, card, pos, scale=None):
        img = self.card_images.get(card.card_id)
        if img:
            if scale:
                scaled = pygame.transform.smoothscale(img, scale)
                self.screen.blit(scaled, pos)
            else:
                self.screen.blit(img, pos)
        else:
            pygame.draw.rect(self.screen, (220, 220, 220), (*pos,120,180))
            self._draw_text(f"ID:{card.card_id}", (pos[0]+10, pos[1]+10))

    def _draw_deck_back(self, level, pos):
        """绘制卡牌背面图 (Lx_C.png)"""
        key = f"L{level}_C"
        img = self.card_images.get(key)
        if img:
            self.screen.blit(img, pos)

    def _draw_noble(self, noble: Noble, pos):
        img = self.noble_images.get(noble.noble_id)
        if img:
            self.screen.blit(img, pos)
        else:
            pygame.draw.rect(self.screen, (255,230,200), (*pos,100,100))
            self._draw_text(f"Noble {noble.noble_id}", (pos[0]+5, pos[1]+5))

    def _draw_gem_pool(self, pos, length):
        """绘制剩余宝石数量横条"""
        x, y = pos
        pygame.draw.rect(self.screen, (245,245,245), (x, y, length, 80), border_radius=10)

        dx = 20
        for color, amount in self.game.board.gems.items():
            gem_img = self.gem_images.get(color.value)
            if gem_img:
                self.screen.blit(gem_img, (x + dx, y + 10))
            self._draw_text(str(amount), (x + dx + 80, y + 30))
            dx += 120

    def _draw_color_block(self, surface, color, discount, gem_count, x, y):
        """
        绘制单个颜色的 UI 小区域（80x80）
        color: GemColor
        discount: 折扣数字（永久宝石）
        gem_count: 该颜色宝石数量
        x, y: 左上角坐标
        """
        COLOR_THEME = {
            GemColor.WHITE: (220, 220, 220),
            GemColor.BLUE:  (90, 140, 230),
            GemColor.GREEN: (80, 180, 120),
            GemColor.RED:   (220, 80, 80),
            GemColor.BLACK: (60, 60, 60),
            GemColor.GOLD:  (230, 200, 70),   # 虽然金色不用于折扣，但备用
        }

        # 小区域尺寸
        w, h = 80, 80

        # 背板（略透明一点）
        pygame.draw.rect(surface, (250, 250, 250), (x, y, w, h), border_radius=8)

        # --- 左侧：折扣数字的框 ---
        pygame.draw.rect(surface, COLOR_THEME[color], (x + 5, y + 5, 35, 70), border_radius=6)
        
        # 折扣数字
        if discount is not None:
            self._draw_text(str(discount), (x + 16, y + 25), size=24)

        # --- 右侧：宝石数量数字 ---
        self._draw_text(str(gem_count), (x + 53, y + 10), size=22, color=(30,30,30))

        # --- 宝石图标 ---
        gem_img = self.gem_images[color.value]
        if gem_img:
            gem_img_scaled = pygame.transform.smoothscale(gem_img, (35, 35))
            surface.blit(gem_img_scaled, (x + 42, y + 40))

    def _draw_reserved_cards(self, player, x, y):
        """在玩家面板内绘制预留卡"""
        dx = x
        for card in player.reserved_cards:
            # 缩略图大小
            self._draw_card(card, (dx, y), scale=(82, 120))
            dx += 90


    def _draw_player_info(self, player: Player, x, y, w, h, highlight=False):
        bg = (180,230,255) if highlight else (230,230,230)
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=10)

        pad = 10

        # 名字 + 分数 + 预留数量
        reserved_count = len(player.reserved_cards)
        self._draw_text(
            f"{player.name} | 分数: {player.get_score()} | 预留: {reserved_count}",
            (x + pad, y + pad),
            size=24
        )

        # 如果当前模式是显示预留卡，画卡牌然后 return
        if self.show_reserved_mode:
            self._draw_reserved_cards(player, x + pad, y + 50)
            return

        # ======================
        # 正常模式：颜色方块显示
        # ======================
        colors = list(player.gems.keys())
        discounts = player.get_card_discounts()

        base_x = x + pad
        base_y = y + 50

        for idx, color in enumerate(colors):
            if color == GemColor.GOLD:
                discount = None
            else:
                discount = discounts[color]
            gem_count = player.gems[color]

            bx = base_x + (idx % 6) * 90
            by = base_y + (idx // 6) * 85

            self._draw_color_block(self.screen, color, discount, gem_count, bx, by)





