import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import pygame
from pygame.locals import QUIT

from agents.human_agent import HumanAgent
from game.card import GemColor
from game.game import Action, ActionType
from game.noble import Noble
from game.player import Player


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJK-Regular.otf")

CARD_SIZE = (175, 255)
CARD_SPACING_X = 195
CARD_SPACING_Y = 270
CARD_START_X = 260
DECK_POSITIONS = {1: (60, 100), 2: (60, 370), 3: (60, 640)}
PLAYER_PANEL_X_OFFSET = 60
PLAYER_PANEL_W = 550
PLAYER_PANEL_H = 200
PLAYER_PANEL_Y = 100
PLAYER_PANEL_GAP = 20
INTERACTION_PANEL_W = 550
INTERACTION_PANEL_H = 500
INTERACTION_PANEL_MARGIN_TOP = 540
BUTTON_HEIGHT = 32
BUTTON_GAP = 8
GRID_COLUMNS = 3
PAGE_SIZE = 30


def load_image(path, scale=None):
    """加载图片，可选缩放。"""
    if not os.path.exists(path):
        print(f"[警告] 贴图不存在: {path}")
        return None

    img = pygame.image.load(path).convert_alpha()
    if scale:
        img = pygame.transform.smoothscale(img, scale)
    return img


class PygameUI:
    def __init__(self, game, agents: Optional[List[Any]] = None, fps: int = 30, fullscreen: bool = False):
        pygame.init()
        self.game = game
        self.fps = fps
        self.fullscreen = fullscreen
        self.running = True
        self.lock = threading.Lock()
        self.agent_map = {agent.player_id: agent for agent in (agents or [])}

        self.windowed_size = (1920, 1080)
        self.screen = self._create_screen()
        pygame.display.set_caption("Splendor - 宝石商人")
        self.clock = pygame.time.Clock()

        bg_path = os.path.join(ASSETS_DIR, "splendor.png")
        self.background_source = load_image(bg_path)
        self.background = None
        self._refresh_background()

        self._load_card_images()
        self._load_noble_images()
        self._load_gem_images()

        self.show_reserved_mode = False
        self.status_message = ""
        self._active_request_id = None
        self._focused_action_options: List[Dict[str, Any]] = []
        self._pending_gem_selection: List[GemColor] = []
        self._reserve_mode = False
        self._draft_response: Any = None
        self._draft_label = ""
        self._button_targets: List[Tuple[pygame.Rect, Dict[str, Any]]] = []
        self._page_targets: Dict[str, pygame.Rect] = {}
        self._board_targets: Dict[str, List[Tuple[pygame.Rect, Dict[str, Any]]]] = {
            "display_cards": [],
            "deck_cards": [],
            "reserved_cards": [],
            "gems": [],
            "nobles": [],
        }
        self._current_page = 0
        self.layout: Dict[str, Any] = {}

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
                self.card_images[cid] = load_image(path, CARD_SIZE)

    def _load_noble_images(self):
        self.noble_images = {}
        noble_dir = os.path.join(ASSETS_DIR, "nobles")
        for fname in os.listdir(noble_dir):
            if fname.endswith(".png"):
                noble_id = fname[:-4]
                path = os.path.join(noble_dir, fname)
                self.noble_images[noble_id] = load_image(path, (170, 170))

    def _load_gem_images(self):
        self.gem_images = {}
        gem_dir = os.path.join(ASSETS_DIR, "gems")
        for gem_name in ["white", "blue", "green", "red", "black", "gold"]:
            path = os.path.join(gem_dir, f"{gem_name}.png")
            self.gem_images[gem_name] = load_image(path, (60, 60))

    def _create_screen(self):
        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode(self.windowed_size)

    def _refresh_background(self):
        if self.background_source is None:
            self.background = None
            return
        self.background = pygame.transform.smoothscale(self.background_source, self.screen.get_size())

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.screen = self._create_screen()
        self._refresh_background()

    def _clamp(self, value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    def _compute_layout(self) -> Dict[str, Any]:
        screen_w, screen_h = self.screen.get_size()
        scale = min(screen_w / 1920, screen_h / 1080)
        margin = self._clamp(int(min(screen_w, screen_h) * 0.02), 18, 36)
        top_bar_y = margin
        top_bar_h = self._clamp(int(screen_h * 0.055), 42, 64)
        top_button_h = self._clamp(int(40 * scale), 34, 46)
        top_button_gap = self._clamp(int(12 * scale), 8, 16)
        display_btn_w = self._clamp(int(150 * scale), 122, 170)
        fullscreen_btn_w = self._clamp(int(170 * scale), 140, 190)
        right_col_w = self._clamp(int(screen_w * 0.27), 320, 540)
        right_col_x = screen_w - margin - right_col_w
        interaction_h = self._clamp(int(screen_h * 0.34), 280, 420)
        interaction_y = screen_h - margin - interaction_h
        player_gap = self._clamp(int(screen_h * 0.014), 12, 24)
        player_y = top_bar_y + top_bar_h + margin
        player_available_h = max(180, interaction_y - player_y - margin)
        player_count = max(1, len(self.game.players))
        player_panel_h = max(140, int((player_available_h - player_gap * (player_count - 1)) / player_count))

        left_x = margin
        left_w = right_col_x - left_x - margin
        gems_h = self._clamp(int(screen_h * 0.095), 76, 104)
        gems_y = screen_h - margin - gems_h
        board_top = player_y
        board_bottom = gems_y - margin
        board_h = max(320, board_bottom - board_top)
        board_gap = self._clamp(int(screen_w * 0.01), 12, 22)
        noble_col_w = self._clamp(int(left_w * 0.15), 110, 180)
        cards_w = left_w - noble_col_w - board_gap

        row_gap = self._clamp(int(board_h * 0.03), 12, 24)
        card_gap = self._clamp(int(screen_w * 0.008), 10, 18)
        card_ratio = 175 / 255
        max_card_h_by_height = int((board_h - 2 * row_gap) / 3)
        max_card_w_by_width = int((cards_w - 4 * card_gap) / 5)
        card_h = max(120, min(max_card_h_by_height, int(max_card_w_by_width / card_ratio)))
        card_w = int(card_h * card_ratio)
        deck_x = left_x
        cards_x = deck_x + card_w + card_gap
        row_pitch = card_h + row_gap

        noble_count = max(1, len(self.game.board.nobles))
        noble_gap = self._clamp(int(board_h * 0.035), 12, 24)
        noble_size = min(
            noble_col_w,
            self._clamp(int((board_h - noble_gap * (noble_count - 1)) / noble_count), 100, 170),
        )
        noble_x = left_x + cards_w + board_gap + max(0, (noble_col_w - noble_size) // 2)

        panel_pad = self._clamp(int(right_col_w * 0.028), 10, 16)
        color_block_gap = self._clamp(int(right_col_w * 0.012), 6, 10)
        color_block_w = self._clamp(int((right_col_w - 2 * panel_pad - color_block_gap * 5) / 6), 42, 80)
        color_block_h = self._clamp(int(player_panel_h * 0.40), 54, 80)
        reserved_card_h = self._clamp(int(player_panel_h * 0.56), 84, 128)
        reserved_card_w = int(reserved_card_h * 82 / 120)
        reserved_gap = self._clamp(int(reserved_card_w * 0.1), 6, 12)
        gem_size = self._clamp(int(gems_h * 0.62), 42, 60)
        interaction_pad = self._clamp(int(right_col_w * 0.03), 14, 18)
        interaction_grid_columns = 3 if right_col_w >= 430 else 2

        return {
            "screen_w": screen_w,
            "screen_h": screen_h,
            "scale": scale,
            "margin": margin,
            "top_bar_y": top_bar_y,
            "top_bar_h": top_bar_h,
            "top_button_h": top_button_h,
            "top_button_gap": top_button_gap,
            "display_btn_w": display_btn_w,
            "fullscreen_btn_w": fullscreen_btn_w,
            "right_col_x": right_col_x,
            "right_col_w": right_col_w,
            "player_y": player_y,
            "player_gap": player_gap,
            "player_panel_h": player_panel_h,
            "interaction_rect": pygame.Rect(right_col_x, interaction_y, right_col_w, interaction_h),
            "left_x": left_x,
            "left_w": left_w,
            "board_top": board_top,
            "card_w": card_w,
            "card_h": card_h,
            "card_gap": card_gap,
            "row_gap": row_gap,
            "deck_x": deck_x,
            "cards_x": cards_x,
            "row_pitch": row_pitch,
            "cards_w": cards_w,
            "noble_x": noble_x,
            "noble_size": noble_size,
            "noble_gap": noble_gap,
            "gems_rect": pygame.Rect(left_x, gems_y, left_w, gems_h),
            "panel_pad": panel_pad,
            "color_block_gap": color_block_gap,
            "color_block_w": color_block_w,
            "color_block_h": color_block_h,
            "reserved_card_w": reserved_card_w,
            "reserved_card_h": reserved_card_h,
            "reserved_gap": reserved_gap,
            "gem_size": gem_size,
            "interaction_pad": interaction_pad,
            "interaction_grid_columns": interaction_grid_columns,
        }

    def run_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_click(event.pos, event.button)

            with self.lock:
                self.render()

            self.clock.tick(self.fps)

    def _handle_mouse_click(self, pos: Tuple[int, int], button: int = 1):
        mx, my = pos
        layout_buttons = self._get_top_button_rects()
        if layout_buttons["display_mode"].collidepoint(mx, my):
            self.show_reserved_mode = not self.show_reserved_mode
            return
        if layout_buttons["fullscreen"].collidepoint(mx, my):
            self.toggle_fullscreen()
            return

        request = self._get_pending_request()
        human_agent = self._get_current_human_agent()
        if request is None or human_agent is None:
            return

        if self._page_targets.get("prev") and self._page_targets["prev"].collidepoint(mx, my):
            self._current_page = max(0, self._current_page - 1)
            return
        if self._page_targets.get("next") and self._page_targets["next"].collidepoint(mx, my):
            self._current_page += 1
            return
        if self._page_targets.get("clear_focus") and self._page_targets["clear_focus"].collidepoint(mx, my):
            self._focused_action_options = []
            self._current_page = 0
            return
        if self._page_targets.get("clear_gems") and self._page_targets["clear_gems"].collidepoint(mx, my):
            self._pending_gem_selection = []
            self._clear_draft()
            self.status_message = "已清空宝石选择。"
            return
        if self._page_targets.get("toggle_reserve") and self._page_targets["toggle_reserve"].collidepoint(mx, my):
            self._reserve_mode = not self._reserve_mode
            self._pending_gem_selection = []
            self._clear_draft()
            self.status_message = "预留卡牌已开启。" if self._reserve_mode else "预留卡牌已关闭。"
            return
        if self._page_targets.get("confirm") and self._page_targets["confirm"].collidepoint(mx, my):
            if self._draft_response is not None:
                self._submit_response(human_agent, self._draft_response)
            return
        if self._page_targets.get("cancel") and self._page_targets["cancel"].collidepoint(mx, my):
            self._pending_gem_selection = []
            self._clear_draft()
            self.status_message = "已取消当前选择。"
            return

        response = self._resolve_board_click(pos, request, button)
        if response is not None:
            self._stage_response(request, response)
            return

        for rect, option in self._button_targets:
            if rect.collidepoint(mx, my):
                self._stage_response(request, option["response"], option.get("label", ""))
                return

    def _submit_response(self, human_agent: HumanAgent, response: Any):
        self.status_message = ""
        self._focused_action_options = []
        self._current_page = 0
        self._pending_gem_selection = []
        self._reserve_mode = False
        self._clear_draft()
        human_agent.submit_response(response)

    def _stage_response(self, request: Dict[str, Any], response: Any, label: str = ""):
        if not label:
            label = self._format_response_label(request, response)
        self._draft_response = response
        self._draft_label = label
        self.status_message = f"已选择：{label}"

    def _clear_draft(self):
        self._draft_response = None
        self._draft_label = ""

    def _get_current_human_agent(self) -> Optional[HumanAgent]:
        current_player = self.game.get_current_player()
        agent = self.agent_map.get(current_player.player_id)
        if isinstance(agent, HumanAgent):
            return agent
        return None

    def _get_pending_request(self) -> Optional[Dict[str, Any]]:
        human_agent = self._get_current_human_agent()
        if human_agent is None:
            return None
        request = human_agent.get_pending_request()
        self._sync_request_state(request)
        return request

    def _sync_request_state(self, request: Optional[Dict[str, Any]]):
        request_id = request.get("request_id") if request else None
        if request_id != self._active_request_id:
            self._active_request_id = request_id
            self._focused_action_options = []
            self._pending_gem_selection = []
            self._reserve_mode = False
            self._current_page = 0
            self._clear_draft()
            self.status_message = ""

    def render(self):
        self.layout = self._compute_layout()
        request = self._get_pending_request()
        self._button_targets = []
        self._page_targets = {}
        for key in self._board_targets:
            self._board_targets[key] = []

        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill((230, 235, 245))

        current_player = self.game.get_current_player()
        round_text = f"第 {self.game.round_number} 回合 | 当前玩家: {current_player.name}"
        self._draw_text(
            round_text,
            (self.layout["margin"], self.layout["top_bar_y"] + 6),
            (255, 255, 255),
            28,
        )

        layout_buttons = self._get_top_button_rects()

        toggle_rect = layout_buttons["display_mode"]
        pygame.draw.rect(self.screen, (200, 200, 220), toggle_rect, border_radius=8)
        btn_text = "显示预购" if not self.show_reserved_mode else "显示属性"
        self._draw_text(btn_text, (toggle_rect.x + 18, toggle_rect.y + 7), size=22)

        fullscreen_rect = layout_buttons["fullscreen"]
        fullscreen_bg = (76, 117, 173) if self.fullscreen else (200, 200, 220)
        fullscreen_fg = (255, 255, 255) if self.fullscreen else (0, 0, 0)
        pygame.draw.rect(self.screen, fullscreen_bg, fullscreen_rect, border_radius=8)
        fullscreen_text = "退出全屏" if self.fullscreen else "进入全屏"
        self._draw_text(fullscreen_text, (fullscreen_rect.x + 18, fullscreen_rect.y + 7), size=22, color=fullscreen_fg)

        for level in sorted(self.game.board.displayed_cards.keys()):
            cards = self.game.board.displayed_cards[level]
            y = self.layout["board_top"] + (level - 1) * self.layout["row_pitch"]
            deck_exists = len(self.game.board.card_decks[level]) > 0
            if deck_exists:
                deck_rect = pygame.Rect(
                    self.layout["deck_x"],
                    y,
                    self.layout["card_w"],
                    self.layout["card_h"],
                )
                clickable = self._has_matching_action(request, lambda act, lvl=level: self._is_deck_reserve_action(act, lvl))
                self._draw_deck_back(level, (deck_rect.x, deck_rect.y), highlight=clickable)
                self._board_targets["deck_cards"].append((deck_rect, {"level": level}))

            x = self.layout["cards_x"]
            for card in cards:
                card_rect = pygame.Rect(x, y, self.layout["card_w"], self.layout["card_h"])
                clickable = self._has_matching_action(
                    request,
                    lambda act, lvl=level, cid=card.card_id: self._is_display_card_action(act, lvl, cid),
                )
                self._draw_card(card, (x, y), scale=(self.layout["card_w"], self.layout["card_h"]), highlight=clickable)
                self._board_targets["display_cards"].append((card_rect, {"level": level, "card_id": card.card_id}))
                x += self.layout["card_w"] + self.layout["card_gap"]

        self._draw_gem_pool(self.layout["gems_rect"], request)

        noble_y = self.layout["board_top"]
        for noble in self.game.board.nobles:
            noble_rect = pygame.Rect(
                self.layout["noble_x"],
                noble_y,
                self.layout["noble_size"],
                self.layout["noble_size"],
            )
            highlight = request is not None and request.get("type") == "select_noble"
            self._draw_noble(
                noble,
                (noble_rect.x, noble_rect.y),
                size=(self.layout["noble_size"], self.layout["noble_size"]),
                highlight=highlight,
            )
            self._board_targets["nobles"].append((noble_rect, {"noble_id": noble.noble_id}))
            noble_y += self.layout["noble_size"] + self.layout["noble_gap"]

        base_x = self.layout["right_col_x"]
        base_y = self.layout["player_y"]
        for idx, player in enumerate(self.game.players):
            highlight = player.player_id == current_player.player_id
            self._draw_player_info(
                player,
                base_x,
                base_y + idx * (self.layout["player_gap"] + self.layout["player_panel_h"]),
                self.layout["right_col_w"],
                self.layout["player_panel_h"],
                request,
                highlight,
            )

        self._draw_interaction_panel(request)
        pygame.display.flip()

    def _get_top_button_rects(self) -> Dict[str, pygame.Rect]:
        layout = self.layout or self._compute_layout()
        right = layout["screen_w"] - layout["margin"]
        top = layout["top_bar_y"]
        button_h = layout["top_button_h"]
        gap = layout["top_button_gap"]
        display_w = layout["display_btn_w"]
        fullscreen_w = layout["fullscreen_btn_w"]
        display_rect = pygame.Rect(right - display_w, top, display_w, button_h)
        fullscreen_rect = pygame.Rect(display_rect.x - gap - fullscreen_w, top, fullscreen_w, button_h)
        return {
            "display_mode": display_rect,
            "fullscreen": fullscreen_rect,
        }

    def _draw_text(self, text, pos, color=(0, 0, 0), size=22):
        scale = self.layout.get("scale", 1.0) if self.layout else 1.0
        min_size = max(12, int(size * 0.82))
        max_size = max(size, int(size * 1.15))
        final_size = max(min_size, min(int(size * scale), max_size))
        font = pygame.font.Font(FONT_PATH, final_size)
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, pos)

    def _draw_card(self, card, pos, scale=None, highlight=False):
        img = self.card_images.get(card.card_id)
        width, height = scale or CARD_SIZE
        if img:
            surface = pygame.transform.smoothscale(img, scale) if scale else img
            self.screen.blit(surface, pos)
        else:
            pygame.draw.rect(self.screen, (220, 220, 220), (*pos, width, height))
            self._draw_text(f"ID:{card.card_id}", (pos[0] + 10, pos[1] + 10))
        if highlight:
            pygame.draw.rect(self.screen, (255, 220, 60), (*pos, width, height), width=4, border_radius=10)

    def _draw_deck_back(self, level, pos, highlight=False):
        key = f"L{level}_C"
        img = self.card_images.get(key)
        card_size = (self.layout["card_w"], self.layout["card_h"]) if self.layout else CARD_SIZE
        if img:
            surface = pygame.transform.smoothscale(img, card_size) if img.get_size() != card_size else img
            self.screen.blit(surface, pos)
        if highlight:
            pygame.draw.rect(self.screen, (255, 220, 60), (*pos, card_size[0], card_size[1]), width=4, border_radius=10)

    def _draw_noble(self, noble: Noble, pos, size=None, highlight=False):
        img = self.noble_images.get(noble.noble_id)
        width, height = size or (170, 170)
        if img:
            surface = pygame.transform.smoothscale(img, (width, height)) if img.get_size() != (width, height) else img
            self.screen.blit(surface, pos)
        else:
            pygame.draw.rect(self.screen, (255, 230, 200), (*pos, width, height))
            self._draw_text(f"Noble {noble.noble_id}", (pos[0] + 5, pos[1] + 5))
        if highlight:
            pygame.draw.rect(self.screen, (255, 220, 60), (*pos, width, height), width=4, border_radius=10)

    def _draw_gem_pool(self, rect: pygame.Rect, request):
        x, y, w, h = rect
        pygame.draw.rect(self.screen, (245, 245, 245), rect, border_radius=10)
        pygame.draw.rect(self.screen, (210, 214, 224), rect, width=2, border_radius=10)

        gem_items = list(self.game.board.gems.items())
        if not gem_items:
            return
        slot_w = w / len(gem_items)
        gem_size = min(self.layout.get("gem_size", 60), int(h - 20), int(slot_w - 26))

        for idx, (color, amount) in enumerate(gem_items):
            slot_x = int(x + idx * slot_w)
            gem_rect = pygame.Rect(slot_x + 4, y + 8, max(72, int(slot_w) - 8), h - 16)
            clickable = self._has_matching_action(
                request,
                lambda act, gem_color=color: self._is_gem_action(act, gem_color),
            )
            if clickable:
                pygame.draw.rect(self.screen, (255, 220, 60), gem_rect, width=3, border_radius=10)

            gem_img = self.gem_images.get(color.value)
            gem_x = gem_rect.x + 10
            gem_y = y + max(8, (h - gem_size) // 2)
            if gem_img:
                scaled = pygame.transform.smoothscale(gem_img, (gem_size, gem_size)) if gem_img.get_size() != (gem_size, gem_size) else gem_img
                self.screen.blit(scaled, (gem_x, gem_y))
            self._draw_text(str(amount), (gem_x + gem_size + 8, gem_y + max(4, gem_size // 4)), size=22)

            self._board_targets["gems"].append((gem_rect, {"color": color}))

    def _draw_color_block(self, surface, color, discount, gem_count, x, y, size: Tuple[int, int]):
        color_theme = {
            GemColor.WHITE: (220, 220, 220),
            GemColor.BLUE: (90, 140, 230),
            GemColor.GREEN: (80, 180, 120),
            GemColor.RED: (220, 80, 80),
            GemColor.BLACK: (60, 60, 60),
            GemColor.GOLD: (230, 200, 70),
        }

        block_w, block_h = size
        accent_w = max(18, int(block_w * 0.42))
        pygame.draw.rect(surface, (250, 250, 250), (x, y, block_w, block_h), border_radius=8)
        pygame.draw.rect(surface, color_theme[color], (x + 5, y + 5, accent_w, block_h - 10), border_radius=6)
        if discount is not None:
            self._draw_text(str(discount), (x + 11, y + max(8, block_h // 4)), size=22)
        self._draw_text(str(gem_count), (x + accent_w + 12, y + 8), size=18, color=(30, 30, 30))
        gem_img = self.gem_images[color.value]
        if gem_img:
            gem_dim = max(20, min(35, block_h - 42))
            gem_img_scaled = pygame.transform.smoothscale(gem_img, (gem_dim, gem_dim))
            surface.blit(gem_img_scaled, (x + accent_w + 8, y + block_h - gem_dim - 8))

    def _draw_reserved_cards(self, player, x, y, max_width, request):
        dx = x
        current_player = self.game.get_current_player()
        scale_h = self.layout.get("reserved_card_h", 120)
        scale_w = self.layout.get("reserved_card_w", 82)
        gap = self.layout.get("reserved_gap", 8)
        card_count = max(1, len(player.reserved_cards))
        if scale_w * card_count + gap * (card_count - 1) > max_width:
            scale_w = max(54, int((max_width - gap * (card_count - 1)) / card_count))
            scale_h = int(scale_w * 120 / 82)
        for idx, card in enumerate(player.reserved_cards):
            scale = (scale_w, scale_h)
            rect = pygame.Rect(dx, y, scale[0], scale[1])
            clickable = (
                request is not None
                and player.player_id == current_player.player_id
                and self._has_matching_action(
                    request,
                    lambda act, cid=card.card_id: self._is_reserved_buy_action(act, cid),
                )
            )
            self._draw_card(card, (dx, y), scale=scale, highlight=clickable)
            if clickable:
                self._board_targets["reserved_cards"].append((rect, {"card_id": card.card_id, "index": idx}))
            dx += scale_w + gap

    def _draw_player_info(self, player: Player, x, y, w, h, request, highlight=False):
        bg = (180, 230, 255) if highlight else (230, 230, 230)
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=10)

        pad = self.layout.get("panel_pad", 10)
        reserved_count = len(player.reserved_cards)
        self._draw_text(
            f"{player.name} | 分数: {player.get_score()} | 预留: {reserved_count}",
            (x + pad, y + pad),
            size=24,
        )

        if self.show_reserved_mode:
            self._draw_reserved_cards(player, x + pad, y + 48, w - 2 * pad, request)
            return

        discounts = player.get_card_discounts()
        base_x = x + pad
        base_y = y + 48
        block_w = self.layout.get("color_block_w", 80)
        block_h = self.layout.get("color_block_h", 80)
        block_gap = self.layout.get("color_block_gap", 8)

        for idx, color in enumerate(player.gems.keys()):
            discount = None if color == GemColor.GOLD else discounts[color]
            gem_count = player.gems[color]
            bx = base_x + (idx % 6) * (block_w + block_gap)
            by = base_y + (idx // 6) * (block_h + 6)
            self._draw_color_block(self.screen, color, discount, gem_count, bx, by, (block_w, block_h))

    def _draw_interaction_panel(self, request: Optional[Dict[str, Any]]):
        panel_rect = self.layout["interaction_rect"]
        x, y, panel_w, panel_h = panel_rect
        pad = self.layout.get("interaction_pad", 18)
        pygame.draw.rect(self.screen, (245, 246, 252), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (90, 100, 140), panel_rect, width=2, border_radius=12)

        title = "当前回合"
        subtitle = "AI 行动中"
        options: List[Dict[str, Any]] = []

        if request is not None:
            title, subtitle, options = self._build_request_view(request)

        self._draw_text(title, (x + pad, y + 16), size=26)
        self._draw_text(subtitle, (x + pad, y + 52), size=18, color=(80, 80, 90))

        if self.status_message:
            self._draw_text(
                self._ellipsis(self.status_message, 42),
                (x + pad, y + 80),
                size=16,
                color=(170, 70, 40),
            )

        content_top = y + 110
        if request is None:
            self._draw_text("等待当前 AI 回合结束，或等待逻辑线程发起真人输入。", (x + pad, content_top), size=18)
            return

        if request.get("type") == "action":
            tips = [
                "左键展示卡：购买这张卡",
                "点“预留卡牌”后，再点卡牌或牌堆：执行预留",
                "左键预留卡：购买自己的预留卡",
                "左键宝石：按点击顺序组合拿取，选好后点确认",
            ]
            line_step = self._clamp(int(panel_h * 0.06), 20, 28)
            for idx, tip in enumerate(tips):
                self._draw_text(tip, (x + pad, content_top + idx * line_step), size=17, color=(55, 60, 78))

            reserve_y = content_top + len(tips) * line_step + 12
            reserve_rect = pygame.Rect(x + pad, reserve_y, 148, 34)
            reserve_bg = (193, 135, 43) if self._reserve_mode else (224, 229, 245)
            pygame.draw.rect(self.screen, reserve_bg, reserve_rect, border_radius=8)
            reserve_text_color = (255, 255, 255) if self._reserve_mode else (0, 0, 0)
            self._draw_text("预留卡牌: 开" if self._reserve_mode else "预留卡牌: 关", (reserve_rect.x + 10, reserve_rect.y + 6), size=18, color=reserve_text_color)
            self._page_targets["toggle_reserve"] = reserve_rect

            if self._pending_gem_selection:
                gems_text = " ".join(self._color_name(color.value) for color in self._pending_gem_selection)
                gems_y = reserve_y + 48
                self._draw_text(f"当前宝石选择：{gems_text}", (x + pad, gems_y), size=19, color=(45, 80, 130))
                clear_rect = pygame.Rect(x + pad, gems_y + 36, 130, 34)
                pygame.draw.rect(self.screen, (224, 229, 245), clear_rect, border_radius=8)
                self._draw_text("清空宝石选择", (clear_rect.x + 12, clear_rect.y + 6), size=18)
                self._page_targets["clear_gems"] = clear_rect

            self._draw_confirm_row(x + pad, y + panel_h - 62, request)

            return

        if self._focused_action_options:
            clear_rect = pygame.Rect(x + panel_w - 128, y + 14, 108, 34)
            pygame.draw.rect(self.screen, (224, 229, 245), clear_rect, border_radius=8)
            self._draw_text("查看全部", (clear_rect.x + 17, clear_rect.y + 6), size=18)
            self._page_targets["clear_focus"] = clear_rect

        total_pages = max(1, (len(options) + PAGE_SIZE - 1) // PAGE_SIZE)
        self._current_page = min(self._current_page, total_pages - 1)

        if total_pages > 1:
            page_text = f"{self._current_page + 1}/{total_pages}"
            self._draw_text(page_text, (x + panel_w - 170, y + 54), size=18)

            prev_rect = pygame.Rect(x + panel_w - 120, y + 50, 40, 30)
            next_rect = pygame.Rect(x + panel_w - 70, y + 50, 40, 30)
            pygame.draw.rect(self.screen, (224, 229, 245), prev_rect, border_radius=8)
            pygame.draw.rect(self.screen, (224, 229, 245), next_rect, border_radius=8)
            self._draw_text("<", (prev_rect.x + 13, prev_rect.y + 2), size=22)
            self._draw_text(">", (next_rect.x + 13, next_rect.y + 2), size=22)
            self._page_targets["prev"] = prev_rect
            self._page_targets["next"] = next_rect

        visible_options = options[self._current_page * PAGE_SIZE:(self._current_page + 1) * PAGE_SIZE]
        grid_columns = self.layout.get("interaction_grid_columns", GRID_COLUMNS)
        col_w = (panel_w - 2 * pad - (grid_columns - 1) * 10) // grid_columns

        for idx, option in enumerate(visible_options):
            col = idx % grid_columns
            row = idx // grid_columns
            btn_x = x + pad + col * (col_w + 10)
            btn_y = content_top + row * (BUTTON_HEIGHT + BUTTON_GAP)
            rect = pygame.Rect(btn_x, btn_y, col_w, BUTTON_HEIGHT)
            pygame.draw.rect(self.screen, (221, 231, 255), rect, border_radius=8)
            pygame.draw.rect(self.screen, (120, 140, 190), rect, width=1, border_radius=8)
            self._draw_text(self._ellipsis(option["label"], 12), (btn_x + 8, btn_y + 6), size=16)
            self._button_targets.append((rect, option))

        self._draw_confirm_row(x + pad, y + panel_h - 62, request)

    def _draw_confirm_row(self, x: int, y: int, request: Dict[str, Any]):
        if self._draft_label:
            self._draw_text(f"待确认：{self._ellipsis(self._draft_label, 26)}", (x, y - 28), size=18, color=(45, 80, 130))
        else:
            self._draw_text("待确认：未选择", (x, y - 28), size=18, color=(110, 110, 120))

        confirm_rect = pygame.Rect(x, y, 110, 36)
        cancel_rect = pygame.Rect(x + 124, y, 110, 36)
        confirm_bg = (78, 154, 96) if self._draft_response is not None else (190, 202, 194)
        confirm_text = (255, 255, 255) if self._draft_response is not None else (235, 235, 235)
        pygame.draw.rect(self.screen, confirm_bg, confirm_rect, border_radius=8)
        pygame.draw.rect(self.screen, (224, 229, 245), cancel_rect, border_radius=8)
        self._draw_text("确认", (confirm_rect.x + 34, confirm_rect.y + 6), size=20, color=confirm_text)
        self._draw_text("取消", (cancel_rect.x + 34, cancel_rect.y + 6), size=20)
        self._page_targets["confirm"] = confirm_rect
        self._page_targets["cancel"] = cancel_rect

    def _build_request_view(self, request: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
        request_type = request["type"]
        if request_type == "action":
            subtitle = "点击棋盘选择操作，选好后点确认"
            return "请直接点击棋盘", subtitle, []

        if request_type == "discard_gems":
            gems = request.get("gems", {})
            num_to_discard = int(request.get("num_to_discard", 0))
            options = self._build_discard_options(gems, num_to_discard)
            subtitle = f"需要弃掉 {num_to_discard} 枚宝石"
            return "请选择弃牌方案", subtitle, options

        if request_type == "select_noble":
            options = [
                {"label": self._format_noble_label(noble), "response": noble["id"]}
                for noble in request.get("available_nobles", [])
            ]
            return "请选择贵族", "点击贵族图块，或从下方面板选择", options

        return "等待输入", "", []

    def _resolve_board_click(self, pos: Tuple[int, int], request: Dict[str, Any], button: int) -> Optional[Any]:
        request_type = request["type"]

        if request_type == "select_noble":
            for rect, payload in self._board_targets["nobles"]:
                if rect.collidepoint(pos):
                    return payload["noble_id"]
            return None

        if request_type != "action":
            return None

        valid_actions = request.get("valid_actions", [])

        for rect, payload in self._board_targets["display_cards"]:
            if rect.collidepoint(pos):
                return self._resolve_display_card_click(valid_actions, payload["level"], payload["card_id"], button)

        for rect, payload in self._board_targets["deck_cards"]:
            if rect.collidepoint(pos):
                if not self._reserve_mode:
                    self.status_message = "如需预留牌堆，请先打开预留卡牌。"
                    return None
                matches = [
                    action
                    for action in valid_actions
                    if self._is_deck_reserve_action(action, payload["level"])
                ]
                if matches:
                    self._pending_gem_selection = []
                    return matches[0]
                self.status_message = "这个牌堆当前不能预留。"
                return None

        for rect, payload in self._board_targets["reserved_cards"]:
            if rect.collidepoint(pos):
                matches = [
                    action
                    for action in valid_actions
                    if self._is_reserved_buy_action(action, payload["card_id"])
                ]
                if matches:
                    self._pending_gem_selection = []
                    return matches[0]
                self.status_message = "这张预留卡当前还买不起。"
                return None

        for rect, payload in self._board_targets["gems"]:
            if rect.collidepoint(pos):
                if button != 1:
                    self.status_message = "宝石只需要左键点击。"
                    return None
                return self._resolve_gem_click(valid_actions, payload["color"])

        return None

    def _build_discard_options(self, gems: Dict[str, int], num_to_discard: int) -> List[Dict[str, Any]]:
        colors = [color for color in sorted(gems.keys()) if gems[color] > 0]
        results: List[Dict[str, int]] = []

        def backtrack(index: int, remaining: int, current: Dict[str, int]):
            if remaining == 0:
                results.append(dict(current))
                return
            if index >= len(colors):
                return

            color = colors[index]
            max_take = min(gems[color], remaining)
            for count in range(max_take, -1, -1):
                if count > 0:
                    current[color] = count
                elif color in current:
                    current.pop(color)
                backtrack(index + 1, remaining - count, current)
            current.pop(color, None)

        backtrack(0, num_to_discard, {})
        options = [{"label": self._format_discard_label(choice), "response": choice} for choice in results]
        options.sort(key=lambda item: item["label"])
        return options

    def _resolve_display_card_click(
        self,
        valid_actions: List[Action],
        level: int,
        card_id: str,
        button: int,
    ) -> Optional[Action]:
        buy_action = next(
            (
                action
                for action in valid_actions
                if action.action_type == ActionType.BUY_CARD
                and action.params.get("level") == level
                and action.params.get("card_id") == card_id
            ),
            None,
        )
        reserve_action = next(
            (
                action
                for action in valid_actions
                if action.action_type == ActionType.RESERVE_CARD
                and action.params.get("level") == level
                and action.params.get("card_id") == card_id
                and not action.params.get("from_deck")
            ),
            None,
        )

        self._pending_gem_selection = []
        if self._reserve_mode:
            if reserve_action is not None:
                return reserve_action
            self.status_message = "这张卡现在不能预留。"
            return None

        if buy_action is not None:
            return buy_action

        if reserve_action is not None:
            self.status_message = "当前买不起这张卡；如需预留，请先打开预留卡牌。"
            return None

        self.status_message = "这张卡当前没有可执行动作。"
        return None

    def _resolve_gem_click(self, valid_actions: List[Action], color: GemColor) -> Optional[Action]:
        gem_actions = [
            action
            for action in valid_actions
            if action.action_type in (ActionType.TAKE_DIFFERENT_GEMS, ActionType.TAKE_SAME_GEMS)
        ]
        if not gem_actions:
            self.status_message = "当前没有可用的拿宝石动作。"
            return None

        candidate = self._pending_gem_selection + [color]
        if not self._selection_is_valid_prefix(candidate, gem_actions):
            fallback = [color]
            if self._selection_is_valid_prefix(fallback, gem_actions):
                candidate = fallback
            else:
                self.status_message = f"{self._color_name(color.value)} 当前不能这样拿。"
                self._pending_gem_selection = []
                return None

        self._pending_gem_selection = candidate
        exact_matches = [action for action in gem_actions if self._selection_exact_matches_action(candidate, action)]
        extendable = [
            action
            for action in gem_actions
            if self._selection_is_prefix_of_action(candidate, action)
            and not self._selection_exact_matches_action(candidate, action)
        ]

        if len(exact_matches) == 1 and not extendable:
            chosen_action = exact_matches[0]
            self.status_message = f"已选择：{self._format_action_label(chosen_action)}"
            return chosen_action

        selected_text = " ".join(self._color_name(item.value) for item in self._pending_gem_selection)
        self.status_message = f"当前宝石选择：{selected_text}"
        return None

    def _selection_is_valid_prefix(self, selection: List[GemColor], actions: List[Action]) -> bool:
        return any(self._selection_is_prefix_of_action(selection, action) for action in actions)

    def _selection_is_prefix_of_action(self, selection: List[GemColor], action: Action) -> bool:
        selection_counts = self._gem_color_counts(selection)
        action_counts = self._action_gem_counts(action)
        if selection_counts is None or action_counts is None:
            return False
        for color, count in selection_counts.items():
            if action_counts.get(color, 0) < count:
                return False
        return True

    def _selection_exact_matches_action(self, selection: List[GemColor], action: Action) -> bool:
        selection_counts = self._gem_color_counts(selection)
        action_counts = self._action_gem_counts(action)
        return selection_counts is not None and action_counts is not None and selection_counts == action_counts

    def _gem_color_counts(self, colors: List[GemColor]) -> Optional[Dict[GemColor, int]]:
        counts: Dict[GemColor, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
            if counts[color] > 2:
                return None
        return counts

    def _action_gem_counts(self, action: Action) -> Optional[Dict[GemColor, int]]:
        if action.action_type == ActionType.TAKE_SAME_GEMS:
            color = action.params.get("color")
            if color is None:
                return None
            return {color: 2}
        if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            colors = action.params.get("colors", [])
            counts: Dict[GemColor, int] = {}
            for color in colors:
                counts[color] = counts.get(color, 0) + 1
            return counts
        return None

    def _format_discard_label(self, discarded: Dict[str, int]) -> str:
        parts = [f"{self._color_name(color)}x{count}" for color, count in discarded.items() if count > 0]
        return "弃 " + " ".join(parts)

    def _format_noble_label(self, noble: Dict[str, Any]) -> str:
        return f"{noble['id']} ({noble['points']}分)"

    def _format_response_label(self, request: Dict[str, Any], response: Any) -> str:
        if request["type"] == "action" and isinstance(response, Action):
            return self._format_action_label(response)
        if request["type"] == "discard_gems" and isinstance(response, dict):
            return self._format_discard_label(response)
        if request["type"] == "select_noble":
            noble = next(
                (item for item in request.get("available_nobles", []) if item.get("id") == response),
                None,
            )
            if noble is not None:
                return self._format_noble_label(noble)
            return str(response)
        return str(response)

    def _format_action_label(self, action: Action) -> str:
        if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            colors = "".join(self._color_name(color.value if isinstance(color, GemColor) else color) for color in action.params.get("colors", []))
            return f"拿三色 {colors}"
        if action.action_type == ActionType.TAKE_SAME_GEMS:
            color = action.params.get("color")
            color_name = self._color_name(color.value if isinstance(color, GemColor) else color)
            return f"拿双色 {color_name}{color_name}"
        if action.action_type == ActionType.RESERVE_CARD:
            level = action.params.get("level", "?")
            if action.params.get("from_deck"):
                return f"预留牌堆 L{level}"
            return f"预留 {action.params.get('card_id', 'Unknown')}"
        if action.action_type == ActionType.BUY_CARD:
            return f"购买 {action.params.get('card_id', 'Unknown')}"
        if action.action_type == ActionType.BUY_RESERVED_CARD:
            return f"购买预留 {action.params.get('card_id', 'Unknown')}"
        return str(action)

    def _color_name(self, color: str) -> str:
        mapping = {
            "white": "白",
            "blue": "蓝",
            "green": "绿",
            "red": "红",
            "black": "黑",
            "gold": "金",
        }
        return mapping.get(color, str(color))

    def _ellipsis(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    def _has_matching_action(self, request: Optional[Dict[str, Any]], predicate) -> bool:
        if request is None or request.get("type") != "action":
            return False
        return any(predicate(action) for action in request.get("valid_actions", []))

    def _is_display_card_action(self, action: Action, level: int, card_id: str) -> bool:
        return (
            action.action_type in (ActionType.BUY_CARD, ActionType.RESERVE_CARD)
            and action.params.get("level") == level
            and action.params.get("card_id") == card_id
        )

    def _is_deck_reserve_action(self, action: Action, level: int) -> bool:
        return (
            action.action_type == ActionType.RESERVE_CARD
            and action.params.get("level") == level
            and bool(action.params.get("from_deck"))
        )

    def _is_reserved_buy_action(self, action: Action, card_id: str) -> bool:
        return action.action_type == ActionType.BUY_RESERVED_CARD and action.params.get("card_id") == card_id

    def _is_gem_action(self, action: Action, color: GemColor) -> bool:
        if action.action_type == ActionType.TAKE_SAME_GEMS:
            return action.params.get("color") == color
        if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            return color in action.params.get("colors", [])
        return False
