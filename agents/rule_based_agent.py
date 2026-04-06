import json
import os
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from game.game import Action, ActionType
from utils.log import CustomLogger


class RuleBasedAgent(BaseAgent):
    """基于手工规则与启发式评分的代理。"""

    def __init__(
        self,
        player_id: str,
        name: str,
        candidate_action_limit: int = 6,
        target_limit: int = 4,
        noble_limit: int = 3,
        run_id: Optional[str] = None,
    ):
        super().__init__(player_id, name)
        self.candidate_action_limit = max(3, int(candidate_action_limit))
        self.target_limit = max(2, int(target_limit))
        self.noble_limit = max(1, int(noble_limit))
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.game_history: List[Dict[str, Any]] = []
        self._latest_summary: Optional[Dict[str, Any]] = None
        self._latest_ranked_actions: List[Dict[str, Any]] = []

        log_dir = os.path.join(".", "log", "llm_agent_runs")
        os.makedirs(log_dir, exist_ok=True)
        safe_player_id = self.player_id.replace("/", "_").replace(" ", "_")
        self.log_file_path = os.path.join(
            log_dir,
            f"{self.run_id}_{safe_player_id}_rank_v2_auto.log"
        )
        self.logger = CustomLogger(self.log_file_path)
        self.logger.enable_file_logging()
        self.logger.log_info(
            {
                "agent_name": self.name,
                "strategy": "rank_v2_auto",
                "candidate_action_limit": self.candidate_action_limit,
                "target_limit": self.target_limit,
                "noble_limit": self.noble_limit,
            },
            is_print=False
        )

    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Action:
        self.logger.log_info("\n====== [DEBUG] ENTER select_action ======", is_print=False)
        self.logger.log_info(
            {
                "round": game_state.get("round"),
                "current_player": game_state.get("current_player"),
                "valid_action_count": len(valid_actions),
                "valid_actions": [str(action) for action in valid_actions],
            },
            is_print=False
        )
        summary = self._summarize_state(game_state, valid_actions)
        self._latest_summary = summary
        self.logger.log_info({"rule_summary_preview": json.dumps(summary, ensure_ascii=False)[:1200]}, is_print=False)

        ranked_actions = self._latest_ranked_actions or []
        if ranked_actions:
            best_index = ranked_actions[0]["index"] - 1
            if 0 <= best_index < len(valid_actions):
                selected_action = valid_actions[best_index]
                self.logger.log_info(
                    {
                        "selection_mode": "rule_based_rank_v2_auto",
                        "chosen_action": str(selected_action),
                    },
                    is_print=False
                )
                self.logger.log_info("====== [DEBUG] EXIT select_action ======\n", is_print=False)
                return selected_action

        selected_action = valid_actions[0]
        self.logger.log_warning(
            {"fallback": "first_valid_action", "chosen_action": str(selected_action)},
            is_print=False
        )
        self.logger.log_info("====== [DEBUG] EXIT select_action ======\n", is_print=False)
        return selected_action

    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """优先丢弃数量最多且非黄金的宝石。"""
        discarded: Dict[str, int] = {}
        remaining = dict(gems)
        colors = sorted(
            [color for color, count in remaining.items() if count > 0 and color != "gold"],
            key=lambda color: (-remaining[color], color),
        )
        while num_to_discard > 0 and colors:
            color = colors[0]
            discarded[color] = discarded.get(color, 0) + 1
            remaining[color] -= 1
            num_to_discard -= 1
            colors = sorted(
                [c for c, count in remaining.items() if count > 0 and c != "gold"],
                key=lambda c: (-remaining[c], c),
            )
        return discarded

    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        if not available_nobles:
            return ""
        return sorted(available_nobles, key=lambda noble: noble.get("id", ""))[0]["id"]

    def _format_compact_gems(self, gems: Dict[str, int]) -> Dict[str, int]:
        return {color: count for color, count in gems.items() if count}

    def _link_targets_to_nobles(
        self,
        targets: List[Dict[str, Any]],
        nobles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        linked_targets = []
        for target in targets:
            linked = dict(target)
            linked["noble_links"] = [
                noble["id"]
                for noble in nobles
                if target.get("color") in noble.get("missing", {})
            ][:2]
            linked_targets.append(linked)
        return linked_targets

    def _get_phase(self, current_player: Dict[str, Any]) -> str:
        score = current_player.get("score", 0)
        discounts = current_player.get("card_discounts", {})
        total_discounts = sum(discounts.values())
        if score >= 10 or total_discounts >= 8:
            return "late"
        if score >= 5 or total_discounts >= 4:
            return "mid"
        return "early"

    def _build_card_target_info(
        self,
        card: Dict[str, Any],
        current_gems: Dict[str, int],
        current_discounts: Dict[str, int],
        current_gold: int,
    ) -> Dict[str, Any]:
        gap = {}
        total_missing = 0
        color_coverage = []
        for color, cost in card.get("cost", {}).items():
            owned = current_gems.get(color, 0) + current_discounts.get(color, 0)
            missing = max(0, cost - owned)
            if missing:
                gap[color] = missing
                total_missing += missing
                color_coverage.append(color)

        return {
            "id": card.get("id"),
            "level": card.get("level"),
            "points": card.get("points"),
            "color": card.get("color"),
            "cost": dict(card.get("cost", {})),
            "missing": gap,
            "missing_total": total_missing,
            "buyable_now": total_missing <= current_gold,
            "needed_colors": color_coverage,
        }

    def _build_noble_info(self, noble: Dict[str, Any], discounts: Dict[str, int]) -> Dict[str, Any]:
        gap = {}
        total_missing = 0
        for color, requirement in noble.get("requirements", {}).items():
            owned_discount = discounts.get(color, 0)
            missing = max(0, requirement - owned_discount)
            if missing:
                gap[color] = missing
                total_missing += missing
        return {
            "id": noble.get("id"),
            "points": noble.get("points"),
            "missing": gap,
            "missing_total": total_missing,
            "needed_colors": list(gap.keys()),
        }

    def _score_target_priority(
        self,
        target: Dict[str, Any],
        phase: str,
        nearest_nobles: List[Dict[str, Any]],
    ) -> float:
        priority = 0.0
        points = target.get("points", 0)
        level = target.get("level", 1)
        missing_total = target.get("missing_total", 0)
        noble_links = len(
            [
                noble for noble in nearest_nobles
                if target.get("color") in noble.get("missing", {})
            ]
        )

        priority += points * 7.0
        priority += noble_links * 2.5
        priority -= missing_total * 1.8

        if target.get("buyable_now"):
            priority += 14.0
        if phase == "early":
            if level == 1:
                priority += 2.5
            if points == 0 and level == 1:
                priority += 1.5
            if points >= 1 and level == 1:
                priority += 2.0
            if level == 3:
                priority -= 7.0
            elif level == 2 and missing_total >= 4:
                priority -= 2.5
        elif phase == "mid":
            priority += points * 1.5
            if level == 2:
                priority += 1.5
        else:
            priority += points * 2.0
            if level == 3:
                priority += 2.0

        if level == 3 and missing_total >= 5:
            priority -= 4.0
        if level == 2 and missing_total >= 5:
            priority -= 2.0

        return round(priority, 3)

    def _simulate_gem_gain(self, current_gems: Dict[str, int], colors: List[str], same_color: Optional[str] = None) -> Dict[str, int]:
        simulated = dict(current_gems)
        if same_color:
            simulated[same_color] = simulated.get(same_color, 0) + 2
        else:
            for color in colors:
                simulated[color] = simulated.get(color, 0) + 1
        return simulated

    def _count_gold_needed(self, target: Dict[str, Any], gems: Dict[str, int]) -> int:
        needed = 0
        for color, missing in target.get("missing", {}).items():
            remaining = max(0, missing - gems.get(color, 0))
            needed += remaining
        return needed

    def _would_be_buyable_after_gems(
        self,
        target: Dict[str, Any],
        simulated_gems: Dict[str, int],
        gold_count: int,
    ) -> bool:
        return self._count_gold_needed(target, simulated_gems) <= gold_count

    def _player_can_afford_card_snapshot(
        self,
        player_snapshot: Dict[str, Any],
        card: Dict[str, Any],
    ) -> bool:
        gems = player_snapshot.get("gems", {})
        discounts = player_snapshot.get("card_discounts", {})
        gold = gems.get("gold", 0)
        missing_total = 0
        for color, cost in card.get("cost", {}).items():
            available = gems.get(color, 0) + discounts.get(color, 0)
            if available < cost:
                missing_total += cost - available
        return missing_total <= gold

    def _prepare_strategy_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        players = game_state.get("players", [])
        current_player_index = game_state.get("current_player", 0)
        current_player = players[current_player_index]
        opponents = [player for i, player in enumerate(players) if i != current_player_index]
        board = game_state.get("board", {})
        current_gems = current_player.get("gems", {})
        current_discounts = current_player.get("card_discounts", {})
        current_gold = current_gems.get("gold", 0)
        phase = self._get_phase(current_player)

        board_cards = []
        for level_cards in board.get("displayed_cards", {}).values():
            board_cards.extend(level_cards)

        nobles = sorted(
            [self._build_noble_info(noble, current_discounts) for noble in board.get("nobles", [])],
            key=lambda item: (item["missing_total"], item["id"])
        )

        all_targets = []
        for card in board_cards + current_player.get("reserved_cards", []):
            target = self._build_card_target_info(card, current_gems, current_discounts, current_gold)
            target["priority"] = self._score_target_priority(target, phase, nobles[:self.noble_limit])
            all_targets.append(target)
        all_targets = self._link_targets_to_nobles(all_targets, nobles)

        buy_now = sorted(
            [item for item in all_targets if item["buyable_now"]],
            key=lambda item: (-item["priority"], -item["points"], item["id"])
        )[:self.target_limit]
        focus_targets = sorted(
            [item for item in all_targets if not item["buyable_now"]],
            key=lambda item: (-item["priority"], item["missing_total"], item["id"])
        )[:self.target_limit]

        available_gems = board.get("gems", {})
        scarce_colors = [
            item["color"]
            for item in sorted(
                [
                    {"color": color, "remaining": count}
                    for color, count in available_gems.items()
                    if color != "gold"
                ],
                key=lambda item: (item["remaining"], item["color"])
            )[:3]
        ]

        card_lookup = {card["id"]: card for card in all_targets if card.get("id")}
        return {
            "current_player": current_player,
            "opponents": opponents,
            "current_gems": current_gems,
            "current_discounts": current_discounts,
            "current_gold": current_gold,
            "phase": phase,
            "nobles": nobles,
            "available_gems": available_gems,
            "scarce_colors": scarce_colors,
            "buy_now": buy_now,
            "focus_targets": focus_targets,
            "card_lookup": card_lookup,
        }

    def _summarize_state(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Dict[str, Any]:
        context = self._prepare_strategy_context(game_state)
        current_player = context["current_player"]
        opponents = context["opponents"]
        focus_targets = context["focus_targets"]
        buy_now = context["buy_now"]
        nearest_nobles = context["nobles"][:self.noble_limit]

        action_evaluations = []
        for i, action in enumerate(valid_actions, start=1):
            action_evaluations.append(
                self._evaluate_action(
                    index=i,
                    action=action,
                    available_gems=context["available_gems"],
                    scarce_colors=context["scarce_colors"],
                    focus_targets=focus_targets,
                    buy_now=buy_now,
                    nearest_nobles=nearest_nobles,
                    card_lookup=context["card_lookup"],
                    current_gems=context["current_gems"],
                    current_gold=context["current_gold"],
                    opponents=opponents,
                    phase=context["phase"],
                )
            )

        top_actions = sorted(
            action_evaluations,
            key=lambda item: (-item["score"], item["action"])
        )[:self.candidate_action_limit]
        self._latest_ranked_actions = top_actions

        return {
            "strategy": "rank_v2_auto",
            "round": game_state.get("round"),
            "phase": context["phase"],
            "me": {
                "score": current_player.get("score"),
                "gems": self._format_compact_gems(context["current_gems"]),
                "discounts": self._format_compact_gems(context["current_discounts"]),
                "gem_slots_left": max(0, 10 - sum(context["current_gems"].values())),
            },
            "scarce_colors": context["scarce_colors"],
            "top_targets": [
                f"{item['id']} pts={item['points']} bonus={item['color']} missing={item['missing_total']} priority={item['priority']} nobles={','.join(item.get('noble_links', [])) or '-'}"
                for item in (buy_now + focus_targets)[:self.target_limit]
            ],
            "top_nobles": [
                f"{item['id']} missing={item['missing_total']} need={item['missing']}"
                for item in nearest_nobles
            ],
            "top_actions": [
                f"{item['index']}. {item['action']} | score={item['score']} | impact={'; '.join(item.get('impact', [])) or 'none'}"
                for item in top_actions
            ],
            "recommended_action_index": top_actions[0]["index"] if top_actions else None,
            "opponent_scores": [f"{opponent['name']}={opponent.get('score', 0)}" for opponent in opponents],
        }

    def _evaluate_action(
        self,
        index: int,
        action: Action,
        available_gems: Dict[str, int],
        scarce_colors: List[str],
        focus_targets: List[Dict[str, Any]],
        buy_now: List[Dict[str, Any]],
        nearest_nobles: List[Dict[str, Any]],
        card_lookup: Dict[str, Dict[str, Any]],
        current_gems: Dict[str, int],
        current_gold: int,
        opponents: List[Dict[str, Any]],
        phase: str,
    ) -> Dict[str, Any]:
        score = 0.0
        impact: List[str] = []
        gem_slots_left = max(0, 10 - sum(current_gems.values()))

        if action.action_type in (ActionType.BUY_CARD, ActionType.BUY_RESERVED_CARD):
            card_id = action.params.get("card_id")
            card = card_lookup.get(card_id)
            score += 20
            if card:
                score += card["priority"] + card["points"] * 2
                impact.extend([f"buy:{card_id}", f"pts:{card['points']}"])
                if card.get("color"):
                    impact.append(f"bonus:{card['color']}")
                if card.get("noble_links"):
                    score += len(card["noble_links"]) * 3
                    impact.append("nobles:" + ",".join(card["noble_links"][:2]))
                if phase == "late" and card["points"] > 0:
                    score += 6
                if phase == "early" and card["level"] == 1:
                    score += 3

        elif action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            colors = [color.value for color in action.params.get("colors", [])]
            score += 6
            noble_hits = set()
            simulated = self._simulate_gem_gain(current_gems, colors)
            for target in focus_targets:
                hit_count = sum(1 for color in colors if color in target.get("missing", {}))
                if hit_count:
                    score += hit_count * (target["priority"] / max(1, target["missing_total"]))
                    if len(impact) < 4:
                        impact.append(f"target:{target['id']}+{hit_count}")
                    if self._would_be_buyable_after_gems(target, simulated, current_gold):
                        score += 5
                        impact.append(f"setup_buy:{target['id']}")
            for noble in nearest_nobles:
                hit_count = sum(1 for color in colors if color in noble.get("missing", {}))
                if hit_count:
                    noble_hits.add(noble["id"])
                    score += hit_count * 2
            scarce_hits = [color for color in colors if color in scarce_colors or available_gems.get(color, 0) <= 2]
            if scarce_hits:
                score += len(scarce_hits)
                impact.append("scarce:" + ",".join(scarce_hits[:3]))
            if noble_hits:
                impact.append("nobles:" + ",".join(sorted(noble_hits)[:2]))
            if gem_slots_left < len(colors):
                score -= 2
            if len(set(colors)) == 3:
                score += 1

        elif action.action_type == ActionType.TAKE_SAME_GEMS:
            color = action.params.get("color")
            color_value = color.value if color else "unknown"
            score += 4
            simulated = self._simulate_gem_gain(current_gems, [], same_color=color_value)
            hit_targets = []
            for target in focus_targets:
                if color_value in target.get("missing", {}):
                    score += min(2, target["missing"][color_value]) * (target["priority"] / max(1, target["missing_total"]))
                    hit_targets.append(target["id"])
                    if self._would_be_buyable_after_gems(target, simulated, current_gold):
                        score += 6
            if hit_targets:
                impact.append("target:" + ",".join(hit_targets[:3]))
            if color_value in scarce_colors:
                score += 1
                impact.append(f"scarce:{color_value}")
            if gem_slots_left < 2:
                score -= 2
            if not hit_targets:
                score -= 3

        elif action.action_type == ActionType.RESERVE_CARD:
            card_id = action.params.get("card_id")
            card = card_lookup.get(card_id)
            score += 3
            impact.append("reserve")
            if available_gems.get("gold", 0) > 0:
                score += 1
                impact.append("gold+1")
            if card:
                score += card["priority"] * 0.8 + card["points"] * 2
                impact.append(f"card:{card_id}")
                if card["missing_total"] <= 2:
                    score += 3
                if phase == "early" and card["level"] == 3:
                    score -= 10
                    impact.append("slow")
                if phase == "early" and card["missing_total"] >= 5:
                    score -= 6
                elif card["missing_total"] >= 6:
                    score -= 3
                if card.get("noble_links"):
                    score += 1
                    impact.append("nobles:" + ",".join(card["noble_links"][:2]))
                for opponent in opponents:
                    if self._player_can_afford_card_snapshot(opponent, card):
                        score += 4
                        impact.append("deny")
                        break
            if action.params.get("from_deck", False):
                score -= 1
                impact.append("blind")

        return {
            "index": index,
            "action": str(action),
            "score": round(score, 3),
            "impact": impact[:4],
        }

    def on_game_start(self, game_state: Dict[str, Any]):
        self.game_history = [{"event": "game_start", "state": game_state}]

    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        self.game_history.append({"event": "game_end", "state": game_state, "winners": winners})

    def on_turn_start(self, game_state: Dict[str, Any]):
        self.game_history.append({"event": "turn_start", "state": game_state})

    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        self.game_history.append(
            {
                "event": "turn_end",
                "state": game_state,
                "action": str(action),
                "success": success,
            }
        )
