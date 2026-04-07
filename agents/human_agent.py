import threading
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from game.game import Action


class HumanAgent(BaseAgent):
    """通过 UI 提交输入的人类玩家代理。"""

    def __init__(self, player_id: str, name: str = "人类玩家"):
        super().__init__(player_id, name)
        self.log_file_path = ""
        self._condition = threading.Condition()
        self._pending_request: Optional[Dict[str, Any]] = None
        self._response: Any = None
        self._request_counter = 0

    def _wait_for_response(self, request_type: str, payload: Dict[str, Any]) -> Any:
        with self._condition:
            self._request_counter += 1
            self._pending_request = {"request_id": self._request_counter, "type": request_type, **payload}
            self._response = None
            self._condition.notify_all()

            while self._response is None:
                self._condition.wait()

            response = self._response
            self._response = None
            self._pending_request = None
            return response

    def get_pending_request(self) -> Optional[Dict[str, Any]]:
        with self._condition:
            if self._pending_request is None:
                return None
            return dict(self._pending_request)

    def submit_response(self, response: Any) -> None:
        with self._condition:
            if self._pending_request is None:
                return
            self._response = response
            self._condition.notify_all()

    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Action:
        return self._wait_for_response(
            "action",
            {
                "game_state": game_state,
                "valid_actions": valid_actions,
            },
        )

    def select_gems_to_discard(
        self,
        game_state: Dict[str, Any],
        gems: Dict[str, int],
        num_to_discard: int,
    ) -> Dict[str, int]:
        return self._wait_for_response(
            "discard_gems",
            {
                "game_state": game_state,
                "gems": dict(gems),
                "num_to_discard": num_to_discard,
            },
        )

    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        return self._wait_for_response(
            "select_noble",
            {
                "game_state": game_state,
                "available_nobles": available_nobles,
            },
        )
