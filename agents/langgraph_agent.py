import random
import operator
import json
import re
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Annotated, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from game.game import Action
from agents.base_agent import BaseAgent
from utils.log import CustomLogger

@dataclass
class ContextSchema:
    models: Dict[str, Any] = None
    llm: Any = None
    store: InMemoryStore = None
    logger: CustomLogger = None
    player_id: str = None

class Plan(BaseModel):
    """结构化输出计划"""
    steps: List[str] = Field(description="接下来要执行的一系列计划")
    reason: str = Field(description="计划的原因")

class Reflexion(BaseModel):
    """每一步进行的反思"""
    summary: str = Field(description="对过去游戏进程与自身行动的总结")
    thought: str = Field(description="接下来指导决策的游戏总体思路")

class ActionChoice(BaseModel):
    action_index: int = Field(description="从 1 开始的动作编号")

class AgentState(TypedDict):
    game_state: Dict[str, Any] = Field(description="当前游戏状态")
    reflexion: Reflexion = Field(description="本局游戏的反思")
    plan: List[str] = Field(description="计划")
    past_steps: List[str] = Field(description="已经执行的任务，以及与任务对应的数个动作")
    action_choice: ActionChoice = Field(description="当前动作索引")
    valid_actions: List[str] = Field(description="当前游戏状态下可用的动作")


def _extract_text_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    return {}


def _invoke_prompt(runtime: Runtime[ContextSchema], prompt_key: str, payload: Dict[str, Any]) -> str:
    prompt = runtime.context.models[prompt_key].invoke(payload)
    response = runtime.context.llm.invoke(prompt)
    text = _extract_text_content(response)
    runtime.context.logger.log_info(
        {
            "prompt_key": prompt_key,
            "raw_response_preview": text[:1200],
            "response_metadata": getattr(response, "response_metadata", {}),
        }
    )
    return text


def _build_compact_state_digest(game_state: Dict[str, Any]) -> str:
    players = game_state.get("players", [])
    current_idx = game_state.get("current_player", 0)
    board = game_state.get("board", {})

    def compact_player(player: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": player.get("name"),
            "score": player.get("score", 0),
            "gems": {k: v for k, v in player.get("gems", {}).items() if v},
            "discounts": {k: v for k, v in player.get("card_discounts", {}).items() if v},
            "reserved": [card.get("id") for card in player.get("reserved_cards", [])[:3]],
        }

    displayed_cards = []
    for level, cards in board.get("displayed_cards", {}).items():
        for card in cards:
            displayed_cards.append(
                {
                    "id": card.get("id"),
                    "level": level,
                    "points": card.get("points", 0),
                    "color": card.get("color"),
                    "cost": card.get("cost", {}),
                }
            )

    digest = {
        "round": game_state.get("round"),
        "current_player": players[current_idx].get("name") if players else None,
        "me": compact_player(players[current_idx]) if players else {},
        "opponents": [compact_player(player) for i, player in enumerate(players) if i != current_idx],
        "board_gems": board.get("gems", {}),
        "displayed_cards": displayed_cards[:12],
        "nobles": board.get("nobles", []),
    }
    return json.dumps(digest, ensure_ascii=False, separators=(",", ":"))


def _invoke_compact_retry(runtime: Runtime[ContextSchema], prompt_name: str, instruction: str, game_state: Dict[str, Any]) -> str:
    compact_state = _build_compact_state_digest(game_state)
    response = runtime.context.llm.invoke(
        f"{instruction}\n当前局面摘要:{compact_state}"
    )
    text = _extract_text_content(response)
    runtime.context.logger.log_info(
        {
            "prompt_key": prompt_name,
            "compact_retry_preview": text[:1200],
            "response_metadata": getattr(response, "response_metadata", {}),
        }
    )
    return text


def _parse_plan_text(text: str) -> Plan:
    data = _parse_json_object(text)
    if data:
        steps = [str(step).strip() for step in data.get("steps", []) if str(step).strip()]
        reason = str(data.get("reason", "")).strip()
        if steps:
            return Plan(steps=steps, reason=reason or "根据当前局势生成计划。")

    steps = []
    reason = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(\d+[\.\)]|[-*])\s*", stripped):
            steps.append(re.sub(r"^(\d+[\.\)]|[-*])\s*", "", stripped).strip())
        elif stripped.startswith("原因") or stripped.startswith("思路"):
            reason = stripped.split(":", 1)[-1].split("：", 1)[-1].strip()

    if not steps:
        steps = ["优先获取能尽快转化为卡牌和分数的资源。"]
    if not reason:
        reason = "根据当前棋盘、资源与可行动作生成计划。"
    return Plan(steps=steps[:5], reason=reason)


def _parse_action_choice_text(text: str, valid_actions: List[str]) -> ActionChoice:
    data = _parse_json_object(text)
    if data and "action_index" in data:
        try:
            idx = int(data["action_index"])
            return ActionChoice(action_index=max(1, min(len(valid_actions), idx)))
        except (TypeError, ValueError):
            pass

    patterns = [
        r"action_index\s*[:：]\s*(\d+)",
        r"选择动作\s*[:：]\s*(\d+)",
        r"动作\s*(\d+)",
        r"^\s*(\d+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            idx = int(match.group(1))
            return ActionChoice(action_index=max(1, min(len(valid_actions), idx)))

    return ActionChoice(action_index=1)


def _parse_reflexion_text(text: str) -> Reflexion:
    data = _parse_json_object(text)
    if data:
        summary = str(data.get("summary", "")).strip()
        thought = str(data.get("thought", "")).strip()
        if summary or thought:
            return Reflexion(
                summary=summary or "需要继续根据最新局势调整策略。",
                thought=thought or "优先保持资源效率并推进高价值目标。",
            )

    summary = ""
    thought = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("总结"):
            summary = stripped.split(":", 1)[-1].split("：", 1)[-1].strip()
        elif stripped.startswith("思路"):
            thought = stripped.split(":", 1)[-1].split("：", 1)[-1].strip()

    return Reflexion(
        summary=summary or "需要继续观察双方资源、目标卡和贵族推进情况。",
        thought=thought or "优先选择能提升下回合买牌概率的动作。",
    )

def plan_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState: 
    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER plan_node ======")
    logger.log_info("game_state keys:" + str(list(state["game_state"].keys())))
    logger.log_info("Raw game_state:" + str(json.dumps(state["game_state"], ensure_ascii=False)))
    
    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    response_text = _invoke_compact_retry(
        runtime,
        "plan_compact",
        "请基于下述璀璨宝石局面摘要生成很短的 JSON 计划。只输出 {\"steps\":[\"步骤1\",\"步骤2\"],\"reason\":\"一句话原因\"}。",
        state["game_state"],
    )
    if not response_text:
        response_text = _invoke_prompt(runtime, "plan", {
            "formatted_state": formatted_state
        })
    response = _parse_plan_text(response_text)

    logger.log_info("====== [DEBUG] EXIT plan_node ======\n")
    logger.log_info("plan_node output plan:" + str(response.steps))

    return {
        "plan": response.steps,
    }


def wait_start_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    update = interrupt("default")
    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER wait_start_node ======")
    logger.log_info("[DEBUG] interrupt returned:" + str(update))
    logger.log_info("====== [DEBUG] EXIT wait_start_node ======\n")
    return {
        "game_state": update["game_state"],
        "valid_actions": update["valid_actions"]
    }


def think_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER think_node ======")
    logger.log_info("[DEBUG] plan:" + str(state["plan"]))
    logger.log_info("[DEBUG] valid_actions:" + str(state["valid_actions"]))

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]

    actions_for_llm = "\n".join(state["valid_actions"])
    response_text = _invoke_prompt(runtime, "think", {
        "formatted_state": formatted_state,
        "plan": plan_str,
        "task": task,
        "valid_actions": actions_for_llm
    })
    if not response_text:
        compact_actions = "\n".join(state["valid_actions"][:12])
        response = runtime.context.llm.invoke(
            f"你是璀璨宝石策略代理。只输出 JSON：{{\"action_index\": 1}}。\n"
            f"当前任务:{task}\n"
            f"当前局面摘要:{_build_compact_state_digest(state['game_state'])}\n"
            f"可用动作(截断前12项):\n{compact_actions}"
        )
        response_text = _extract_text_content(response)
        logger.log_info(
            {
                "prompt_key": "think_retry",
                "compact_retry_preview": response_text[:1200],
                "response_metadata": getattr(response, "response_metadata", {}),
            }
        )
    response = _parse_action_choice_text(response_text, state["valid_actions"])

    logger.log_info("[DEBUG] think_node response:" + str(response))
    logger.log_info("====== [DEBUG] EXIT think_node ======\n")

    return {
        "action_choice": response
    }


def execute_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    update = interrupt({
        "action_choice": state["action_choice"]
    })

    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER execute_node ======")

    action_index = state["action_choice"].action_index
    action = state["valid_actions"][action_index-1]

    logger.log_info(f"[DEBUG] chosen action index = {action_index}, action = {action}")
    logger.log_info("[DEBUG] interrupt returned:" + str(update))
    logger.log_info("====== [DEBUG] EXIT execute_node ======\n")

    task = state["plan"][0]
    if state.get("past_steps"):
        past_steps = state["past_steps"] + [f"task: {task}, action: {action}"]
    else:
        past_steps = [f"task: {task}, action: {action}"]
    past_steps = past_steps[-5:]
    return {
        "game_state": update["game_state"],
        "past_steps": past_steps,
        "valid_actions": update["valid_actions"]
    }


def reflexion_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER reflexion_node ======")
    logger.log_info("[DEBUG] past_steps:" + str(state["past_steps"]))
    logger.log_info("[DEBUG] current reflexion:" + str(state.get("reflexion")))

    store = runtime.context.store
    items = store.search((runtime.context.player_id, "memories"))
    logger.log_info(f"[DEBUG] memory items found: {len(items)}")

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    memory = items[-5:]
    reflexion = state.get("reflexion", Reflexion(summary="", thought=""))

    response_text = _invoke_prompt(runtime, "reflexion", {
        "formatted_state": formatted_state,
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "history": memory,
        "valid_actions": state["valid_actions"],
        "summary": reflexion.summary,
        "thought": reflexion.thought
    })
    if not response_text:
        response_text = _invoke_compact_retry(
            runtime,
            "reflexion_retry",
            "请基于下述璀璨宝石局面摘要输出 JSON：{\"summary\":\"一句话总结\",\"thought\":\"一句话思路\"}。",
            state["game_state"],
        )
    response = _parse_reflexion_text(response_text)

    logger.log_info("[DEBUG] reflexion_node new reflexion:" + str(response))
    logger.log_info("====== [DEBUG] EXIT reflexion_node ======\n")
    return {
        "reflexion": response
    }


def replan_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    logger = runtime.context.logger
    logger.log_info("\n====== [DEBUG] ENTER replan_node ======")
    logger.log_info("[DEBUG] old plan:" + str(state["plan"]))
    logger.log_info("[DEBUG] past_steps:" + str(state["past_steps"]))

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    response_text = _invoke_prompt(runtime, "replan", {
        "formatted_state": formatted_state,
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "valid_actions": state["valid_actions"]
    })
    if not response_text:
        response_text = _invoke_compact_retry(
            runtime,
            "replan_retry",
            "请基于下述璀璨宝石局面摘要输出很短的 JSON 新计划：{\"steps\":[\"步骤1\",\"步骤2\"],\"reason\":\"一句话原因\"}。",
            state["game_state"],
        )
    response = _parse_plan_text(response_text)

    logger.log_info("[DEBUG] new plan:" + str(response.steps))
    logger.log_info("====== [DEBUG] EXIT replan_node ======\n")
    return {"plan": response.steps}


def should_end(state: AgentState):
    if state["game_state"]["game_over"]:
        return END
    else:
        return "think"


class LanggraphAgent(BaseAgent):
    """基于LLM的代理"""
    def __init__(
        self,
        player_id: str,
        name: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.5,
        max_tokens: int = 500,
        base_url: str = None,
        model_type: str = "openai_compatible",
        api_version: str = None,
        deployment_name: str = None,
        run_id: str = None,
    ):
        super().__init__(player_id, name)
        max_tokens = max(int(max_tokens), 512)
        provider = self._resolve_model_provider(model_type)
        chat_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
        }
        if provider == "azure_openai":
            if base_url:
                chat_kwargs["azure_endpoint"] = base_url
            if api_version:
                chat_kwargs["api_version"] = api_version
            if deployment_name:
                chat_kwargs["azure_deployment"] = deployment_name
        elif base_url:
            chat_kwargs["base_url"] = base_url

        self.llm = init_chat_model(
            model_name,
            model_provider=provider,
            **chat_kwargs
        )

        self.models = {
            "plan": self._construct_plan_prompt(),
            "think": self._construct_think_prompt(),
            "reflexion": self._construct_reflexion_prompt(),
            "replan": self._construct_replan_prompt(),
        }
        self.store = InMemoryStore()
        self.key_count = 0

        builder = StateGraph(AgentState, context_schema=ContextSchema)
        builder.add_node("plan", plan_node)
        builder.add_node("wait_start", wait_start_node)
        builder.add_node("think", think_node)
        builder.add_node("execute", execute_node)
        builder.add_node("reflexion", reflexion_node)
        builder.add_node("replan", replan_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "wait_start")
        builder.add_edge("wait_start", "think")
        builder.add_edge("think", "execute")
        builder.add_edge("execute", "reflexion")
        builder.add_edge("reflexion", "replan")
        builder.add_conditional_edges("replan", should_end, ["think", END])
        
        checkpointer = MemorySaver()
        self.agent = builder.compile(checkpointer=checkpointer)
        self.config = {"configurable": {"thread_id": f"{self.player_id}"}}

        log_path = "./log/llm.log"
        if run_id:
            log_path = f"./log/langgraph_{run_id}.log"
        self.log_file_path = log_path
        self.logger = CustomLogger(log_path)
        self.logger.enable_file_logging()

        self.ctx = ContextSchema(models=self.models, llm=self.llm, store=self.store, logger=self.logger, player_id=self.player_id)   

    def _resolve_model_provider(self, model_type: str) -> str:
        mapping = {
            "openai": "openai",
            "doubao": "openai",
            "ark": "openai",
            "openai_compatible": "openai",
            "azure_openai": "azure_openai",
            "azure": "azure_openai",
        }
        return mapping.get(model_type, "openai")

    def select_action(self, game_state, valid_actions):
        logger = self.logger
        logger.log_info("\n====== [DEBUG] ENTER select_action ======")

        formatted_actions = []
        for i, action in enumerate(valid_actions):
            formatted_actions.append(f"动作 {i+1}: {str(action)}")
        logger.log_info("[DEBUG] valid_actions:" + str(formatted_actions))

        resume = {
            "game_state": game_state,
            "valid_actions": formatted_actions
        }

        logger.log_info("[DEBUG] resume passed to agent.invoke:" + str(resume))

        result = self.agent.invoke(
            Command(resume=resume),
            config=self.config,
            context=self.ctx,
        )

        logger.log_info("[DEBUG] agent.invoke result keys:" + str(list(result.keys())))
        logger.log_info("[DEBUG] raw __interrupt__:" + str(result["__interrupt__"]))

        action_index = result["__interrupt__"][0].value["action_choice"].action_index
        action = valid_actions[action_index-1]

        logger.log_info(f"[DEBUG] FINAL ACTION INDEX = {action_index}, ACTION = {action}")
        logger.log_info("====== [DEBUG] EXIT select_action ======\n")

        return action

    
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

    def _construct_plan_prompt(self) -> str:
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是璀璨宝石策略代理。目标是尽快获得 15 分。
只输出极短的 JSON，不要解释，不要 markdown。
计划只需要未来 1 到 3 步，优先考虑买牌、折扣、贵族和分数。""",
                ),
                (
                    "human",
                    """
                    根据当前游戏状态，生成一个短计划。
                    {formatted_state}
                    请只输出 JSON，格式如下：
                    {{"steps":["步骤1","步骤2"],"reason":"一句话原因"}}
                    """
                )
            ]
        )
        return planner_prompt

    def _construct_think_prompt(self) -> str:
        think_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是璀璨宝石策略代理。
从给定合法动作中选一个最优动作。
优先：立即买牌 > 推进高价值目标 > 推进贵族 > 提高下回合买牌概率。
只输出 JSON，不要解释。""",
                ),
                (
                    "human",
                    """
                    当前游戏状态:
                    {formatted_state}
                    当前计划：
                    {plan}
                    当前任务：{task}
                    可用动作：
                    {valid_actions}
                    请只输出 JSON，格式如下：
                    {{"action_index": 1}}
                    """
                )
            ]
        )
        return think_prompt
            
    def _construct_reflexion_prompt(self) -> str:
        reflexion_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是璀璨宝石策略代理。
根据最近几步和当前局势，给出一条简短总结和一条下一步思路。
只输出 JSON，不要解释。""",
                ),
                (
                    "human",
                    """
                    旧总结：
                    {summary}
                    旧思路：
                    {thought}
                    最近记忆：
                    {history}
                    当前游戏状态:
                    {formatted_state}
                    当前计划：
                    {plan}
                    最近执行：
                    {past_steps}
                    当前动作：
                    {valid_actions}
                    请只输出 JSON，格式如下：
                    {{"summary":"一句话总结","thought":"一句话策略思路"}}
                    """
                )
            ]
        )
        return reflexion_prompt
    
    def _construct_replan_prompt(self) -> str:
        replanner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是璀璨宝石策略代理。
根据当前局势对计划做短更新，只保留未来 1 到 3 步。
只输出 JSON，不要解释。""",
                ),
                (
                    "human",
                    """
                    当前游戏状态:
                    {formatted_state}
                    旧计划：
                    {plan}
                    当前动作：
                    {valid_actions}
                    最近执行：
                    {past_steps}
                    请只输出 JSON，格式如下：
                    {{"steps":["步骤1","步骤2"],"reason":"一句话原因"}}
                    """
                )
            ]
        )
        return replanner_prompt
    
    def on_game_start(self, game_state: Dict[str, Any]):
        """游戏开始时的回调"""
        logger = self.logger
        logger.log_info("\n====== [DEBUG] on_game_start ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:game_start,state:{formatted_state}"})
        self.agent.invoke({"game_state": game_state}, config=self.config, context=self.ctx)
        logger.log_info("\n====== [DEBUG] on_game_start END ======")

    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        """游戏结束时的回调"""
        logger = self.logger
        logger.log_info("\n====== [DEBUG] on_game_end ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:game_end,state:{formatted_state},winners:{winners}"})
        items = self.store.search((self.player_id, "memories"),limit=9999)
        memory_list = [
            {
                "key": item.key,
                "text": item.value["text"],
            }
            for item in items
        ]
        with open("agents/langgraph_agent_memory.json", "w", encoding="utf-8") as f:
            json.dump(memory_list, f, ensure_ascii=False, indent=2)
        logger.log_info("\n====== [DEBUG] on_game_end END ======")
    
    def on_turn_start(self, game_state: Dict[str, Any]):
        """回合开始时的回调"""
        logger = self.logger
        logger.log_info("\n====== [DEBUG] on_turn_start ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:turn_start,state:{formatted_state}"})
        logger.log_info("\n====== [DEBUG] on_turn_start END ======")
    
    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        """回合结束时的回调"""
        logger = self.logger
        logger.log_info("\n====== [DEBUG] on_turn_end ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:turn_end,state:{formatted_state},action:{str(action)},success:{success}"})
        logger.log_info("\n====== [DEBUG] on_turn_end END ======")
        


