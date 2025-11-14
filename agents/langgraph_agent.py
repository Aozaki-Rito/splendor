from agents.base_agent import BaseAgent
from langchain.chat_models import init_chat_model
from typing import TypedDict, List, Optional, Dict, Any, Literal, Annotated, Tuple
import random
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from game.game import Action, ActionType, Game
import operator
import json

class AgentState(TypedDict):
    game_state: Dict[str, Any] # 当前游戏状态
    plan: List[str]  # 计划
    action: Action  # 当前动作
    past_steps: Annotated[List[Tuple], operator.add] # 已经执行的步骤，自动添加

def plan_node(state: AgentState, llm) -> AgentState: #要改
    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    plan = llm.invoke({"messages": [("user", formatted_state)]})
    return {"plan": plan.steps}

def think_node(state: AgentState) -> AgentState:
    pass

def execute_node(state: AgentState) -> AgentState:
    """执行动作节点"""
    update = interrupt({
        "action": state["action"]
    })
    state["game_state"] = update["game_state"]
    return state

def replan_node(state: AgentState) -> AgentState:
    pass

def memory_node(state: AgentState) -> AgentState:
    pass

def should_end(state: AgentState):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"





class LLMAgent(BaseAgent):
    """基于LLM的代理"""
    def __init__(self, player_id: str, name: str, model_name: str, api_key: str, temperature: float = 0.5, max_tokens: int = 500, prompts: str = None):
        super().__init__(player_id, name)
        self.model_name = model_name
        self.api_key = api_key
        self.llm = init_chat_model(model_name,temperature=temperature,max_tokens=max_tokens)
        self.system_prompt = prompts["system"]
        self.action_prompt = prompts["action"]
        self.discard_prompt = prompts["discard"]
        self.noble_prompt = prompts["noble"]
        self.game_history = []

        builder = StateGraph(AgentState)
        builder.add_node("plan", plan_node)
        builder.add_node("think", think_node)
        builder.add_node("execute", execute_node)
        builder.add_node("replan", replan_node)
        builder.add_node("memory", memory_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "think")
        builder.add_edge("think", "execute")
        builder.add_edge("execute", "replan")
        builder.add_edge("replan", "memory")
        builder.add_conditional_edges("memory", should_end, ["think", END])
        
        checkpointer = MemorySaver()
        self.agent = builder.compile(checkpointer=checkpointer)
        self.config = {"configurable": {"thread_id": f"{self.player_id}"}}

        initial = self.agent.invoke({"game_state": self.game_state}, config=self.config) # 初始状态，待输入

    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Any]) -> Any:

        pass
    
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

