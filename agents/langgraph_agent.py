import random
import operator
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Annotated, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from game.game import Action
from agents.base_agent import BaseAgent

@dataclass
class ContextSchema:
    models: Dict[str, Any] = None
    store: InMemoryStore = None
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

def plan_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState: 
    print("\n====== [DEBUG] ENTER plan_node ======")
    print("game_state keys:", list(state["game_state"].keys()))
    print("Raw game_state:", json.dumps(state["game_state"], ensure_ascii=False))
    
    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    model = runtime.context.models["plan"]
    response = model.invoke({
        "formatted_state": formatted_state
    })

    print("[DEBUG] plan_node output plan:", response.steps)
    print("====== [DEBUG] EXIT plan_node ======\n")

    return {
        "plan": response.steps,
    }


def wait_start_node(state: AgentState) -> AgentState:
    print("\n====== [DEBUG] ENTER wait_start_node ======")
    update = interrupt("default")
    print("[DEBUG] interrupt returned:", update)
    print("====== [DEBUG] EXIT wait_start_node ======\n")
    return {
        "game_state": update["game_state"],
        "valid_actions": update["valid_actions"]
    }


def think_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    print("\n====== [DEBUG] ENTER think_node ======")
    print("[DEBUG] plan:", state["plan"])
    print("[DEBUG] valid_actions:", state["valid_actions"])

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    model = runtime.context.models["think"]
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]

    actions_for_llm = "\n".join(state["valid_actions"])
    response = model.invoke({
        "formatted_state": formatted_state,
        "plan": plan_str,
        "task": task,
        "valid_actions": actions_for_llm
    })

    print("[DEBUG] think_node response:", response)
    print("====== [DEBUG] EXIT think_node ======\n")

    return {
        "action_choice": response
    }


def execute_node(state: AgentState) -> AgentState:
    print("\n====== [DEBUG] ENTER execute_node ======")

    update = interrupt({
        "action_choice": state["action_choice"]
    })

    action_index = state["action_choice"].action_index
    action = state["valid_actions"][action_index-1]

    print(f"[DEBUG] chosen action index = {action_index}, action = {action}")
    print("[DEBUG] interrupt returned:", update)
    print("====== [DEBUG] EXIT execute_node ======\n")

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
    print("\n====== [DEBUG] ENTER reflexion_node ======")
    print("[DEBUG] past_steps:", state["past_steps"])
    print("[DEBUG] current reflexion:", state.get("reflexion"))

    store = runtime.context.store
    items = store.search((runtime.context.player_id, "memories"))
    print(f"[DEBUG] memory items found: {len(items)}")

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    model = runtime.context.models["reflexion"]
    memory = items[-5:]
    reflexion = state.get("reflexion", Reflexion(summary="", thought=""))

    response = model.invoke({
        "formatted_state": formatted_state,
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "history": memory,
        "valid_actions": state["valid_actions"],
        "summary": reflexion.summary,
        "thought": reflexion.thought
    })

    print("[DEBUG] reflexion_node new reflexion:", response)
    print("====== [DEBUG] EXIT reflexion_node ======\n")
    return {
        "reflexion": response
    }


def replan_node(state: AgentState, runtime: Runtime[ContextSchema]) -> AgentState:
    print("\n====== [DEBUG] ENTER replan_node ======")
    print("[DEBUG] old plan:", state["plan"])
    print("[DEBUG] past_steps:", state["past_steps"])

    formatted_state = json.dumps(state["game_state"], indent=2, ensure_ascii=False)
    model = runtime.context.models["replan"]
    response = model.invoke({
        "formatted_state": formatted_state,
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "valid_actions": state["valid_actions"]
    })

    print("[DEBUG] new plan:", response.steps)
    print("====== [DEBUG] EXIT replan_node ======\n")
    return {"plan": response.steps}


def should_end(state: AgentState):
    if state["game_state"]["game_over"]:
        return END
    else:
        return "think"


class LLMAgent(BaseAgent):
    """基于LLM的代理"""
    def __init__(self, player_id: str, name: str, model_name: str, api_key: str, temperature: float = 0.5, max_tokens: int = 500):
        super().__init__(player_id, name)
        self.llm = init_chat_model(model_name,temperature=temperature,max_tokens=max_tokens,api_key=api_key)

        self.plan_chain = self._construct_plan_prompt() | self.llm.with_structured_output(Plan)
        self.think_chain = self._construct_think_prompt() | self.llm.with_structured_output(ActionChoice)
        self.reflexion_chain = self._construct_reflexion_prompt() | self.llm.with_structured_output(Reflexion)
        self.replan_chain = self._construct_replan_prompt() | self.llm.with_structured_output(Plan)
        self.models = {
            "plan": self.plan_chain,
            "think": self.think_chain,
            "reflexion": self.reflexion_chain,
            "replan": self.replan_chain
        }
        embeddings = init_embeddings("openai:text-embedding-3-small",api_key=api_key)
        self.store = InMemoryStore(
            index={
                "embed": embeddings,
                "dims": 1536,
            }
        )
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

        self.ctx = ContextSchema(models=self.models,store=self.store,player_id=self.player_id)

    def select_action(self, game_state, valid_actions):
        print("\n====== [DEBUG] ENTER select_action ======")

        formatted_actions = []
        for i, action in enumerate(valid_actions):
            formatted_actions.append(f"动作 {i+1}: {str(action)}")
        print("[DEBUG] valid_actions:", formatted_actions)

        resume = {
            "game_state": game_state,
            "valid_actions": formatted_actions
        }

        print("[DEBUG] resume passed to agent.invoke:", resume)

        result = self.agent.invoke(
            Command(resume=resume),
            config=self.config,
            context=self.ctx,
        )

        print("[DEBUG] agent.invoke result keys:", list(result.keys()))
        print("[DEBUG] raw __interrupt__:", result["__interrupt__"])

        action_index = result["__interrupt__"][0].value["action_choice"].action_index
        action = valid_actions[action_index-1]

        print(f"[DEBUG] FINAL ACTION INDEX = {action_index}, ACTION = {action}")
        print("====== [DEBUG] EXIT select_action ======\n")

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
                    """你是一名璀璨宝石(Splendor)游戏的AI玩家。你的目标是通过策略性地收集宝石、购买卡牌和吸引贵族，尽可能快地获得15分。
                        游戏规则:
                        1. 每回合你可以执行以下操作之一:
                        - 拿取3个不同颜色的宝石代币
                        - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
                        - 购买一张面朝上的发展卡或预留的卡
                        - 预留一张发展卡并获得一个金色宝石(黄金)
                        2. 你最多持有10个宝石代币，超过需要丢弃
                        3. 当你的发展卡达到一位贵族的要求时，该贵族会立即访问你，提供额外的胜利点数，每次执行操作后你至多可以选择被一位满足条件的贵族访问
                        4. 游戏在一位玩家达到15分后，剩余玩家依次完成一个回合之后游戏结束
                        策略提示:
                        - 注意平衡短期与长期利益
                        - 考虑其他玩家可能的行动
                        - 关注贵族卡的要求
                        - 预留对你重要或对对手有价值的卡牌
                        - 留意游戏板上的卡牌分布
                        请你想出一个一步一步的计划，这个计划应该最终指引你获得游戏胜利。
                        由于游戏局势不断变化，你的计划可能只是需要在未来数个回合中被完成，不需要完美地预测到所有情况，你只需要保证你接下来的几个计划对于最终胜利是有帮助的。 
                        这个计划应该包括单独的任务，这个任务可能需要多个回合去完成，
                        例如你可能需要三到四个回合来搜集宝石并兑换第二级别的第一张卡，或者你可能需要预先扣留第三级别的某一张七分卡并再接下来三个回合兑换它。
                        确保每一步都有所有需要的信息——不要跳过任何步骤。""",
                ),
                (
                    "human",
                    """
                    请分析当前的游戏状态，按照要求制定游戏计划。
                    当前游戏开始，状态如下:
                    {formatted_state}
                    请给出计划以及对应的思路。
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
                    """你是一名璀璨宝石(Splendor)游戏的AI玩家。你的目标是通过策略性地收集宝石、购买卡牌和吸引贵族，尽可能快地获得15分。
                        游戏规则:
                        1. 每回合你可以执行以下操作之一:
                        - 拿取3个不同颜色的宝石代币
                        - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
                        - 购买一张面朝上的发展卡或预留的卡
                        - 预留一张发展卡并获得一个金色宝石(黄金)
                        2. 你最多持有10个宝石代币，超过需要丢弃
                        3. 当你的发展卡达到一位贵族的要求时，该贵族会立即访问你，提供额外的胜利点数，每次执行操作后你至多可以选择被一位满足条件的贵族访问
                        4. 游戏在一位玩家达到15分后，剩余玩家依次完成一个回合之后游戏结束
                        策略提示:
                        - 注意平衡短期与长期利益
                        - 考虑其他玩家可能的行动
                        - 关注贵族卡的要求
                        - 预留对你重要或对对手有价值的卡牌
                        - 留意游戏板上的卡牌分布
                        你需要自己分析当前的游戏局势，在合法动作中选择一项输出，以期达到最终的胜利。""",
                ),
                (
                    "human",
                    """
                    请分析当前的游戏状态，并从以下可用动作中选择最佳动作。
                    当前游戏状态:
                    {formatted_state}
                    你正在执行的计划是这样的：
                    {plan}
                    你目前需要执行计划的第一步，{task}。
                    你的可用的动作是：
                    {valid_actions}
                    请你根据上述信息给出当前步骤的动作编号。
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
                    """你是一名璀璨宝石(Splendor)游戏的AI玩家。你的目标是通过策略性地收集宝石、购买卡牌和吸引贵族，尽可能快地获得15分。
                        游戏规则:
                        1. 每回合你可以执行以下操作之一:
                        - 拿取3个不同颜色的宝石代币
                        - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
                        - 购买一张面朝上的发展卡或预留的卡
                        - 预留一张发展卡并获得一个金色宝石(黄金)
                        2. 你最多持有10个宝石代币，超过需要丢弃
                        3. 当你的发展卡达到一位贵族的要求时，该贵族会立即访问你，提供额外的胜利点数，每次执行操作后你至多可以选择被一位满足条件的贵族访问
                        4. 游戏在一位玩家达到15分后，剩余玩家依次完成一个回合之后游戏结束
                        策略提示:
                        - 注意平衡短期与长期利益
                        - 考虑其他玩家可能的行动
                        - 关注贵族卡的要求
                        - 预留对你重要或对对手有价值的卡牌
                        - 留意游戏板上的卡牌分布
                        你需要根据已有的信息对当前的游戏做出反思，反思包括总结和思路两个方面。
                        总结的内容包括对游戏局势的分析，对自己打法的总结，对对手打法的分析或者其他你认为有必要的信息。
                        思路则是你当前正在遵循的策略，这是你整局游戏的行动方针，它将指导你修改计划并在每一步思考做出何种行动。
                        """,
                ),
                (
                    "human",
                    """
                    在过去的回合中，你已经做出了如下总结：
                    {summary}
                    当前你的思路是：
                    {thought}
                    过去一段时间内游戏的进程是（仅包含过去五步）：
                    {history}
                    当前游戏状态:
                    {formatted_state}
                    你正在实施的计划是这样的：
                    {plan}
                    您目前针对之前的任务列表已经完成了如下工作(仅包含过去五步）：
                    {past_steps}
                    当前可用的动作是：
                    {valid_actions}
                    请你根据以上信息，给出新的总结和思路。
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
                    """你是一名璀璨宝石(Splendor)游戏的AI玩家。你的目标是通过策略性地收集宝石、购买卡牌和吸引贵族，尽可能快地获得15分。
                        游戏规则:
                        1. 每回合你可以执行以下操作之一:
                        - 拿取3个不同颜色的宝石代币
                        - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
                        - 购买一张面朝上的发展卡或预留的卡
                        - 预留一张发展卡并获得一个金色宝石(黄金)
                        2. 你最多持有10个宝石代币，超过需要丢弃
                        3. 当你的发展卡达到一位贵族的要求时，该贵族会立即访问你，提供额外的胜利点数，每次执行操作后你至多可以选择被一位满足条件的贵族访问
                        4. 游戏在一位玩家达到15分后，剩余玩家依次完成一个回合之后游戏结束
                        策略提示:
                        - 注意平衡短期与长期利益
                        - 考虑其他玩家可能的行动
                        - 关注贵族卡的要求
                        - 预留对你重要或对对手有价值的卡牌
                        - 留意游戏板上的卡牌分布
                        请你对已有的计划进行修改，这个计划应该最终指引你获得游戏胜利。
                        由于游戏局势不断变化，你的计划可能只是需要在未来数个回合中被完成，不需要完美地预测到所有情况，你只需要保证你接下来的几个计划对于最终胜利是有帮助的。 
                        这个计划应该包括单独的任务，这个任务可能需要多个回合去完成，
                        例如你可能需要三到四个回合来搜集宝石并兑换第二级别的第一张卡，或者你可能需要预先扣留第三级别的某一张七分卡并再接下来三个回合兑换它。
                        确保每一步都有所有需要的信息——不要跳过任何步骤。""",
                ),
                (
                    "human",
                    """
                    请分析当前的游戏状态，按照要求修改游戏计划。
                    当前游戏状态:
                    {formatted_state}
                    你正在实施的计划是这样的：
                    {plan}
                    当前可用的动作是：
                    {valid_actions}
                    你目前针对之前的任务列表已经完成了如下工作(仅包含过去五步）：
                    {past_steps}
                    请你注意最近一个任务，它可能需要多步去完成，而已经记录的可能只是多步中的一部分，这代表该任务可能还没有完成。
                    并且你需要注意如果最近一个任务的内容与可用动作列表有冲突，要及时修改当前任务。
                    如果你认为它还没有完成且无需修改，请你在新制定的计划中依然将它放在第一位。
                    相应地更新你的计划。只向计划中添加仍需完成的步骤。不要返回先前完成的步骤作为计划的一部分。
                    请给出计划以及对应的思路。
                    """
                )
            ]
        )
        return replanner_prompt
    
    def on_game_start(self, game_state: Dict[str, Any]):
        """游戏开始时的回调"""
        print("\n====== [DEBUG] on_game_start ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:game_start,state:{formatted_state}"})
        self.agent.invoke({"game_state": game_state}, config=self.config, context=self.ctx)
        print("\n====== [DEBUG] on_game_start END ======")

    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        """游戏结束时的回调"""
        print("\n====== [DEBUG] on_game_end ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:game_end,state:{formatted_state},winners:{winners}"})
        items = self.store.search((self.player_id, "memories"))
        memory_list = [
            {
                "key": item.key,
                "text": item.value["text"],
            }
            for item in items
        ]
        with open("agents/langgraph_agent_memory.json", "w", encoding="utf-8") as f:
            json.dump(memory_list, f, ensure_ascii=False, indent=2)
        print("\n====== [DEBUG] on_game_end END ======")
    
    def on_turn_start(self, game_state: Dict[str, Any]):
        """回合开始时的回调"""
        print("\n====== [DEBUG] on_turn_start ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:turn_start,state:{formatted_state}"})
        print("\n====== [DEBUG] on_turn_start END ======")
    
    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        """回合结束时的回调"""
        print("\n====== [DEBUG] on_turn_end ======")
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        self.key_count += 1
        self.store.put((self.player_id,"memories"),f"key{self.key_count}",{"text":f"event:turn_end,state:{formatted_state},action:{str(action)},success:{success}"})
        print("\n====== [DEBUG] on_turn_end END ======")
        


