from __future__ import annotations

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from .prompts import SYSTEM_PROMPT, PLAN_PROMPT, VERIFY_PROMPT
from .tools.browser import screenshot_page, load_image
from .tools.ocr import ocr_image


class AgentState(TypedDict):
    url: str
    goal: str
    ocr: str
    plan: List[str]
    logs: List[str]
    answer: str


def llm_client() -> OpenAI:
    return OpenAI()


def node_perceive(state: AgentState) -> AgentState:
    png = screenshot_page(state["url"])
    img = load_image(png)
    state["ocr"] = ocr_image(img)
    state.setdefault("logs", []).append("OCR长度=" + str(len(state["ocr"])) )
    return state


def node_plan(state: AgentState) -> AgentState:
    client = llm_client()
    prompt = PLAN_PROMPT.format(ocr=state["ocr"], goal=state["goal"])
    resp = client.chat.completions.create(
        model="qwen2-7b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    steps = [s.strip("- ") for s in content.split("\n") if s.strip()]
    state["plan"] = steps[:3]
    state.setdefault("logs", []).append("计划=" + str(state["plan"]))
    return state


def node_act(state: AgentState) -> AgentState:
    # 简化：本示例只做只读操作（截图+OCR），并不真正执行点击/输入
    state.setdefault("logs", []).append("执行=示例未对页面进行写操作")
    return state


def node_verify(state: AgentState) -> AgentState:
    client = llm_client()
    prompt = VERIFY_PROMPT.format(logs="\n".join(state.get("logs", [])))
    resp = client.chat.completions.create(
        model="qwen2-7b",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    state["answer"] = resp.choices[0].message.content
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("perceive", node_perceive)
    g.add_node("plan", node_plan)
    g.add_node("act", node_act)
    g.add_node("verify", node_verify)
    g.set_entry_point("perceive")
    g.add_edge("perceive", "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "verify")
    g.add_edge("verify", END)
    return g.compile()


