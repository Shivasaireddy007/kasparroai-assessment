"""Scripted fake provider for agent, reliability and API tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from claimcheck.providers import ModelTurn, ToolCall, ToolResult

Script = Callable[[dict[str, Any], int], Any]


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, script: Script, delay_s: float = 0.0):
        self.script = script
        self.delay_s = delay_s
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self.payloads: list[str] = []  # everything sent to the "model", serialized

    def settings(self) -> dict[str, Any]:
        return {}

    def new_conversation(self, system: str, user: str) -> dict[str, Any]:
        conv = {"system": system, "messages": [{"role": "user", "text": user}], "turns": 0}
        self.payloads += [system, user]
        return conv

    def add_user(self, conv: dict[str, Any], text: str) -> None:
        conv["messages"].append({"role": "user", "text": text})
        self.payloads.append(text)

    def add_tool_results(self, conv: dict[str, Any], results: list[ToolResult]) -> None:
        for r in results:
            conv["messages"].append({"role": "tool", "name": r.call.name, "response": r.response})
            self.payloads.append(json.dumps(r.response))

    async def generate(self, conv: dict[str, Any], tools: list[dict[str, Any]]) -> ModelTurn:
        self.calls += 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.payloads.append(json.dumps(tools))
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            conv["turns"] += 1
            result = self.script(conv, conv["turns"])
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            self.active -= 1


def lookup(brand: str = "cleartrail") -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall("get_brand_facts", {"brand_id": brand}, "c1")])


def submit(claims: list[dict[str, Any]]) -> ModelTurn:
    return ModelTurn(tool_calls=[ToolCall("submit_claims", {"claims": claims}, "c2")])


def claim(i: int, label: str = "supported", fact_ids: list[str] | None = None, reason: str = "Matches the catalogue.") -> dict[str, Any]:
    return {"sentence_index": i, "label": label, "fact_ids": fact_ids or [], "reason": reason}


def standard(claims: list[dict[str, Any]]) -> Script:
    """Turn 1: look up facts. Turn 2: submit."""

    def s(conv: dict[str, Any], turn: int) -> ModelTurn:
        return lookup() if turn == 1 else submit(claims)

    return s
