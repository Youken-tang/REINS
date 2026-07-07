"""PlannerAgent — wraps a single planner request as an Agent turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from high_agent.agent.protocol import AgentContext, AgentTurnResult
from high_agent.agent.tool_calls import ToolCallNormalizer, sanitize_tool_protocol_messages
from high_agent.llm.client import ModelClient
from high_agent.tools.registry import ToolRegistry


@dataclass
class PlannerAgent:
    """Single-turn planner sharing MainAgent's objective.

    Encapsulates one model call that returns tool_calls or a final answer.
    Satisfies the Agent protocol.
    """

    objective: str
    model_client: ModelClient
    tools: ToolRegistry
    system_prompt: str = ""
    context: AgentContext = field(init=False)
    _base_messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.context = AgentContext(objective=self.objective)

    def run_one_turn(self, ledger_digest: str) -> AgentTurnResult:
        messages = list(self._base_messages)
        messages.append({
            "role": "user",
            "content": self._build_planning_prompt(ledger_digest),
        })

        api_messages = sanitize_tool_protocol_messages(messages)
        response = self.model_client.complete(
            api_messages,
            tools=self.tools.definitions(),
        )

        if response.tool_calls:
            normalizer = ToolCallNormalizer()
            normalized = normalizer.normalize(response.tool_calls)
            tool_call_dicts = [
                {
                    "id": item.call.id,
                    "name": item.call.name,
                    "arguments": item.call.arguments,
                    "original_name": item.original_name,
                    "parent_call_id": item.parent_call_id,
                }
                for item in normalized
            ]
            return AgentTurnResult.with_tool_calls(tool_call_dicts, content=response.content or "")

        return AgentTurnResult.final_answer(response.content or "")

    def _build_planning_prompt(self, ledger_digest: str) -> str:
        context_text = self.context.render(max_chars=8_000)
        return (
            f"Runtime planning snapshot.\n"
            f"Ledger:\n{ledger_digest}\n\n"
            f"Durable context:\n{context_text}\n\n"
            "Use tools now for any concrete work that can progress. "
            "Only provide a final answer when the ledger shows the requested work is complete."
        )

    @staticmethod
    def from_main_agent(main_agent: Any, *, base_messages: list[dict[str, Any]] | None = None) -> "PlannerAgent":
        """Factory: create a PlannerAgent sharing MainAgent's config."""
        planner = PlannerAgent(
            objective=main_agent.objective,
            model_client=main_agent.model_client,
            tools=main_agent.tools,
            system_prompt=getattr(main_agent, "system_prompt", ""),
        )
        planner._base_messages = list(base_messages or [])
        planner.context = AgentContext.from_total(main_agent.context)
        return planner
