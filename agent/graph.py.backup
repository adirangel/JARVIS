import asyncio
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SessionState:
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    conversation_history: list = field(default_factory=list)
    timers: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    speech_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

class Node:
    def __init__(self, name: str): self.name = name
    async def process(self, state, **kwargs): raise NotImplementedError

class AgentGraph:
    def __init__(self, entry_node): self.entry_node = entry_node; self.nodes = {entry_node.name: entry_node}
    def add_node(self, node, depends_on=None): self.nodes[node.name] = node
    async def run(self, state, max_turns=15):
        input_node = self.entry_node
        reflector = self.nodes.get("reflector")
        tool_node = self.nodes.get("tool")
        output_node = self.nodes.get("output")
        for turn in range(max_turns):
            if state.stopped.is_set():
                break
            # Input
            result = await input_node.process(state)
            yield result
            if not state.current_input:
                break
            # Tool (if handles, sets last_response)
            if tool_node:
                tool_result = await tool_node.process(state)
                if tool_result is not None:
                    state.last_response = tool_result
            # Reflector (LLM) if no tool response yet
            if reflector and not getattr(state, "last_response", None):
                result = await reflector.process(state)
                state.last_response = result
                yield result
            # Output
            if output_node:
                await output_node.process(state)
                yield ""
            break  # One turn per wake
