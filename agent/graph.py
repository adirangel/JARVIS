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
    def __init__(self, entry_node): 
        self.entry_node = entry_node
        self.nodes = {entry_node.name: entry_node}
    
    def add_node(self, node, depends_on=None): 
        self.nodes[node.name] = node
    
    async def run(self, state, max_turns=15):
        # Ensure nodes exist
        input_node = self.entry_node
        reflector = self.nodes.get("reflector")
        tool_node = self.nodes.get("tool")
        output_node = self.nodes.get("output")
        
        for turn in range(max_turns):
            if state.stopped.is_set():
                break
            
            # Process input node (already set by handle_wake or CLI)
            input_result = await input_node.process(state)
            yield input_result
            
            if not getattr(state, 'current_input', ''):
                break
            
            # Reset last_response
            state.last_response = None
            
            # Tool node (if handles, returns a response string)
            if tool_node:
                tool_result = await tool_node.process(state)
                if tool_result is not None:
                    state.last_response = tool_result
            
            # Reflector node (LLM) if no tool response yet
            if reflector and not state.last_response:
                async with asyncio.timeout(30):
                    llm_response = await reflector.process(state)
                    state.last_response = llm_response
                yield llm_response
            
            # Output node (prints response and timers)
            if output_node:
                await output_node.process(state)
                yield ""
            
            break  # One turn per invocation
