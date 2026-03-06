"""JARVIS Multi-Agent Manager.

Spawns background agents that work independently, report results back to JARVIS,
and can communicate with each other via a shared message bus.

Each agent runs in its own thread and can:
- Execute tools from tool_registry
- Run arbitrary commands/scripts
- Send messages to other agents
- Report results back to JARVIS

Architecture:
    AgentManager (singleton)
    ├── Agent "researcher" (thread) — doing web research
    ├── Agent "monitor" (thread) — watching system metrics
    ├── Agent "coder" (thread) — writing/running code
    └── Message Bus (thread-safe dict of queues)
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"     # Waiting for message / event
    STOPPED = "stopped"


@dataclass
class AgentMessage:
    """A message between agents or from an agent to JARVIS."""
    sender: str           # Agent name or "jarvis" or "user"
    recipient: str        # Agent name, "jarvis", or "all" (broadcast)
    content: str          # Message body
    timestamp: float = field(default_factory=time.time)
    msg_type: str = "text"  # "text", "result", "command", "error"


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    description: str
    task_type: str = "general"     # "general", "command", "script", "monitor", "research"
    params: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """A background agent that executes tasks independently."""

    def __init__(
        self,
        name: str,
        task: AgentTask,
        agent_id: str = "",
        on_result: Optional[Callable[[str, str], None]] = None,
        on_status_change: Optional[Callable[[str, str, str], None]] = None,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.task = task
        self.status = AgentStatus.IDLE
        self.result: str = ""
        self.error: str = ""
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks
        self._on_result = on_result          # (agent_name, result) -> None
        self._on_status_change = on_status_change  # (agent_name, status, desc) -> None
        self._on_progress = on_progress      # (agent_name, message) -> None
        self._on_interim_report: Optional[Callable[[str, str], None]] = None  # periodic reports to JARVIS

        # Message inbox
        self.inbox: List[AgentMessage] = []
        self._inbox_lock = threading.Lock()

        # Progress log
        self.progress_log: List[str] = []

    def start(self):
        """Start the agent in a background thread."""
        self.status = AgentStatus.RUNNING
        self.started_at = time.time()
        self._notify_status("running", f"Working on: {self.task.description[:60]}")
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"Agent-{self.name}")
        self._thread.start()

    def stop(self):
        """Signal the agent to stop."""
        self._stop_event.set()
        self.status = AgentStatus.STOPPED
        self._notify_status("stopped", "Agent stopped by user")

    def send_message(self, msg: AgentMessage):
        """Put a message in this agent's inbox."""
        with self._inbox_lock:
            self.inbox.append(msg)

    def get_messages(self) -> List[AgentMessage]:
        """Read and clear inbox."""
        with self._inbox_lock:
            msgs = list(self.inbox)
            self.inbox.clear()
            return msgs

    def _log_progress(self, msg: str):
        """Log a progress step."""
        self.progress_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.debug(f"[Agent:{self.name}] {msg}")
        # Notify GUI live
        if self._on_progress:
            try:
                self._on_progress(self.name, msg)
            except Exception:
                pass

    def _notify_status(self, status: str, desc: str):
        """Notify status change callback."""
        if self._on_status_change:
            try:
                self._on_status_change(self.name, status, desc)
            except Exception:
                pass

    def _run(self):
        """Main agent execution loop."""
        try:
            self._log_progress(f"Started task: {self.task.description}")

            if self.task.task_type == "command":
                self.result = self._run_command()
            elif self.task.task_type == "script":
                self.result = self._run_script()
            elif self.task.task_type == "monitor":
                self.result = self._run_monitor()
            elif self.task.task_type == "research":
                self.result = self._run_research()
            elif self.task.task_type == "tool":
                self.result = self._run_tool()
            elif self.task.task_type == "multi_step":
                self.result = self._run_multi_step()
            elif self.task.task_type == "autonomous":
                self.result = self._run_autonomous()
            else:
                self.result = self._run_general()

            self.status = AgentStatus.COMPLETED
            self.finished_at = time.time()
            elapsed = self.finished_at - (self.started_at or self.finished_at)
            self._log_progress(f"Completed in {elapsed:.1f}s")
            self._notify_status("completed", f"Done: {self.result[:60]}")

            # Report result back
            if self._on_result:
                self._on_result(self.name, self.result)

        except Exception as e:
            self.status = AgentStatus.FAILED
            self.error = str(e)
            self.finished_at = time.time()
            self._log_progress(f"Failed: {e}")
            self._notify_status("failed", f"Error: {str(e)[:60]}")
            if self._on_result:
                self._on_result(self.name, f"Agent error: {e}")

    def _run_command(self) -> str:
        """Execute a system command."""
        cmd = self.task.params.get("command", self.task.description)
        self._log_progress(f"Running command: {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            output = (result.stdout or "") + (result.stderr or "")
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out (120s)."
        except Exception as e:
            return f"Command error: {e}"

    def _run_script(self) -> str:
        """Run a script file."""
        import os, platform
        file_path = self.task.params.get("file_path", "")
        language = self.task.params.get("language", "python")
        visible = self.task.params.get("visible", False)

        if not file_path:
            return "No file_path specified."
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        abs_path = os.path.abspath(file_path)
        work_dir = os.path.dirname(abs_path)

        runners = {
            "python": f'python "{abs_path}"',
            "javascript": f'node "{abs_path}"',
            "typescript": f'npx ts-node "{abs_path}"',
        }
        ext = os.path.splitext(file_path)[1].lower()
        ext_map = {".py": "python", ".js": "javascript", ".ts": "typescript"}
        language = ext_map.get(ext, language)
        run_cmd = runners.get(language, f'python "{abs_path}"')

        self._log_progress(f"Running: {run_cmd}")

        if visible and platform.system() == "Windows":
            full_cmd = f'start "Agent:{self.name}" cmd /k "cd /d {work_dir} && {run_cmd}"'
            subprocess.Popen(full_cmd, shell=True)
            return f"Script launched in visible terminal: {file_path}"

        try:
            result = subprocess.run(
                run_cmd, shell=True, capture_output=True, text=True,
                timeout=120, cwd=work_dir
            )
            output = (result.stdout or "") + (result.stderr or "")
            return output.strip() or "(script completed with no output)"
        except subprocess.TimeoutExpired:
            return "Script timed out (120s)."
        except Exception as e:
            return f"Script error: {e}"

    def _run_monitor(self) -> str:
        """Monitor a condition periodically and report.

        Supports both shell commands AND JARVIS tools (browser_control, etc.).
        Sends interim reports back to JARVIS after every check.
        """
        condition = self.task.params.get("condition", "")
        interval = self.task.params.get("interval_seconds", 10)
        max_checks = self.task.params.get("max_checks", 30)
        command = self.task.params.get("command", "")
        tool_name = self.task.params.get("tool_name", "")
        tool_args = self.task.params.get("tool_args", {})

        source = condition or command or tool_name or "condition"
        self._log_progress(f"Monitoring: {source} every {interval}s (max {max_checks} checks)")
        results = []

        for i in range(max_checks):
            if self._stop_event.is_set():
                self._log_progress("Stopped by user")
                break

            # Check for incoming messages
            msgs = self.get_messages()
            for msg in msgs:
                self._log_progress(f"Message from {msg.sender}: {msg.content}")
                if msg.content.lower() in ("stop", "quit", "cancel"):
                    return f"Stopped by {msg.sender}. Collected {len(results)} readings."

            # Run the check — either shell command or JARVIS tool
            output = ""
            check_label = f"Check {i+1}/{max_checks}"

            if tool_name:
                try:
                    from tool_registry import execute_tool
                    output = execute_tool(tool_name, tool_args)
                    self._log_progress(f"{check_label}: {output[:120]}")
                except Exception as e:
                    output = f"Tool error: {e}"
                    self._log_progress(f"{check_label}: error - {e}")
            elif command:
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True, timeout=15
                    )
                    output = (result.stdout or "").strip()
                    self._log_progress(f"{check_label}: {output[:120]}")
                except Exception as e:
                    output = f"Command error: {e}"
                    self._log_progress(f"{check_label}: error - {e}")

            if output:
                results.append(output)

            # Send interim report back to JARVIS so he can react in real-time
            if output and self._on_interim_report:
                try:
                    report = (
                        f"[Monitor '{self.name}' — {check_label}]\n"
                        f"{output[:1500]}"
                    )
                    self._on_interim_report(self.name, report)
                except Exception:
                    pass

            # Wait for next interval (interruptible)
            self._stop_event.wait(interval)

        summary = f"Monitoring complete. {len(results)} checks performed."
        if results:
            summary += f"\nLast reading: {results[-1][:200]}"
        return summary

    # ── Autonomous agent ──────────────────────────────────────────────────

    def _run_autonomous(self) -> str:
        """Autonomous agent with its own LLM brain.

        Loops: screenshot → think → act → report → repeat.
        Uses Gemini text API for decision-making and can execute JARVIS tools.
        Sends interim reports to JARVIS after every iteration.
        """
        goal = self.task.params.get("goal", self.task.description)
        max_iterations = self.task.params.get("max_iterations", 20)
        tools_allowed = self.task.params.get("tools", ["browser_control"])

        self._log_progress(f"Autonomous mode. Goal: {goal}")
        self._log_progress(f"Max iterations: {max_iterations}, Tools: {tools_allowed}")

        conversation: list = []
        iteration_reports: list = []
        last_action_results: list = []

        system_instruction = (
            "You are an autonomous browser agent working for JARVIS (an AI voice assistant). "
            f"Your goal: {goal}\n\n"
            "Each turn you receive a screenshot of the current screen. "
            "Decide what to do next to achieve the goal.\n\n"
            "Respond ONLY with a JSON object (no markdown, no extra text):\n"
            "{\n"
            '  "thought": "What I observe and my reasoning for next action",\n'
            '  "actions": [\n'
            '    {"tool": "browser_control", "args": {"action": "...", ...}}\n'
            '  ],\n'
            '  "report": "Brief status update for the user (1-2 sentences)",\n'
            '  "done": false\n'
            "}\n\n"
            "Available browser_control actions:\n"
            "- go_to: Navigate to URL. Args: {action:'go_to', url:'...'}\n"
            "- click: Click at coordinates. Args: {action:'click', selector:'x,y'}\n"
            "- double_click: Double-click. Args: {action:'double_click', selector:'x,y'}\n"
            "- type: Type text into focused field. Args: {action:'type', text:'...'}\n"
            "- type_and_enter: Type and press Enter. Args: {action:'type_and_enter', text:'...'}\n"
            "- press_key: Press a key. Args: {action:'press_key', key:'tab'|'enter'|'escape'|etc}\n"
            "- hotkey: Key combo. Args: {action:'hotkey', keys:'ctrl+a'|'ctrl+enter'}\n"
            "- scroll: Scroll page. Args: {action:'scroll', direction:'down'|'up'}\n"
            "- wait: Wait for page load. Args: {action:'wait', seconds:2}\n"
            "- new_tab: Open tab. Args: {action:'new_tab', url:'...'}\n"
            "- focus: Bring browser to front. Args: {action:'focus'}\n"
            "- back / forward / refresh: Navigation\n\n"
            "Rules:\n"
            "- Execute 1-3 actions per turn, then wait for the next screenshot\n"
            "- For chat interfaces: click the input box or Tab to it, then type_and_enter\n"
            "- After typing/clicking, your next turn shows the updated screen\n"
            "- Set done=true ONLY when the goal is fully achieved or clearly impossible\n"
            "- Keep reports concise — the user hears them spoken aloud\n"
            "- If something fails, try a different approach\n"
            "- Coordinates in screenshots are screen pixels (top-left = 0,0)\n"
        )

        for i in range(max_iterations):
            if self._stop_event.is_set():
                self._log_progress("Stopped by user")
                break

            # Check for messages from JARVIS / user
            jarvis_messages: list = []
            for msg in self.get_messages():
                self._log_progress(f"Message from {msg.sender}: {msg.content}")
                if msg.content.lower() in ("stop", "quit", "cancel"):
                    return f"Stopped by {msg.sender} after {i} iterations."
                jarvis_messages.append(f"[Message from {msg.sender}]: {msg.content}")

            # 1. Capture screen
            self._log_progress(f"Iteration {i + 1}/{max_iterations}: Capturing screen...")
            screenshot_part = self._take_screenshot_part()
            if screenshot_part is None:
                self._log_progress("Screenshot failed, retrying in 3s...")
                self._stop_event.wait(3)
                continue

            # 2. Build user message: action results + messages + screenshot
            parts: list = []
            if last_action_results:
                parts.append("Results from your last actions:\n" + "\n".join(last_action_results))
            for jm in jarvis_messages:
                parts.append(jm)
            parts.append(f"Iteration {i + 1}/{max_iterations}. Current screen:")
            parts.append(screenshot_part)

            conversation.append({"role": "user", "parts": parts})

            # Trim conversation to avoid token overload (keep last ~12 exchanges)
            if len(conversation) > 24:
                conversation = conversation[:2] + conversation[-22:]

            # 3. Ask LLM what to do
            self._log_progress(f"Iteration {i + 1}: Thinking...")
            try:
                response_text = self._agent_llm_call(system_instruction, conversation)
            except Exception as e:
                self._log_progress(f"LLM error: {e}")
                conversation.pop()  # Remove user msg to keep alternation valid
                self._stop_event.wait(5)
                continue

            conversation.append({"role": "model", "parts": [response_text]})

            # 4. Parse JSON response
            try:
                json_str = response_text.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                decision = json.loads(json_str)
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                self._log_progress(f"Bad JSON from LLM: {e}")
                last_action_results = [
                    f"(Your response was not valid JSON: {str(e)[:80]}. "
                    "Please respond with ONLY a JSON object.)"
                ]
                continue

            thought = decision.get("thought", "")
            actions = decision.get("actions", [])
            report = decision.get("report", "")
            done = decision.get("done", False)

            self._log_progress(f"Thought: {thought[:120]}")

            # 5. Execute actions
            last_action_results = []
            for act in actions:
                tool = act.get("tool", "browser_control")
                tool_args = act.get("args", {})

                if tool not in tools_allowed:
                    last_action_results.append(f"Tool '{tool}' not allowed. Allowed: {tools_allowed}")
                    continue

                try:
                    from tool_registry import execute_tool
                    result = execute_tool(tool, tool_args)
                    short = result[:200] if result else "(no output)"
                    last_action_results.append(f"{tool}({tool_args.get('action', '')}): {short}")
                    self._log_progress(f"  -> {tool}({tool_args.get('action', '')}): {result[:80]}")
                except Exception as e:
                    last_action_results.append(f"{tool} error: {e}")
                    self._log_progress(f"  -> {tool} error: {e}")

                time.sleep(0.3)

            # 6. Send interim report to JARVIS
            if report and self._on_interim_report:
                try:
                    full_report = (
                        f"[Autonomous Agent '{self.name}' — Step {i + 1}/{max_iterations}]\n"
                        f"{report[:1500]}"
                    )
                    self._on_interim_report(self.name, full_report)
                except Exception:
                    pass
            if report:
                iteration_reports.append(report)

            # 7. Done?
            if done:
                self._log_progress(f"Goal achieved: {report}")
                break

            # Brief pause before next iteration
            self._stop_event.wait(1.5)

        summary = f"Autonomous agent finished after {len(iteration_reports)} iterations."
        if iteration_reports:
            summary += f"\nFinal: {iteration_reports[-1][:500]}"
        return summary

    def _take_screenshot_part(self):
        """Capture screen and return as a genai Part for LLM input."""
        try:
            from PIL import ImageGrab
            import io
            from google.genai import types as gtypes

            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return gtypes.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
        except Exception as e:
            self._log_progress(f"Screenshot error: {e}")
            return None

    def _agent_llm_call(self, system_instruction: str, conversation: list) -> str:
        """Call Gemini text API for autonomous decision-making."""
        from google import genai
        from google.genai import types as gtypes

        api_key = self._get_api_key()
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=conversation,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return response.text or ""

    def _get_api_key(self) -> str:
        """Load the Gemini API key from config files."""
        for path in ["config/api_keys.json", "api_key.txt"]:
            try:
                if path.endswith(".json"):
                    with open(path) as f:
                        key = json.load(f).get("gemini_api_key", "")
                else:
                    with open(path) as f:
                        key = f.read().strip()
                if key:
                    return key
            except Exception:
                continue
        raise RuntimeError("No Gemini API key found for autonomous agent")

    # ── Research / Tool / Multi-step runners ──────────────────────────────────

    def _run_research(self) -> str:
        """Research a topic using web search."""
        query = self.task.params.get("query", self.task.description)
        max_results = self.task.params.get("max_results", 5)

        self._log_progress(f"Researching: {query}")
        try:
            from tool_registry import _exec_web_search
            results = _exec_web_search({"query": query, "max_results": max_results})
            self._log_progress(f"Found results")
            return results
        except Exception as e:
            return f"Research error: {e}"

    def _run_tool(self) -> str:
        """Execute a specific tool from tool_registry."""
        tool_name = self.task.params.get("tool_name", "")
        tool_args = self.task.params.get("tool_args", {})

        self._log_progress(f"Running tool: {tool_name}({tool_args})")
        try:
            from tool_registry import execute_tool
            return execute_tool(tool_name, tool_args)
        except Exception as e:
            return f"Tool error: {e}"

    def _run_multi_step(self) -> str:
        """Execute multiple steps sequentially."""
        steps = self.task.params.get("steps", [])
        results = []

        for i, step in enumerate(steps):
            if self._stop_event.is_set():
                break
            step_type = step.get("type", "command")
            step_desc = step.get("description", f"Step {i+1}")
            self._log_progress(f"Step {i+1}/{len(steps)}: {step_desc}")

            if step_type == "command":
                cmd = step.get("command", "")
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=60
                    )
                    output = (result.stdout or "") + (result.stderr or "")
                    results.append(f"Step {i+1} ({step_desc}): {output.strip()[:300]}")
                except Exception as e:
                    results.append(f"Step {i+1} ({step_desc}): ERROR - {e}")
            elif step_type == "tool":
                try:
                    from tool_registry import execute_tool
                    output = execute_tool(step.get("tool_name", ""), step.get("tool_args", {}))
                    results.append(f"Step {i+1} ({step_desc}): {output[:300]}")
                except Exception as e:
                    results.append(f"Step {i+1} ({step_desc}): ERROR - {e}")
            elif step_type == "wait":
                wait_time = step.get("seconds", 5)
                self._stop_event.wait(wait_time)
                results.append(f"Step {i+1}: Waited {wait_time}s")

        return "\n".join(results) or "No steps to execute."

    def _run_general(self) -> str:
        """General task — run as command."""
        return self._run_command()

    def to_dict(self) -> dict:
        """Serialize agent state for reporting."""
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "task": self.task.description,
            "task_type": self.task.task_type,
            "result": self.result[:500] if self.result else "",
            "error": self.error,
            "elapsed_seconds": elapsed,
            "messages_pending": len(self.inbox),
            "progress": self.progress_log[-3:] if self.progress_log else [],
        }


class AgentManager:
    """Manages all spawned agents and the inter-agent message bus.

    Singleton pattern — one manager per JARVIS instance.
    """

    _instance: Optional["AgentManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()
        self._result_log: List[Dict[str, str]] = []  # History of completed results

        # SQLite persistence
        self._db_path = "data/jarvis.db"
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._ensure_table()

        # GUI callback for agent status updates
        self.on_agent_status: Optional[Callable[[str, str, str], None]] = None
        # Callback when an agent reports a result (for JARVIS to speak)
        self.on_agent_result: Optional[Callable[[str, str], None]] = None
        # Callback for live progress updates (agent_name, message)
        self.on_agent_progress: Optional[Callable[[str, str], None]] = None
        # Callback for periodic monitor reports sent to Gemini
        self.on_agent_interim_report: Optional[Callable[[str, str], None]] = None

        # Load persisted agents from previous sessions
        self._load_persisted_agents()

        logger.info("[AgentManager] Initialized")

    # ── SQLite persistence ────────────────────────────────────────────────────

    def _ensure_table(self):
        """Create agents table if it doesn't exist."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                task_type TEXT DEFAULT 'general',
                task_params TEXT DEFAULT '{}',
                status TEXT DEFAULT 'idle',
                result TEXT DEFAULT '',
                error TEXT DEFAULT '',
                created_at REAL,
                started_at REAL,
                finished_at REAL
            )
        """)
        self._db.commit()

    def _persist_agent(self, agent: "Agent"):
        """Save or update an agent record in SQLite."""
        try:
            self._db.execute("""
                INSERT OR REPLACE INTO agents
                    (name, agent_id, task_description, task_type, task_params,
                     status, result, error, created_at, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent.name,
                agent.id,
                agent.task.description,
                agent.task.task_type,
                json.dumps(agent.task.params),
                agent.status.value,
                agent.result[:5000] if agent.result else "",
                agent.error,
                agent.created_at,
                agent.started_at,
                agent.finished_at,
            ))
            self._db.commit()
        except Exception as e:
            logger.error(f"[AgentManager] Persist error: {e}")

    def _delete_persisted_agent(self, name: str):
        """Remove an agent from SQLite."""
        try:
            self._db.execute("DELETE FROM agents WHERE name = ?", (name,))
            self._db.commit()
        except Exception as e:
            logger.error(f"[AgentManager] Delete persist error: {e}")

    def _load_persisted_agents(self):
        """Load agents from SQLite on startup (completed/failed/stopped only, not running)."""
        try:
            rows = self._db.execute(
                "SELECT * FROM agents ORDER BY created_at"
            ).fetchall()
            for row in rows:
                name = row["name"]
                task = AgentTask(
                    description=row["task_description"],
                    task_type=row["task_type"],
                    params=json.loads(row["task_params"] or "{}"),
                )
                agent = Agent(name=name, task=task, agent_id=row["agent_id"])
                agent.status = AgentStatus(row["status"])
                agent.result = row["result"] or ""
                agent.error = row["error"] or ""
                agent.created_at = row["created_at"] or time.time()
                agent.started_at = row["started_at"]
                agent.finished_at = row["finished_at"]

                # Running agents from a previous session are now stale — mark stopped
                if agent.status in (AgentStatus.RUNNING, AgentStatus.WAITING):
                    agent.status = AgentStatus.STOPPED
                    agent.error = "JARVIS restarted — agent was interrupted"
                    agent.finished_at = time.time()

                self._agents[name] = agent
            if rows:
                logger.info(f"[AgentManager] Loaded {len(rows)} agents from database")
        except Exception as e:
            logger.warning(f"[AgentManager] Could not load persisted agents: {e}")

    def spawn_agent(
        self,
        name: str,
        task_description: str,
        task_type: str = "general",
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Spawn a new background agent.

        Args:
            name: Agent display name (e.g. "researcher", "monitor-cpu")
            task_description: What the agent should do
            task_type: "command", "script", "monitor", "research", "tool", "multi_step", "general"
            params: Task-specific parameters

        Returns:
            Status message
        """
        with self._lock:
            # If agent with same name exists and is done, replace it
            if name in self._agents:
                existing = self._agents[name]
                if existing.status in (AgentStatus.RUNNING, AgentStatus.WAITING):
                    return f"Agent '{name}' is already running. Stop it first or use a different name."
                # Remove completed/stopped agent
                del self._agents[name]

            task = AgentTask(
                description=task_description,
                task_type=task_type,
                params=params or {},
            )

            agent = Agent(
                name=name,
                task=task,
                on_result=self._on_agent_result,
                on_status_change=self._on_agent_status_change,
                on_progress=self._on_agent_progress,
            )
            agent._on_interim_report = self._on_agent_interim_report

            self._agents[name] = agent
            agent.start()

            # Persist to database
            self._persist_agent(agent)

            logger.info(f"[AgentManager] Spawned agent '{name}': {task_description}")
            return f"Agent '{name}' spawned and running. Task: {task_description}"

    def stop_agent(self, name: str) -> str:
        """Stop a running agent."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                return f"Agent '{name}' not found."
            if agent.status not in (AgentStatus.RUNNING, AgentStatus.WAITING):
                return f"Agent '{name}' is not running (status: {agent.status.value})."
            agent.stop()
            self._persist_agent(agent)
            return f"Agent '{name}' stopped."

    def get_agent_status(self, name: str = "") -> str:
        """Get status of one or all agents."""
        with self._lock:
            if name:
                agent = self._agents.get(name)
                if not agent:
                    return f"Agent '{name}' not found."
                info = agent.to_dict()
                lines = [
                    f"Agent: {info['name']} ({info['id']})",
                    f"Status: {info['status']}",
                    f"Task: {info['task']}",
                    f"Type: {info['task_type']}",
                ]
                if info['elapsed_seconds']:
                    lines.append(f"Elapsed: {info['elapsed_seconds']}s")
                if info['result']:
                    lines.append(f"Result: {info['result']}")
                if info['error']:
                    lines.append(f"Error: {info['error']}")
                if info['progress']:
                    lines.append("Recent progress:")
                    for p in info['progress']:
                        lines.append(f"  {p}")
                return "\n".join(lines)
            else:
                if not self._agents:
                    return "No agents spawned yet."
                lines = [f"Agents ({len(self._agents)}):"]
                for n, a in self._agents.items():
                    status_icon = {
                        "running": "🟢", "completed": "✅",
                        "failed": "❌", "stopped": "⏹",
                        "idle": "⚪", "waiting": "🟡",
                    }.get(a.status.value, "⚪")
                    elapsed = ""
                    if a.started_at:
                        end = a.finished_at or time.time()
                        elapsed = f" ({round(end - a.started_at, 1)}s)"
                    result_preview = ""
                    if a.result:
                        result_preview = f" — {a.result[:80]}"
                    lines.append(
                        f"  {status_icon} {n} [{a.status.value}]{elapsed}: "
                        f"{a.task.description[:60]}{result_preview}"
                    )
                return "\n".join(lines)

    def get_agent_result(self, name: str) -> str:
        """Get the full result of a completed agent."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                return f"Agent '{name}' not found."
            if agent.status == AgentStatus.RUNNING:
                return f"Agent '{name}' is still running. Progress: {agent.progress_log[-1] if agent.progress_log else 'working...'}"
            if agent.result:
                return f"Agent '{name}' result:\n{agent.result}"
            if agent.error:
                return f"Agent '{name}' failed: {agent.error}"
            return f"Agent '{name}' has no result yet."

    def send_message(self, sender: str, recipient: str, content: str) -> str:
        """Send a message from one agent to another (or broadcast).

        sender/recipient can be agent names, "jarvis", or "all" for broadcast.
        """
        with self._lock:
            msg = AgentMessage(
                sender=sender,
                recipient=recipient,
                content=content,
            )

            if recipient == "all":
                # Broadcast to all agents
                count = 0
                for name, agent in self._agents.items():
                    if name != sender:
                        agent.send_message(msg)
                        count += 1
                return f"Broadcast message sent to {count} agents."
            else:
                agent = self._agents.get(recipient)
                if not agent:
                    return f"Agent '{recipient}' not found."
                agent.send_message(msg)
                return f"Message sent to agent '{recipient}'."

    def list_agents(self) -> List[Dict]:
        """Return all agents as dicts (for GUI)."""
        with self._lock:
            return [a.to_dict() for a in self._agents.values()]

    def remove_agent(self, name: str) -> str:
        """Remove a stopped/completed agent from the registry."""
        with self._lock:
            agent = self._agents.get(name)
            if not agent:
                return f"Agent '{name}' not found."
            if agent.status == AgentStatus.RUNNING:
                agent.stop()
            del self._agents[name]
            self._delete_persisted_agent(name)
            return f"Agent '{name}' removed."

    def _on_agent_result(self, agent_name: str, result: str):
        """Called when an agent finishes."""
        self._result_log.append({
            "agent": agent_name,
            "result": result[:1000],
            "timestamp": time.time(),
        })
        # Persist updated state
        agent = self._agents.get(agent_name)
        if agent:
            self._persist_agent(agent)
        # Notify JARVIS
        if self.on_agent_result:
            try:
                self.on_agent_result(agent_name, result)
            except Exception as e:
                logger.error(f"[AgentManager] Result callback error: {e}")

    def _on_agent_progress(self, agent_name: str, message: str):
        """Called when an agent logs progress."""
        if self.on_agent_progress:
            try:
                self.on_agent_progress(agent_name, message)
            except Exception as e:
                logger.error(f"[AgentManager] Progress callback error: {e}")

    def _on_agent_interim_report(self, agent_name: str, report: str):
        """Called when a monitor agent sends a periodic check result."""
        if self.on_agent_interim_report:
            try:
                self.on_agent_interim_report(agent_name, report)
            except Exception as e:
                logger.error(f"[AgentManager] Interim report callback error: {e}")

    def _on_agent_status_change(self, agent_name: str, status: str, desc: str):
        """Called when an agent's status changes."""
        if self.on_agent_status:
            try:
                self.on_agent_status(agent_name, status, desc)
            except Exception as e:
                logger.error(f"[AgentManager] Status callback error: {e}")
