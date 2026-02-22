"""Task-based multi-LLM routing logic.
Selects the appropriate LLM based on the agent's current task type.
"""

from typing import Dict, Literal
from config_validation import validate_config

TaskType = Literal["conversation", "planning", "reflection", "coding", "default"]

class MultiModelRouter:
    """Routes each task to a specific model configuration."""

    def __init__(self, config):
        self.config = config
        self.models: Dict[TaskType, dict] = {}
        self._init_models()

    def _init_models(self):
        llm_cfg = self.config.get('llm', {})
        models_cfg = llm_cfg.get('models', {})
        for task, model_cfg in models_cfg.items():
            self.models[task] = model_cfg

    def get_model_config(self, task: TaskType) -> dict:
        """Return the model config for the given task."""
        if task in self.models:
            return self.models[task]
        # Fallback to default single model
        default_model = self.config.get('llm', {}).get('model', 'qwen3:4b')
        default_host = self.config.get('llm', {}).get('host', 'http://localhost:11434')
        return {
            'model': default_model,
            'base_url': default_host,
            'temperature': self.config.get('llm', {}).get('temperature', 0.7),
            'num_ctx': self.config.get('llm', {}).get('context_window', 8192),
        }

# Global router instance (set in main after config loaded)
_router = None


def get_llm_for_task(task: TaskType, config):
    """Get the appropriate LLM callable or client for the given task."""
    global _router
    if _router is None:
        _router = MultiModelRouter(config)
    model_cfg = _router.get_model_config(task)
    # Return config; caller (agent node) will instantiate LLM client
    return model_cfg
