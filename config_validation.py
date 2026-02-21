from pydantic import BaseModel, Field
from typing import Optional, List

class LLMConfig(BaseModel):
    provider: str = "ollama"  # "ollama" | "openrouter"
    model: str = "qwen3:4b"
    api_key: Optional[str] = None  # OpenRouter API key, or use OPENROUTER_API_KEY env
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 256
    context_window: int = 6
    planner_temperature: float = 0.5
    planner_max_tokens: int = 512
    reflector_temperature: float = 0.5
    num_ctx_planner: int = 8192
    num_ctx_reflector: int = 4096
    num_ctx_tool: int = 4096

class VoiceConfig(BaseModel):
    stt_model: str = "large-v3-turbo"
    stt_device: str = "cuda"
    stt_language: Optional[str] = "en"
    stt_beam_size: int = 3
    stt_compute_type: str = "int8"
    tts_engine: str = "piper"
    tts_quality: str = "medium"
    tts_speed: float = 1.0
    preload_tts: bool = True
    stream_tts: bool = True
    max_response_words: int = 0
    min_rms: float = 0.0010
    min_audio_length: float = 1.0
    min_transcript_words: int = 3
    use_vad: bool = True
    vad_aggressiveness: int = 2
    vad_min_speech_ratio: float = 0.08
    recorder_sample_rate: int = 16000
    recorder_silence_threshold: float = 0.012
    recorder_silence_duration: float = 2.5
    speech_start_timeout: float = 3.5
    max_record_seconds: int = 30
    push_to_talk_seconds: int = 5
    silence_timeout: int = 15
    session_end_commands: List[str] = ["goodbye", "stop", "end session", "end conversation"]
    listening_prompt: str = "Listening, Sir."
    wake_ack_prompt: str = ""
    session_end_prompt: str = "Very well, Sir. Standing by."
    ready_beep: bool = True
    ready_beep_only: bool = False
    ready_beep_volume: float = 0.08
    ready_beep_seconds: float = 0.08
    ready_beep_frequency_hz: float = 920
    thinking_prompt: str = "As you wish, Sir..."
    thinking_prompt_each_turn: bool = False

class WakeWordConfig(BaseModel):
    models: List[str] = ["hey_jarvis_v0.1"]
    threshold: float = 0.0015
    wake_confidence: float = 0.0015
    device: Optional[str] = None
    cooldown_seconds: int = 3
    noise_gate_rms: float = 0.005
    show_scores: bool = False

class MemoryConfig(BaseModel):
    db_path: str = "data/jarvis.db"
    chroma_path: str = "data/chroma"
    embedding_model: str = "nomic-embed-text"
    max_memories: int = 3
    chroma_cache_recent: bool = True
    chroma_cache_size: int = 50

class ToolsConfig(BaseModel):
    allowed_directories: List[str] = ["~"]
    max_search_results: int = 5
    command_timeout: int = 30
    time_verify: str = "chrome"

class HeartbeatConfig(BaseModel):
    interval_minutes: int = 30

class ContextConfig(BaseModel):
    max_tokens: int = 256,000
    warning_threshold: float = 0.85
    show_after_each_turn: bool = True

class JarvisConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    debug: bool = False
    log_level: str = "INFO"
    timing: bool = False
    show_latency: bool = True


def validate_config(config_dict: dict) -> JarvisConfig:
    return JarvisConfig(**config_dict)
