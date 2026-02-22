import psutil
import platform
from typing import Dict, Any

def get_cpu_percent(interval: float = 0.5) -> float:
    return psutil.cpu_percent(interval=interval)

def get_ram_usage() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percent': mem.percent,
        'available_gb': mem.available / (1024**3)
    }

def get_disk_usage(path: str = '/') -> Dict[str, Any]:
    disk = psutil.disk_usage(path)
    return {
        'total_gb': disk.total / (1024**3),
        'used_gb': disk.used / (1024**3),
        'free_gb': disk.free / (1024**3),
        'percent': disk.percent
    }

def get_network_stats() -> Dict[str, Any]:
    net_io = psutil.net_io_counters()
    return {
        'bytes_sent_mb': net_io.bytes_sent / (1024**2),
        'bytes_recv_mb': net_io.bytes_recv / (1024**2),
        'packets_sent': net_io.packets_sent,
        'packets_recv': net_io.packets_recv
    }

def get_system_summary() -> str:
    cpu = get_cpu_percent()
    ram = get_ram_usage()
    disk = get_disk_usage()
    return (
        f"System Status:\n"
        f"- CPU Usage: {cpu:.1f}%\n"
        f"- RAM: {ram['used_gb']:.1f}/{ram['total_gb']:.1f} GB ({ram['percent']:.0f}%)\n"
        f"- Disk: {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB ({disk['percent']:.0f}%)\n"
        f"- OS: {platform.system()} {platform.release()}"
    )

def format_for_speech() -> str:
    cpu = get_cpu_percent()
    ram = get_ram_usage()
    return f"CPU is at {cpu:.0f} percent, RAM at {ram['percent']:.0f} percent."
