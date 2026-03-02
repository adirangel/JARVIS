"""System stats endpoint - CPU, RAM, disk."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system")
async def get_system():
    """Return CPU, RAM, and disk usage."""
    from tools.system_monitor import get_cpu_percent, get_ram_usage, get_disk_usage
    import platform

    cpu = get_cpu_percent()
    ram = get_ram_usage()
    disk_path = "C:\\" if platform.system() == "Windows" else "/"
    disk = get_disk_usage(disk_path)

    return {
        "cpu_percent": cpu,
        "ram": {
            "used_gb": round(ram["used_gb"], 2),
            "total_gb": round(ram["total_gb"], 2),
            "percent": ram["percent"],
        },
        "disk": {
            "used_gb": round(disk["used_gb"], 2),
            "total_gb": round(disk["total_gb"], 2),
            "percent": disk["percent"],
        },
        "os": f"{platform.system()} {platform.release()}",
    }
