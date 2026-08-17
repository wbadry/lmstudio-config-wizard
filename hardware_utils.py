import csv
import io
import json
import platform
import re
import subprocess
from pathlib import Path

import psutil
from rich.console import Console
from rich.table import Table


UNKNOWN_GPU = "Unknown"


def _run_command(command, timeout=5):
    """Run a hardware probe and return its output, or None if unavailable."""
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def _parse_nvidia_smi(output):
    gpus = []
    if not output:
        return gpus

    for row in csv.reader(io.StringIO(output)):
        if len(row) < 2:
            continue

        name = row[0].strip()
        memory_match = re.search(r"\d+(?:\.\d+)?", row[1])
        if not name or not memory_match:
            continue

        gpus.append((name, round(float(memory_match.group()))))

    return gpus


def _get_nvidia_gpus():
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,nounits,noheader",
        ]
    )
    return _parse_nvidia_smi(output)


def _parse_windows_gpu_json(output):
    if not output or not output.strip():
        return []

    try:
        records = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return []

    gpus = []
    for record in records:
        if not isinstance(record, dict):
            continue

        name = str(record.get("name") or "").strip()
        try:
            memory_bytes = int(record.get("memory_bytes") or 0)
        except (TypeError, ValueError):
            memory_bytes = 0

        if name:
            gpus.append((name, round(memory_bytes / (1024 * 1024))))

    return gpus


def _get_windows_gpus():
    # Driver registry data uses a 64-bit VRAM value when available. This avoids
    # the 4 GiB limit of Win32_VideoController.AdapterRAM on modern GPUs.
    script = r"""
$videoRoot = 'HKLM:\SYSTEM\CurrentControlSet\Control\Video'
$adapters = Get-ChildItem -Path $videoRoot -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.PSChildName -eq '0000' } |
    ForEach-Object {
        $properties = Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue
        $name = $properties.'HardwareInformation.AdapterString'
        if ($name) {
            $memory = $properties.'HardwareInformation.qwMemorySize'
            if (-not $memory) {
                $legacyMemory = $properties.'HardwareInformation.MemorySize'
                if ($legacyMemory -is [byte[]] -and $legacyMemory.Length -ge 4) {
                    $memory = [BitConverter]::ToUInt32($legacyMemory, 0)
                } elseif ($legacyMemory) {
                    $memory = [uint64]$legacyMemory
                }
            }
            [PSCustomObject]@{
                name = [string]$name
                memory_bytes = [uint64]($memory | Select-Object -First 1)
            }
        }
    }
@($adapters) | ConvertTo-Json -Compress
"""
    output = _run_command(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    )
    return _parse_windows_gpu_json(output)


def _parse_lspci(output):
    gpus = []
    if not output:
        return gpus

    controller_pattern = re.compile(
        r"(?:VGA compatible controller|3D controller|Display controller):\s*(.+)"
    )
    for line in output.splitlines():
        match = controller_pattern.search(line)
        if match:
            name = re.sub(r"\s+\(rev [^)]+\)$", "", match.group(1)).strip()
            if name:
                gpus.append((name, 0))

    return gpus


def _get_linux_vram_mb():
    memory_values = []
    for memory_file in Path("/sys/class/drm").glob(
        "card[0-9]*/device/mem_info_vram_total"
    ):
        try:
            memory_values.append(int(memory_file.read_text(encoding="ascii").strip()))
        except (OSError, ValueError):
            continue

    if not memory_values:
        return 0
    return round(max(memory_values) / (1024 * 1024))


def _get_linux_gpus():
    gpus = _parse_lspci(_run_command(["lspci"]))
    if not gpus:
        return []

    vram_mb = _get_linux_vram_mb()
    # Prefer a discrete AMD/NVIDIA adapter over integrated graphics.
    preferred_index = next(
        (
            index
            for index, (name, _) in enumerate(gpus)
            if any(vendor in name.lower() for vendor in ("amd", "ati", "nvidia"))
        ),
        0,
    )
    name, _ = gpus[preferred_index]
    return [(name, vram_mb)]


def _get_macos_gpus():
    output = _run_command(["system_profiler", "SPDisplaysDataType"])
    if not output:
        return []

    gpu_name = None
    gpu_memory_mb = 0
    for line in output.splitlines():
        if "Chipset Model:" in line or "Model:" in line:
            gpu_name = line.split(":", 1)[1].strip()
        elif "VRAM" in line:
            match = re.search(r"(\d+(?:\.\d+)?)\s*([MG])B", line, re.IGNORECASE)
            if match:
                multiplier = 1024 if match.group(2).upper() == "G" else 1
                gpu_memory_mb = round(float(match.group(1)) * multiplier)

    return [(gpu_name, gpu_memory_mb)] if gpu_name else []


def _select_gpu(gpus):
    if not gpus:
        return UNKNOWN_GPU, 0
    return max(gpus, key=lambda gpu: gpu[1])


def get_gpu_info():
    """Return the best detected GPU name and its dedicated memory in MiB."""
    system = platform.system()

    if system in {"Windows", "Linux"}:
        nvidia_gpus = _get_nvidia_gpus()
        if nvidia_gpus:
            return _select_gpu(nvidia_gpus)

    if system == "Windows":
        return _select_gpu(_get_windows_gpus())
    if system == "Linux":
        return _select_gpu(_get_linux_gpus())
    if system == "Darwin":
        return _select_gpu(_get_macos_gpus())

    return UNKNOWN_GPU, 0


def get_system_info():
    cpu = platform.processor() or platform.machine()
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 1)
    gpu_name, gpu_memory_mb = get_gpu_info()

    return {
        "CPU": cpu,
        "Cores (Physical/Logical)": f"{physical_cores} / {logical_cores}",
        "RAM (GB)": ram_gb,
        "GPU": gpu_name,
        "GPU Memory (MB)": gpu_memory_mb,
    }


def display_system_info():
    console = Console()
    table = Table(title="System Hardware Info", title_style="bold cyan")
    table.add_column("Component", style="bold magenta")
    table.add_column("Details", style="green")

    info = get_system_info()
    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


def get_hardware_profile():
    system_info = get_system_info()
    return {
        "cpu": system_info["CPU"],
        "physical_cores": system_info["Cores (Physical/Logical)"].split(" / ")[0],
        "logical_cores": system_info["Cores (Physical/Logical)"].split(" / ")[1],
        "ram_gb": system_info["RAM (GB)"],
        "gpu": system_info["GPU"],
        "gpu_memory_mb": system_info["GPU Memory (MB)"],
    }


if __name__ == "__main__":
    display_system_info()
