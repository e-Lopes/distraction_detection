"""
Monitor de recursos computacionais (CPU, RAM, GPU) amostrado em uma thread
separada, para não interferir na medição de latência de inferência.

Funciona tanto para o processo atual (mediapipe/insightface, que rodam
in-process) quanto para um processo filho por PID (openface, que roda via
subprocess).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


@dataclass
class ResourceSummary:
    avg_cpu_pct: Optional[float]
    peak_rss_mb: Optional[float]
    avg_gpu_util_pct: Optional[float]
    peak_gpu_mem_mb: Optional[float]
    gpu_available: bool


class ResourceMonitor:
    """
    Amostra CPU%, RSS (MB) do processo (e opcionalmente de um PID de subprocesso)
    e utilização/memória de GPU (via NVML, se disponível) em intervalos fixos.

    Uso:
        mon = ResourceMonitor(interval_s=0.2)
        mon.start()
        ... trabalho a ser medido ...
        summary = mon.stop()
    """

    def __init__(self, interval_s: float = 0.2, pid: Optional[int] = None, gpu_index: int = 0):
        self.interval_s = interval_s
        self._proc = psutil.Process(pid) if pid is not None else psutil.Process()
        self._gpu_index = gpu_index
        self._cpu_samples: List[float] = []
        self._rss_samples: List[float] = []
        self._gpu_util_samples: List[float] = []
        self._gpu_mem_samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_handle = None
        if _NVML_OK:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            except Exception:
                self._gpu_handle = None

    def _sample_once(self):
        try:
            # primeira chamada de cpu_percent sempre retorna 0.0 (baseline);
            # isso é aceitável pois descartamos essa amostra abaixo.
            cpu = self._proc.cpu_percent(interval=None)
            rss = self._proc.memory_info().rss / (1024 * 1024)
            self._cpu_samples.append(cpu)
            self._rss_samples.append(rss)
        except psutil.NoSuchProcess:
            pass

        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self._gpu_util_samples.append(float(util.gpu))
                self._gpu_mem_samples.append(mem.used / (1024 * 1024))
            except Exception:
                pass

    def _run(self):
        # prime cpu_percent (primeira leitura é sempre 0)
        try:
            self._proc.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            pass
        while not self._stop_event.wait(self.interval_s):
            self._sample_once()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
        # amostra final
        self._sample_once()

        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None
        peak_rss = max(self._rss_samples) if self._rss_samples else None
        avg_gpu_util = (
            sum(self._gpu_util_samples) / len(self._gpu_util_samples)
            if self._gpu_util_samples
            else None
        )
        peak_gpu_mem = max(self._gpu_mem_samples) if self._gpu_mem_samples else None

        return ResourceSummary(
            avg_cpu_pct=round(avg_cpu, 2) if avg_cpu is not None else None,
            peak_rss_mb=round(peak_rss, 2) if peak_rss is not None else None,
            avg_gpu_util_pct=round(avg_gpu_util, 2) if avg_gpu_util is not None else None,
            peak_gpu_mem_mb=round(peak_gpu_mem, 2) if peak_gpu_mem is not None else None,
            gpu_available=self._gpu_handle is not None,
        )


def gpu_is_available() -> bool:
    return _NVML_OK
