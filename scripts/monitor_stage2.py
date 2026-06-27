"""Live monitor for stage-2 training."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]


def _latest_event(log_dir: Path) -> Path | None:
    files = sorted(log_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _read_scalars(event_path: Path) -> dict:
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    out = {}
    for tag in ("train/loss", "val/qwk_dr", "val/qwk_me", "val/qwk_mean"):
        vals = ea.Scalars(tag)
        out[tag] = vals[-1] if vals else None
    return out


def _find_stage2_pids() -> list[int]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'stage2_(finetune|crossval)\\.py' } | "
            "ForEach-Object { $_.ProcessId }"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    pids = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _process_stats(pid: int) -> str:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
            "if ($null -eq $p) { return }; "
            "[string]::Format('pid={0} cpu={1:N1}s ws={2:N1}GB', "
            "$p.Id, $p.CPU, $p.WorkingSet64 / 1GB)"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return res.stdout.strip()


def _gpu_stats() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    line = res.stdout.splitlines()[0].strip() if res.stdout.splitlines() else ""
    if not line:
        return "gpu=n/a"
    util, mem_used, mem_total, temp, power = [x.strip() for x in line.split(",")]
    return f"gpu={util}% mem={mem_used}/{mem_total}MiB temp={temp}C power={power}W"


def _ckpt_status(ckpt_dir: Path) -> str:
    files = sorted(ckpt_dir.rglob("best_qwk.pth"), key=lambda p: p.stat().st_mtime)
    if not files:
        return "best_qwk=missing"
    ckpt = files[-1]
    ts = time.strftime("%H:%M:%S", time.localtime(ckpt.stat().st_mtime))
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    return f"best_qwk={ts} {size_mb:.1f}MB"


def _fmt_scalar(name: str, item) -> str:
    if item is None:
        return f"{name}=n/a"
    return f"{name}@{item.step}={item.value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="runs/stage2")
    ap.add_argument("--ckpt-dir", default="checkpoints/stage2")
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()

    log_dir = (ROOT / args.log_dir).resolve()
    ckpt_dir = (ROOT / args.ckpt_dir).resolve()
    print(f"monitor log_dir={log_dir}", flush=True)
    print(f"monitor ckpt_dir={ckpt_dir}", flush=True)

    last_stamp = None
    while True:
        event_path = _latest_event(log_dir)
        pids = _find_stage2_pids()
        gpu = _gpu_stats()
        ckpt = _ckpt_status(ckpt_dir)
        now = time.strftime("%H:%M:%S")

        if event_path is None:
            print(f"[{now}] no event file yet | {gpu} | {ckpt}", flush=True)
            time.sleep(args.interval)
            continue

        stamp = (str(event_path), event_path.stat().st_mtime, event_path.stat().st_size)
        try:
            scalars = _read_scalars(event_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{now}] event read failed: {exc} | {gpu} | {ckpt}", flush=True)
            time.sleep(args.interval)
            continue

        proc = "proc=n/a"
        if pids:
            proc = " | ".join(filter(None, (_process_stats(pid) for pid in pids))) or proc

        changed = stamp != last_stamp
        marker = "*" if changed else "-"
        print(
            f"[{now}] {marker} "
            f"{_fmt_scalar('loss', scalars['train/loss'])} "
            f"{_fmt_scalar('qwk_dr', scalars['val/qwk_dr'])} "
            f"{_fmt_scalar('qwk_me', scalars['val/qwk_me'])} "
            f"{_fmt_scalar('qwk_mean', scalars['val/qwk_mean'])} | "
            f"{gpu} | {proc} | {ckpt}",
            flush=True,
        )
        last_stamp = stamp
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
