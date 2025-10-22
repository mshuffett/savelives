#!/usr/bin/env python3
"""
Training Monitor and Auto-Restart Script
Monitors RunPod training pod and auto-restarts on failure.
"""

import os
import time
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import runpod

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.runpod_manager import RunPodManager

load_dotenv()


class TrainingMonitor:
    """Monitors training pod and auto-restarts on failure."""

    def __init__(self, pod_id: str, check_interval: int = 300):
        """
        Initialize training monitor.

        Args:
            pod_id: RunPod pod ID to monitor
            check_interval: Seconds between health checks (default 5 min)
        """
        self.pod_id = pod_id
        self.check_interval = check_interval
        self.manager = RunPodManager()
        self.start_time = datetime.now()
        self.check_count = 0
        self.restart_count = 0

    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def check_pod_health(self) -> bool:
        """
        Check if pod is healthy.

        Returns:
            True if pod is running and healthy, False otherwise
        """
        pod = self.manager.get_pod(self.pod_id)
        if not pod:
            self.log(f"❌ Pod {self.pod_id} not found")
            return False

        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime", {})
        uptime = runtime.get("uptimeInSeconds", 0)

        if status != "RUNNING":
            self.log(f"⚠️  Pod status: {status} (not RUNNING)")
            return False

        if uptime < 10:
            self.log(f"⚠️  Pod uptime too low: {uptime}s")
            return False

        # Check GPU utilization (if available)
        gpu_util = runtime.get("gpuUtilization")
        if gpu_util is not None:
            self.log(f"✓ Pod healthy - Uptime: {uptime}s, GPU: {gpu_util}%")
        else:
            self.log(f"✓ Pod healthy - Uptime: {uptime}s")

        return True

    def check_training_logs(self, pod_id: str) -> bool:
        """
        Check if training is making progress via logs.

        Returns:
            True if training appears to be progressing
        """
        # TODO: Implement log checking via SSH or RunPod API
        # For now, just check pod health
        return True

    def restart_training(self):
        """Attempt to restart training on the pod."""
        self.log("🔄 Attempting to restart training...")
        self.restart_count += 1

        # Resume pod if stopped
        pod = self.manager.get_pod(self.pod_id)
        if pod and pod.get("desiredStatus") == "EXITED":
            self.log("Resuming stopped pod...")
            self.manager.resume_pod(self.pod_id)
            time.sleep(30)  # Wait for pod to start

        # TODO: Implement automatic training restart via SSH
        # For now, just log the issue
        self.log(f"⚠️  Manual intervention may be required")
        self.log(f"   SSH into pod and run: python src/train.py --config config/salm_lora.yaml --resume")

    def run(self):
        """Run monitoring loop."""
        self.log(f"🔍 Starting training monitor for pod {self.pod_id}")
        self.log(f"   Check interval: {self.check_interval}s ({self.check_interval/60:.1f} min)")

        try:
            while True:
                self.check_count += 1
                elapsed = (datetime.now() - self.start_time).total_seconds()

                self.log(f"\n--- Check #{self.check_count} (Elapsed: {elapsed/60:.1f} min) ---")

                # Health check
                if not self.check_pod_health():
                    self.log("❌ Health check failed!")
                    self.restart_training()
                else:
                    # Check training progress
                    if not self.check_training_logs(self.pod_id):
                        self.log("⚠️  Training may be stalled")

                # Sleep until next check
                self.log(f"💤 Sleeping for {self.check_interval}s...\n")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            self.log("\n🛑 Monitor stopped by user")
            self.print_stats()

    def print_stats(self):
        """Print monitoring statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log("\n=== Monitoring Statistics ===")
        self.log(f"Total checks: {self.check_count}")
        self.log(f"Total restarts: {self.restart_count}")
        self.log(f"Total runtime: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Monitor RunPod training and auto-restart on failure")
    parser.add_argument("pod_id", help="RunPod pod ID to monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 min)",
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(args.pod_id, args.interval)
    monitor.run()


if __name__ == "__main__":
    main()
