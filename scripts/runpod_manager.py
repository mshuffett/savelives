#!/usr/bin/env python3
"""
RunPod Pod Lifecycle Manager
Handles creation, monitoring, and termination of GPU pods for training.
"""

import os
import time
import sys
from typing import Optional, Dict
from dotenv import load_dotenv
import runpod

# Load environment variables
load_dotenv()


class RunPodManager:
    """Manages RunPod pod lifecycle for training jobs."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize RunPod manager with API key."""
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found in environment")
        runpod.api_key = self.api_key

    def list_pods(self) -> list:
        """List all active pods."""
        try:
            pods = runpod.get_pods()
            return pods
        except Exception as e:
            print(f"Error listing pods: {e}")
            return []

    def create_pod(
        self,
        name: str,
        gpu_type: str = "NVIDIA A100-SXM4-80GB",
        image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
        disk_size: int = 50,
        volume_path: str = "/workspace",
        ports: str = "22/tcp,8888/http,6006/http",
        docker_args: str | None = None,
        inject_env: bool = True,
        cloud: str = "COMMUNITY",
    ) -> Optional[Dict]:
        """
        Create a new GPU pod for training.

        Args:
            name: Pod name
            gpu_type: GPU type to provision
            image: Docker image to use
            disk_size: Disk size in GB
            volume_path: Path to mount volume

        Returns:
            Pod information dictionary or None on error
        """
        try:
            print(f"Creating pod '{name}' with {gpu_type}...")
            env = None
            if inject_env:
                # Pass through a minimal set of env vars to the container
                keys = [
                    "WANDB_API_KEY",
                    "WANDB_PROJECT",
                    "WANDB_ENTITY",
                    "HF_TOKEN",
                    "CHECKPOINT_DIR",
                    "DATA_DIR",
                    "LOG_DIR",
                ]
                env = {k: os.getenv(k) for k in keys if os.getenv(k)}

            pod = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu_type,
                cloud_type=cloud,
                support_public_ip=True,
                start_ssh=True,
                volume_in_gb=disk_size,
                container_disk_in_gb=disk_size,
                volume_mount_path=volume_path,
                ports=ports,
                docker_args=docker_args or "",
                env=env,
            )

            print(f"✓ Pod created: {pod['id']}")
            print(f"  Status: {pod.get('desiredStatus', 'unknown')}")

            return pod
        except Exception as e:
            print(f"✗ Error creating pod: {e}")
            return None

    def get_pod(self, pod_id: str) -> Optional[Dict]:
        """Get pod information by ID."""
        try:
            pod = runpod.get_pod(pod_id)
            return pod
        except Exception as e:
            print(f"Error getting pod {pod_id}: {e}")
            return None

    def wait_for_running(self, pod_id: str, timeout: int = 600) -> bool:
        """
        Wait for pod to reach running state.

        Args:
            pod_id: Pod ID to monitor
            timeout: Maximum wait time in seconds

        Returns:
            True if pod is running, False on timeout
        """
        print(f"Waiting for pod {pod_id} to start (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            pod = self.get_pod(pod_id)
            if not pod:
                return False

            status = pod.get("desiredStatus", "unknown")
            runtime = pod.get("runtime") or {}
            # Some accounts return `uptimeSeconds` at top-level instead
            uptime = runtime.get("uptimeInSeconds") if isinstance(runtime, dict) else None
            if not uptime:
                uptime = pod.get("uptimeSeconds")

            print(f"  Status: {status}, Uptime: {uptime}s", end="\r")

            if status == "RUNNING" and uptime is not None and int(uptime) >= 10:
                print(f"\n✓ Pod is running (uptime: {runtime_status}s)")
                return True

            time.sleep(5)

        print(f"\n✗ Timeout waiting for pod to start")
        return False

    def stop_pod(self, pod_id: str) -> bool:
        """Stop a running pod."""
        try:
            print(f"Stopping pod {pod_id}...")
            runpod.stop_pod(pod_id)
            print(f"✓ Pod stopped")
            return True
        except Exception as e:
            print(f"✗ Error stopping pod: {e}")
            return False

    def resume_pod(self, pod_id: str) -> bool:
        """Resume a stopped pod."""
        try:
            print(f"Resuming pod {pod_id}...")
            runpod.resume_pod(pod_id)
            print(f"✓ Pod resumed")
            return True
        except Exception as e:
            print(f"✗ Error resuming pod: {e}")
            return False

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate and delete a pod."""
        try:
            print(f"Terminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
            print(f"✓ Pod terminated")
            return True
        except Exception as e:
            print(f"✗ Error terminating pod: {e}")
            return False

    def get_ssh_command(self, pod_id: str) -> Optional[str]:
        """Get SSH command to connect to pod."""
        pod = self.get_pod(pod_id)
        if not pod:
            return None

        runtime = pod.get("runtime") or {}
        ssh_port = None
        pod_ip = None
        if isinstance(runtime, dict):
            ssh_info = runtime.get("ports", {}).get("22/tcp", [{}])
            if ssh_info:
                ssh_port = ssh_info[0].get("publicPort")
            pod_ip = runtime.get("ip")

        if ssh_port and pod_ip:
            return f"ssh root@{pod_ip} -p {ssh_port}"
        return None


def main():
    """CLI interface for RunPod manager."""
    import argparse

    parser = argparse.ArgumentParser(description="RunPod Pod Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List pods
    subparsers.add_parser("list", help="List all pods")

    # Create pod
    create_parser = subparsers.add_parser("create", help="Create a new pod")
    create_parser.add_argument("name", help="Pod name")
    create_parser.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB", help="GPU type")
    create_parser.add_argument("--image", default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel")
    create_parser.add_argument("--disk", type=int, default=50, help="Disk size in GB")
    create_parser.add_argument("--ports", default="22/tcp,8888/http,6006/http")
    create_parser.add_argument("--cmd", default="")
    create_parser.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY","SECURE","ALL"]) 

    # Get pod info
    info_parser = subparsers.add_parser("info", help="Get pod information")
    info_parser.add_argument("pod_id", help="Pod ID")

    # Stop pod
    stop_parser = subparsers.add_parser("stop", help="Stop a pod")
    stop_parser.add_argument("pod_id", help="Pod ID")

    # Resume pod
    resume_parser = subparsers.add_parser("resume", help="Resume a pod")
    resume_parser.add_argument("pod_id", help="Pod ID")

    # Terminate pod
    terminate_parser = subparsers.add_parser("terminate", help="Terminate a pod")
    terminate_parser.add_argument("pod_id", help="Pod ID")

    # SSH command
    ssh_parser = subparsers.add_parser("ssh", help="Get SSH command")
    ssh_parser.add_argument("pod_id", help="Pod ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = RunPodManager()

    if args.command == "list":
        pods = manager.list_pods()
        print(f"\nFound {len(pods)} pods:")
        for pod in pods:
            print(f"  - {pod['name']} ({pod['id']}): {pod.get('desiredStatus', 'unknown')}")

    elif args.command == "create":
        pod = manager.create_pod(
            name=args.name,
            gpu_type=args.gpu,
            image=args.image,
            disk_size=args.disk,
            ports=args.ports,
            docker_args=args.cmd,
        )
        if pod:
            manager.wait_for_running(pod["id"])
            ssh_cmd = manager.get_ssh_command(pod["id"])
            if ssh_cmd:
                print(f"\nSSH command: {ssh_cmd}")

    elif args.command == "info":
        pod = manager.get_pod(args.pod_id)
        if pod:
            print(f"\nPod: {pod['name']} ({pod['id']})")
            print(f"Status: {pod.get('desiredStatus', 'unknown')}")
            print(f"GPU: {pod.get('gpuCount', 0)}x {pod.get('gpuType', 'unknown')}")
            runtime = pod.get("runtime", {})
            print(f"Uptime: {runtime.get('uptimeInSeconds', 0)}s")

    elif args.command == "stop":
        manager.stop_pod(args.pod_id)

    elif args.command == "resume":
        manager.resume_pod(args.pod_id)

    elif args.command == "terminate":
        manager.terminate_pod(args.pod_id)

    elif args.command == "ssh":
        ssh_cmd = manager.get_ssh_command(args.pod_id)
        if ssh_cmd:
            print(f"\n{ssh_cmd}")
        else:
            print("Could not get SSH command")


if __name__ == "__main__":
    main()
