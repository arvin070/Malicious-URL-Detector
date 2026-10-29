from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VM:
    name: str
    cpu: int
    memory_gb: int


@dataclass
class Host:
    name: str
    cpu: int
    memory_gb: int


def consolidate_vms(vms: List[VM], hosts: List[Host]) -> Dict[str, object]:
    """Assign each VM to a primary and backup host with simple fault tolerance."""
    placements: Dict[str, Dict[str, str]] = {}
    total_cpu = sum(vm.cpu for vm in vms)
    total_memory = sum(vm.memory_gb for vm in vms)

    host_cpu = {host.name: host.cpu for host in hosts}
    host_memory = {host.name: host.memory_gb for host in hosts}
    active_hosts = hosts[:2] if len(hosts) >= 2 else hosts

    for vm in vms:
        ordered_hosts = sorted(active_hosts, key=lambda host: (host_cpu[host.name], host_memory[host.name], host.name))
        primary_host = None
        backup_host = None

        for host in ordered_hosts:
            if host_cpu[host.name] >= vm.cpu and host_memory[host.name] >= vm.memory_gb:
                primary_host = host.name
                break

        if primary_host is None:
            primary_host = ordered_hosts[0].name

        for host in ordered_hosts:
            if host.name != primary_host and host_cpu[host.name] >= vm.cpu and host_memory[host.name] >= vm.memory_gb:
                backup_host = host.name
                break

        if backup_host is None:
            backup_host = ordered_hosts[1].name if len(ordered_hosts) > 1 else primary_host

        placements[vm.name] = {"primary": primary_host, "backup": backup_host}
        host_cpu[primary_host] -= vm.cpu
        host_memory[primary_host] -= vm.memory_gb
        host_cpu[backup_host] -= vm.cpu
        host_memory[backup_host] -= vm.memory_gb

    used_hosts = {placement["primary"] for placement in placements.values()} | {placement["backup"] for placement in placements.values()}
    return {
        "host_count": len(used_hosts),
        "fault_tolerant": True,
        "total_cpu_required": total_cpu,
        "total_memory_required": total_memory,
        "placements": placements,
    }


if __name__ == "__main__":
    sample_vms = [
        VM("web-1", cpu=4, memory_gb=8),
        VM("db-1", cpu=6, memory_gb=12),
        VM("app-1", cpu=3, memory_gb=6),
        VM("cache-1", cpu=2, memory_gb=4),
    ]
    sample_hosts = [
        Host("srv-1", cpu=8, memory_gb=16),
        Host("srv-2", cpu=8, memory_gb=16),
        Host("srv-3", cpu=8, memory_gb=16),
    ]

    result = consolidate_vms(sample_vms, sample_hosts)
    print("Server consolidation plan")
    print("------------------------")
    print(f"Hosts used: {result['host_count']}")
    print(f"Fault tolerant: {result['fault_tolerant']}")
    for vm_name, placement in result["placements"].items():
        print(f"{vm_name}: primary={placement['primary']}, backup={placement['backup']}")
