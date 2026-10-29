import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server_consolidation_demo import Host, VM, consolidate_vms


class ConsolidationDemoTests(unittest.TestCase):
    def test_consolidation_assigns_primary_and_backup_hosts(self):
        vms = [
            VM("web-1", cpu=4, memory_gb=8),
            VM("db-1", cpu=6, memory_gb=12),
            VM("app-1", cpu=3, memory_gb=6),
            VM("cache-1", cpu=2, memory_gb=4),
        ]
        hosts = [
            Host("srv-1", cpu=8, memory_gb=16),
            Host("srv-2", cpu=8, memory_gb=16),
            Host("srv-3", cpu=8, memory_gb=16),
        ]

        plan = consolidate_vms(vms, hosts)

        self.assertEqual(plan["host_count"], 2)
        self.assertTrue(plan["fault_tolerant"])

        for vm in vms:
            placement = plan["placements"][vm.name]
            self.assertNotEqual(placement["primary"], placement["backup"])
            self.assertIn(placement["primary"], {host.name for host in hosts})
            self.assertIn(placement["backup"], {host.name for host in hosts})


if __name__ == "__main__":
    unittest.main()
