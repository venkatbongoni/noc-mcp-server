# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number],
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from .syslog_patterns import CiscoSyslogModel, CiscoSyslogMnemonicModel, InputMessageHeaderModel
from .hardware_patterns import IOSXRDevicePathModel


print("=== Testing CiscoSyslogModel ===")
syslog_examples = [
    "Mar 18 11:10:45 10.128.18.94 33408 *Mar 18 07:10:45.192: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : test message",
    "Sep  1 12:04:40 10.90.91.66 1071: *Sep  1 09:31:33.939: %OSPF-4-ERRRCV: Received invalid packet",
]

for example in syslog_examples:
    result = CiscoSyslogModel.try_parse(example)
    if result:
        print(result)
        print("  Log:", result.system_message)
    else:
        print("Invalid syslog message:", example)

print("=== Testing CiscoSyslogMnemonicModel ===")
syslog_examples = [
    "%OSPF-4-FLOOD_WAR",
    "%OS-XTC-5-SR_POLICY_UPDOWN",
    "%BGP-3-ADJCHANGE",
    "%SYS-5-CONFIG_I",
    "Mar 18 11:10:45 10.128.18.94 33408 *Mar 18 07:10:45.192: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : test",
    "Sep  1 12:04:40 10.90.91.66 1071: *Sep  1 09:31:33.939: %OSPF-4-ERRRCV: Received invalid packet: ",
    "Invalid-Example",  # negative test
]

for example in syslog_examples:
    result = CiscoSyslogMnemonicModel.try_parse(example)
    if result:
        print(result)
        print("  Facility:", result.facility)
        print("  Severity:", result.severity)
        print("  Mnemonic:", result.mnemonic)
        print("  Signature:", result.signature)
        print("  As dict:", result.model_dump())
    else:
        print("Invalid syslog mnemonic:", example)


print("\n=== Testing IOSXRDevicePathModel ===")
path_examples = [
    " RP/0/RP0/CPU0",
    " RP/0/RP1/CPU1",
    " LC/0/0/CPU0",
    " 0/0/0/CPU0",
    "Mar 18 11:10:45 10.128.18.94 33408 *Mar 18 07:10:45.192: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : test",
    " GigabitEthernet0/0/1",  # negative test
    " TenGigE0/0/0/1",        # negative test
    " INVALID/PATH"           # negative test
]

for path in path_examples:
    result = IOSXRDevicePathModel.try_parse(path)
    if result:
        print(result)
        print("  Path:", result.path)
    else:
        print("Invalid device path:", path)


print("\n=== Testing InputMessageHeaderModel ===")
syslog_examples = [
    "Aug 29 13:47:30 10.90.91.10 117: *Aug 29 13:26:31.762: %OSPF-4-FLOOD_WAR: Process 1 re-originates LSA ID 10.90.91.41 type-2 adv-rtr 10.90.91.30 in area 0",
    "Sep  1 12:04:40 10.90.91.66 1071: *Sep  1 09:31:33.939: %OSPF-4-ERRRCV: Received invalid packet: mismatched area ID from backbone area from 10.90.91.73, GigabitEthernet0/1",
    "Mar 18 11:10:42 10.128.18.94 33392 *Mar 18 07:10:42.482: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : SR policy 'test_a_97_192.168.252.36' (color 97, end-point 192.168.252.36) state changed to DOWN (no valid candidate-path)",
    "Mar 18 11:10:45 10.128.18.94 33408 *Mar 18 07:10:45.192: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : SR policy 'test_a_93_192.168.252.34' (color 93, end-point 192.168.252.34) state changed to UP",
    "Mar 18 11:10:45 10.128.18.94 *Mar 18 07:10:45.192: RP/0/RP0/CPU0 %OS-XTC-5-SR_POLICY_UPDOWN : Invalid input header",  # negative test
]

for example in syslog_examples:
    result = InputMessageHeaderModel.try_parse(example)
    if result:
        print(result)
        print("  Timestamp:", result.timestamp)
        print("  System_IP:", result.system_ip)
        print("  Message_ID:", result.message_id)
        print("  As dict:", result.model_dump())
    else:
        print("Invalid input header:", example)
