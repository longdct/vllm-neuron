#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the first logical NeuronCore on a device with no active process."""

import json
import subprocess


devices = json.loads(subprocess.check_output(["neuron-ls", "-j"], text=True))
for device in devices:
    if not device.get("neuron_processes") and device.get("neuroncore_ids"):
        print(device["neuroncore_ids"][0])
        break
else:
    raise SystemExit("no free logical NeuronCore is available")
