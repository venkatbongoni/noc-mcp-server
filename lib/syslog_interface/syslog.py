# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import json
import logging
import requests

netio_logger = logging.getLogger("netio")


class SyslogAPI:
    """
    Syslog API for use with Agent Tools.
    """

    def __init__(self, api_base: str):
        self.api_base = api_base

    def get_syslog(self, ip_address: str, rows: str) -> str:
        data = {"ip_address": ip_address, "rows": rows}
        netio_logger.info(f" 📃 Retrieving syslog from {ip_address}")

        try:
            response = requests.post(
                self.api_base,
                json=data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except requests.RequestException as e:
            netio_logger.error(
                f" ❌ Failed to get syslog from '{ip_address}'. Exception: {e}"
            )
            return ""

        try:
            response_data = response.json()
        except json.JSONDecodeError as json_err:
            netio_logger.error(
                f" ❌ Invalid JSON response for '{ip_address}'. Exception: {json_err}\nResponse content: {response.text}"
            )
            return ""
        return response_data.get("output", "")
