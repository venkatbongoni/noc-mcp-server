from pathlib import Path
import requests
from typing import Dict, Any


class SplunkICSAPI:
    name = "splunk_ics_api"
    description = "Retrieves relevant ICS information by querying a splunk ICS API"
    model_path: Path = None

    def __init__(self, lm: dict, api_url: str, original_query: str = None, 
                signature: str = None, timeout: int = 30):
        """
        Initialize the API Tool.

        Args:
            lm: Language model (not used directly but often required in DSpy tool pattern)
            api_url (str): The URL of the ICS API endpoint
            original_query (str): The original query to be used if provided
            signature (str): A signature to be included in the query
            timeout (int): Timeout in seconds for the API request
        """
        self.lm = lm
        self.api_url = api_url
        self.original_query = original_query
        self.signature = signature
        self.timeout = timeout

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Send a query to the splunk ICS API and return the response.
        Args:
            user_query (str): The user's question to be sent to the splunk ICS API
        Returns:
            dict: The splunk ICS API response containing retrieved rows
        """
        query_to_use = user_query
        
        # Ensure signature is included in the query
        if self.signature:
            if self.signature not in query_to_use:
                query_to_use = f"Looking for information about {self.signature}: {query_to_use}"
            
        print(f"ICS API query: {query_to_use}")
        
        try:
            payload = {"key": query_to_use}
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"ICS API request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"Failed to parse ICS API response: {str(e)}"}
