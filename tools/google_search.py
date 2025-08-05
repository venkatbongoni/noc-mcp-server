# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import dspy
import json
import logging
import requests
from bs4 import BeautifulSoup
from agents.syslog_agent import SyslogAgent

netio_logger = logging.getLogger("netio")


class GoogleSearchRAG:
    """
    Retrieval Augmented Google Search based on Custom Search Engine (CSE).
    """
    def __init__(self, lm, api_key, cse_id):
        self.google_api_key = api_key
        self.cse_id = cse_id
        self.lm = lm
        self.TRUNCATE_SCRAPED_TEXT = 50000  # Adjust based on your model's context window

        prompt = "Provide a short, factoid google search term based on search query."
        self.query_expansion = dspy.Predict("search_query -> search_term", prompt)

    def google_search(self, search_item, api_key, cse_id, search_depth, site_filter=None):
        """Running a web search using the Google API and Custom Search Engine."""
        service_url = 'https://www.googleapis.com/customsearch/v1'
        params = {'q': search_item, 'key': api_key, 'cx': cse_id, 'num': search_depth}
        try:
            response = requests.get(service_url, params=params)
            response.raise_for_status()
            results = response.json()

            if 'items' in results:
                if site_filter is not None:
                    filtered_results = [result for result in results['items'] if site_filter in result['link']]
                    if filtered_results:
                        return filtered_results
                    else:
                        print(f"No results with {site_filter} found.")
                        return []
                else:
                    if 'items' in results:
                        return results['items']
                    else:
                        print("No search results found.")
                        return []
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the search: {e}")
            return []

    def retrieve_content(self, url, max_tokens=None):
        """Scrape Web Page Content"""
        if not max_tokens:
            max_tokens = self.TRUNCATE_SCRAPED_TEXT
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            text = soup.get_text(separator=' ', strip=True)
            characters = max_tokens * 4  # Approximate conversion
            text = text[:characters]
            return text
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return None

    def extract_content(self, content, search_term, character_limit):
        """Summarize Content"""
        prompt = (
            f"You are an AI assistant tasked with extracting the answer relevant to '{search_term}'. "
            f"Please provide a short, factoid answer in {character_limit} characters or less."
            )
        content_search = dspy.Predict("content, search_term -> answer", prompt)
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            answer = content_search(content=content, search_term=search_term).answer
        return answer

    def get_search_results(self, search_items, character_limit) -> list[dict]:
        """Create a Structured Dictionary of the summarized content."""
        results_list = []
        for idx, item in enumerate(search_items, start=1):
            url = item.get('link')
            snippet = item.get('snippet', '')
            web_content = self.retrieve_content(url, self.TRUNCATE_SCRAPED_TEXT)
            if web_content is None:
                print(f"Error: skipped URL: {url}")
            else:
                summary = self.extract_content(web_content, self.search_term, character_limit)
                result_dict = {
                    'order': idx,
                    'link': url,
                    'title': snippet,
                    'Summary': summary
                }
                results_list.append(result_dict)
        return results_list

    def print_google_search_results(self, results):
        for result in results:
            print(f"Search order: {result['order']}")
            print(f"Link: {result['link']}")
            print(f"Snippet: {result['title']}")
            print(f"Summary: {result['Summary']}")
            print('-' * 80)

    def augmented_generation(self, search_query, results: list[dict]):
        """Generate a RAG Response to the search query."""
        prompt = (
            f"The user will provide a dictionary of search results in JSON format for "
            f"search query {self.search_term}. Based on the search results provided "
            f"by the user, provide a detailed answer to this query: **'{search_query}'**. "
            f"Make sure to cite all the sources at the end of your answer."
        )
        augment_content = dspy.Predict("content -> answer", prompt)
        with dspy.context(lm=self.lm["gpt4o_mini_temp0"]):
            answer = augment_content(content=json.dumps(results)).answer
        return answer

    def search(self, query, top_k=10, character_limit=500, verbose=False):
        """Retrieval Augmented Google Search"""

        netio_logger.info(" 🌐 Running Google search")

        # Make a google search term from search query
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            self.search_term = self.query_expansion(search_query=query).search_term

        # Scrape web page content
        search_items = self.google_search(
            search_item=self.search_term,
            api_key=self.google_api_key,
            cse_id=self.cse_id,
            search_depth=top_k,
            site_filter=None
        )

        # Build a search dictionary
        results = self.get_search_results(search_items, character_limit)
        if verbose:
            self.print_google_search_results(results)

        # Generate a RAG style answer
        answer = self.augmented_generation(search_query=query, results=results)

        return answer


def main():
    agent = SyslogAgent()
    lm = agent.setup_environment()
    api_key, cse_id = agent.setup_google_search()

    query = "Which commands should I use to troubleshoot OSPF adjacency issues and dead timer expirations. Do not include any debug commands in your answer."
    search_agent = GoogleSearchRAG(lm=lm, api_key=api_key, cse_id=cse_id)
    answer = search_agent.search(query=query, top_k=5, character_limit=2000)
    print(answer)


if __name__ == "__main__":
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(__name__)
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    try:
        main()
    except Exception as e:
        logger.exception(f"An exception occurred during execution: {e}")
