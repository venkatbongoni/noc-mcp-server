# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import os
import re
import argparse
import logging
import random
import ipaddress
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
from typing import Optional


class SyslogNormalization:

    def __init__(self, logger: Optional[logging.Logger] = None,):
        self.MONTH_PATTERN = re.compile(
            r"\b(?:Jan|Feb|Mar|Mär|Apr|May|Mai|Jun|Jul|Aug|Sep|Oct|Okt|Nov|Dec|Dez)\b"
        )
        self.IPV6_PATTERN = re.compile(r'\b[\da-fA-F]{1,4}(?::[\da-fA-F]{1,4}){1,7}\b')
        INTERFACE_NUMERIC_PART = r"\d+(?:/\d+)*(?:\.\d+)?"
        self.MESSAGE_ENTITIES_PATTERN = re.compile(
            r"UPDOWN|UP|DOWN|ACTIVE|STANDBY|FAILED|"        # Status codes and Syslog severity levels
            r"ENABLED|DISABLED|CHANGED|"
            r"OK|ON|OFF|NO|up|down|on|off|no|administratively|"
            r"success|failure|successfully|failed|"
            r"error|warning|critical|alert|emergency|notice|informational|debug|trace|"
            r"\b[A-Za-z]{3,30}\b|"                          # Any alpha string 3 .. 30 chars
            r"\bveth[0-9a-fA-F]{7,12}\b|"                   # Linux virtual interfaces like veth77305bfa
            rf"GigabitEthernet{INTERFACE_NUMERIC_PART}|"    # Cisco interfaces
            rf"FastEthernet{INTERFACE_NUMERIC_PART}|"
            rf"Ethernet{INTERFACE_NUMERIC_PART}|"
            rf"Serial{INTERFACE_NUMERIC_PART}|"
            rf"Loopback{INTERFACE_NUMERIC_PART}|"
            rf"Tunnel{INTERFACE_NUMERIC_PART}|"
            rf"Vlan{INTERFACE_NUMERIC_PART}|"
            rf"Port-channel{INTERFACE_NUMERIC_PART}|"
            rf"Bundle-Ether{INTERFACE_NUMERIC_PART}|"
            rf"ATM{INTERFACE_NUMERIC_PART}|"
            rf"POS{INTERFACE_NUMERIC_PART}|"
            rf"Dialer{INTERFACE_NUMERIC_PART}|"
            rf"Async{INTERFACE_NUMERIC_PART}|"
            rf"BVI{INTERFACE_NUMERIC_PART}|"
            rf"SVI{INTERFACE_NUMERIC_PART}|"
            rf"GE{INTERFACE_NUMERIC_PART}|"
            rf"Gi{INTERFACE_NUMERIC_PART}|"
            rf"FE{INTERFACE_NUMERIC_PART}|"
            rf"Tengigabitethernet{INTERFACE_NUMERIC_PART}|"
            rf"TenGig{INTERFACE_NUMERIC_PART}|"
            rf"HundredGig{INTERFACE_NUMERIC_PART}"
        )
        self.logger = logger or logging.getLogger(__name__)

    def whitelist_matches(self, text):
        interface_matches = re.findall(self.MESSAGE_ENTITIES_PATTERN, text)
        return ' '.join(interface_matches).strip()

    def dir_reader(self, path, blacklist=[]):
        for dir_, _, files in os.walk(path):
            for file in files:
                _, extension = os.path.splitext(file)
                if file in blacklist or extension in blacklist:
                    continue
                abs_path = os.path.join(dir_, file)
                with open(abs_path, "r", encoding="utf-8") as log_file:
                    content = log_file.read()
                    text = "".join(content).splitlines(True)
                    self.logger.debug("X input raw text: %d lines.", len(text))
                    yield abs_path, text

    def normalize_tokenize(self, text):
        out = []
        for line in text:

            # After the timestamp there must be a IPv4 address.
            parts = line.split()
            if len(parts) < 4:
                self.logger.warning("Line does not have enough parts, skipping line..")
                out.append("")
                continue

            ip_str = parts[3]

            try:
                ipaddress.IPv4Address(ip_str)
            except ipaddress.AddressValueError:
                self.logger.warning(f"Invalid input: {ip_str}, must be IPv4 address, skipping line..")
                out.append("")
                continue

            clean_month = self.MONTH_PATTERN.sub("", line)
            clean_ipv6 = self.IPV6_PATTERN.sub("", clean_month)
            clean_line = self.whitelist_matches(clean_ipv6)
            if not clean_line:
                clean_line = ""
            out.append(clean_line)

        self.logger.info("Normalized text %d lines.", len(out))
        return out

    def loader(
        self,
        input_data,
        blacklist,
        max_lines=30000,
        sample=False,
        sample_size=32,
        stream=False,
    ):
        if stream:
            if not input_data:
                # raise ValueError("Input string is empty.")
                self.logger.warning("Input is empty.")
            self.logger.info("Processing input from string...")
            corpus = []
            total_lines = 0
            lines = input_data.splitlines(True)
            for line in lines:
                corpus.append(line)
                total_lines += 1
                if total_lines > max_lines:
                    self.logger.warning(f"Reached the max number of {max_lines} input lines.")
                    break
            self.logger.info(f"Total number of input lines: {len(corpus)} lines.")
        else:
            self.logger.info(f"Processing input from directory: {input_data}")
            corpus = []
            file_num, total_lines = 0, 0
            for _, content in self.dir_reader(path=input_data, blacklist=blacklist):
                lines = "".join(content).splitlines(True)
                corpus.extend(lines)
                total_lines += len(lines)
                file_num += 1
                if total_lines > max_lines:
                    self.logger.warning(f"Reached the max number of {max_lines} input lines.")
                    break
            self.logger.info(f"File collection has {file_num} files, total length: {len(corpus)} lines.")

        if sample:
            X_raw = random.sample(corpus, k=sample_size)
            self.logger.info(f"Random sample {len(X_raw)} lines.")
        else:
            X_raw = corpus
        cleaned_messages = [message.strip() for message in X_raw]
        return cleaned_messages


class MessageSyntaxTree:

    def __init__(self, logger: Optional[logging.Logger] = None,):
        self.graph = nx.DiGraph()
        self.root = "ROOT"
        self.graph.add_node(self.root, name="ROOT")
        self._num_messages = 0
        self._next_message_index = 0
        self.logger = logger or logging.getLogger(__name__)

    def insert_message(self, message, index):
        words = message.split()
        if not words:
            return

        current_node = self.root
        for i, word in enumerate(words):
            word_node = f"{word}_{i}"
            if self.graph.has_node(word_node):
                if not nx.has_path(self.graph, current_node, word_node):
                    word_node = f"{word}_{i}_{current_node}"
            self.graph.add_node(word_node, name=word)
            self.graph.nodes[word_node].setdefault("index", []).append(index)
            self.graph.add_edge(current_node, word_node)
            current_node = word_node

    def create_from(self, messages):
        self._num_messages = len(messages)
        for msg in messages:
            self.insert_message(msg, self._next_message_index)
            self._next_message_index += 1
        return self

    def plot_tree(self):
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        labels = nx.get_node_attributes(self.graph, "name")
        plt.figure(figsize=(16, 14))
        nx.draw(
            self.graph,
            pos,
            labels=labels,
            with_labels=True,
            arrows=True,
            node_size=220,
            node_color="lightblue",
            font_size=6,
            font_weight="normal",
        )
        plt.title("Directed Message Graph")
        plt.show()

    def get_nodes_at_distance(self, distance):
        distances = nx.single_source_shortest_path_length(self.graph, self.root)
        nodes_at_distance = {
            self.graph.nodes[node]["name"]: {
                "name": self.graph.nodes[node]["name"],
                "index": self.graph.nodes[node]["index"],
            }
            for node, dist in distances.items()
            if dist == distance
        }
        return nodes_at_distance

    def get_node_indices_at_distance(self, distance):
        nodes = self.get_nodes_at_distance(distance)
        indices = [attributes["index"] for attributes in nodes.values()]
        return sorted(indices)

    def get_nodes_attributes(self):
        return {node: dict(self.graph.nodes[node]) for node in self.graph.nodes}

    def get_tail_end_node_indices(self):
        tail_end_indices = [
            self.graph.nodes[node]["index"]
            for node in self.graph.nodes
            if self.graph.out_degree(node) == 0
        ]
        return tail_end_indices

    def get_total_message_count(self):
        unique_indices = set()
        node_attributes = self.get_nodes_attributes()
        for attributes in node_attributes.values():
            indices = attributes.get("index", [])
            unique_indices.update(indices)
        return len(unique_indices)

    def sum_node_degrees_per_branch(self):
        branch_degree_sums = {}
        for child in self.graph.successors(self.root):
            degree_sum = 0
            for node in nx.descendants(self.graph, child) | {child}:
                degree_sum += self.graph.degree(node)
            branch_name = self.graph.nodes[child]["name"]
            branch_degree_sums[branch_name] = degree_sum
        return branch_degree_sums

    def get_tail_end_nodes(self):
        return [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]

    def get_branch_degree_sum(self, tail_node):
        return sum(
            self.graph.degree[node]
            for node in nx.shortest_path(self.graph, source=self.root, target=tail_node)
        )

    def get_highest_degree_node(self, tail_node):
        path = nx.shortest_path(self.graph, source=self.root, target=tail_node)
        if not path:
            return None
        path = [node for node in path if node != self.root]
        max_degree_node = max(path, key=lambda node: self.graph.degree[node])
        return max_degree_node

    def sample_from_index(self, out_degree, index_attr):
        if 1 < out_degree < 4:
            sample_size = 2
        else:
            sample_size = 1
        sample_size = min(sample_size, len(index_attr))
        return random.sample(index_attr, sample_size)

    def sample_node_indices(self, data):
        unique_indices = set()

        for tail_node in self.get_tail_end_nodes():
            path = nx.shortest_path(self.graph, source=self.root, target=tail_node)
            self.logger.debug(f"tail-end node {tail_node}, path: {path}")
            outgoing_degrees = OrderedDict()
            intermediate_path_sum_degree = 0
            intermediate_path = path[1:-1]  # Skip root and tail nodes

            if len(intermediate_path) == 0:
                continue

            for node in intermediate_path:
                degree = self.graph.out_degree(node)
                intermediate_path_sum_degree += degree
                outgoing_degrees[node] = degree
            self.logger.debug(
                "\tOutgoing degrees of nodes (excluding root and tail):",
                outgoing_degrees,
            )

            """
            Sample k=1 from a linear leaf path
            """
            self.logger.debug(
                "\t\tlen inter path: %s, sum out degrees: %s",
                len(intermediate_path),
                intermediate_path_sum_degree,
            )
            if len(intermediate_path) == intermediate_path_sum_degree:
                last_node_in_path = next(reversed(outgoing_degrees))
                indices = self.graph.nodes[last_node_in_path]["index"]
                samples = random.sample(indices, k=1)
                unique_indices.update(samples)
                self.logger.debug(f"\tSampled single path indices: {samples}")
                continue

            """
            Heuristic for finding the best node for sampling node attributes
            """
            degrees_list = list(outgoing_degrees.values())
            mean_degree = np.mean(degrees_list)
            std_degree = np.std(degrees_list)
            threshold = mean_degree + std_degree

            sampled_indices = OrderedDict()
            for node, out_degree in outgoing_degrees.items():
                indices = self.graph.nodes[node]["index"]
                sample_done = self.graph.nodes[node].get("sample_done", False)
                if not sample_done and out_degree > 1 and out_degree > threshold:
                    sampled_indices[node] = self.sample_from_index(out_degree, indices)
                    self.logger.debug(
                        f"\t\tMean degree:{mean_degree}, Stdev:{std_degree}, threshold:{threshold}"
                    )
                    self.logger.debug(
                        f"\t\tSampled out_degree > 1 indices:{sampled_indices[node]}"
                    )
                    self.graph.nodes[node]["sample_done"] = True
                    continue
                else:
                    continue

            for indices in sampled_indices.values():
                unique_indices.update(indices)

        """
        Deduplicate
        """
        deduplicated_dict = {}
        seen_messages = {}
        for index in unique_indices:
            message = data[index]
            if message not in seen_messages:
                seen_messages[message] = index
                deduplicated_dict[index] = message
        indices_list = list(deduplicated_dict.keys())
        messages_list = list(deduplicated_dict.values())
        self.logger.info(
            f"Nodes: {len(self.graph)}, "
            f"messages: {self.next_message_index}, "
            f"high informational messages: "
            f"{len(messages_list)} "
            f"({((len(messages_list) + 1e-6) * 100.0 / self.next_message_index):.1f}%)."
        )
        return indices_list, messages_list

    def sample_tail_node_indices(self, random=False):
        unique_indices = set()
        for tail_node in self.get_tail_end_nodes():
            if "index" not in self.graph.nodes[tail_node]:
                self.logger.info(f"Tail node {tail_node} does not have an 'index' key. Skipping.")
                continue
            indices = self.graph.nodes[tail_node]["index"]
            if not indices:
                self.logger.warning(f"Tail node {tail_node} has an empty 'index' list. Skipping.")
                continue
            if random:
                unique_indices.update(random.sample(indices, k=1))
            else:
                unique_indices.add(indices[-1])
        unique_indices_list = list(unique_indices)
        if self.next_message_index > 0:
            percentage = ((len(unique_indices_list) + 1e-6) * 100.0 / self.next_message_index)
        else:
            percentage = 0.0
        self.logger.info(
            f"Nodes: {len(self.graph)}, "
            f"messages: {self.next_message_index}, "
            f"high informational messages: "
            f"{len(unique_indices_list)} "
            f"({percentage:.1f}%)."
        )
        return unique_indices_list

    def dataloader(self, messages, batch_size):
        for i in range(0, len(messages), batch_size):
            yield messages[i:i + batch_size]

    def subgraph_match(self, search_sequence):
        matched_branches = []
        matched_indices = []
        for node in self.graph.nodes:
            matched_nodes = []
            current_node = node
            for word in search_sequence:
                found = False
                matched_node_indices = []
                for neighbor in self.graph.successors(current_node):
                    if self.graph.nodes[neighbor]["name"] == word:
                        matched_nodes.append(neighbor)
                        matched_node_indices.extend(self.graph.nodes[neighbor]["index"])
                        current_node = neighbor
                        found = True
                        break
                if not found:
                    break
            if len(matched_nodes) == len(search_sequence):
                matched_branches.append(matched_nodes)
                matched_indices.extend(matched_node_indices)
        return matched_branches, matched_indices

    @property
    def num_messages(self):
        return self._num_messages

    @property
    def next_message_index(self):
        return self._next_message_index

    @property
    def num_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def num_edges(self):
        return self.graph.number_of_edges()


def main(dataset):
    DATA_DIR = "./datasets/"
    DATASET_PATH = Path(DATA_DIR).joinpath(dataset)

    text_retrieval = SyslogNormalization()
    X_raw = text_retrieval.loader(input_data=DATASET_PATH, blacklist=[".txt"])
    X = text_retrieval.normalize_tokenize(X_raw)
    G = MessageSyntaxTree().create_from(X)
    tail_node_indices = G.sample_tail_node_indices()
    messages_raw = [f"{X_raw[index]}" for index in tail_node_indices]
    for i in messages_raw:
        print(i)


if __name__ == "__main__":
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Deduplicate and extract high informational syslog messages."
    )

    # Required positional argument
    parser.add_argument("dataset", help="the dataset to process.")

    # Optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-D", "--debug", action="store_true", help="more verbose")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    try:
        main(args.dataset)
    except Exception as e:
        logger.exception(f"An exception occurred during execution: {e}")
