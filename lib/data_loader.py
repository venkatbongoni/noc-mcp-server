# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import logging
import ast
import random
import json
import dspy
from pathlib import Path
from typing import Any, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DataLoader:
    """Data sampling and loading for optimization of DSPy programs and Avatar agents."""

    def __init__(self, data_dir: str = "./datasets/") -> None:
        self.data_dir = data_dir

    def load_examples(self, filename: str) -> List[dspy.Example]:
        """Load a QA dataset from file containing (Question, Answer) tuples into
        dspy.Example format for Avatar agent optimization."""
        file_path = Path(self.data_dir).joinpath(filename)
        logger.info(f"Loading examples from {file_path}...")
        examples = []
        try:
            with open(file_path, "r") as file:
                for line in file:
                    try:
                        q, a = ast.literal_eval(line.strip())
                        examples.append(
                            dspy.Example(question=q, answer=a).with_inputs("question")
                        )
                    except (ValueError, SyntaxError) as e:
                        logger.error(f"Error parsing line: {line} - {e}")
            logger.info(f"Loaded {len(examples)} examples from {file_path}.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
        return examples

    def actor_dataset_sampling(
        self,
        filename_prefix: str,
        *input_data: Any,
        max_train_samples: int,
        max_dev_samples: int,
    ) -> None:
        """Sample a train and dev dataset in the format of dspy.Example for DSPy program
        optimization."""

        data_dir = Path(self.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        trainset_file = data_dir / f"{filename_prefix}_trainset.json"
        devset_file = data_dir / f"{filename_prefix}_devset.json"

        # Ensure all input variables are represented, even if not initialized.
        data_to_write = {
            f"input_{i}": (item.model_dump() if isinstance(item, BaseModel) else item)
            for i, item in enumerate(input_data)
        }

        split_choice = random.choices(["train", "dev"], weights=[20, 80], k=1)[0]

        target_file = trainset_file if split_choice == "train" else devset_file
        max_samples = max_train_samples if split_choice == "train" else max_dev_samples

        if target_file.exists():
            with target_file.open("r+") as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []

                if len(existing_data) >= max_samples:
                    existing_data = existing_data[1:]

                existing_data.append(data_to_write)
                file.seek(0)
                file.truncate()
                json.dump(existing_data, file, indent=2)
        else:
            with target_file.open("w") as file:
                json.dump([data_to_write], file, indent=2)

    def actor_dataset_loading(
        self, filename: str, fields=[], with_inputs=[]
    ) -> List[dspy.Example]:
        """Load a dspy.Example dataset from file for DSPy program optimization."""

        file_path = Path(self.data_dir).joinpath(filename)
        logger.info(f"Loading examples from {file_path}...")
        examples = []
        try:
            with open(file_path, "r") as file:
                data_entries = json.load(file)
                if not isinstance(data_entries, list):
                    logger.error(f"Invalid data format in file: {file_path}")
                    raise ValueError("Data entries should be a list of dictionaries.")
                for entry in data_entries:
                    if isinstance(entry, dict):
                        inputs = list(entry.values())
                        if len(inputs) >= len(fields):
                            field_values = {field: inputs[idx] for idx, field in enumerate(fields)}
                            example = dspy.Example(**field_values).with_inputs(*with_inputs)
                            examples.append(example)
            logger.info(f"Loaded {len(examples)} examples from {file_path}.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Error loading JSON from file: {file_path} - {e}")
            raise e
        return examples
