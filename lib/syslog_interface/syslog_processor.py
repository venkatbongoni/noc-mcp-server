# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from .syslog import SyslogAPI
from .message_compression import SyslogNormalization, MessageSyntaxTree
from ..syslog_modeler.model_builder import SyslogJsonOutputBuilder
from ..syslog_modeler.models import SyslogJsonOutput
from config import Settings


class SyslogProcessor:
    """
    A class to process syslog data for reasoning tasks.
    """

    def __init__(
        self,
        data_dir: str = "./datasets/",
        blacklist: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SyslogProcessor with the specified data directory and blacklist.

        Args:
            data_dir (str): Directory containing the datasets.
            blacklist (Optional[List[str]]): List of file patterns to ignore
                during data loading.
            logger (Optional[logging.Logger]): Logger for handling log messages.
        """
        self.data_dir = data_dir
        self.blacklist = blacklist if blacklist is not None else []
        self.X_index: List[str] = []
        self.G: Optional[MessageSyntaxTree] = None
        self.text_retrieval = SyslogNormalization()
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self, dataset: str, stream=False) -> None:
        """
        Loads the syslog data from the specified dataset or input stream.

        Args:
            dataset (str): Name of the dataset to process.
            stream (bool): String input
        """
        if stream:
            dataset_path = dataset
        else:
            dataset_path = Path(self.data_dir).joinpath(dataset)
        try:
            self.X_new = self.text_retrieval.loader(
                input_data=dataset_path,
                blacklist=self.blacklist,
                stream=stream,
            )
        except Exception as e:
            self.logger.error(f"Failed to load input data: {e}")
            raise

    def preprocess_data(self) -> None:
        """
        Preprocesses the loaded syslog data by normalizing and tokenizing.
        """
        try:
            self.X = self.text_retrieval.normalize_tokenize(self.X_new)
            self.logger.info("Data preprocessing completed successfully.")
            if self.logger.isEnabledFor(logging.DEBUG):
                for i in self.X:
                    self.logger.debug(f"Graph input: {i}")
        except Exception as e:
            self.logger.error(f"Failed to preprocess data: {e}")
            raise

    def create_or_update_graph(self, reset: bool = False) -> None:
        """
        Creates or updates the message tree with the preprocessed data.
        """
        try:
            if not len(self.X_index) or reset:
                self.G = MessageSyntaxTree().create_from(self.X)
                self.logger.info(
                    f"Graph created. Messages: {self.G.num_messages}."
                )
            else:
                if self.G is not None:
                    self.G.create_from(self.X)
                    self.logger.info(
                        f"Graph updated. Messages: {self.G.num_messages}."
                    )
                else:
                    self.G = MessageSyntaxTree().create_from(self.X)
                    self.logger.info(
                        f"Graph created. Messages: {self.G.num_messages}."
                    )
            self.X_index.extend(self.X_new)
        except Exception as e:
            self.logger.error(f"Failed to create or update the graph: {e}")
            raise

    def sample_logs(self) -> None:
        """
        Samples log data from the message tree.
        """
        try:
            if self.G is None:
                self.logger.warning("Graph is None, cannot sample logs.")
                self.logs = []
                return

            self.tail_node_indices = self.G.sample_tail_node_indices()
            if not self.tail_node_indices:
                self.logger.info("No messages found, skipping log sampling.")
                self.logs = []
                return

            self.logs = [f"{self.X_index[i]}" for i in self.tail_node_indices]
            self.logger.info(f"Sampled {len(self.logs)} log entries.")
        except Exception as e:
            self.logger.error(f"Failed to sample logs: {e}")
            raise

    def build_json_model(self) -> Optional[SyslogJsonOutput]:
        """
        Builds a JSON model from the sampled logs.
        """
        model = None

        try:
            builder = SyslogJsonOutputBuilder()
            builder.add_syslog_lines(self.logs)
            model = builder.build()
            self.logger.info("JSON model built successfully.")
        except Exception as e:
            self.logger.error(f"Failed to build SyslogJsonOutput model: {e}")

        return model

    async def run_workflow(
        self,
        dataset: str,
        delete_context: bool = False,
        stream: bool = False,
    ) -> str:
        """
        Executes the full workflow from data loading to log
        post-processing for the given dataset.

        Args:
            dataset (str): Name of the dataset to process.
            delete_context (bool): Delete the context store then update.
            stream (bool): Load input data from string.

        Returns:
            str: The final context string for the reasoning task.
        """
        context = ""
        try:
            self.load_data(dataset, stream=stream)
            self.preprocess_data()
            self.create_or_update_graph(reset=delete_context)
            self.sample_logs()
            model = self.build_json_model()
            if model:
                context = model.model_dump_json(indent=2)
                self.logger.info("Context JSON model created successfully.")
            else:
                self.logger.warning("Failed to build JSON model: model is None")
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")

        return context


async def main() -> None:
    """Main function to process syslog data from API endpoint."""
    logger = logging.getLogger(__name__)

    try:
        # Initialize the syslog API
        settings = Settings()
        if not hasattr(settings, 'syslog_endpoint'):
            logger.error("Settings missing syslog_endpoint configuration")
            return

        syslog = settings.syslog_endpoint

        # Validate required attributes
        required_attrs = ['api_base', 'host', 'buffer_name']
        for attr in required_attrs:
            if not hasattr(syslog, attr):
                logger.error(f"syslog_endpoint missing required attribute: {attr}")
                return

        syslog_api = SyslogAPI(api_base=str(syslog.api_base))
        log_messages = syslog_api.get_syslog(syslog.host, syslog.buffer_name)

        logger.info(f"syslog_api host: {syslog.host}")
        logger.info(f"syslog_api buffer_name: {syslog.buffer_name}")
        logger.info(f"syslog_api api_base: {syslog.api_base}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"syslog_api messages:\n{log_messages}")

        # Create the syslog processor reading from the syslog API
        processor = SyslogProcessor(logger=logger)
        context = await processor.run_workflow(log_messages, stream=True)
        logger.info(f"Context JSON model created with {len(context)} characters")

    except Exception as e:
        logger.error(f"Failed to process syslog data: {e}")
        raise


if __name__ == "__main__":
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Syslog processing workflow."
    )

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
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"An exception occurred during execution: {e}")
