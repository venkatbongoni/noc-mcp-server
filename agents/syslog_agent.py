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
import time
import tqdm
import dspy
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from lib import ShortTermMem, OptimizationBase, ModeType, DataLoader
from lib.syslog_interface.syslog_processor import SyslogProcessor
from settings.settings import setup_tracing, setup_llm, setup_google_search
from metrics.metric import LLMMetric, RougeMetric, SemanticF1Metric
from dspy.teleprompt import AvatarOptimizer, BootstrapFewShotWithRandomSearch, MIPROv2
from dspy.evaluate import Evaluate
from .syslog_tools import define_tools
from .main_agent import Avatar

logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class SyslogAgent(OptimizationBase):
    """
    Encapsulates the Avatar AI agent workflow.
    """

    def __init__(self, data_dir: str = "./datasets/", model_dir: str = "./models/") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.context: Optional[str] = None
        self.llm: Optional[dict] = None
        self.google_api: Optional[dict] = None
        self.trainset: Optional[List[dspy.Example]] = None
        self.devset: Optional[List[dspy.Example]] = None
        self.agent_metric = None
        self.actor_metric = None
        self.tools = []
        self.avatar: Optional[Avatar] = None
        self.optimizer: Optional[AvatarOptimizer] = None
        self.optimized_avatar: Optional[Avatar] = None
        self.phoenix_session = None
        self.tracer_provider = None
        self.abs_cost = 0.0
        self.processor = SyslogProcessor(
            data_dir=self.data_dir, blacklist=["experiment_description.txt", ".txt"]
        )

    def update_context(self, dataset_name: str) -> None:
        """
        Load, preprocess, and update the context with new data.
        """
        logger.info("Loading and preprocessing context...")
        self.context = self.processor.run_workflow(dataset_name)
        logger.info("Context loaded and processed.")

    def replace_context(self, dataset_name: str) -> None:
        """
        Load and preprocessed data. Existing context will be replaced with a new one.
        """
        logger.info("Load, preprocess and replace existing context...")
        self.context = self.processor.run_workflow(dataset_name, delete_context=True)
        logger.info("Loaded, processed and replaced existing context.")

    def setup_tracing(self) -> None:
        """
        Enable OpenTelemetry tracing.
        """
        logger.info("Setting up OpenTelemetry tracing...")
        self.phoenix_session, self.tracer_provider = setup_tracing()

    def setup_environment(self, api_key: str = None) -> dict:
        """
        Set up the LLM OpenAI API key and model for DSPy.
        """
        logger.info("Setting up LLM...")
        self.llm = setup_llm(api_key=api_key)
        return self.llm

    def setup_google_search(self) -> tuple:
        """
        Set up search keys for agent tool web search.
        """
        return setup_google_search()

    def load_datasets(self, train, dev) -> None:
        """
        Load the train and test datasets for Avatar agent workflow optimization.
        """
        logger.info("Loading Avatar datasets...")
        data_loader = DataLoader()
        self.trainset = data_loader.load_examples(train)
        self.devset = data_loader.load_examples(dev)

    def load_actor_datasets(
        self, actor_name: str = None, fields: List[str] = [], with_inputs: List[str] = []
    ) -> None:
        """
        Load the train and test datasets for DSPy program optimization.
        """
        logger.info(f"Loading {actor_name} DSPy Example datasets...")
        data_loader = DataLoader()
        trainset = f"{actor_name}_trainset.json"
        devset = f"{actor_name}_devset.json"
        self.actor_trainset = data_loader.actor_dataset_loading(
            filename=trainset, fields=fields, with_inputs=with_inputs
        )
        self.actor_devset = data_loader.actor_dataset_loading(
            filename=devset, fields=fields, with_inputs=with_inputs
        )

    def evaluate_actor(self, actor: dspy.Module = None) -> float:
        """Accuracy of the model."""
        actor = actor or self.optimized_actor
        if not actor:
            raise ValueError("Actor not defined.")
        evaluator = Evaluate(
            devset=self.actor_devset,
            num_threads=4,
            display_progress=True,
            display_table=False,
        )
        score = evaluator(actor, metric=self.actor_metric)
        return score

    def optimize_actor_bootstrap_few_shot(self, actor: dspy.Module) -> None:
        """
        Optimize a DSPy program used as an actor in the agent workflow using
        BootstrapFewShotWithRandomSearch.
        """
        logger.info("Optimizing zero-shot program with BootstrapFewShotWithRandomSearch...")
        if not self.actor_metric or not self.actor_trainset:
            raise ValueError(
                "Actor metric or trainset not defined. Please ensure they are loaded."
            )
        config = dict(
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=10,
            num_threads=6,
        )
        optimizer = BootstrapFewShotWithRandomSearch(metric=self.actor_metric, **config)
        self.optimized_actor = optimizer.compile(
            student=actor,
            teacher=actor,
            trainset=self.actor_trainset,
            valset=self.actor_devset,
        )

    def optimize_actor_mipro_v2(self, actor: dspy.Module) -> None:
        """
        Optimize a DSPy program used as an actor in the agent workflow using MIPROv2.
        """
        logger.info("Optimizing zero-shot program with MIPROv2...")
        if not self.actor_metric or not self.actor_trainset:
            raise ValueError(
                "Actor metric or trainset not defined. Please ensure they are loaded."
            )
        optimizer = MIPROv2(metric=self.actor_metric)
        self.optimized_actor = optimizer.compile(
            student=actor.deepcopy(),
            trainset=self.actor_trainset,
            valset=self.actor_devset,
            max_bootstrapped_demos=3,
            max_labeled_demos=4,
            minibatch_size=4,
            requires_permission_to_run=False,
        )

    def save_actor(self, actor_name: str) -> None:
        """
        Save the optimized DSPy program.
        """
        if not self.optimized_actor:
            raise ValueError(
                "Optimized actor does not exist. Please run optimization first."
            )
        model_path = Path(self.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        file_path = model_path.joinpath(f"{actor_name}.json")
        self.optimized_actor.save(file_path)
        logger.info(f"Optimized actor saved to {file_path}")

    def load_actor(self, actor: dspy.Module) -> dspy.Module:
        """
        Load an optimized DSPy program if a saved model exists.
        """
        model_path = Path(self.model_dir)
        model = model_path.joinpath(f"{actor.__class__.__name__}.json")
        if model.exists():
            actor.load(model)
            self.set_optimized()
            netio_logger.info(f" ♨️ Loaded optimized actor from {model}")
        else:
            self.reset_optimization_state()
            netio_logger.info(f" ❄️ Running unoptimized actor {actor.__class__.__name__}")
        return actor

    def synthesize_actor_optimization_sets(self, status: bool = False) -> None:
        """
        Sample train and dev sets for optimizing the DSPy actors.
        """
        ShortTermMem.set("synthesize_actor_optimization_sets", status)

    def define_metric(self, lm, role: str = "agent", metric_type: str = "semantic_f1") -> None:
        """
        Define the metric for program evaluation and optimization based on the
        provided metric_type and role. Depending on the role, this sets either
        self.actor_metric or self.agent_metric.
        """
        logger.info("Defining metric for %s evaluation using '%s'...", role, metric_type)

        if metric_type == "llm":
            metric = LLMMetric(lm=lm)
        elif metric_type == "rouge_l":
            metric = RougeMetric()
        elif metric_type == "semantic_f1":
            metric = SemanticF1Metric(decompositional=True)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        if role == "actor":
            self.actor_metric = metric
        elif role == "agent":
            self.agent_metric = metric
        else:
            raise ValueError(f"Unknown role: {role}")

    def create_optimizer(self, lb, ub) -> None:
        """
        Create Avatar optimizer instance.
        """
        logger.info(f"New optimizer, lb: {lb}, ub: {ub}")
        if not self.agent_metric:
            raise ValueError("Agent metric not defined. Please ensure they are loaded.")

        self.optimizer = AvatarOptimizer(
            metric=self.agent_metric,
            max_iters=2,
            lower_bound=lb,
            upper_bound=ub,
            max_negative_inputs=10,
            max_positive_inputs=10,
        )

    def define_tools(self, lm: dict, mode: ModeType = "inference") -> None:
        """
        Define the agent tools and context.
        """
        logger.info("Defining tools context...")
        self.tools = define_tools(self.context, lm, mode, model_path=Path(self.model_dir))

    def create_agent(self, verbose: bool = True) -> None:
        """
        Create an instance of Avatar agent.
        """
        logger.info("Creating Avatar agent...")
        if not self.tools:
            raise ValueError("Tools not defined. Please ensure they are loaded.")

        class LogRootifyReasoner(dspy.Signature):
            """
            You are a Cisco network expert and you have the tools to interact
            with the live network, enabling you to perform real time analysis and
            troubleshooting of problems in your network. If the user asks a technical
            question about your network, then answer as a Cisco expert. Provide a direct
            answer, without reading the network logs, for device specific factoid questions
            (e.g. user asking to run some command on node A, B or C). Conduct a
            comprehensive root cause analysis if the user asking to read and analyze the
            latest log messages from the network. If you have gathered enough diagnostic
            data, evaluate whether this information allows you to fully answer the user's
            question.
            """
            question: str = dspy.InputField(
                desc="question to ask or statement to follow up on",
                format=lambda x: x.strip(),
            )
            answer: str = dspy.OutputField(
                desc=(
                    "direct factoid answer for device specific questions, OR, root cause "
                    "assessment when analyzing network wide syslog messages"
                )
            )

        self.avatar = Avatar(
            signature=LogRootifyReasoner,
            tools=self.tools,
            max_iters=20,
            verbose=verbose,
        )
        self.reset_optimization_state()

    def optimize_agent(self) -> None:
        """
        Optimize the Avatar agent using the defined metric.
        """
        logger.info("Optimizing agent")
        if not self.agent_metric or not self.trainset:
            raise ValueError(
                "Agent metric or trainset not defined. Please ensure they are loaded."
            )
        if not self.optimizer:
            raise ValueError("No optimizer. Please ensure it exists.")

        self.optimized_avatar = self.optimizer.compile(
            student=self.avatar, trainset=self.trainset
        )
        self.set_optimized()

    def evaluate_agent(self, agent: Avatar = None, dataset: str = "dev") -> float:
        """
        Evaluate the un-optimized agent on the specified development dataset.
        """
        logger.info(f"Evaluating un-optimized agent on {dataset} dataset...")
        agent_to_use = self.avatar if agent is None else agent
        if not agent_to_use:
            raise ValueError("Agent does not exist.")

        if self.is_optimized():
            raise ValueError("Agent is optimized.")

        dataset_to_use = self.devset if dataset == "dev" else None
        if not dataset_to_use:
            raise ValueError(f"{dataset.capitalize()} set is not loaded.")

        total_score = 0
        total_examples = len(dataset_to_use)

        def process_example(example):
            try:
                prediction = agent_to_use(**example.inputs().toDict())
                return self.agent_metric(example, prediction)
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                return 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_example, example) for example in dataset_to_use
            ]
            for future in tqdm.tqdm(
                futures, total=total_examples, desc="Processing examples"
            ):
                total_score += future.result()

        avg_score = total_score / total_examples
        logger.info(f"Average Score on {dataset} dataset: {avg_score:.2f}")
        return avg_score

    def evaluate_optimized_agent(self, agent: Avatar = None, dataset: str = "dev") -> float:
        """
        Evaluate the optimized agent on the specified development dataset.
        """
        logger.info(f"Evaluating optimized agent on {dataset} dataset...")
        agent_to_use = self.optimized_avatar if agent is None else agent
        if not agent_to_use:
            raise ValueError("Agent does not exist.")

        if not self.is_optimized():
            raise ValueError("Agent is not optimized.")

        dataset_to_use = self.devset if dataset == "dev" else None
        if not dataset_to_use:
            raise ValueError(f"{dataset.capitalize()} set is not loaded.")

        if not self.optimizer:
            raise ValueError("Avatar optimizer does not exist.")
        avg_score = self.optimizer.thread_safe_evaluator(dataset_to_use, agent_to_use)
        logger.info(f"Average Score on {dataset} dataset: {avg_score:.2f}")
        return avg_score

    def save_agent(self, filename: str = "SyslogAvatarAgent.json") -> None:
        """
        Save the optimized agent to a file.
        """
        if not self.optimized_avatar:
            raise ValueError(
                "Optimized Avatar agent does not exist. Please run optimize_agent() first."
            )
        model_path = Path(self.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        file_path = model_path.joinpath(filename)
        self.optimized_avatar.save(file_path)
        logger.info(f"Optimized Agent saved to {file_path}")

    def load_agent(self, filename: str = "SyslogAvatarAgent.json", verbose: bool = True) -> Optional[Avatar]:
        """
        Creates a new Avatar agent instance and then trying to load an optimized agent
        model from file.
        """
        self.create_agent(verbose=verbose)
        if self.avatar:
            model = Path(self.model_dir).joinpath(filename)
            if model.exists():
                self.avatar.load(model)
                self.set_optimized()
                netio_logger.info(f" ♨️ Loaded optimized agent from {model}")
            else:
                self.reset_optimization_state()
                netio_logger.info(" ❄️ Running unoptimized agent")
            logger.info("Avatar agent is created.")
        else:
            logger.error("Avatar agent is not created.")
        return self.avatar

    def run_inference(self, question: str) -> Tuple[str, str]:
        """
        Run inference on a given question using the Avatar agent.
        """
        logger.info(f"Running inference on question: {question}")
        if not self.avatar:
            raise ValueError("Avatar agent does not exist.")

        ShortTermMem.set("user_query", question)
        start = time.time()
        pred = self.avatar(question=question)
        exec_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        return exec_time, pred.answer

    def set_agent_context(self, dataset_name: str, mode: ModeType = "inference", verbose: bool = True) -> Avatar | None:
        """
        Set a new context in the tools and agent and load a trained model.
        """
        self.replace_context(dataset_name=dataset_name)
        self.define_tools(mode)
        optimized_agent = self.load_agent(verbose=verbose)
        return optimized_agent

    def get_lm_cost(self) -> Tuple[float, float]:
        """
        Obtain the cost in USD of all LM calls, as calculated by LiteLLM for certain
        providers, made by the DSPy extractor program so far.
        """
        total_cost = sum(
            sum(
                x.get('cost', 0.0) for x in lm.history
                if x.get('cost') is not None
            )
            for lm in self.llm.values()
        )
        previous_cost = getattr(self, 'abs_cost', 0.0)
        delta_cost = total_cost - previous_cost
        self.abs_cost = total_cost
        return round(self.abs_cost, 4), round(delta_cost, 4)
