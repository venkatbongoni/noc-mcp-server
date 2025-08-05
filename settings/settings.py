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
import os
import dspy
import phoenix as px
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from getpass import getpass
from config import Settings

logger = logging.getLogger(__name__)


def setup_tracing():
    """
    Enable OpenTelemetry tracing with a configurable endpoint.
    """
    endpoint = os.getenv("OTLP_TRACE_ENDPOINT", "http://127.0.0.1:6006/v1/traces")

    if not endpoint:
        logger.error(
            "OTLP_TRACE_ENDPOINT is not configured. Tracing will not be enabled."
        )
        raise ValueError(
            "OTLP_TRACE_ENDPOINT must be set in the environment or .env file."
        )

    logger.info(f" ✅ Setting up tracing with endpoint: {endpoint}")

    phoenix_session = px.launch_app()
    tracer_provider = trace_sdk.TracerProvider()
    span_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_exporter))
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    DSPyInstrumentor().instrument()

    return phoenix_session, tracer_provider


def create_llm_instances(settings: Settings, api_key: str):
    """
    Create LLM instances based on configuration in Settings.
    """
    return {
        "default_openai": dspy.LM(
            settings.default_openai_llm.model,
            api_key=api_key,
            api_base=settings.default_openai_llm.api_base,
            max_tokens=settings.default_openai_llm.max_tokens,
            temperature=settings.default_openai_llm.temperature,
            cache=settings.default_openai_llm.cache,
        ),
        "metric_openai": dspy.LM(
            settings.openai_metric_llm.model,
            api_key=api_key,
            api_base=settings.openai_metric_llm.api_base,
            max_tokens=settings.openai_metric_llm.max_tokens,
            temperature=settings.openai_metric_llm.temperature,
            cache=settings.openai_metric_llm.cache,
        ),
        "llama33_70b": dspy.LM(
            settings.llama33_70b_llm.model,
            api_base=settings.llama33_70b_llm.api_base,
            temperature=settings.llama33_70b_llm.temperature,
            max_tokens=settings.llama33_70b_llm.max_tokens,
            api_key=settings.llama33_70b_llm.api_key,
            model_type=settings.llama33_70b_llm.model_type,
            cache=settings.llama33_70b_llm.cache,
        ),
    }


def setup_llm(api_key: str = None):
    """
    Set up the LLM OpenAI API key and model for DSPy.
    """
    settings = Settings()

    api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass(prompt="Enter your OpenAI API key: ")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    llm_instances = create_llm_instances(settings, api_key)

    dspy.configure(lm=llm_instances.get("default_openai"), trace=[])

    llm_mapping = {
        "gpt4o_mini": llm_instances["default_openai"],
        "gpt4o_mini_temp0": llm_instances["metric_openai"],
        "llama33_70b": llm_instances["llama33_70b"],
    }

    return llm_mapping


def setup_google_search(api_key: str = None, cse_id: str = None):
    """
    Google Custom Search API key and Custom Search Engine ID (CSE ID).
    """
    settings = Settings()
    api_key = api_key or settings.google_api_key or os.getenv("GOOGLE_API_KEY")
    cse_id = cse_id or settings.google_cse_id or os.getenv("GOOGLE_CSE_ID")

    if not api_key:
        api_key = getpass(prompt="Enter your Google Custom Search API key: ")

    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")

    if not cse_id:
        cse_id = getpass(prompt="Enter your Google Custom Search Engine ID: ")

    if not cse_id:
        raise EnvironmentError("GOOGLE_CSE_ID is not set.")

    return api_key, cse_id
