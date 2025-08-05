# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class OpenAIDefaultLLMConfig(BaseSettings):
    """The LLM used as the default for the Agent and DSPy programs."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_DEFAULT_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    model: str = Field("openai/gpt-4o-mini")
    api_base: Optional[HttpUrl] = Field(
        default=None,
        description="Base URL for the OpenAI API.",
    )
    api_key: Optional[str] = None
    max_tokens: int = 16384
    temperature: float = 0.7
    cache: bool = True

    @field_validator("api_base", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class OpenAIEvaluationLLMConfig(BaseSettings):
    """The LLM used as evaluation metric function."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_METRIC_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    model: str = Field("openai/gpt-4o-mini")
    api_base: Optional[HttpUrl] = Field(
        default=None,
        description="Base URL for the OpenAI API.",
    )
    api_key: Optional[str] = None
    max_tokens: int = 16384
    temperature: float = 0.0
    cache: bool = True

    @field_validator("api_base", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class Llama33_70BConfig(BaseSettings):
    """70B model served by SGL."""

    model_config = SettingsConfigDict(
        env_prefix="LLAMA33_70B_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    model: str
    api_base: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="Base URL for the model on SGLang.",
    )
    temperature: float = 0.7
    max_tokens: int = 16384
    api_key: str
    model_type: str = Field("chat")
    cache: bool = True

    @field_validator("api_key")
    def validate_api_key(cls, value):
        if not value:
            raise ValueError(
                "LLAMA33_70B_API_KEY must be provided in the environment or .env file."
            )
        return value


class SyslogConfig(BaseSettings):
    """Syslog configuration to fetch system logs."""
    model_config = SettingsConfigDict(
        env_prefix="SYSLOG_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    host: str = Field(..., description="Syslog API endpoint.")
    buffer_name: str
    api_base: HttpUrl


class CommandLineConfig(BaseSettings):
    """
    Configuration for Command-Line API to access network devices.
    """
    model_config = SettingsConfigDict(
        env_prefix="COMMANDLINE_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    api_base: str = Field(
        default="http://127.0.0.1:8000/v1",
        description="Base URL for Command-Line network device access.",
    )


class ICSConfig(BaseSettings):
    """
    Configuration for ICS APIs (RAG and SQL).
    """
    model_config = SettingsConfigDict(
        env_prefix="ICS_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    rag_url: str = Field(
        default="http://localhost:9000/ics/rag/query/faiss",
        description="URL for ICS RAG API.",
    )
    sql_url: str = Field(
        default="http://localhost:9000/ics/sql/query",
        description="URL for ICS SQL API.",
    )
    rag_store: str = Field(
        default="faiss",
        description="RAG Store for ICS. Options: 'faiss', 'elastic'.",
    )


class Settings(BaseSettings):
    """LM and RM settings."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    # LM settings
    openai_api_key: str = Field(default=None, env="OPENAI_API_KEY")
    default_openai_llm: OpenAIDefaultLLMConfig = OpenAIDefaultLLMConfig()
    openai_metric_llm: OpenAIEvaluationLLMConfig = OpenAIEvaluationLLMConfig()
    llama33_70b_llm: Llama33_70BConfig = Llama33_70BConfig()

    # Google Custom Search Engine
    google_api_key: str = Field(default=None, env="GOOGLE_API_KEY")
    google_cse_id: str = Field(default=None, env="GOOGLE_CSE_ID")

    # Network API settings
    syslog_endpoint: SyslogConfig = SyslogConfig()
    command_endpoint: CommandLineConfig = CommandLineConfig()

    # ICS and logs API settings
    ics_config: ICSConfig = ICSConfig()
