"""
logger.py
---------
Structured logging module for the multi-source research agent.

Usage:
    from logger import get_logger, log_tool_call, log_agent_action, log_warning

    logger = get_logger(__name__)

    # Log a tool invocation
    log_tool_call(logger, tool_name="web_search", inputs={"query": "transformer efficiency"})

    # Log an agent reasoning action
    log_agent_action(logger, action="plan", detail="Searching web first, then memory")

    # Log a warning with full agent context
    log_warning(logger, error=e, context={"tool": "fetch_page", "url": url, "step": 3})
"""

import logging
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Log levels available
# ---------------------------------------------------------------------------
# DEBUG   — fine-grained internals: chunk counts, similarity scores, token usage
# INFO    — every tool call and agent action (the standard operational level)
# WARNING — errors with full context of what the agent was trying to do
# ---------------------------------------------------------------------------


class StructuredFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Fields always present:
        timestamp   ISO-8601 UTC
        level       DEBUG | INFO | WARNING | ERROR | CRITICAL
        logger      the logger name (usually the module)
        message     the human-readable message
    Extra fields passed via the `extra` dict are merged at the top level,
    making them easy to filter in log aggregators (Datadog, Loki, CloudWatch).
    """

    RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach exception details if present
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "value": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Merge any extra fields the caller supplied
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """
    Readable formatter for local development.
    Example output:
        2025-04-23 14:02:11 UTC  INFO  agent.tools  [web_search] query="transformer efficiency"
    """

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        base = f"{ts}  {level}  {record.name}  {record.getMessage()}"

        # Append any extra structured fields as key=value pairs
        reserved = StructuredFormatter.RESERVED
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in reserved and not k.startswith("_")
        }
        if extras:
            kv = "  ".join(f'{k}="{v}"' for k, v in extras.items())
            base = f"{base}  |  {kv}"

        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"

        return base


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------


def _build_handler(
    stream=sys.stdout, structured: bool = False
) -> logging.StreamHandler:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter() if structured else HumanFormatter())
    return handler


def configure_logging(
    level: int = logging.DEBUG,
    structured: bool = False,
) -> None:
    """
    Call once at application startup (e.g. in main.py or config.py).

    Args:
        level:       Root log level. Use logging.DEBUG for development,
                     logging.INFO for production.
        structured:  True  → JSON output (production / log aggregators).
                     False → human-readable coloured output (development).

    Example:
        from logger import configure_logging
        import logging
        configure_logging(level=logging.DEBUG, structured=False)
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers that were added by imported libraries
    root.handlers.clear()

    # INFO and below → stdout
    info_handler = _build_handler(sys.stdout, structured)
    info_handler.setLevel(logging.DEBUG)
    info_handler.addFilter(lambda r: r.levelno < logging.WARNING)
    root.addHandler(info_handler)

    # WARNING and above → stderr
    warn_handler = _build_handler(sys.stderr, structured)
    warn_handler.setLevel(logging.WARNING)
    root.addHandler(warn_handler)

    # Silence noisy third-party loggers at WARNING so they don't bury agent logs
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for a module.

    Args:
        name: Typically __name__ from the calling module.

    Example:
        logger = get_logger(__name__)
        logger.debug("Chunk count after splitting", extra={"chunks": 42})
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
# These keep call sites clean and enforce a consistent schema for the three
# main event types: tool calls, agent actions, and warnings with context.
# ---------------------------------------------------------------------------


def log_tool_call(
    logger: logging.Logger,
    tool_name: str,
    inputs: dict[str, Any],
    job_id: str | None = None,
) -> None:
    """
    Log a tool invocation at INFO level.

    Call this BEFORE executing the tool so the log appears even if the tool
    raises an exception.

    Args:
        logger:    Logger from get_logger(__name__).
        tool_name: The registered name of the LangChain tool being called.
        inputs:    The arguments passed to the tool (will be serialised).
        job_id:    Optional research job ID for cross-request correlation.

    Example:
        log_tool_call(logger, "web_search", {"query": "transformer efficiency"}, job_id=job_id)
    """
    extra: dict[str, Any] = {
        "event": "tool_call",
        "tool": tool_name,
        "inputs": inputs,
    }
    if job_id:
        extra["job_id"] = job_id

    logger.info("[tool_call] %s", tool_name, extra=extra)


def log_tool_result(
    logger: logging.Logger,
    tool_name: str,
    result_summary: str,
    duration_ms: float | None = None,
    job_id: str | None = None,
) -> None:
    """
    Log the outcome of a tool call at INFO level.

    Args:
        logger:         Logger from get_logger(__name__).
        tool_name:      The tool that was called.
        result_summary: A short human-readable summary of what was returned
                        (e.g. "5 sources found", "3 chunks retrieved").
                        Do NOT log full raw content here — it bloats logs.
        duration_ms:    How long the tool took, in milliseconds.
        job_id:         Optional research job ID.

    Example:
        log_tool_result(logger, "web_search", "5 sources found", duration_ms=340.2)
    """
    extra: dict[str, Any] = {
        "event": "tool_result",
        "tool": tool_name,
        "result_summary": result_summary,
    }
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    if job_id:
        extra["job_id"] = job_id

    logger.info("[tool_result] %s → %s", tool_name, result_summary, extra=extra)


def log_agent_action(
    logger: logging.Logger,
    action: str,
    detail: str,
    step: int | None = None,
    job_id: str | None = None,
) -> None:
    """
    Log a high-level agent reasoning action at INFO level.

    Use this for the agent's Thought / Plan / Observation events in the
    ReAct loop — not for individual tool calls (use log_tool_call for those).

    Args:
        logger:  Logger from get_logger(__name__).
        action:  Short label for the reasoning step: "plan", "thought",
                 "observation", "finish", "retry", "skip_duplicate".
        detail:  A sentence describing what the agent decided to do and why.
        step:    The current iteration number in the ReAct loop (1-indexed).
        job_id:  Optional research job ID.

    Example:
        log_agent_action(logger, "plan", "Will search web first, then retrieve from memory", step=1)
        log_agent_action(logger, "finish", "Gathered 6 sources, confidence=high", step=9)
    """
    extra: dict[str, Any] = {
        "event": "agent_action",
        "action": action,
        "detail": detail,
    }
    if step is not None:
        extra["step"] = step
    if job_id:
        extra["job_id"] = job_id

    logger.info("[agent:%s] %s", action, detail, extra=extra)


def log_warning(
    logger: logging.Logger,
    error: Exception,
    context: dict[str, Any],
    job_id: str | None = None,
) -> None:
    """
    Log a recoverable error at WARNING level with full agent context.

    This is the primary error-logging function. It captures the exception AND
    everything the agent was doing at the time, making traces in LangSmith and
    log aggregators self-contained — you never need to correlate multiple log
    lines to understand a failure.

    Args:
        logger:  Logger from get_logger(__name__).
        error:   The caught exception instance.
        context: A dict describing the agent's state at failure time. Include
                 at minimum: the tool name, the inputs, and the current step.
                 More is better — this is the full forensic record.
        job_id:  Optional research job ID.

    Example:
        try:
            result = fetch_page(url)
        except Exception as e:
            log_warning(logger, e, context={
                "tool": "fetch_page",
                "url": url,
                "step": current_step,
                "query": original_query,
                "sources_gathered_so_far": len(sources),
            })
    """
    extra: dict[str, Any] = {
        "event": "agent_warning",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "agent_context": context,
    }
    if job_id:
        extra["job_id"] = job_id

    logger.warning(
        "[warning] %s — %s (context: %s)",
        type(error).__name__,
        str(error),
        context,
        exc_info=True,
        extra=extra,
    )


def log_debug(
    logger: logging.Logger,
    message: str,
    data: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> None:
    """
    Log fine-grained internals at DEBUG level.

    Use for: chunk counts, similarity scores, token usage, cache hits,
    embedding batch sizes — anything too noisy for INFO but useful when
    diagnosing retrieval quality or cost issues.

    Args:
        logger:  Logger from get_logger(__name__).
        message: Short description of what is being logged.
        data:    Optional dict of numeric or structural values.
        job_id:  Optional research job ID.

    Example:
        log_debug(logger, "Chunks after splitting", {"count": 42, "avg_chars": 870})
        log_debug(logger, "Embedding batch complete", {"batch_size": 50, "tokens_used": 12400})
    """
    extra: dict[str, Any] = {"event": "debug"}
    if data:
        extra.update(data)
    if job_id:
        extra["job_id"] = job_id

    logger.debug("[debug] %s", message, extra=extra)
