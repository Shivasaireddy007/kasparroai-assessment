"""Tracing hooks. One trace per answer, with generation observations per model call and tool
observations per get_brand_facts execution. The no-op tracer keeps tests independent of Langfuse."""

from __future__ import annotations

import os
from typing import Any, Protocol


class Observation(Protocol):
    def end(self, output: Any = None, usage: dict[str, int] | None = None, error: str | None = None) -> None: ...


class AnswerTrace(Protocol):
    trace_id: str | None

    def generation(self, name: str, model: str, input: Any) -> Observation: ...

    def tool(self, name: str, input: Any) -> Observation: ...

    def event(self, name: str, metadata: dict[str, Any]) -> None: ...

    def end(self, output: Any, error: str | None = None) -> None: ...


class Tracer(Protocol):
    def start_answer(self, name: str, input: Any, metadata: dict[str, Any]) -> AnswerTrace: ...

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None, metadata: Any = None) -> None: ...

    def flush(self) -> None: ...


class _NoopObs:
    def end(self, output: Any = None, usage: dict[str, int] | None = None, error: str | None = None) -> None:
        pass


class _NoopTrace:
    trace_id = None

    def generation(self, name: str, model: str, input: Any) -> Observation:
        return _NoopObs()

    def tool(self, name: str, input: Any) -> Observation:
        return _NoopObs()

    def event(self, name: str, metadata: dict[str, Any]) -> None:
        pass

    def end(self, output: Any, error: str | None = None) -> None:
        pass


class NoopTracer:
    def start_answer(self, name: str, input: Any, metadata: dict[str, Any]) -> AnswerTrace:
        return _NoopTrace()

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None, metadata: Any = None) -> None:
        pass

    def flush(self) -> None:
        pass


NOOP_TRACER = NoopTracer()


class _LfObs:
    def __init__(self, obs: Any):
        self._obs = obs

    def end(self, output: Any = None, usage: dict[str, int] | None = None, error: str | None = None) -> None:
        kwargs: dict[str, Any] = {"output": output}
        if usage:
            kwargs["usage_details"] = usage
        if error:
            kwargs["level"] = "ERROR"
            kwargs["status_message"] = error
        self._obs.update(**kwargs)
        self._obs.end()


class _LfTrace:
    def __init__(self, client: Any, name: str, input: Any, metadata: dict[str, Any]):
        from langfuse import propagate_attributes

        self._client = client
        tags = [str(metadata[k]) for k in ("agent_version", "split") if metadata.get(k)]
        str_meta = {k: str(v)[:200] for k, v in metadata.items()}
        self._ctx = propagate_attributes(
            session_id=str(metadata.get("run_id") or "") or None, metadata=str_meta, tags=tags, trace_name=name
        )
        self._ctx.__enter__()
        self._root = client.start_observation(name=name, as_type="agent", input=input, metadata=metadata)
        self.trace_id: str | None = getattr(self._root, "trace_id", None)

    def generation(self, name: str, model: str, input: Any) -> Observation:
        return _LfObs(self._root.start_observation(name=name, as_type="generation", model=model, input=input))

    def tool(self, name: str, input: Any) -> Observation:
        return _LfObs(self._root.start_observation(name=name, as_type="tool", input=input))

    def event(self, name: str, metadata: dict[str, Any]) -> None:
        self._root.create_event(name=name, metadata=metadata, level="WARNING")

    def end(self, output: Any, error: str | None = None) -> None:
        kwargs: dict[str, Any] = {"output": output}
        if error:
            kwargs["level"] = "ERROR"
            kwargs["status_message"] = error
        self._root.update(**kwargs)
        self._root.end()
        try:
            self._ctx.__exit__(None, None, None)
        except ValueError:
            pass  # context detached from a different task; attributes already applied


class LangfuseTracer:
    def __init__(self) -> None:
        from langfuse import get_client

        self._client = get_client()

    def start_answer(self, name: str, input: Any, metadata: dict[str, Any]) -> AnswerTrace:
        return _LfTrace(self._client, name, input, metadata)

    def score(self, trace_id: str, name: str, value: float, comment: str | None = None, metadata: Any = None) -> None:
        self._client.create_score(trace_id=trace_id, name=name, value=value, comment=comment, metadata=metadata, data_type="NUMERIC")

    def trace_url(self, trace_id: str) -> str | None:
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def flush(self) -> None:
        self._client.flush()


def tracer_from_env() -> Tracer:
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        return LangfuseTracer()
    return NOOP_TRACER
