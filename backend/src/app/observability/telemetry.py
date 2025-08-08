from __future__ import annotations

import os
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


def init_opentelemetry(app: Any) -> None:  # FastAPI app type
    """Initialize OpenTelemetry if SDK is available.

    - Instruments FastAPI, httpx, and psycopg if packages are present.
    - Exports OTLP over gRPC or HTTP depending on available exporters.
    - Adds a structlog processor to inject trace/span IDs into logs.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Exporters (prefer OTLP)
        exporter = None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HTTPOTLPSpanExporter,
            )

            exporter = HTTPOTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
            )
        except Exception:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as GRPCOTLPSpanExporter,
                )

                exporter = GRPCOTLPSpanExporter()
            except Exception:
                exporter = None

        service_name = os.getenv("OTEL_SERVICE_NAME", os.getenv("APP_APP_NAME", "forgehub-backend"))
        resource = Resource.create({"service.name": service_name, "service.version": "0.1.0"})
        provider = TracerProvider(resource=resource)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Instrument libraries
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception:
            pass
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except Exception:
            pass
        try:
            from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

            PsycopgInstrumentor().instrument()
        except Exception:
            pass

        # Add a structlog processor that injects trace/span ids
        def _otel_ids(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
            try:
                span = trace.get_current_span()
                span_ctx = span.get_span_context()
                if span_ctx and span_ctx.is_valid:
                    event_dict.setdefault("trace_id", format(span_ctx.trace_id, "032x"))
                    event_dict.setdefault("span_id", format(span_ctx.span_id, "016x"))
            except Exception:
                pass
            return event_dict

        # Prepend so IDs are available to downstream processors
        processors = list(structlog.get_config()["processors"])  # type: ignore[index]
        processors.insert(0, _otel_ids)
        structlog.configure(processors=processors)

        logger.info("opentelemetry_initialized", exporter=exporter.__class__.__name__ if exporter else None)
    except Exception as e:  # SDK not installed or failed to init
        logger.warning("opentelemetry_not_initialized", reason=str(e))
