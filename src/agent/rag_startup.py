"""Include the selected online endpoint in RAG startup."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from config import EMBEDDING_PROVIDER
from providers.endpoints import EndpointRegistry
from rag.rerank import reranker_for_route
from utils.run_logging import RUN_LOGGER, log_event

from .memory_context import _initialize_rag_for_docs
from .rag_routes import resolve_rag_routes, runtime_for_online_selection


def _knowledge_base_initializer(base_fn: Any, route: Any, reranker: Any):
    def initialize(path, *, index_path=None, confirm_rebuild=None):
        return base_fn(
            path,
            index_path=index_path,
            confirm_rebuild=confirm_rebuild,
            embedding_provider=route.label if route is not None else "none",
            embedding_model=route.model if route is not None else "",
            embedding_base_url=route.base_url if route is not None else "",
            reranker=reranker,
        )

    return initialize


async def initialize_rag_session(
    *,
    docs_input: str,
    local_runtimes: Any,
    console: Any,
    interactive: bool,
    input_fn: Any,
    rag_index_path: str,
    app_config: Any,
    initialize_knowledge_base_fn: Any,
    knowledge_base_signature_fn: Any,
) -> tuple[Any, bool, Any, float, Any]:
    """Resolve the startup RAG routes and start indexing."""
    from .provider_report import record_embedding_route

    embedding_route, rerank_route = resolve_rag_routes(
        runtimes=local_runtimes,
        registry=EndpointRegistry.from_config(app_config),
        console=console,
        interactive=interactive,
        embedding_enabled=str(EMBEDDING_PROVIDER).strip().lower() != "none",
    )
    embedding_provider = (
        embedding_route.label if embedding_route is not None else "none"
    )
    embedding_model = embedding_route.model if embedding_route is not None else ""
    embedding_base_url = embedding_route.base_url if embedding_route is not None else ""
    record_embedding_route(embedding_provider, embedding_model, embedding_base_url)
    reranker = (
        reranker_for_route(rerank_route, app_config)
        if rerank_route is not None
        else None
    )

    return await _initialize_rag_for_docs(
        docs_input,
        rag_index_path=rag_index_path,
        console=console,
        input_fn=input_fn,
        log_event=log_event,
        run_logger=RUN_LOGGER,
        uuid=uuid,
        logging=logging,
        initialize_knowledge_base_fn=_knowledge_base_initializer(
            initialize_knowledge_base_fn, embedding_route, reranker
        ),
        knowledge_base_signature_fn=knowledge_base_signature_fn,
    )


async def _reinitialize(
    *,
    docs_input: str,
    local_runtimes: Any,
    online_selection: dict | None,
    console: Any,
    interactive: bool,
    input_fn: Any,
    rag_index_path: str,
    app_config: Any,
    initialize_knowledge_base_fn: Any,
    knowledge_base_signature_fn: Any,
    previous_task: Any = None,
) -> tuple[Any, bool, Any, float, Any] | None:
    online_runtime = runtime_for_online_selection(online_selection)
    if online_runtime is None:
        return None
    if previous_task is not None and not previous_task.done():
        previous_task.cancel()
        await asyncio.gather(previous_task, return_exceptions=True)
    embedding_route, rerank_route = resolve_rag_routes(
        runtimes=[*local_runtimes, online_runtime],
        registry=EndpointRegistry.from_config(app_config),
        console=console,
        interactive=interactive,
        embedding_enabled=str(EMBEDDING_PROVIDER).strip().lower() != "none",
    )
    reranker = (
        reranker_for_route(rerank_route, app_config)
        if rerank_route is not None
        else None
    )

    return await _initialize_rag_for_docs(
        docs_input,
        rag_index_path=rag_index_path,
        console=console,
        input_fn=input_fn,
        log_event=log_event,
        run_logger=RUN_LOGGER,
        uuid=uuid,
        logging=logging,
        initialize_knowledge_base_fn=_knowledge_base_initializer(
            initialize_knowledge_base_fn, embedding_route, reranker
        ),
        knowledge_base_signature_fn=knowledge_base_signature_fn,
    )


async def finalize_online_rag(
    *,
    docs_input: str,
    local_runtimes: Any,
    online_selection: dict | None,
    console: Any,
    interactive: bool,
    input_fn: Any,
    loaded_skills: Any,
    rag_index_path: str,
    app_config: Any,
    initialize_knowledge_base_fn: Any,
    knowledge_base_signature_fn: Any,
    previous_task: Any = None,
) -> tuple[tuple[Any, bool, Any, float, Any], str] | None:
    """Rebuild RAG with the online endpoint and return its state and mode label."""
    state = await _reinitialize(
        docs_input=docs_input,
        local_runtimes=local_runtimes,
        online_selection=online_selection,
        console=console,
        interactive=interactive,
        input_fn=input_fn,
        rag_index_path=rag_index_path,
        app_config=app_config,
        initialize_knowledge_base_fn=initialize_knowledge_base_fn,
        knowledge_base_signature_fn=knowledge_base_signature_fn,
        previous_task=previous_task,
    )
    if state is None:
        return None
    vectorstore, is_rag_active, rag_signature, rag_last_check, rag_task = state
    if rag_task is not None:
        with console.status("[bold info]Finishing RAG index...[/bold info]"):
            vectorstore = await rag_task
        is_rag_active = vectorstore is not None
        rag_signature = (
            knowledge_base_signature_fn(docs_input) if is_rag_active else None
        )
        rag_last_check = time.monotonic()
        state = (vectorstore, is_rag_active, rag_signature, rag_last_check, None)
    mode_label = (
        "[info]Skilled Private Agent[/info]"
        if loaded_skills and is_rag_active
        else "[info]Skilled Agent[/info]"
        if loaded_skills
        else "[info]Private Agent[/info]"
        if is_rag_active
        else "[info]Agent[/info]"
    )
    return state, mode_label


__all__ = ["finalize_online_rag", "initialize_rag_session"]
