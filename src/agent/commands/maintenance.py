"""Handler for the /maintenance slash command."""


async def handle_maintenance_command(
    *,
    memory,
    llm,
    allow_model_episode_context,
    local_model_suggestions_allowed,
    rag_docs_path=None,
    vectorstore=None,
    on_data_reset=None,
):
    """Run memory maintenance and return the resulting vectorstore."""
    from agent.runtime import _run_memory_maintenance

    return await _run_memory_maintenance(
        memory,
        llm,
        allow_model_episode_context=allow_model_episode_context,
        local_model_suggestions_allowed=local_model_suggestions_allowed,
        rag_docs_path=rag_docs_path,
        vectorstore=vectorstore,
        on_data_reset=on_data_reset,
    )
