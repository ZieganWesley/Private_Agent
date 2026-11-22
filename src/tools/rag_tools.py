"""Document search tool for the agent."""

from langchain_core.tools import tool

from . import get_tool_context


@tool
def search_documents(query: str, k: int = 5, collection: str | None = None) -> str:
    """Search the indexed local documents (RAG) for relevant passages.

    When ``collection`` is given, only chunks from that collection are returned.
    """
    context = get_tool_context()
    vectorstore = getattr(context, "vectorstore", None)
    if vectorstore is None:
        return "No document index is available for this session."
    try:
        results = vectorstore.similarity_search(query, k=k, collection=collection)
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        return f"Document search failed: {exc}"
    if not results:
        return "No relevant documents found."
    lines = []
    for index, doc in enumerate(results, 1):
        source = getattr(doc, "metadata", {}).get("source", "unknown")
        coll = getattr(doc, "metadata", {}).get("collection")
        label = f"{coll}: {source}" if coll else source
        content = getattr(doc, "page_content", "") or ""
        lines.append(f"[{index}] {label}\n{content}")
    return "\n\n".join(lines)
