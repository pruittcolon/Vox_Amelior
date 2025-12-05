# RAG Service Database Artifacts

- `docker/rag_instance/rag.db` (mounted to `/app/instance/rag.db` at runtime) is the **only** database used by the running rag-service container. This file lives outside of the `services/` tree so it survives container rebuilds.
- `services/rag-service/rag.db.container` is an **old snapshot** checked in for reference. It is not attached to the docker-compose stack.

Use `scripts/show_rag_db.py` to print the active host-side database path (and optionally run a read-only sanity query) before inspecting transcript data.
