# Cancellation Model

Cancellation is cooperative and durable. The API writes a cancellation request to PostgreSQL first,
then publishes a short-lived Redis signal for prompt delivery to an active worker. Live work uses a
composite token, so a worker restart cannot lose the request.

Workers check the token before and after safe boundaries. Local validation and Git subprocesses run
in their own process groups; cancellation sends `SIGTERM`, waits briefly, then kills the group if
needed. Coding and GitHub provider requests cannot be forcibly cancelled by every provider, so the
worker stops waiting, records an uncertain outcome if needed, and never starts later commit, push,
or PR operations.

Feature cancellation moves through `cancellation_requested`, `cancellation_in_progress`, and a
terminal `cancelled` or `cancelled_with_external_side_effects` state. A retained branch, commit,
push, or PR is exposed as a cleanup requirement; the platform never deletes it automatically.
