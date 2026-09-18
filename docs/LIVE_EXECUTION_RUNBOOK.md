# Live Execution Runbook

1. Apply migrations and verify `/readyz`. Do not send live work while readiness is unavailable.
2. On a failed or cancelled feature, inspect `/features/{id}`, `/workstreams`, `/pull-requests`, and
   `/timeline`, then query `external_operations` through approved operator access.
3. For `UNKNOWN_EXTERNAL_STATE`, inspect the repository branch/commit and GitHub PR before any retry.
   Never force-push, delete a PR, or recreate a branch as automated cleanup.
4. For a cancelled feature with cleanup requirements, review each retained resource and decide
   manually whether to retain or close it under repository policy.
5. Restarting the API invokes bounded recovery. It does not retain user provider tokens, so a later
   resume requires fresh request-scoped OpenAI and GitHub credentials. Submit the normal resume
   endpoint with `{"answers": []}` when no human clarification is pending.
6. Escalate when recovery remains unresolved. Readiness intentionally stays degraded instead of
   accepting a duplicate live operation.
