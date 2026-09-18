# Implementation completeness

Repository plans carry an `ImplementationExpectation` for every scoped requirement. It names the
expected source categories, source areas, and whether tests are required. Backend API requirements
normally include `route`, `controller`, `service`, and `test` expectations.

Before repository review, the completion check compares expectations with production, test, and
configuration files reported by the Engineer. It rejects `tests_only_change`,
`documentation_only_change`, `configuration_only_change`, `missing_required_source_change`,
`missing_required_test_change`, and `unexpected_scope_change`.

The completion artifact records production/test/configuration files, implemented and missing
requirements, satisfied expectations, and requirement-to-file evidence. Evidence names the files,
available symbols, description, and relevant validation results. Tests support production evidence;
they cannot replace it unless the requirement is explicitly test-only.

Reviewers run the same gate before approval. A backend change that omits required production files
emits actionable findings such as `BACKEND_ROUTE_NOT_IMPLEMENTED`,
`BACKEND_CONTROLLER_NOT_IMPLEMENTED`, and `TESTS_ONLY_CHANGE`.
