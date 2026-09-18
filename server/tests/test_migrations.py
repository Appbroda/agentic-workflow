"""The migration chain the deployment refuses to start without.

Readiness compares the database's recorded revision against the head packaged in the image,
so a branched or mislinked chain does not fail a test -- it fails the container, after it has
been built and pushed. These checks are cheap and run everywhere, unlike applying the
migrations themselves, which needs the PostgreSQL they are written for.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory

from storage.models import Base

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def script_directory() -> ScriptDirectory:
    """Read the packaged migrations the way the readiness probe does."""
    configuration = AlembicConfig()
    configuration.set_main_option("script_location", str(MIGRATIONS))
    return ScriptDirectory.from_config(configuration)


def test_the_migration_chain_has_exactly_one_head() -> None:
    """Two heads mean two people added a migration against the same parent.

    The deployment reads a single head to compare against the database, so a branch makes
    every container unready with a message about a revision mismatch rather than about the
    branch that caused it.
    """
    assert len(script_directory().get_heads()) == 1


def test_every_revision_links_to_the_one_before_it() -> None:
    """A chain with a gap applies some migrations and silently skips others."""
    directory = script_directory()
    revisions = list(directory.walk_revisions())
    known = {item.revision for item in revisions}

    for revision in revisions:
        parent = revision.down_revision
        assert parent is None or parent in known, (
            f"{revision.revision} is linked to {parent}, which is not a migration in this tree"
        )

    roots = [item for item in revisions if item.down_revision is None]
    assert len(roots) == 1, "a second root would create an unreachable second chain"


def test_every_migration_can_be_downgraded() -> None:
    """A migration with no downgrade cannot be backed out of a running deployment."""
    for revision in script_directory().walk_revisions():
        source = Path(revision.path).read_text(encoding="utf-8")
        assert "def downgrade()" in source, f"{revision.revision} has no downgrade"
        body = source.split("def downgrade()", 1)[1]
        assert "pass" not in body.split("\n")[1:3], (
            f"{revision.revision} declares a downgrade that does nothing"
        )


def test_every_mapped_table_is_created_by_a_migration() -> None:
    """A model with no migration works in tests, which build the schema from metadata.

    It then fails in the deployment, which builds the schema from migrations only. The two
    are checked against each other here because nothing else does: the test suite calls
    ``create_all`` and never applies a migration at all.
    """
    # The table name is the first argument, which the formatter usually puts on its own
    # line -- so this reads the whole call rather than matching a single line.
    created_table = re.compile(r"op\.create_table\(\s*\"([a-z_]+)\"")
    created: set[str] = set()
    for revision in script_directory().walk_revisions():
        source = Path(revision.path).read_text(encoding="utf-8")
        created.update(created_table.findall(source))

    missing = set(Base.metadata.tables) - created
    assert missing == set(), f"these mapped tables have no migration: {sorted(missing)}"


def _literal_string(node: ast.expr | None) -> str | None:
    """Return a literal string, or nothing for anything computed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _column_names(arguments: list[ast.expr]) -> list[str]:
    """Return the names of the `sa.Column(...)` calls among these arguments."""
    names: list[str] = []
    for argument in arguments:
        if (
            isinstance(argument, ast.Call)
            and isinstance(argument.func, ast.Attribute)
            and argument.func.attr == "Column"
            and argument.args
        ):
            name = _literal_string(argument.args[0])
            if name is not None:
                names.append(name)
    return names


class _SchemaReplay(ast.NodeVisitor):
    """Replay one revision's `upgrade()` onto a set of column names per table.

    A visitor rather than a flat scan because this repository's migrations name their table
    three different ways, and a scan that understands only the first silently reports a
    column as missing when it is not -- which would make this check noise, and noise gets
    deleted. The three:

    * `op.add_column("feature_actions", sa.Column(...))` -- the table is a literal.
    * `for table in ("feature_workflows", "feature_execution_queue"): op.add_column(table, ...)`
      -- the table is a loop variable over a literal tuple, sometimes named as a
      module-level constant and sometimes `reversed()`.
    * `with op.batch_alter_table("feature_child_workflows") as batch: batch.add_column(...)`
      -- the table is on the `with`, and the call is on the batch operator, not `op`.

    Only `upgrade()` is replayed. A `downgrade()` that drops a column is not evidence the
    column does not exist.
    """

    def __init__(self, columns: dict[str, set[str]], constants: dict[str, list[str]]) -> None:
        """Bind the accumulating schema and the module's string constants."""
        self._columns = columns
        self._constants = constants
        self._bindings: dict[str, list[str]] = {}
        self._batch: list[str] = []

    def visit_For(self, node: ast.For) -> None:  # noqa: N802 - ast's own naming
        """Bind a loop variable to the table names it iterates, for the body only."""
        names = self._iterated_strings(node.iter)
        target = node.target.id if isinstance(node.target, ast.Name) else None
        if target is not None and names:
            self._bindings[target] = names
        for statement in node.body:
            self.visit(statement)
        if target is not None:
            self._bindings.pop(target, None)
        for statement in node.orelse:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802 - ast's own naming
        """Push the table a `batch_alter_table` block operates on, for the body only."""
        pushed = False
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "batch_alter_table"
                and call.args
            ):
                table = self._resolve(call.args[0])
                if table:
                    self._batch.append(table[0])
                    pushed = True
        for statement in node.body:
            self.visit(statement)
        if pushed:
            self._batch.pop()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast's own naming
        """Apply one schema operation, if this call is one."""
        if isinstance(node.func, ast.Attribute):
            self._apply(node.func.attr, node)
        self.generic_visit(node)

    def _apply(self, operation: str, node: ast.Call) -> None:
        """Apply `create_table`, `drop_table`, `add_column` or `drop_column`."""
        if operation == "create_table" and node.args:
            for table in self._resolve(node.args[0]):
                self._columns[table] = set(_column_names(node.args[1:]))
        elif operation == "drop_table" and node.args:
            for table in self._resolve(node.args[0]):
                self._columns.pop(table, None)
        elif operation == "add_column":
            # `op.add_column(table, Column(...))` names the table first;
            # `batch.add_column(Column(...))` takes it from the enclosing `with`.
            tables = self._resolve(node.args[0]) if node.args else []
            arguments = node.args[1:]
            if not tables and self._batch:
                tables = [self._batch[-1]]
                arguments = node.args
            for table in tables:
                self._columns.setdefault(table, set()).update(_column_names(arguments))
        elif operation == "drop_column":
            tables = self._resolve(node.args[0]) if node.args else []
            name = _literal_string(node.args[1]) if len(node.args) > 1 else None
            if not tables and self._batch:
                tables = [self._batch[-1]]
                name = _literal_string(node.args[0]) if node.args else None
            for table in tables:
                if name is not None:
                    self._columns.get(table, set()).discard(name)

    def _resolve(self, node: ast.expr) -> list[str]:
        """Return the table names one expression can stand for."""
        literal = _literal_string(node)
        if literal is not None:
            return [literal]
        if isinstance(node, ast.Name):
            return self._bindings.get(node.id) or self._constants.get(node.id) or []
        return []

    def _iterated_strings(self, node: ast.expr) -> list[str]:
        """Return the strings a `for` iterates over, unwrapping `reversed(...)`."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reversed"
            and node.args
        ):
            return self._iterated_strings(node.args[0])
        if isinstance(node, ast.Tuple | ast.List):
            return [
                value
                for value in (_literal_string(element) for element in node.elts)
                if value is not None
            ]
        if isinstance(node, ast.Name):
            return self._constants.get(node.id, [])
        return []


def _module_constants(module: ast.Module) -> dict[str, list[str]]:
    """Return the module-level names bound to a string or a tuple or list of them."""
    constants: dict[str, list[str]] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literal = _literal_string(statement.value)
        if literal is not None:
            constants[target.id] = [literal]
        elif isinstance(statement.value, ast.Tuple | ast.List):
            values = [
                value
                for value in (_literal_string(element) for element in statement.value.elts)
                if value is not None
            ]
            if values:
                constants[target.id] = values
    return constants


def test_every_mapped_column_is_created_by_a_migration() -> None:
    """A column with no migration passes this entire suite and fails in a deployment.

    Tests build the schema with ``create_all`` from the models, so a column that exists only
    in `models.py` is present everywhere the suite looks. The deployment builds its schema
    from the migrations alone, so the first thing that reads that column is a 500 in
    production -- which is exactly the shape of failure the table check above was written
    for, one level down.

    The migrations are replayed in revision order rather than scanned as a set, because these
    operations are order-dependent: a column added by one revision and dropped by a later one
    is not a column the deployment has.
    """
    columns: dict[str, set[str]] = {}
    for revision in reversed(list(script_directory().walk_revisions())):
        module = ast.parse(Path(revision.path).read_text(encoding="utf-8"))
        upgrade = next(
            (
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
            ),
            None,
        )
        if upgrade is None:  # pragma: no cover - the chain tests prove every revision has one
            continue
        replay = _SchemaReplay(columns, _module_constants(module))
        for statement in upgrade.body:
            replay.visit(statement)

    missing: dict[str, list[str]] = {}
    for name, table in Base.metadata.tables.items():
        absent = sorted({column.name for column in table.columns} - columns.get(name, set()))
        if absent:
            missing[name] = absent
    assert missing == {}, f"these mapped columns have no migration: {missing}"
