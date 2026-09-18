# Test Tiers and the Orchestration Canary

Four tiers now exist. They cost very different amounts of wall-clock and they answer very
different questions, so this document says what each one is for, what it costs, and how a
continuous-integration job runs them.

**All four tiers now run in CI**, in `.github/workflows/ci.yml`, on every push and every pull
request; the canary runs nightly in `.github/workflows/canary.yml`. Section 3 describes the
pipeline that exists rather than one somebody should build. `.pre-commit-config.yaml` is
deliberately unchanged — it still runs only `ruff check`, `ruff format --check` and `mypy`,
because a pre-commit hook that costs four minutes is a hook that gets bypassed.

Every command below is still one a person can type, and the pipeline types exactly these.

---

## 1. The tiers

| Tier | Command | Tests | Wall-clock | What only it can answer |
|---|---|---|---|---|
| default | `uv run --directory server pytest` | 1414 | ~299 s | everything that does not need a real database, a real toolchain or a second process |
| crash windows | `uv run --directory server pytest tests/test_crash_windows.py` | 13 | ~24 s | what a worker killed at a named boundary leaves behind, and what recovery then does |
| postgres | `uv run --directory server pytest -m postgres` | 30 | ~22 s | row locking, `FOR UPDATE SKIP LOCKED`, and every race two workers can produce |
| real repository | `uv run --directory server pytest -m realrepo` | 57 | ~130 s | whether a checkout's own commands actually *run*, not merely whether they were chosen |

The crash-window tests are not separately marked: they are part of the default run and the
row above is the cost of selecting that one file. The other two tiers are deselected by
default in `pyproject.toml` (`-m 'not postgres and not realrepo'`) and each is selected by
its own marker.

### Prerequisites, and how an absent one behaves

| Tier | Needs | Absent locally | Absent in CI |
|---|---|---|---|
| postgres | a PostgreSQL server | skips with a reason | fails, when `REQUIRE_POSTGRES_TESTS=1` |
| real repository | `git`, `node`, `npm` on PATH (`pnpm`, `ruff` and `pytest` for four of the tests), **and a PostgreSQL server**, for the three `tests/test_canary.py` cases | skips with a reason | fails, when `REQUIRE_REAL_REPOSITORY_TESTS=1` |

Both variables exist for the same reason: a tier that passes by skipping certifies nothing,
and reads to everyone afterwards as coverage.

The realrepo tier's database requirement is easy to miss, because the marker's own
documentation is about a toolchain. `tests/test_canary.py` is in that tier and creates its own
database, and `REQUIRE_REAL_REPOSITORY_TESTS=1` covers it too — with the variable set and no
server, those three fail rather than skipping. That is why the `server` job runs the tier with
the PostgreSQL service still attached rather than after tearing it down.

Neither tier ever touches the deployment's database. `tests/postgres_support.py` creates
`pytest_*` databases beside it and refuses by name to create, drop or migrate anything else.

---

## 2. The tier nobody will run

**The default tier is at three minutes and forty-six seconds.** That is already past the point
where it gets run once before a commit rather than continuously while working, and it is the
one number in this table worth watching: nothing else here is on the inner loop.

It is not slow because of any one test. It is 1208 tests of which a large minority build real
git checkouts in `tmp_path` and shell out to `git`, which is the right trade — a mock-only
suite is what let a JavaScript repository be validated with Python tooling in production.
Two things would buy the loop back without giving up coverage:

* **Run it in parallel.** `pytest-xdist` with `-n auto` is the obvious first move. Nothing in
  the default tier shares mutable state across tests: every database is a `tmp_path` SQLite
  file and every checkout is a `tmp_path` directory. This is a dependency addition, so it is
  out of this task's scope, but it is where the three minutes goes.
* **Give people a smaller default to type.** A `-m 'not slow'` marker on the heaviest
  file-system tests would make the inner loop seconds again while leaving the full run
  as the pre-commit and pre-PR gate.

The other three tiers are all at or under a minute and none of them is on the inner loop. They
are fine as they are. In CI the default tier is the whole `server` job's critical path, which
is why the pipeline is structured so that adding `-n auto` or a `slow` marker later is an edit
to one command rather than a change to the pipeline.

---

## 3. What the pipeline does

`.github/workflows/ci.yml`, on **every push and every pull request**, blocking. Three jobs,
running concurrently:

```text
server      ruff check . && ruff format --check . && mypy .
            pytest                                        # default tier + crash windows
            REQUIRE_POSTGRES_TESTS=1        pytest -m postgres
            REQUIRE_REAL_REPOSITORY_TESTS=1 pytest -m realrepo

client      npm ci && npm run typecheck && npm run lint && npm test && npm run build

container   docker build --build-arg BUILD_REVISION=$(git rev-parse HEAD) --target runtime .
```

`server/` is the working directory for every Python command; the repository root is not a
Python project. The three tiers share one job on purpose — the extra two are about eighty
seconds and a second runner costs more in setup than it saves.

The `server` job takes a `postgres:16-alpine` service container, matching the major
`docker-compose.yml` runs; a tier that passes on a different major than the deployment is a
false green. It keeps that service attached for the realrepo tier as well, for the reason in
section 1. `node`, `npm` and `pnpm@9.15.4` come from the same versions the `Dockerfile`
installs.

**Nothing in the pipeline needs a secret.** The postgres tier provisions its own databases,
the realrepo tier uses local bare repositories as origins and a scripted model double, the
canary reaches no provider, and the container job pulls only public base images and pushes
nothing.

Two things it does *not* do, deliberately:

* **No migration job.** The postgres tier already applies the packaged Alembic migrations to a
  template database and asserts the recorded revision equals `migration_head()`
  (`tests/postgres_support.py`), which is what a migrations job would check.
* **No pytest in `.pre-commit-config.yaml`.** See section 2.

### The one accommodation, which is really a finding

`Settings` has five required fields with no defaults — `openai_reasoning_model`,
`openai_coding_model`, `platform_api_key`, `database_url`, `redis_url` — and
`configs/settings.py` reads them from a `.env` at the repository root. That file is gitignored,
so a clean checkout has none, and **65 tests in the default tier call `load_settings()` without
supplying them.** On a clean machine they fail with `5 validation errors for
_DirectoryScopedSettings`; on a developer's machine they silently adopt that developer's real
`PLATFORM_API_KEY` and live `DATABASE_URL`, which is the half of this worth fixing.

The `server` and `canary` jobs supply the five as literal placeholders — no test authenticates
with the key, no provider is called, the database URL is in-process SQLite and the cache URL is
the runner's own loopback. They are not secrets and are not GitHub Secrets. The proper fix is
for the suite to supply its own configuration rather than inherit an operator's, and that is a
change under `server/` rather than to the pipeline.

---

## 4. The canary

`server/tests/canary.py` is a runnable target, not a test. It runs three whole features through
the real orchestrator, the real control plane, the real queue with two dispatchers, and the
real child executor against real git checkouts with real subprocesses and a local bare
repository as each origin. Then it answers six questions from the durable record and prints
the answers as JSON.

```bash
cd server
uv run python -m tests.canary --json canary.json
```

Exit codes are deliberately three-valued:

```text
0   every invariant held
1   an invariant was violated          -> notify
2   the canary could not run at all    -> notify, and fix the runner
```

Two is separate from one because a canary that could not run is not a canary that passed,
and a job that collapses them will eventually report a green build for a machine with no
PostgreSQL on it.

### The six invariants

Each is a question production has already answered wrongly. The count in each message is the
production baseline, so a future failure reads as a regression against a known number.

```text
every_approved_child_has_a_pull_request                          9 approved children had no PR
no_child_left_running_under_a_terminal_feature                   16 child rows were left running
every_terminal_feature_is_classified_and_diagnosed               10 of 50 unclassified, 5 with no diagnostics
every_attempted_workstream_has_a_current_revision                165 of 281 workstreams had no revision
no_duplicate_clone_commit_push_or_pull_request                   7 of 7 cross-links re-attempted after success
no_feature_exceeds_its_runtime_ceiling_without_a_terminal_state   live features averaged 86 min, reached 571
```

### What it cannot tell you

There is no model call and no remote side effect. A green canary says the platform did not
lose, duplicate or misreport work. It says nothing about whether the work was any good —
that is a question only a live run with a real model can answer.

**It does not exercise a second agent platform.** A feature now chooses the provider it runs
on, but the canary substitutes a deterministic model double for the Engineer and the Reviewer
and never constructs a provider client at all, so running it on `anthropic` would exercise the
same orchestration against the same double and assert nothing new. A genuine Anthropic canary
needs a live key, which this repository does not hold.

That is a bound on coverage, logged rather than assumed: this platform's rule is that a limit
which silently truncates coverage gets written down. What *is* covered without a key is every
layer around the call — the request shape, `max_tokens`, the two classified stop reasons, the
credential gate, and the provider surviving a restart — in `tests/test_anthropic_adapter.py`,
`tests/test_agent_platform.py` and the executor-vanished case in `tests/test_crash_windows.py`.

### Where it runs

`.github/workflows/canary.yml`, at 03:17 UTC nightly, and on `workflow_dispatch` for anyone
who wants one now.

**It cannot block a merge**, and not by convention: the workflow has no `push` and no
`pull_request` trigger, so it never produces a check on a pull request and cannot be selected
as a required one. A flaky canary that blocks merges gets disabled within a week.

The job branches on the exit code rather than collapsing it. `1` annotates *the platform
regressed*; `2` annotates *the canary could not run*, which is not a canary that passed and is
a different thing to go and fix. Both fail the job so the scheduled-run notification fires.

`canary.json` is uploaded as a run artifact with a thirty-day retention, always — including
after exit 2, because the canary writes its report before returning and that report names what
stopped it. Without the artifact a failure is only diagnosable by running it again, which is
the one thing a nightly cannot do for you.

`tests/test_canary.py` also runs it as part of the `realrepo` tier, so the target cannot rot
silently between the nights it is meant to run on.
