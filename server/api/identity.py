"""Who is acting, and what they are allowed to do.

The platform used to authenticate one shared key and record nothing about who held it. Every
audit answer was therefore the same answer: somebody with the key. This module replaces that
with individual identity, while keeping the shared key working as an explicit administrative
compatibility mode -- removing it outright would break the operator console, the browser
flows and every deployment that has one configured, for no security gain over disabling it.

Authorization is checked here and enforced in the routes. Frontend button visibility is a
convenience; it is never the control.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

# The identity a request authenticated with the deployment's shared platform key resolves to.
PLATFORM_ADMIN_ID = "platform-admin"

# How a request proved who it is. Recorded on actions so an audit can tell a named person
# from somebody holding the shared key -- which is a real distinction, and was invisible.
AUTH_USER_TOKEN = "user_token"
AUTH_PLATFORM_KEY = "platform_key"


class Permission(StrEnum):
    """Every distinct thing an actor may be allowed to do.

    Named after the decision rather than the endpoint, because more than one route reaches
    the same decision: a repository is retried from a button and from a confirmed chat
    proposal, and both must be the same check.
    """

    FEATURE_READ = "feature:read"
    FEATURE_CREATE = "feature:create"
    FEATURE_ANSWER_CLARIFICATION = "feature:answer_clarification"
    FEATURE_RETRY = "feature:retry"
    # Opening a pull request on a repository this platform has write access to, for work that
    # did not pass review. Deliberately not a reuse of FEATURE_RETRY: buying another attempt
    # spends money inside this platform, and this puts rejected code in front of a merge
    # button on somebody's repository. A strictly larger grant, and the two must be separable.
    FEATURE_PUBLISH = "feature:publish"
    FEATURE_CANCEL = "feature:cancel"
    FEATURE_RETIRE = "feature:retire"
    CONTRACT_APPROVE = "contract:approve"
    CONTRACT_REJECT = "contract:reject"
    REPAIR_APPROVE = "repair:approve"
    REPAIR_REJECT = "repair:reject"
    ACTION_EXECUTE = "action:execute"
    ACTION_RECONCILE = "action:reconcile"
    CREDENTIAL_MANAGE = "credential:manage"
    # Authoring model setups is its own decision rather than a reuse of CREDENTIAL_MANAGE: a
    # setup names models and efforts, not secrets, and naming the grant after the decision
    # keeps it honest. A viewer may read the resolved table and author nothing.
    MODEL_SETUP_MANAGE = "model_setup:manage"
    # Pointing the deployment's Slack delivery at a workspace and channel. Its own decision
    # for MODEL_SETUP_MANAGE's reason: the configuration names a channel, not a secret --
    # the bot token itself stays behind CREDENTIAL_MANAGE like every other credential.
    SLACK_CONFIGURATION_MANAGE = "slack_configuration:manage"
    # Pointing the deployment's design citations at one Figma account, and choosing which
    # files may be cited. Its own decision for SLACK_CONFIGURATION_MANAGE's reason: the
    # configuration names an account and an allowlist, not a secret -- the personal access
    # token itself stays behind CREDENTIAL_MANAGE like every other credential.
    DESIGN_SOURCE_MANAGE = "design_source:manage"
    # Reading a feature that is in somebody else's workspace. Its own decision because it is
    # the deployment operator's grant rather than an ordinary one: the disaster-recovery
    # runbook has an administrator reconciling unresolved external operations across every
    # feature, and that has to keep working.
    #
    # Read only, and deliberately not paired with a write. An administrator may look at
    # anybody's feature; retrying, publishing or retiring it would spend that person's money
    # and push with that person's token, so it is refused. If a real need appears it gets its
    # own permission and its own audit record rather than being folded into this one.
    #
    # Named as a grant so nothing has to test a role string. `Actor.roles` holds raw strings
    # and an unrecognised one grants nothing; a scattered `"admin" in actor.roles` would be a
    # second authorization authority beside `ROLE_PERMISSIONS`.
    WORKSPACE_READ_ANY = "workspace:read_any"
    USER_MANAGE = "user:manage"


class Role(StrEnum):
    """The grants a deployment hands out, rather than a permission at a time."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


_VIEWER_PERMISSIONS = frozenset({Permission.FEATURE_READ})

# Everything that changes a feature. Deliberately excludes user management: somebody who can
# retry a repository should not thereby be able to grant themselves more.
_OPERATOR_PERMISSIONS = _VIEWER_PERMISSIONS | {
    Permission.FEATURE_CREATE,
    Permission.FEATURE_ANSWER_CLARIFICATION,
    Permission.FEATURE_RETRY,
    Permission.FEATURE_PUBLISH,
    Permission.FEATURE_CANCEL,
    Permission.FEATURE_RETIRE,
    Permission.CONTRACT_APPROVE,
    Permission.CONTRACT_REJECT,
    Permission.REPAIR_APPROVE,
    Permission.REPAIR_REJECT,
    Permission.ACTION_EXECUTE,
    Permission.CREDENTIAL_MANAGE,
    Permission.MODEL_SETUP_MANAGE,
}

# The deployment operator. Everything an operator may do, plus the three decisions that are
# about the deployment rather than about one person's work.
#
# `SLACK_CONFIGURATION_MANAGE` and `DESIGN_SOURCE_MANAGE` moved here from the operator set
# when workspaces became isolated, and that is a *narrowing* of an existing grant -- it is in
# the rollout notes for that reason. Both point a deployment-wide singleton at a workspace:
# `slack_workspace_configurations` permits one enabled row and
# `design_source_configurations` the same, so under multi-user an ordinary operator holding
# these was an ordinary user reconfiguring everybody's Slack delivery and everybody's design
# citations. Neither names a secret -- the bot token and the Figma token stay behind
# `CREDENTIAL_MANAGE` like every other credential -- which is why they were operator grants
# in the first place, and why that reasoning stopped holding once there was more than one
# person in the deployment.
_ADMIN_PERMISSIONS = _OPERATOR_PERMISSIONS | {
    Permission.ACTION_RECONCILE,
    Permission.SLACK_CONFIGURATION_MANAGE,
    Permission.DESIGN_SOURCE_MANAGE,
    Permission.WORKSPACE_READ_ANY,
    Permission.USER_MANAGE,
}

# What each action a person can confirm in chat actually asks the platform to do.
#
# A confirmed proposal reaches the same control-plane method the ordinary button does, so it
# has to meet the same bar. One blanket "may execute chat actions" permission would have let
# anybody allowed to answer a clarification also approve a repository repair, purely because
# they asked for it in a sentence rather than by pressing the button that does it.
ACTION_PERMISSIONS: dict[str, Permission] = {
    "ANSWER_CLARIFICATION": Permission.FEATURE_ANSWER_CLARIFICATION,
    "RESUME_WORKFLOW": Permission.FEATURE_ANSWER_CLARIFICATION,
    "RETRY_WORKSTREAM": Permission.FEATURE_RETRY,
    # Deciding a design conflict answers a question *and* starts an attempt on a stopped
    # repository. One decision, one check: it is guarded by the permission that covers the
    # effect, because answering a clarification costs nothing and running an attempt does.
    "ANSWER_DESIGN_VERDICT": Permission.FEATURE_RETRY,
    "CANCEL_WORKFLOW": Permission.FEATURE_CANCEL,
    "APPROVE_REPOSITORY_REPAIR": Permission.REPAIR_APPROVE,
    "REJECT_REPOSITORY_REPAIR": Permission.REPAIR_REJECT,
    "APPROVE_CONTRACT_CHANGE": Permission.CONTRACT_APPROVE,
    "REJECT_CONTRACT_CHANGE": Permission.CONTRACT_REJECT,
    "RETIRE_FEATURE": Permission.FEATURE_RETIRE,
    # Guarded by the permission that covers its effect, on the same rule as
    # ANSWER_DESIGN_VERDICT above: this opens pull requests, some of them holding code a
    # review rejected, so it is not a retry and is not checked as one.
    "PUBLISH_FEATURE": Permission.FEATURE_PUBLISH,
    # A revision is a new run in everything but identity: it plans, codes, pushes and opens
    # pull requests. Guarded by the permission that covers that effect -- creating feature
    # work -- rather than as a retry, which buys attempts on an existing run.
    "REVISE_FEATURE": Permission.FEATURE_CREATE,
}


def permission_for_action(action_type: str) -> Permission:
    """Return the permission one action type requires.

    An unrecognised type falls back to the most restrictive answer rather than the most
    permissive: a build that does not know what an action does must not conclude that anybody
    may do it.
    """
    return ACTION_PERMISSIONS.get(action_type, Permission.USER_MANAGE)


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset(_VIEWER_PERMISSIONS),
    Role.OPERATOR: frozenset(_OPERATOR_PERMISSIONS),
    Role.ADMIN: frozenset(_ADMIN_PERMISSIONS),
}


@dataclass(frozen=True, slots=True)
class Actor:
    """The authenticated identity a request acts as."""

    actor_id: str
    display_name: str
    authentication: str = AUTH_PLATFORM_KEY
    roles: frozenset[str] = field(default_factory=frozenset)

    @property
    def permissions(self) -> frozenset[Permission]:
        """Return everything this actor's roles allow, ignoring roles nobody defined."""
        granted: set[Permission] = set()
        for name in self.roles:
            try:
                granted |= ROLE_PERMISSIONS[Role(name)]
            except ValueError:
                # A role a newer build wrote and this one does not know grants nothing.
                # Failing open on an unrecognised name is how privilege gets invented.
                continue
        return frozenset(granted)

    def may(self, permission: Permission) -> bool:
        """Return whether this actor holds one permission."""
        return permission in self.permissions

    @property
    def is_platform_key(self) -> bool:
        """Return whether this is the shared administrative credential rather than a person."""
        return self.authentication == AUTH_PLATFORM_KEY


@dataclass(frozen=True, slots=True)
class WorkspaceScope:
    """Whose features a caller may reach.

    One value, built once per request from the authenticated actor, and applied in the SQL
    `WHERE` of every feature read and every feature mutation. A row that does not match is
    `WorkflowNotFoundError`, which the routes already answer `404` -- so a caller who does
    not own a feature is told the same thing as a caller who named one that never existed,
    and that cannot be forgotten route by route.

    404 rather than 403 deliberately: `AB-Feature-N` is a dense global sequence, so a 403
    confirms existence and lets anybody count how much work the platform is doing and for
    whom.

    `owner_id` is always the actor's own id and never anything a client sent. There is no
    constructor here that takes an owner from a request body or a query string, which is what
    makes "ownership is derived from the authenticated actor" a property of the type rather
    than a rule each route remembers.
    """

    owner_id: str
    may_read_any: bool = False

    def applies(self) -> bool:
        """Return whether a query must be narrowed to one owner."""
        return not self.may_read_any

    def for_mutation(self) -> WorkspaceScope:
        """Return this scope narrowed to its own workspace, whatever it may read.

        `WORKSPACE_READ_ANY` is a read grant and only a read grant. An administrator is the
        deployment operator: the disaster-recovery runbook has them reading across every
        workspace and reconciling unresolved operations, and that has to keep working.

        Acting is different. Retrying, publishing or retiring somebody else's feature spends
        that person's money on model calls and pushes to their repository with their GitHub
        token -- the queue entry names the feature's owner precisely so it does. One person's
        credential doing another person's work is the thing this whole change exists to
        prevent, and an administrator is not an exception to it.

        So every mutation asks with this, and an administrator acting on a feature they do
        not own gets the same `404` anybody else would. If a real need appears -- an operator
        having to cancel a runaway feature on somebody's behalf -- it gets its own permission
        and its own audit record rather than being folded into the read grant.
        """
        return WorkspaceScope(owner_id=self.owner_id, may_read_any=False)

    @classmethod
    def of(cls, actor: Actor) -> WorkspaceScope:
        """Return the scope one authenticated identity acts within."""
        return cls(
            owner_id=actor.actor_id,
            may_read_any=actor.may(Permission.WORKSPACE_READ_ANY),
        )

    @classmethod
    def unscoped(cls) -> WorkspaceScope:
        """Return the scope a worker acts within: every workspace, because it acts on none.

        Named and explicit rather than available as a scope-less overload of each store
        method. Two overloads is how the scoped one gets forgotten -- a background sweep that
        happened to call the short form would silently read the whole deployment, which is
        correct for the sweep and catastrophic for a route.

        `owner_id` is empty because it is never consulted: `applies()` is false, so no query
        narrows by it. An empty string is safer here than a plausible id, which could match a
        row if some future caller read the field without checking `may_read_any`.
        """
        return cls(owner_id="", may_read_any=True)


def platform_admin() -> Actor:
    """Return the identity behind the deployment's shared platform key.

    It holds every permission, which is what the shared key has always effectively been.
    Naming it makes that explicit in the audit record instead of leaving it unrecorded.
    """
    return Actor(
        actor_id=PLATFORM_ADMIN_ID,
        display_name="Platform operator",
        authentication=AUTH_PLATFORM_KEY,
        roles=frozenset({Role.ADMIN.value}),
    )


def issue_token() -> tuple[str, str]:
    """Return a new token and the digest to store for it.

    The token is returned exactly once, to the caller who asked for it. Only the digest is
    persisted, so a copy of the database does not let somebody act as its users.
    """
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    """Return the stored form of a token.

    A plain digest rather than a password hash on purpose: this is a 256-bit random value,
    not something a person chose, so there is nothing for an attacker to guess more cheaply
    than the token itself and no work factor worth the latency on every request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_is_usable(*, expires_at: datetime | None, revoked_at: datetime | None) -> bool:
    """Return whether a stored token may still authenticate a request."""
    if revoked_at is not None:
        return False
    return expires_at is None or expires_at > datetime.now(UTC)


__all__ = [
    "ACTION_PERMISSIONS",
    "AUTH_PLATFORM_KEY",
    "AUTH_USER_TOKEN",
    "PLATFORM_ADMIN_ID",
    "ROLE_PERMISSIONS",
    "Actor",
    "Permission",
    "Role",
    "WorkspaceScope",
    "hash_token",
    "issue_token",
    "permission_for_action",
    "platform_admin",
    "token_is_usable",
]
