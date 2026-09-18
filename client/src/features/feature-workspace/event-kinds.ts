import type { TimelineEvent } from '@/schemas/feature';
import { humanise } from '@/utils/text';

/**
 * Reading one timeline event: what kind of thing it is, and how it went.
 *
 * The categories exist so the history can be filtered the way somebody debugging thinks --
 * "just the validation", "just the human decisions" -- and so a failure is red without every
 * row being coloured.
 *
 * Unknown events fall through to a neutral working row with their own name humanised. The
 * server owns this vocabulary and adds to it; a client that hid what it did not recognise
 * would hide the newest thing in the run.
 */

export type EventCategory = 'agent' | 'repository' | 'validation' | 'human' | 'git' | 'artifact' | 'lifecycle';

export interface EventKind {
  category: EventCategory;
  tone: 'done' | 'stopped' | 'attention' | 'working';
  label?: string;
}

const KNOWN: Record<string, EventKind> = {
  feature_started: { category: 'lifecycle', tone: 'working', label: 'Feature started' },
  feature_queued: { category: 'lifecycle', tone: 'working', label: 'Feature queued' },
  // A refusal somebody has to act on, not a platform fault: the fallthrough would colour it
  // by the word "missing", which is neither stopped nor merely working.
  feature_credentials_missing: {
    category: 'human',
    tone: 'attention',
    label: 'Provider credentials not configured',
  },
  feature_completed: { category: 'lifecycle', tone: 'done', label: 'Feature completed' },
  feature_failed: { category: 'lifecycle', tone: 'stopped', label: 'Feature failed' },
  feature_cancelled: { category: 'lifecycle', tone: 'stopped', label: 'Feature cancelled' },
  feature_retry_failed: { category: 'lifecycle', tone: 'stopped', label: 'Retry failed' },
  feature_waiting_for_human: { category: 'human', tone: 'attention', label: 'Waiting for a person' },
  clarification_requested: { category: 'human', tone: 'attention', label: 'Clarification requested' },
  clarification_answered: { category: 'human', tone: 'done', label: 'Clarification answered' },
  repository_repair_proposed: { category: 'human', tone: 'attention', label: 'Repository repair proposed' },
  repository_repair_approved: { category: 'human', tone: 'done', label: 'Repository repair approved' },
  repository_repair_rejected: { category: 'human', tone: 'stopped', label: 'Repository repair rejected' },
  contract_change_requested: { category: 'human', tone: 'attention', label: 'Contract change requested' },
  contract_change_approved: { category: 'human', tone: 'done', label: 'Contract change approved' },
  contract_change_rejected: { category: 'human', tone: 'stopped', label: 'Contract change rejected' },
  contract_created: { category: 'agent', tone: 'done', label: 'Shared contract written' },
  contract_approved: { category: 'agent', tone: 'done', label: 'Shared contract approved' },
  // Reconnaissance failed and the plan was written without that repository's evidence.
  // Attention rather than stopped: the feature continues, and the reader's question is
  // whether to trust the plan it continued with.
  repository_planned_blind: {
    category: 'repository',
    tone: 'attention',
    label: 'Repository planned without reconnaissance',
  },
  child_workflow_started: { category: 'repository', tone: 'working', label: 'Repository workstream started' },
  child_workflow_completed: { category: 'repository', tone: 'done', label: 'Repository workstream finished' },
  child_workflow_failed: { category: 'repository', tone: 'stopped', label: 'Repository workstream stopped' },
  integration_review_started: { category: 'agent', tone: 'working', label: 'Integration review started' },
  integration_review_completed: { category: 'agent', tone: 'done', label: 'Integration review completed' },
  pull_request_created: { category: 'git', tone: 'done', label: 'Pull request opened' },
  validation_failed: { category: 'validation', tone: 'stopped', label: 'Validation failed' },
  validation_succeeded: { category: 'validation', tone: 'done', label: 'Validation passed' },
};

export function classify(event: TimelineEvent): EventKind {
  const known = KNOWN[event.event];
  if (known) return known;

  // An artifact row is named for the artifact it produced, which is more useful than the word
  // "artifact" repeated twenty-six times.
  if (event.event_type === 'artifact') {
    return { category: 'artifact', tone: 'working', label: humanise(event.event) };
  }
  if (event.event.includes('validation')) return { category: 'validation', tone: 'working' };
  if (event.event.includes('pull_request') || event.event.includes('commit')) {
    return { category: 'git', tone: 'working' };
  }
  if (event.event.includes('failed') || event.event.includes('rejected')) {
    return { category: 'lifecycle', tone: 'stopped' };
  }
  return { category: 'lifecycle', tone: 'working' };
}

export const CATEGORY_LABELS: Record<EventCategory | 'all' | 'errors', string> = {
  all: 'All',
  agent: 'Agents',
  repository: 'Repositories',
  validation: 'Validation',
  human: 'Human actions',
  git: 'Git & PRs',
  artifact: 'Artifacts',
  lifecycle: 'Lifecycle',
  errors: 'Errors',
};

export type TimelineFilter = EventCategory | 'all' | 'errors';

export const TIMELINE_FILTERS: TimelineFilter[] = [
  'all',
  'agent',
  'repository',
  'validation',
  'human',
  'git',
  'artifact',
  'errors',
];

export function matchesFilter(event: TimelineEvent, filter: TimelineFilter): boolean {
  if (filter === 'all') return true;
  const kind = classify(event);
  if (filter === 'errors') return kind.tone === 'stopped';
  return kind.category === filter;
}
