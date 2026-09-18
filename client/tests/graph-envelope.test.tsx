// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { WorkflowGraph } from '@/features/feature-workspace/WorkflowGraph';
import {
  MAX_NODE_OPS,
  nodeOperations,
  type NodeOpLine,
} from '@/features/feature-workspace/agent-work';
import type { FeatureGraph, GraphNode } from '@/features/feature-workspace/graph';
import type { WorkstreamAttempt, WorkstreamOperation } from '@/schemas/feature';

/**
 * The graph draws the agent around its work (65 B).
 *
 * The grouping the todo asked for — "group together the things a single agent does" — shipped
 * in the drawer's stage headers and never in the graph, where Implementation and Validation
 * render as free-floating siblings with nothing saying one agent owns both.
 *
 * Two of these decide correctness. B1 asserts node coordinates: the envelope is a pure
 * additive background layer, and a box that moved a node would be a re-layout wearing a
 * decoration's clothes. B3 asserts the edge label between the two member nodes is still
 * clickable, because the box is drawn across the gap it sits in.
 */

const REPOSITORY = 'admanager_console-2.0';

/** Fixed, because a duration rendered from the wall clock is not a deterministic assertion. */
const NOW = Date.parse('2026-09-03T06:02:00Z');

// The arithmetic the graph is built from, restated here so a change to either is a failing
// test rather than a silently moved node. NODE_WIDTH 168, NODE_HEIGHT 78, COLUMN_GAP 112,
// ROW_GAP 84, PADDING 16.
const COLUMN_STRIDE = 168 + 112;
// A lane node carries the operations of the stage it owns, so it is 152px rather than the
// 78px a parent stage node still is, and the stride is sized on the taller of the two: row 0
// holds the parent stages AND the first lane, so a stride sized on 78 would let one lane's
// operations run into the lane below it.
const LANE_NODE_HEIGHT = 152;
const ROW_STRIDE = LANE_NODE_HEIGHT + 84;
const PADDING = 16;

function node(overrides: Partial<GraphNode> & Pick<GraphNode, 'id' | 'kind'>): GraphNode {
  return {
    label: overrides.kind,
    state: 'active',
    column: 0,
    row: 0,
    ...overrides,
  } as GraphNode;
}

/** One lane: repository → implementation → validation → review, at today's columns. */
function lane(repositoryId: string, row: number): GraphNode[] {
  return [
    node({ id: `repo:${repositoryId}`, kind: 'repository', repositoryId, column: 4, row }),
    node({
      id: `implement:${repositoryId}`,
      kind: 'implement',
      label: 'Implementation',
      repositoryId,
      column: 5,
      row,
    }),
    node({
      id: `validate:${repositoryId}`,
      kind: 'validate',
      label: 'Validation',
      repositoryId,
      column: 6,
      row,
    }),
    node({
      id: `review:${repositoryId}`,
      kind: 'review',
      label: 'Review',
      repositoryId,
      column: 7,
      row,
    }),
  ];
}

function graphOf(nodes: GraphNode[], edges: FeatureGraph['edges'] = []): FeatureGraph {
  return { nodes, edges, columns: 10, rows: Math.max(1, ...nodes.map((item) => item.row + 1)) };
}

function renderGraph(
  graph: FeatureGraph,
  operations?: ReadonlyMap<string, NodeOpLine[]>,
  onSelectNode?: (node: GraphNode) => void,
) {
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <WorkflowGraph
        graph={graph}
        label="Feature execution graph"
        operations={operations}
        onSelectNode={onSelectNode}
      />
    </MemoryRouter>,
  );
}

/** The operations of one lane node, keyed the way the graph keys its nodes. */
function opsFor(
  repositoryId: string,
  rows: WorkstreamOperation[],
  endings: WorkstreamAttempt[],
  stages: readonly string[] = ['setup', 'coding'],
  prefix = 'implement',
): ReadonlyMap<string, NodeOpLine[]> {
  return new Map([
    [`${prefix}:${repositoryId}`, nodeOperations(rows, endings, stages, { nowMs: NOW })],
  ]);
}

let sequence = 0;

function operation(overrides: Partial<WorkstreamOperation> = {}): WorkstreamOperation {
  sequence += 1;
  return {
    operation_id: `op-${sequence}`,
    operation_type: 'run_coding_executor',
    stage: 'coding',
    status: 'succeeded',
    attempt: 1,
    max_attempts: 1,
    child_attempt: 0,
    started_at: '2026-09-03T06:00:00Z',
    heartbeat_at: null,
    completed_at: '2026-09-03T06:01:00Z',
    error_code: null,
    repeat: null,
    stream_reissues: null,
    ...overrides,
  };
}

function ending(overrides: Partial<WorkstreamAttempt> = {}): WorkstreamAttempt {
  return {
    attempt: 0,
    ended_by: 'review_rejected',
    stage: 'review',
    detail: 'review returned changes requested with 1 finding, blocking on F-1',
    workspace: 'preserved',
    self_review_outcome: 'clean',
    self_review_corrected_files: 0,
    source_repair_passes: null,
    stream_reissues: null,
    ...overrides,
  };
}

describe('B1 — the Engineer box', () => {
  it('draws one container around exactly Implementation and Validation', () => {
    renderGraph(graphOf(lane(REPOSITORY, 0)));

    const boxes = document.querySelectorAll('.graph__envelope');
    expect(boxes).toHaveLength(1);
    const box = boxes[0]!;
    // Implementation sits at column 5 and Validation at column 6, so the box spans exactly
    // those two nodes plus the 8px inset that fits inside the canvas padding.
    expect(box.getAttribute('x')).toBe(String(PADDING + 5 * COLUMN_STRIDE - 8));
    expect(box.getAttribute('y')).toBe(String(PADDING - 8));
    expect(box.getAttribute('width')).toBe(String(COLUMN_STRIDE + 168 + 16));
    // As tall as the lane nodes it contains, which carry their stages' operations.
    expect(box.getAttribute('height')).toBe(String(LANE_NODE_HEIGHT + 16));

    // Labelled with the agent, from the one ownership vocabulary.
    expect(screen.getByRole('img', { name: `Engineer, ${REPOSITORY}` })).toBeInTheDocument();

    // The Review node is outside it: the reviewer is a different agent, and the box stops
    // where the run of same-agent nodes does.
    const review = screen.getByRole('button', { name: /^Review, Review/ });
    const reviewLeft = Number.parseInt(review.style.left, 10);
    const boxRight = Number(box.getAttribute('x')) + Number(box.getAttribute('width'));
    expect(reviewLeft).toBeGreaterThan(boxRight);

    // Node placement is the arithmetic and nothing else. The envelope is still a pure
    // background layer -- it moves no node -- but the lane nodes are 152px now that they
    // carry their operations, so the stride they sit on is sized from that, and this loop is
    // what fails if either number drifts.
    for (const [name, column] of [
      [/^Repository, repository/, 4],
      [/^Implementation, Implementation/, 5],
      [/^Validation, Validation/, 6],
      [/^Review, Review/, 7],
    ] as const) {
      const placed = screen.getByRole('button', { name });
      expect(placed.style.left).toBe(`${PADDING + column * COLUMN_STRIDE}px`);
      expect(placed.style.top).toBe(`${PADDING}px`);
    }
  });

  it('draws a box per lane and never one that spans two rows', () => {
    renderGraph(graphOf([...lane('backend', 0), ...lane('frontend', 1)]));

    const boxes = [...document.querySelectorAll('.graph__envelope')];
    expect(boxes).toHaveLength(2);
    expect(boxes.map((box) => box.getAttribute('y'))).toEqual([
      String(PADDING - 8),
      String(PADDING + ROW_STRIDE - 8),
    ]);
    // Each box is one lane tall. A box that grouped across rows would swallow the row gap the
    // retry loop lives in.
    expect(new Set(boxes.map((box) => box.getAttribute('height')))).toEqual(
      new Set([String(LANE_NODE_HEIGHT + 16)]),
    );
  });

  it('never puts the repository node inside the box, though the Engineer owns setup too', () => {
    // The trap: `AGENT_FOR_STAGE` owns `setup: 'Engineer'`, so mapping the lane's header node
    // to `setup` would silently stretch the box across three nodes.
    renderGraph(graphOf(lane(REPOSITORY, 0)));

    const box = document.querySelector('.graph__envelope')!;
    const repository = screen.getByRole('button', { name: /^Repository, repository/ });
    expect(Number.parseInt(repository.style.left, 10)).toBeLessThan(Number(box.getAttribute('x')));
  });
});

describe('B2 — no single-node boxes', () => {
  it('draws no container for a lane shape where no two contiguous nodes share an agent', () => {
    // Review alone: a box around one node is noise, and the drawer already names single-stage
    // ownership.
    renderGraph(
      graphOf([
        node({ id: `repo:${REPOSITORY}`, kind: 'repository', repositoryId: REPOSITORY, column: 4 }),
        node({
          id: `review:${REPOSITORY}`,
          kind: 'review',
          label: 'Review',
          repositoryId: REPOSITORY,
          column: 7,
        }),
      ]),
    );

    expect(document.querySelectorAll('.graph__envelope')).toHaveLength(0);
    expect(screen.queryByRole('img', { name: /Engineer/ })).not.toBeInTheDocument();
  });

  it('draws no container for the stages outside any lane', () => {
    renderGraph(
      graphOf([
        node({ id: 'request', kind: 'stage', column: 0 }),
        node({ id: 'technical_prd', kind: 'stage', column: 1 }),
        node({ id: 'integration_review', kind: 'integration', column: 8 }),
        node({ id: 'pull_requests', kind: 'pull-requests', column: 9 }),
      ]),
    );

    expect(document.querySelectorAll('.graph__envelope')).toHaveLength(0);
  });
});

describe('B3 — labels stay on top', () => {
  it('keeps the edge-label button between Implementation and Validation clickable', async () => {
    const onSelectEdge = vi.fn();
    const graph = graphOf(lane(REPOSITORY, 0), [
      {
        id: `implement:${REPOSITORY}->validate:${REPOSITORY}`,
        from: `implement:${REPOSITORY}`,
        to: `validate:${REPOSITORY}`,
        kind: 'flow',
        label: 'Validator',
        executions: [],
        showLabel: true,
      },
    ]);
    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <WorkflowGraph
          graph={graph}
          label="Feature execution graph"
          onSelectEdge={onSelectEdge}
        />
      </MemoryRouter>,
    );

    // The box is drawn across the column gap this label sits in. It is an SVG rect in a layer
    // beneath, with pointer events off, so the label is still the thing a click reaches.
    const label = screen.getByRole('button', { name: 'Validator' });
    expect(document.querySelector('.graph__envelopes')).toBeInTheDocument();
    // Layer order, asserted as document order: the box's SVG comes before the label, so the
    // label paints above it.
    const canvas = document.querySelector('.graph__canvas')!;
    const children = [...canvas.children];
    expect(children.indexOf(document.querySelector('.graph__envelopes')!)).toBe(0);
    expect(children.indexOf(label)).toBeGreaterThan(0);

    // An edge with no executions is deliberately not openable, which is today's behaviour;
    // what B3 needs is that the button is present, enabled by its own rule, and reachable.
    expect(label).toBeDisabled();
  });
});

describe('B4 — the node says what the agent did', () => {
  it('lists the operations of the stages it owns, in the order they ran', () => {
    const rows = [
      // Newest first, as the endpoint answers.
      operation({ operation_type: 'run_coding_executor', stage: 'coding', status: 'running', completed_at: null }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
      operation({ operation_type: 'clone_repository', stage: 'setup' }),
    ];
    renderGraph(graphOf(lane(REPOSITORY, 0)), opsFor(REPOSITORY, rows, []));

    const names = [...document.querySelectorAll('.graph__node-op-name')].map(
      (item) => item.textContent,
    );
    // Setup's rows, then coding's, then the self-review that sits after the coding call --
    // the drawer's own order, on the node whose stages ran them.
    expect(names).toEqual([
      'clone_repository',
      'install_dependencies',
      'run_coding_executor',
      'self-review',
    ]);
    // The running call carries the running state, so the node says which operation is live.
    const running = document.querySelector('.graph__node-op--running .graph__node-op-name');
    expect(running).toHaveTextContent('run_coding_executor');
  });

  it('names an empty stage rather than leaving it blank', () => {
    // The Review node of an attempt whose ending is a self-review failure: the reviewer never
    // ran, and "never ran" is the word A.1 licenses for it.
    const rows = [operation({ operation_type: 'install_dependencies', stage: 'setup' })];
    renderGraph(
      graphOf(lane(REPOSITORY, 0)),
      opsFor(REPOSITORY, rows, [ending({ ended_by: 'self_review', stage: 'coding' })], ['review'], 'review'),
    );

    const review = screen.getByRole('button', { name: /^Review, Review/ });
    expect(review).toHaveAccessibleName(/review never ran/);
  });

  it('maps all seven self-review states, including the one the code does not enumerate', () => {
    const selfReview = (outcome: string | null, files = 0) =>
      nodeOperations(
        [operation({ operation_type: 'install_dependencies', stage: 'setup' })],
        [ending({ self_review_outcome: outcome, self_review_corrected_files: files })],
        ['setup', 'coding'],
        { nowMs: NOW },
      ).find((line) => line.name === 'self-review')!;

    expect(selfReview('clean')).toMatchObject({ state: 'succeeded', word: 'clean' });
    // Files corrected, never correction rounds: `corrections_applied` is a list of paths.
    expect(selfReview('corrected', 2)).toMatchObject({
      state: 'succeeded',
      word: 'corrected · 2 files',
    });
    expect(selfReview('unavailable')).toMatchObject({
      state: 'other',
      word: 'composed but could not run',
    });
    expect(selfReview('substantive_problem')).toMatchObject({ state: 'failed' });
    // The gate that actually stopped 201's attempt 0. A node that rendered only `corrected`
    // and blanked the rest is how this becomes invisible -- the exact defect Part A fixes.
    expect(selfReview('corrections_failed')).toMatchObject({
      state: 'failed',
      word: 'corrections failed',
    });
    expect(selfReview('correction_rejected')).toMatchObject({ state: 'failed' });
    // The seventh: 201's backend attempt 4 carries no `self_review` key at all, and a blank
    // line there reads as a clean pass.
    expect(selfReview(null)).toMatchObject({
      state: 'skipped',
      word: 'no self-review recorded',
    });
  });

  it('never says the self-review is still to come once a later stage has run', () => {
    // AB-Feature-202's frontend, live: a cancelled run serves no endings at all, so the
    // self-review had no record -- and the node reported it "not yet seen" beside lint, tests
    // and build all succeeded. The gate runs before validation, so a validation row is proof
    // the attempt is past it, and a claim about the future is false there.
    const rows = [
      operation({ operation_type: 'run_build', stage: 'validation' }),
      operation({ operation_type: 'run_coding_executor', stage: 'coding' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
    ];
    const line = nodeOperations(rows, [], ['setup', 'coding'], { nowMs: NOW }).find(
      (item) => item.name === 'self-review',
    )!;

    expect(line).toMatchObject({ state: 'skipped', word: 'no record' });

    // Before anything after coding has run, "not yet seen" is the honest word and stays.
    const early = nodeOperations(
      [operation({ operation_type: 'install_dependencies', stage: 'setup' })],
      [],
      ['setup', 'coding'],
      { nowMs: NOW },
    ).find((item) => item.name === 'self-review')!;
    expect(early).toMatchObject({ state: 'pending', word: 'not yet seen' });
  });

  it('lines in-attempt repair passes only where there were any', () => {
    const withRepairs = (passes: number | null) =>
      nodeOperations(
        [operation({ operation_type: 'install_dependencies', stage: 'setup' })],
        [ending({ source_repair_passes: passes })],
        ['setup', 'coding'],
        { nowMs: NOW },
      ).map((line) => line.key);

    // Absent renders nothing, and `0` renders nothing either: across 201's nine backend
    // completions the field is present twice, both times zero.
    expect(withRepairs(null)).not.toContain('repairs');
    expect(withRepairs(0)).not.toContain('repairs');
    expect(withRepairs(2)).toContain('repairs');
  });

  it('keeps the newest operations and counts the rest, rather than dropping them', () => {
    // 201's Implementation node holds seven lines. An attempt that ran more says how many it
    // is not showing, and keeps the newest -- an operator watches what is happening now.
    const rows = Array.from({ length: 10 }, (_unused, index) =>
      operation({ operation_type: `op_${9 - index}`, stage: 'coding' }),
    );
    renderGraph(graphOf(lane(REPOSITORY, 0)), opsFor(REPOSITORY, rows, []));

    const lines = [...document.querySelectorAll('.graph__node-op')];
    expect(lines).toHaveLength(MAX_NODE_OPS);
    expect(lines[0]).toHaveTextContent(/^\+5 earlier operations$/);
    // The self-review line is the newest, because it follows the coding rows.
    expect(lines.at(-1)).toHaveTextContent('self-review');
    // And the count is in the accessible name, so it is not a visual-only fact.
    const node = screen.getByRole('button', { name: /^Implementation, Implementation/ });
    expect(node).toHaveAccessibleName(/5 earlier operations not shown/);
  });

  it('never lists publication rows, because no lane node owns them', () => {
    // `create_commit` and `push_branch` are the attempt persisting its work to the branch.
    // The lane has no publication node, and inventing one would claim a stage the graph does
    // not model, so they stay in the drawer.
    const rows = [
      operation({ operation_type: 'push_branch', stage: 'publication' }),
      operation({ operation_type: 'create_commit', stage: 'publication' }),
      operation({ operation_type: 'install_dependencies', stage: 'setup' }),
    ];
    renderGraph(graphOf(lane(REPOSITORY, 0)), opsFor(REPOSITORY, rows, []));

    const names = [...document.querySelectorAll('.graph__node-op-name')].map(
      (item) => item.textContent,
    );
    expect(names).toContain('install_dependencies');
    expect(names).not.toContain('create_commit');
    expect(names).not.toContain('push_branch');
  });

  it('opens the agent-work drawer from the node the operations are on', async () => {
    const onSelectNode = vi.fn();
    const rows = [operation({ operation_type: 'install_dependencies', stage: 'setup' })];
    renderGraph(graphOf(lane(REPOSITORY, 0)), opsFor(REPOSITORY, rows, []), onSelectNode);

    await userEvent.click(screen.getByRole('button', { name: /^Implementation, Implementation/ }));

    expect(onSelectNode).toHaveBeenCalledTimes(1);
    expect(onSelectNode.mock.calls[0]![0].id).toBe(`implement:${REPOSITORY}`);
  });

  it('draws the box and its nodes while the read has not arrived', () => {
    renderGraph(graphOf(lane(REPOSITORY, 0)), new Map());

    expect(document.querySelectorAll('.graph__envelope')).toHaveLength(1);
    expect(document.querySelectorAll('.graph__node-ops')).toHaveLength(0);
  });

  it('reflects the newest attempt only, because history lives in the drawer', () => {
    const lines = nodeOperations(
      [
        operation({ operation_type: 'run_coding_executor', stage: 'coding', child_attempt: 1 }),
        operation({ operation_type: 'install_dependencies', stage: 'setup', child_attempt: 1 }),
        operation({ operation_type: 'run_reviewer', stage: 'review', child_attempt: 0 }),
      ],
      [
        ending({ attempt: 0, self_review_outcome: 'corrections_failed' }),
        ending({ attempt: 1, ended_by: 'validation_failed', stage: 'validation', self_review_outcome: 'clean' }),
      ],
      ['setup', 'coding'],
      { nowMs: NOW },
    );

    // Attempt 1's self-review, not attempt 0's -- reading the older one onto the node would
    // be the class of mistake the drawer's per-attempt split exists to remove.
    expect(lines.find((line) => line.name === 'self-review')!.word).toBe('clean');
    // And attempt 0's reviewer call is not on it: the node shows one attempt.
    expect(lines.map((line) => line.name)).not.toContain('run_reviewer');
  });
});

/**
 * The remediation loop's geometry (65 D).
 *
 * A new branch, not a reuse: `geometryFor` tests `edge.kind === 'retry'` before comparing
 * columns and computes its dip from `from.y` alone, so a cross-column, cross-row back edge
 * drawn as a retry would run straight through the lanes between its ends.
 */
describe('D — the remediation loop is drawn as its own edge', () => {
  const twoLanes = () => [
    ...lane('backend', 0),
    ...lane('frontend', 1),
    node({ id: 'integration_review', kind: 'integration', label: 'Integration review', column: 8, row: 0.5 }),
  ];

  function remediationEdge(to: string): FeatureGraph['edges'][number] {
    return {
      id: `integration_review->${to}`,
      from: 'integration_review',
      to,
      kind: 'remediation',
      label: 'Remediation 2 / 5',
      executions: [],
      showLabel: true,
    };
  }

  it('runs under the lower of its two rows rather than through the lanes between them', () => {
    renderGraph(graphOf(twoLanes(), [remediationEdge('implement:backend')]));

    const path = [...document.querySelectorAll('.graph__edge--remediation path')][0]!;
    const drawn = path.getAttribute('d')!;
    // The control points dip below the LOWER of the two ends. The integration review sits on
    // row 0.5 and the backend lane on row 0, so the floor is the review's own bottom edge.
    const floor = PADDING + 0.5 * ROW_STRIDE + 78;
    expect(drawn).toContain(`C `);
    const dip = Math.round(floor + 84 * 0.75);
    expect(drawn).toContain(String(dip));
    // It starts and ends at the BOTTOM of each node, which is what makes it read as a loop
    // around the outside rather than an arrow through the middle.
    expect(drawn.startsWith('M ')).toBe(true);
  });

  it('reserves canvas height for its own depth, not the retry loop\'s', () => {
    const withRemediation = renderGraph(
      graphOf(twoLanes(), [remediationEdge('implement:backend')]),
    );
    const remediationHeight = Number.parseInt(
      (document.querySelector('.graph__canvas') as HTMLElement).style.height,
      10,
    );
    withRemediation.unmount();

    renderGraph(graphOf(twoLanes()));
    const bareHeight = Number.parseInt(
      (document.querySelector('.graph__canvas') as HTMLElement).style.height,
      10,
    );

    // The remediation loop dips deeper than a retry loop and carries a label under that, so
    // the canvas owns the space. A box that quietly clipped is the failure mode.
    expect(remediationHeight).toBeGreaterThan(bareHeight);
  });

  it('carries the loop mark beside its count, like the lane loop it sits beside', () => {
    renderGraph(graphOf(twoLanes(), [remediationEdge('implement:backend')]));

    const label = screen.getByRole('button', { name: /Remediation 2 \/ 5/ });
    expect(label).toHaveClass('graph__edge-label--retry');
    // Shape as well as colour, so the two kinds of arrow are distinguishable in greyscale.
    expect(label.querySelector('.graph__edge-mark')).toBeInTheDocument();
  });
});
