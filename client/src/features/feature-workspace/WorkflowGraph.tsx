import { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { AGENT_FOR_STAGE, MAX_NODE_OPS, STAGE_GLYPHS, type NodeOpLine } from './agent-work';
import type { FeatureGraph, GraphEdge, GraphNode } from './graph';
import { edgeLabel, edgeTone, fallbackEdgeLabel, type EdgeLabel } from './executions';
import type { NodeLiveness } from './operations';

/**
 * The execution flow, drawn.
 *
 * Nodes are positioned from the layered model rather than by a layout library: the graph has
 * a known shape -- stages, then one lane per repository, then convergence -- so the geometry
 * is arithmetic, and the whole thing stays a pure function of backend state.
 *
 * Nodes and edges answer different questions and are both interactive for that reason. A node
 * says *what state this stage is in* and opens the stage. An edge says *who moved the feature
 * here* -- which model, which attempt, or which deterministic handler -- and opens the
 * execution behind it. The edge labels are real buttons positioned over the SVG rather than
 * SVG text, which is what makes them focusable, tab-ordered and readable by a screen reader;
 * the drawn path underneath carries a wide transparent hit stroke so clicking the arrow itself
 * works too.
 */

const NODE_WIDTH = 168;
// Tall enough for the three lines a node can carry -- what it is, which repository, and the
// count -- without any of them being clipped. A node that hid "Retrying" was worse than one
// that said nothing.
const NODE_HEIGHT = 78;
/**
 * How tall a repository lane node is, now that it carries the operations of the stage it owns.
 *
 * The graph showed which stage the work was in and nothing about the work itself: clone,
 * install, the coding call, lint, tests and build lived one drawer away, so the surface a
 * person actually watches said the least. Each lane node now lists its own stage's
 * operations, which costs height -- `MAX_NODE_OPS` lines of 11px text plus the two the node
 * already had.
 *
 * Fixed rather than measured, deliberately. A per-node height would leave the lanes ragged
 * and every arrow arriving at a different altitude; one height keeps the rows aligned and the
 * stride arithmetic single-valued, and the overflow line says what did not fit.
 */
const LANE_NODE_HEIGHT = 152;
/** The kinds that always carry operations, and therefore the taller box. */
const LANE_KINDS = new Set<GraphNode['kind']>(['implement', 'validate', 'review']);

/**
 * A lane node is always tall. Any other node -- the planning stages, whose journaled calls
 * arrive with the executions read -- takes the tall box only while it has lines to show, so
 * a queued feature's planning row is not a strip of empty boxes.
 */
function nodeHeight(node: GraphNode, operations?: ReadonlyMap<string, NodeOpLine[]>): number {
  if (LANE_KINDS.has(node.kind)) return LANE_NODE_HEIGHT;
  return (operations?.get(node.id)?.length ?? 0) > 0 ? LANE_NODE_HEIGHT : NODE_HEIGHT;
}
// Wide enough for an edge label to sit between two columns without touching either node. This
// is the whole reason the graph is wider than it used to be: an arrow that names its model has
// to have somewhere to write it, and writing it on top of a node is not somewhere.
const COLUMN_GAP = 112;
// Tall enough for a retry loop, its label, and clear air before the next lane. A graph whose
// loop label overlapped the next repository read as a line through it.
const ROW_GAP = 84;
const PADDING = 16;

const COLUMN_STRIDE = NODE_WIDTH + COLUMN_GAP;
// From the tallest node a row can hold, not from the shortest. Row 0 carries the parent
// stages AND the first repository lane, so a stride sized on the 78px stage node would let a
// lane node's operations run into the lane below it.
const ROW_STRIDE = LANE_NODE_HEIGHT + ROW_GAP;

const LABEL_WIDTH = COLUMN_GAP - 12;
// Three rows of 11px text plus padding and a border. Kept as a constant because the row gap
// below is chosen to clear it: a loop label taller than this overlapped the next lane.
const LABEL_MAX_HEIGHT = 50;
const RETRY_LABEL_WIDTH = 176;
// Labels in the same gap are nudged apart when they would otherwise sit on each other. A
// dense graph must stay readable without any execution information being dropped.
// Enough for the tallest label -- a wrapped handler name plus a status row -- so nudging one
// clear of another cannot leave them touching.
const LABEL_MIN_GAP = LABEL_MAX_HEIGHT + 14;

/**
 * How far an agent envelope sits outside the nodes it contains.
 *
 * Node positions do not move -- the envelope is a pure additive background layer -- so the
 * inset has to come out of space the canvas already owns. `PADDING` is 16, so 8 fits in every
 * direction: the top lane's box has 16px of margin above it, the first column's has 16px to
 * its left, and the canvas dimensions are unchanged. A box that quietly clipped at the canvas
 * edge is the failure mode this number is chosen against.
 *
 * The agent's name therefore does NOT ride the box's top edge. A chip there needs about 20px
 * above the node row, which the top lane does not have, and growing the canvas upwards would
 * move every node. It rides the column gap to the left of the box instead -- the same
 * positioned-DOM idiom the edge labels use, in the 112px of gap the entry arrow leaves.
 */
const ENVELOPE_INSET = 8;
const ENVELOPE_LABEL_WIDTH = 86;

/**
 * Which stage each lane node stands for, so ownership comes from one vocabulary.
 *
 * `repository` maps to nothing on purpose. It is the lane's header rather than a stage the
 * Engineer runs, and since `AGENT_FOR_STAGE` also owns `setup: 'Engineer'`, mapping it there
 * would silently stretch the box across three nodes. `stage`, `integration` and
 * `pull-requests` map to nothing for the same reason.
 */
const STAGE_FOR_KIND: Partial<Record<GraphNode['kind'], string>> = {
  implement: 'coding',
  validate: 'validation',
  review: 'review',
};

interface Envelope {
  key: string;
  /** The agent that owns every node inside it. */
  agent: string;
  repositoryId: string;
  left: number;
  top: number;
  width: number;
  height: number;
  /** The lane node the strip and the box open, which is the lane's Implementation node. */
  opens: GraphNode;
}

/**
 * One labelled container per contiguous run of same-agent lane nodes.
 *
 * For today's shape that is one box per lane, around Implementation and Validation, labelled
 * Engineer. The grouping the todo asked for -- "group together the things a single agent
 * does" -- shipped in the drawer's stage headers and never in the graph, where the two nodes
 * render as free-floating siblings with nothing saying one agent owns both.
 *
 * A run of one node draws nothing: a box around a single Review node is noise, and the drawer
 * already names single-stage ownership.
 */
function envelopes(graph: FeatureGraph): Envelope[] {
  const lanes = new Map<string, GraphNode[]>();
  for (const node of graph.nodes) {
    const stage = STAGE_FOR_KIND[node.kind];
    if (!node.repositoryId || !stage || !AGENT_FOR_STAGE[stage]) continue;
    const existing = lanes.get(node.repositoryId);
    if (existing) existing.push(node);
    else lanes.set(node.repositoryId, [node]);
  }
  const found: Envelope[] = [];
  for (const [repositoryId, nodes] of lanes) {
    const ordered = [...nodes].sort((left, right) => left.column - right.column);
    let run: GraphNode[] = [];
    const flush = () => {
      // Two or more, and every member owned by the same agent as the first.
      if (run.length > 1) found.push(envelope(repositoryId, run));
      run = [];
    };
    for (const node of ordered) {
      const agent = AGENT_FOR_STAGE[STAGE_FOR_KIND[node.kind]!]!;
      const previous = run.at(-1);
      const contiguous =
        previous !== undefined &&
        previous.row === node.row &&
        previous.column + 1 === node.column &&
        AGENT_FOR_STAGE[STAGE_FOR_KIND[previous.kind]!] === agent;
      if (previous !== undefined && !contiguous) flush();
      run.push(node);
    }
    flush();
  }
  return found;
}

function envelope(repositoryId: string, run: GraphNode[]): Envelope {
  const first = position(run[0]!);
  const last = position(run.at(-1)!);
  return {
    key: `envelope:${repositoryId}:${run[0]!.column}`,
    agent: AGENT_FOR_STAGE[STAGE_FOR_KIND[run[0]!.kind]!]!,
    repositoryId,
    left: first.x - ENVELOPE_INSET,
    top: first.y - ENVELOPE_INSET,
    width: last.x + NODE_WIDTH - first.x + ENVELOPE_INSET * 2,
    height:
      Math.max(...run.map((node) => position(node).y + nodeHeight(node))) -
      first.y +
      ENVELOPE_INSET * 2,
    // The lane's first member, which for today's shape is Implementation: the drill-in it
    // opens is per-repository, so any lane node reaches the same rows.
    opens: run[0]!,
  };
}

/**
 * How far below the lowest lane a remediation loop runs.
 *
 * Deeper than a retry loop's own dip, because it is a different edge: a retry loops under one
 * lane, while a remediation runs from the integration review back across every lane between
 * them. Drawing it at the retry depth would put it through the lanes it passes under.
 */
const REMEDIATION_DIP = ROW_GAP * 0.75;

function position(node: GraphNode) {
  return {
    x: PADDING + node.column * COLUMN_STRIDE,
    y: PADDING + node.row * ROW_STRIDE,
  };
}

export function WorkflowGraph({
  graph,
  label,
  selectedEdgeId,
  onSelectEdge,
  selectedNodeId,
  onSelectNode,
  operations,
  liveness,
}: {
  graph: FeatureGraph;
  label: string;
  /** The arrow whose execution is open, so the graph shows where the drawer came from. */
  selectedEdgeId?: string | null;
  onSelectEdge?: (edge: GraphEdge) => void;
  /** The lane node whose agent-work drill-in is open, so the graph shows where it came from. */
  selectedNodeId?: string | null;
  /**
   * Opens a repository lane node's drill-in: the journal's operations for that repository,
   * grouped by agent and stage. When provided, a node with a repository opens this instead of
   * navigating -- the drawer links onward to everything the navigation reached, so nothing is
   * lost. Nodes outside a lane keep navigating: the journal is per-repository and they have
   * no rows to show. The geometry is untouched either way; the drill-in is a drawer beside
   * the page, not a re-layout.
   */
  onSelectNode?: (node: GraphNode) => void;
  /**
   * The operations each node lists, keyed by the node that owns them: lane nodes carry the
   * newest attempt's journal rows from the same polled response the liveness overlay and the
   * drawer read, and the planning stage nodes carry the journaled pre-coding calls from the
   * executions read the edge chips already show. One data source per surface pair: these
   * lines and the drawer's stage sections -- or the arrow beside a planning node -- are the
   * same records, never derived twice. Absent where a read has not arrived, and a node then
   * renders as it did before.
   */
  operations?: ReadonlyMap<string, NodeOpLine[]>;
  /**
   * The journal's truthful overlay, keyed by node id: which operation is running right now
   * and whether its heartbeat is fresh. Drawn on the node of the stage the journal names,
   * even when the node states have not caught up — the node states update at attempt
   * boundaries and are deliberately not reworked here.
   */
  liveness?: ReadonlyMap<string, NodeLiveness>;
}) {
  const navigate = useNavigate();
  const boxes = useMemo(() => envelopes(graph), [graph]);
  const placed = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, { node, ...position(node) }])),
    [graph],
  );
  const drawn = useMemo(() => layout(graph, placed, operations), [graph, placed, operations]);

  const width = PADDING * 2 + (graph.columns - 1) * COLUMN_STRIDE + NODE_WIDTH;
  // The last lane's retry loop dips below it, so the canvas has to own that space too. A
  // remediation loop dips deeper -- it runs under every lane between the integration review
  // and the one it points at -- so it owns the height it needs rather than borrowing the
  // retry reservation and clipping when it is the only loop there is.
  const loops = graph.edges.some((edge) => edge.kind === 'retry');
  const remediations = graph.edges.some((edge) => edge.kind === 'remediation');
  const height =
    PADDING * 2 +
    (graph.rows - 1) * ROW_STRIDE +
    LANE_NODE_HEIGHT +
    Math.max(loops ? ROW_GAP * 0.9 : 0, remediations ? REMEDIATION_DIP + LABEL_MAX_HEIGHT : 0);

  return (
    <div className="graph" role="group" aria-label={label}>
      <div className="graph__canvas" style={{ width, height }}>
        {/* The agent envelopes: one background layer, drawn before the edges and therefore
            before the nodes and the edge-label buttons, so nothing it contains is covered by
            it. `aria-hidden` for the same reason the edge canvas is -- the announced text is
            the positioned label below, which is real DOM. */}
        <svg className="graph__envelopes" width={width} height={height} aria-hidden="true">
          {boxes.map((box) => (
            <rect
              key={box.key}
              className="graph__envelope"
              x={box.left}
              y={box.top}
              width={box.width}
              height={box.height}
              rx={12}
            />
          ))}
        </svg>
        <svg className="graph__edges" width={width} height={height} aria-hidden="true">
          <defs>
            <marker
              id="graph-arrow"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L8 4 L0 8 z" fill="currentColor" />
            </marker>
          </defs>
          {drawn.map((item) => (
            <Edge
              key={item.edge.id}
              drawn={item}
              selected={item.edge.id === selectedEdgeId}
              onOpen={
                onSelectEdge && item.edge.executions.length > 0
                  ? () => onSelectEdge(item.edge)
                  : undefined
              }
            />
          ))}
        </svg>

        {/* The agent's name. Real DOM over the canvas, the same idiom the edge labels use: an
            aria label inside the background SVG above would be dead air, because that layer
            is hidden. The sub-stage strip that used to sit under each box is gone -- the
            operations it summarised are now inside the nodes that ran them, and two
            renderings of one lifecycle is the duplication this layer was meant to avoid. */}
        {boxes.map((box) => (
          <EnvelopeLabel key={`envelope-label-${box.key}`} box={box} />
        ))}

        {drawn.map((item) =>
          item.label && item.edge.showLabel ? (
            <EdgeLabelButton
              key={`label-${item.edge.id}`}
              drawn={item}
              label={item.label}
              selected={item.edge.id === selectedEdgeId}
              onOpen={
                onSelectEdge && item.edge.executions.length > 0
                  ? () => onSelectEdge(item.edge)
                  : undefined
              }
            />
          ) : null,
        )}

        {graph.nodes.map((node) => {
          const { x, y } = position(node);
          return (
            <Node
              key={node.id}
              node={node}
              live={liveness?.get(node.id) ?? null}
              opensDrawer={Boolean(onSelectNode && node.repositoryId)}
              selected={node.id === selectedNodeId}
              operations={operations?.get(node.id) ?? null}
              style={{ left: x, top: y, width: NODE_WIDTH, height: nodeHeight(node, operations) }}
              onOpen={() => {
                if (onSelectNode && node.repositoryId) onSelectNode(node);
                else if (node.href) navigate(node.href);
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

/**
 * One envelope's name, and the strip of sub-stages inside it.
 *
 * The name sits in the column gap to the left of the box rather than on its top edge, which
 * `ENVELOPE_INSET` explains: the top lane has only the canvas padding above it. The strip sits
 * along the bottom inside the box, where the row gap gives it room without touching the retry
 * loop that lives below the lane.
 *
 * The strip is annotation, never nodes. The envelope's two full nodes stay the only nodes, so
 * a two-lane graph gains no width from it. It sits immediately under the box rather than
 * inside it: `ENVELOPE_INSET` is 8, which is not room for a row of text, and the alternative
 * would be to move the nodes -- which this whole layer exists not to do. Both it and the name
 * are drawn before the edge-label buttons, so a label that ever landed on either still wins.
 */
function EnvelopeLabel({ box }: { box: Envelope }) {
  const spoken = `${box.agent}, ${box.repositoryId}`;
  return (
    <span
      className="graph__envelope-label"
      style={{
        left: Math.max(0, box.left - ENVELOPE_LABEL_WIDTH - 6),
        top: box.top,
        width: ENVELOPE_LABEL_WIDTH,
      }}
      role="img"
      aria-label={spoken}
      title={spoken}
    >
      {box.agent}
    </span>
  );
}

type Placed = Map<string, { node: GraphNode; x: number; y: number }>;

interface DrawnEdge {
  edge: GraphEdge;
  path: string;
  /** Where the label sits, already nudged clear of its neighbours. */
  anchor: { x: number; y: number } | null;
  label: EdgeLabel | null;
  tone: string;
}

/**
 * Turn every edge into a path, a label and somewhere to put it.
 *
 * Done in one pass so labels can be de-collided against each other: two arrows leaving the
 * same column can land on the same point, and a graph with five repositories used to stack
 * three model names on top of one another. Nudging is preferred to hiding -- the requirement
 * is that no execution information disappears because the graph got busy.
 */
function layout(
  graph: FeatureGraph,
  placed: Placed,
  operations?: ReadonlyMap<string, NodeOpLine[]>,
): DrawnEdge[] {
  const drawn: DrawnEdge[] = [];
  for (const edge of graph.edges) {
    const from = placed.get(edge.from);
    const to = placed.get(edge.to);
    if (!from || !to) continue;
    const label = edgeLabel(edge.executions) ?? fallbackEdgeLabel(edge.label);
    const geometry = geometryFor(edge, from, to, operations);
    drawn.push({
      edge,
      path: geometry.path,
      anchor: geometry.anchor,
      label,
      tone: label?.record ? edgeTone(label.record) : 'pending',
    });
  }

  // Nudge overlapping labels apart, column band by column band, top to bottom.
  const bands = new Map<number, DrawnEdge[]>();
  for (const item of drawn) {
    if (!item.label || !item.anchor || !item.edge.showLabel) continue;
    const band = Math.round(item.anchor.x / 8);
    const existing = bands.get(band);
    if (existing) existing.push(item);
    else bands.set(band, [item]);
  }
  for (const group of bands.values()) {
    group.sort((left, right) => (left.anchor?.y ?? 0) - (right.anchor?.y ?? 0));
    let lowest = -Infinity;
    for (const item of group) {
      if (!item.anchor) continue;
      const y = Math.max(item.anchor.y, lowest + LABEL_MIN_GAP);
      item.anchor = { x: item.anchor.x, y };
      lowest = y;
    }
  }
  return drawn;
}

function geometryFor(
  edge: GraphEdge,
  from: { node: GraphNode; x: number; y: number },
  to: { node: GraphNode; x: number; y: number },
  operations?: ReadonlyMap<string, NodeOpLine[]>,
): { path: string; anchor: { x: number; y: number } | null } {
  // Read once per end, with the same operations the boxes render: a node that carries its
  // calls is taller, so an arrow between two heights meets each box at its own middle rather
  // than at a shared constant -- which is what made every edge look slightly detached.
  const fromHeight = nodeHeight(from.node, operations);
  const toHeight = nodeHeight(to.node, operations);
  if (edge.kind === 'remediation') {
    // A back edge across columns AND rows: the integration review sits to the right of every
    // lane and on the middle row, and the lane it sends work back to is neither. Its own
    // branch rather than the retry one, which computes its dip from `from.y` alone and would
    // draw a line straight through the lanes between them.
    //
    // Under the lowest of the two rows, so it never crosses a lane it does not belong to.
    const startX = from.x + NODE_WIDTH / 2;
    const endX = to.x + NODE_WIDTH / 2;
    const floor = Math.max(from.y + fromHeight, to.y + toHeight);
    const dip = floor + REMEDIATION_DIP;
    return {
      path: `M ${startX} ${from.y + fromHeight} C ${startX} ${dip}, ${endX} ${dip}, ${endX} ${to.y + toHeight}`,
      anchor: { x: (startX + endX) / 2, y: dip + 4 },
    };
  }

  if (edge.kind === 'retry') {
    // Backwards, so it is drawn under its own lane rather than through it: a loop that
    // overlapped the forward arrow read as a line, which is the one thing it must not.
    const startX = to.x + NODE_WIDTH / 2;
    const endX = from.x + NODE_WIDTH / 2;
    const dip = Math.max(from.y + fromHeight, to.y + toHeight) + ROW_GAP * 0.55;
    return {
      path: `M ${endX} ${from.y + fromHeight} C ${endX} ${dip}, ${startX} ${dip}, ${startX} ${to.y + toHeight}`,
      anchor: { x: (startX + endX) / 2, y: dip + 4 },
    };
  }

  if (from.node.column === to.node.column) {
    // A dependency between two lanes: same column, different rows. Drawn down the left of
    // the column rather than through it, so it reads as one lane waiting for another, and
    // fanned by distance so several dependents of the same repository stay distinguishable.
    const rows = Math.abs(to.node.row - from.node.row);
    const x = from.x - COLUMN_GAP / 3 - Math.min(rows, 4) * 4;
    // Each end at its own node's middle, because the two boxes are no longer the same height:
    // a lane node carries its operations and a stage node does not.
    const startY = from.y + fromHeight / 2;
    const endY = to.y + toHeight / 2;
    return {
      path: `M ${from.x} ${startY} L ${x} ${startY} L ${x} ${endY} L ${to.x} ${endY}`,
      anchor: null,
    };
  }

  const startX = from.x + NODE_WIDTH;
  const startY = from.y + fromHeight / 2;
  const endX = to.x;
  const endY = to.y + toHeight / 2;
  const midX = startX + (endX - startX) / 2;
  return {
    path: `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`,
    anchor: { x: midX, y: (startY + endY) / 2 },
  };
}

function Edge({
  drawn,
  selected,
  onOpen,
}: {
  drawn: DrawnEdge;
  selected: boolean;
  onOpen?: () => void;
}) {
  const { edge, path, tone, label } = drawn;
  const classes = [
    'graph__edge',
    `graph__edge--${edge.kind}`,
    `graph__edge--${tone}`,
    // One treatment, two reasons to have it. `--active` is the solid, thicker, toned edge a
    // retry loop already got while a lane was going round it; `--live` generalises it to the
    // flow edge, which never received it, and adds the motion on top.
    edge.active || edge.live ? 'graph__edge--active' : '',
    edge.live ? 'graph__edge--live' : '',
    selected ? 'graph__edge--selected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <g className={classes}>
      <path d={path} fill="none" markerEnd="url(#graph-arrow)" />
      {/* A wide transparent stroke over the same path, so the arrow itself is clickable and
          not only its label. Keyboard users reach the label button instead; this is a mouse
          convenience and is hidden from assistive technology with the rest of the canvas. */}
      {label && onOpen ? (
        <path className="graph__edge-hit" d={path} fill="none" onClick={onOpen} />
      ) : null}
    </g>
  );
}

function EdgeLabelButton({
  drawn,
  label,
  selected,
  onOpen,
}: {
  drawn: DrawnEdge;
  label: EdgeLabel;
  selected: boolean;
  onOpen?: () => void;
}) {
  if (!drawn.anchor) return null;
  const { x, y } = drawn.anchor;
  // A loop label sits in the row gap under its lane, which spans several columns, so it is not
  // squeezed into the width a forward label between two nodes has to fit.
  const isLoop = drawn.edge.kind === 'retry' || drawn.edge.kind === 'remediation';
  const width = isLoop ? RETRY_LABEL_WIDTH : LABEL_WIDTH;
  const classes = [
    'graph__edge-label',
    `graph__edge-label--${drawn.tone}`,
    isLoop ? 'graph__edge-label--retry' : '',
    label.record ? `graph__edge-label--${label.record.handler_type}` : '',
    selected ? 'graph__edge-label--selected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const spoken = label.description;
  return (
    <button
      type="button"
      className={classes}
      style={{ left: x - width / 2, top: y, width }}
      onClick={onOpen}
      disabled={!onOpen}
      aria-label={spoken}
      title={spoken}
    >
      {label.lines.map((line, index) => (
        <span key={line.text} className={`graph__edge-line graph__edge-line--${line.kind}`}>
          {/* Shape as well as colour: a retry arrow carries a loop mark beside its attempt
              count, so the two kinds of arrow are distinguishable in greyscale and to anyone
              who cannot use the colour at all. Inline rather than on its own row, because a
              row of its own made a three-line loop label tall enough to reach the next lane. */}
          {index === 0 && isLoop ? (
            <span className="graph__edge-mark" aria-hidden="true">
              ↺{' '}
            </span>
          ) : null}
          {line.text}
          {/* The effort, as its own token: it may fall to the next line, and the model
              identifier above it may not. Named "effort:" because a bare word beside a model
              name reads as a performance tier -- the tier is on the feature header, once. */}
          {line.effort ? <span className="graph__edge-effort">effort: {line.effort}</span> : null}
        </span>
      ))}
    </button>
  );
}

const KIND_LABELS: Record<GraphNode['kind'], string> = {
  stage: 'Stage',
  repository: 'Repository',
  implement: 'Implementation',
  validate: 'Validation',
  review: 'Review',
  integration: 'Integration review',
  'pull-requests': 'Pull requests',
};

const STATE_WORDS: Record<GraphNode['state'], string> = {
  done: 'done',
  active: 'in progress',
  pending: 'not started',
  stopped: 'stopped',
  attention: 'needs attention',
};

function Node({
  node,
  live,
  opensDrawer,
  selected,
  operations,
  style,
  onOpen,
}: {
  node: GraphNode;
  live: NodeLiveness | null;
  /** Whether opening this node shows the agent-work drawer rather than navigating. */
  opensDrawer: boolean;
  selected: boolean;
  /**
   * The operations this node owns — a lane node's stage rows, or a planning node's journaled
   * calls. Absent until the backing read arrives, and the node then renders as it did before.
   */
  operations: NodeOpLine[] | null;
  style: React.CSSProperties;
  onOpen: () => void;
}) {
  const classes = [
    'graph__node',
    `graph__node--${node.state}`,
    `graph__node--${node.kind}`,
    node.retrying ? 'graph__node--retrying' : '',
    selected ? 'graph__node--selected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  // A node is at most three lines. The second says which repository it belongs to, or -- on
  // the nodes that are not inside a lane -- what it is about. The third carries how it is
  // going, and is where the retry state and its count share a line.
  const inLane = Boolean(node.repositoryId) && node.kind !== 'repository';
  const second = inLane ? node.repositoryId : node.detail;
  const third = [
    node.retrying ? 'Retrying' : '',
    node.counter
      ? `${node.counter.label} ${node.counter.current}/${node.counter.limit}`
      : inLane
        ? (node.detail ?? '')
        : '',
  ]
    .filter(Boolean)
    .join(' · ');

  // A fixed box holds a fixed number of lines. Where an attempt ran more operations than
  // that, the newest are the ones worth seeing -- an operator watches what is happening now --
  // so the tail is kept and the remainder is counted rather than dropped.
  const lines = operations ?? [];
  const hidden = Math.max(0, lines.length - MAX_NODE_OPS);
  const shown = hidden > 0 ? lines.slice(lines.length - (MAX_NODE_OPS - 1)) : lines;

  // The accessible name carries everything the shape and colour carry, in order, so the
  // graph is readable without seeing it -- including every operation the box lists, each
  // with the word its glyph stands for.
  const spoken = [
    KIND_LABELS[node.kind],
    node.label,
    node.repositoryId && node.kind !== 'repository' ? node.repositoryId : '',
    STATE_WORDS[node.state],
    node.retrying ? 'retrying' : '',
    node.counter ? `${node.counter.label} ${node.counter.current} of ${node.counter.limit}` : '',
    node.detail ?? '',
    hidden > 0 ? `${hidden} earlier operations not shown` : '',
    ...shown.map((line) => `${line.name} ${line.word}`),
  ]
    .filter(Boolean)
    .join(', ');

  return (
    <button
      type="button"
      className={classes}
      style={style}
      onClick={onOpen}
      aria-label={spoken}
      // The drill-in is a dialog beside the page, and the button says so the same way the
      // drawer's own opener idiom does; a navigating node stays a plain button.
      aria-haspopup={opensDrawer ? 'dialog' : undefined}
      aria-expanded={opensDrawer ? selected : undefined}
      // A node is a fixed box, so a long status is truncated in it. The whole line stays
      // reachable without opening anything.
      title={spoken}
    >
      <span className="graph__node-head">
        <span className="graph__marker" aria-hidden="true" />
        <span className="graph__node-label">{node.label}</span>
      </span>
      {/* The journal's truthful overlay: a live dot for a fresh heartbeat, a warning for a
          quiet one, with the operation's own facts in the tooltip. It sits on the node the
          journal names -- run 194's reviewer was running while the Review node sat grey,
          and this is the element that tells that truth without reworking node states. */}
      {live ? (
        <span
          className={`graph__liveness graph__liveness--${live.state}`}
          title={`${live.summary} · ${live.detail}`}
          aria-label={`${live.summary} · ${live.detail}`}
          role="img"
        >
          <span className="graph__liveness-dot" aria-hidden="true" />
        </span>
      ) : null}
      {second ? (
        <span className={inLane ? 'graph__node-repository' : 'graph__node-detail'}>{second}</span>
      ) : null}
      {/* Three lines before the operations: what it is, which repository, and how it is
          going. The retry state shares the last line with its count rather than pushing it
          out. */}
      {third ? (
        <span
          className={
            node.retrying ? 'graph__node-counter graph__node-counter--retry' : 'graph__node-counter'
          }
        >
          {third}
        </span>
      ) : null}
      {/* What the agent actually did, on the node whose stage did it. The count line is the
          honest end of a fixed box: an attempt with more operations than fit says how many
          it is not showing rather than silently dropping the newest ones. */}
      {shown.length > 0 ? (
        <span className="graph__node-ops">
          {hidden > 0 ? (
            <span className="graph__node-op graph__node-op--more">
              +{hidden} earlier {hidden === 1 ? 'operation' : 'operations'}
            </span>
          ) : null}
          {shown.map((line) => (
            <span key={line.key} className={`graph__node-op graph__node-op--${line.state}`}>
              <span aria-hidden="true">{STAGE_GLYPHS[line.state]}</span>
              <span className="graph__node-op-name">{line.name}</span>
              {line.meta ? <span className="graph__node-op-meta">{line.meta}</span> : null}
            </span>
          ))}
        </span>
      ) : null}
    </button>
  );
}

/** The key, so the colours are not the only thing saying what a state is. */
export function GraphLegend() {
  const states: GraphNode['state'][] = ['done', 'active', 'attention', 'stopped', 'pending'];
  return (
    <ul className="graph__legend">
      {states.map((state) => (
        <li key={state} className={`graph__legend-item graph__legend-item--${state}`}>
          <span className="graph__marker" aria-hidden="true" />
          {STATE_WORDS[state]}
        </li>
      ))}
    </ul>
  );
}
