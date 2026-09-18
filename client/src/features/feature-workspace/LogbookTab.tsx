import { Link } from 'react-router-dom';
import { Async, EmptyState, TableSkeleton } from '@/components/common/States';
import { Panel } from '@/components/ui/Layout';
import { AgentBadge, RepositoryBadge } from '@/components/ui/Badge';
import { absoluteTime, eventTime } from '@/utils/time';
import type { LogbookEntry } from '@/schemas/feature';
import { useLogbook } from './hooks';
import { RECORD_LABELS, groupByDay, recordHref, toneClass } from './logbook';

/**
 * The feature's run, as a conversation between the agents that ran it.
 *
 * Beside History rather than instead of it, because they answer different questions. History
 * is the record in order — every event, every artifact, filterable, for somebody debugging.
 * This is the story: what was asked for, what each agent did about it, what they said, and
 * where it ended. During the 185-194 cycle three separate "what is actually going on?"
 * questions were each answered by hand-translating journal rows into exactly these sentences.
 *
 * Nothing here writes one of them. Every sentence, attribution and quote arrives composed
 * from `GET /features/{id}/logbook`, which reads only durable records and makes no model
 * call — so the worst this tab can do is show a record faithfully. What it adds is the
 * reading: who spoke, when, about which repository, and the link to the record underneath.
 */
export function LogbookTab({ featureId, at }: { featureId: string; at: number | null }) {
  const logbook = useLogbook(featureId, at);

  return (
    <Panel
      title="Logbook"
      meta={
        logbook.data
          ? `${logbook.data.entries.length} entr${logbook.data.entries.length === 1 ? 'y' : 'ies'}`
          : undefined
      }
      flush
    >
      <Async query={logbook} skeleton={<TableSkeleton rows={8} label="Reading the run…" />}>
        {(data) =>
          data.entries.length === 0 ? (
            <EmptyState
              title="Nothing has happened yet"
              detail="This feature's story is written as its agents work. It will appear here as they do."
            />
          ) : (
            <div className="logbook">
              {groupByDay(data.entries).map((day) => (
                <section key={day.label} className="logbook__day">
                  <h3 className="logbook__date">{day.label}</h3>
                  <ol className="logbook__thread" aria-label={`Logbook, ${day.label}`}>
                    {day.entries.map((entry) => (
                      <Bubble key={entry.sequence} featureId={featureId} entry={entry} />
                    ))}
                  </ol>
                </section>
              ))}
              {/* The server pages by entry. A run long enough to be paged says so rather
                  than quietly ending, because a story that stops mid-sentence reads as a
                  finished one. */}
              {data.next_cursor !== null && data.next_cursor !== undefined ? (
                <p className="logbook__more subtle">
                  This run is longer than one page. The rest of it is in{' '}
                  <Link to={`/features/${encodeURIComponent(featureId)}/history`}>History</Link>.
                </p>
              ) : null}
            </div>
          )
        }
      </Async>
    </Panel>
  );
}

/**
 * One bubble: who acted, what they did, and the record that says so.
 *
 * The quote is set apart because it is not this platform's sentence — it is what the agent
 * itself wrote, clipped by the server at a whole word, with the rest behind the record link.
 * Presenting the two the same way would make the platform's summary and a model's prose
 * indistinguishable, which is the one thing a logbook must never do.
 */
function Bubble({ featureId, entry }: { featureId: string; entry: LogbookEntry }) {
  const href = recordHref(featureId, entry.record);
  return (
    <li className={`logbook__entry logbook__entry--${toneClass(entry.tone)}`}>
      <div className="logbook__meta">
        <AgentBadge agent={entry.agent} />
        {entry.repository_id ? <RepositoryBadge repositoryId={entry.repository_id} /> : null}
        <span className="logbook__time" title={absoluteTime(entry.timestamp)}>
          {eventTime(entry.timestamp)}
        </span>
      </div>
      <div className="logbook__bubble">
        <p className="logbook__text">{entry.text}</p>
        {entry.detail ? <p className="logbook__detail">{entry.detail}</p> : null}
        {entry.quote ? (
          <blockquote className="logbook__quote">
            {entry.quote}
            {entry.quote_source ? (
              <cite className="logbook__source" title={`Read from ${entry.quote_source}`}>
                {entry.quote_source}
              </cite>
            ) : null}
          </blockquote>
        ) : null}
        {href ? (
          <Link className="logbook__record subtle" to={href}>
            {RECORD_LABELS[entry.record.kind] ?? 'View record'}
          </Link>
        ) : (
          // Anchored but not yet linkable: a record kind the server added after this
          // shipped. The reference is still shown, because "which row is this?" is the
          // technical reader's whole question.
          <span className="logbook__record subtle mono">{entry.record.id}</span>
        )}
      </div>
    </li>
  );
}
