import type { Clarification } from '@/schemas/feature';
import { Markdown } from '@/components/common/Markdown';

/**
 * What was already asked and answered about this feature.
 *
 * These answers are not incidental: they are where somebody told the platform which endpoint
 * to change, which status value to reuse, and which of two parallel module layouts is the live
 * one. The platform planned from them, so when the result is wrong this is the first thing to
 * re-read — and it was reachable only by querying the API by hand.
 *
 * It stays visible after the feature has moved on, and appears above the open questions while
 * answering, so round two is answered knowing what round one said.
 */
export function ClarificationHistory({ clarification }: { clarification: Clarification }) {
  const answers = Object.entries(clarification.previous_answers);
  if (answers.length === 0) return null;

  const summary = `${answers.length} ${answers.length === 1 ? 'answer' : 'answers'} over ${
    clarification.clarification_rounds
  } ${clarification.clarification_rounds === 1 ? 'round' : 'rounds'}`;

  return (
    <details
      className="artifact__section"
      // A run against two repositories collects a dozen of these, each several paragraphs of
      // repository detail. Open by default they are longer than the document they explain, so
      // they start closed once there are more than a couple -- and open when a person is
      // answering, which is exactly when round one matters.
      open={answers.length <= 2}
    >
      <summary className="muted">
        Answered already — {summary}. The platform planned from these.
      </summary>
      <dl className="answers" style={{ marginTop: 'var(--space-4)' }}>
        {answers.map(([questionId, answer]) => (
          <div key={questionId} className="answers__item">
            <dt>
              <code className="mono">{questionId}</code>
            </dt>
            <dd>
              {/* Answers are typed by a person and are markdown in practice -- paths, code
                  spans, several paragraphs. Rendered as elements, never as HTML. */}
              <Markdown>{String(answer)}</Markdown>
            </dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
