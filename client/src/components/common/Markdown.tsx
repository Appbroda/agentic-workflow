import { Fragment, type ReactNode } from 'react';
import { safeMarkdownHref } from '@/utils/url';

/**
 * Markdown rendered as React elements, never as HTML.
 *
 * Everything this displays is untrusted: PRDs a person pasted, prose a model wrote, text read
 * out of a repository. The usual approach -- a markdown library that emits an HTML string,
 * then a sanitiser, then `dangerouslySetInnerHTML` -- makes safety depend on the sanitiser
 * being configured correctly and staying that way. This never produces HTML at all: it parses
 * to React elements, so a `<script>` or an `onerror=` in the source has nowhere to become
 * markup and arrives on the page as the characters it is.
 *
 * The supported subset is what the platform's documents actually contain: headings, lists,
 * fenced and inline code, bold, italic, links and paragraphs. Anything unrecognised renders as
 * its own text rather than disappearing, because silently dropping part of a requirement is
 * worse than showing an unstyled line.
 *
 * Links get `rel="noopener noreferrer"` and are restricted to http(s): a `javascript:` URL in
 * agent output must not become a clickable script.
 */
export function Markdown({ children }: { children: string }) {
  return <div className="markdown">{renderBlocks(children)}</div>;
}

function renderBlocks(source: string): ReactNode[] {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let index = 0;
  let key = 0;

  while (index < lines.length) {
    const line = lines[index] ?? '';

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = /^```(\S*)\s*$/.exec(line);
    if (fence) {
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index] ?? '')) {
        body.push(lines[index] ?? '');
        index += 1;
      }
      // A file that ends mid-fence still shows its content rather than nothing.
      index += 1;
      blocks.push(
        <pre className="markdown__code" key={key++}>
          <code>{body.join('\n')}</code>
        </pre>,
      );
      continue;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      // Clamped to h3-h6: the page owns h1 and h2, and a document that starts at `#` must not
      // insert a second top-level heading into the outline a screen reader follows.
      const level = Math.min(6, heading[1]!.length + 2);
      const Tag = `h${level}` as 'h3' | 'h4' | 'h5' | 'h6';
      blocks.push(<Tag key={key++}>{renderInline(heading[2] ?? '')}</Tag>);
      index += 1;
      continue;
    }

    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items: string[] = [];
      while (index < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[index] ?? '')) {
        items.push((lines[index] ?? '').replace(/^\s*([-*+]|\d+[.)])\s+/, ''));
        index += 1;
      }
      const children = items.map((item, position) => (
        <li key={position}>{renderInline(item)}</li>
      ));
      blocks.push(
        ordered ? <ol key={key++}>{children}</ol> : <ul key={key++}>{children}</ul>,
      );
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoted: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index] ?? '')) {
        quoted.push((lines[index] ?? '').replace(/^>\s?/, ''));
        index += 1;
      }
      blocks.push(<blockquote key={key++}>{renderInline(quoted.join(' '))}</blockquote>);
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length &&
      (lines[index] ?? '').trim() &&
      !/^(#{1,6}\s|```|>\s?|\s*([-*+]|\d+[.)])\s)/.test(lines[index] ?? '')
    ) {
      paragraph.push(lines[index] ?? '');
      index += 1;
    }
    blocks.push(<p key={key++}>{renderInline(paragraph.join(' '))}</p>);
  }

  return blocks;
}

// Ordered so the first match wins on overlapping syntax: code before emphasis, because
// `` `**not bold**` `` is code containing asterisks.
const INLINE = /(`[^`]+`)|(\[[^\]]+\]\([^)\s]+\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(_[^_]+_)/;

function renderInline(source: string): ReactNode {
  const parts: ReactNode[] = [];
  let rest = source;
  let key = 0;

  while (rest) {
    const match = INLINE.exec(rest);
    if (!match || match.index === undefined) break;

    if (match.index > 0) parts.push(rest.slice(0, match.index));
    const token = match[0];

    if (token.startsWith('`')) {
      parts.push(<code key={key++}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith('[')) {
      const link = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(token);
      const href = link?.[2] ?? '';
      parts.push(
        safeMarkdownHref(href) ? (
          <a key={key++} href={href} target="_blank" rel="noopener noreferrer">
            {link?.[1]}
          </a>
        ) : (
          // A `javascript:` or `data:` URL is shown as text. Refusing to link it is the point;
          // hiding it would leave a person unable to see what the document actually said.
          <Fragment key={key++}>{token}</Fragment>
        ),
      );
    } else if (token.startsWith('**')) {
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else {
      parts.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }

    rest = rest.slice(match.index + token.length);
  }

  if (rest) parts.push(rest);
  return parts.length > 0 ? parts : source;
}
