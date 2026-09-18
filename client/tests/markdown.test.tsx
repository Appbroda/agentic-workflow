import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Markdown } from '@/components/common/Markdown';

describe('markdown from untrusted sources', () => {
  it('never turns source into markup', () => {
    render(
      <Markdown>{'A PRD said <img src=x onerror="alert(1)"> and <script>go()</script>.'}</Markdown>,
    );

    // Parsed to React elements rather than to an HTML string, so there is no sanitiser to
    // misconfigure: the tags arrive as characters because they were never markup.
    expect(
      screen.getByText(/<img src=x onerror="alert\(1\)"> and <script>go\(\)<\/script>/),
    ).toBeInTheDocument();
    expect(document.querySelector('img')).toBeNull();
    expect(document.querySelector('script')).toBeNull();
  });

  it('refuses to link a javascript: URL but still shows it', () => {
    const { container } = render(
      <Markdown>{'Click [here](javascript:alert(1)) or [docs](https://example.com/d).'}</Markdown>,
    );

    const links = [...container.querySelectorAll('a')];
    // The real link works; the script URL is shown as text rather than hidden, because a
    // person reading a suspicious document needs to see what it actually said.
    expect(links.map((item) => item.getAttribute('href'))).toEqual(['https://example.com/d']);
    expect(links[0]?.rel).toContain('noopener');
    expect(screen.getByText(/\[here\]\(javascript:alert\(1\)\)/)).toBeInTheDocument();
  });

  it('does not treat protocol-relative or credential-bearing URLs as safe links', () => {
    const { container } = render(
      <Markdown>
        {'[lookalike](//evil.example/login) [credential](https://token@evil.example/) [local](/features/f-1)'}
      </Markdown>,
    );

    expect([...container.querySelectorAll('a')].map((item) => item.getAttribute('href'))).toEqual([
      '/features/f-1',
    ]);
  });

  it('renders the structure the platform documents actually use', () => {
    const { container } = render(
      <Markdown>
        {[
          '# Objective',
          '',
          'Deactivate an ad unit from the **console**, using `PATCH /ad-units/:id`.',
          '',
          '- One requirement',
          '- Another requirement',
          '',
          '1. First step',
          '2. Second step',
          '',
          '```ts',
          'const x = 1;',
          '```',
        ].join('\n')}
      </Markdown>,
    );

    // Headings start at h3: the page owns h1 and h2, and a document beginning at `#` must not
    // insert a second top-level heading into the outline.
    expect(container.querySelector('h3')?.textContent).toBe('Objective');
    expect(container.querySelectorAll('ul li')).toHaveLength(2);
    expect(container.querySelectorAll('ol li')).toHaveLength(2);
    expect(container.querySelector('pre code')?.textContent).toBe('const x = 1;');
    expect(container.querySelector('strong')?.textContent).toBe('console');
    expect(container.querySelector('code')?.textContent).toBeTruthy();
  });

  it('shows an unrecognised line rather than dropping it', () => {
    // Silently losing part of a requirement is worse than showing it unstyled.
    render(<Markdown>{'| a | table | we | do | not | style |'}</Markdown>);
    expect(screen.getByText(/table/)).toBeInTheDocument();
  });

  it('does not treat asterisks inside code as emphasis', () => {
    const { container } = render(<Markdown>{'Use `**literal**` here.'}</Markdown>);
    expect(container.querySelector('code')?.textContent).toBe('**literal**');
    expect(container.querySelector('strong')).toBeNull();
  });
});
