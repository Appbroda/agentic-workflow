// Click the retry grant and the chat composer, capturing what they send.
//
// `fetch` is replaced for the mutating call only, so the request the client builds -- path,
// body, headers -- is inspected exactly as it would leave the browser, and the platform's
// refusal is fed back to check the client renders it. The server half of both paths is already
// covered by its own tests and by curl against the running API; what has never been exercised
// is a person pressing the button.
import { connect, SET_VALUE } from './drive.mjs';

const UI = process.env.UI ?? 'http://localhost:8000/ui';
const KEY = process.env.PLATFORM_API_KEY;
const FEATURE = process.env.FEATURE ?? 'admin-server-status-live-083';

const INTERCEPT = `
  window.__captured = [];
  window.__realFetch = window.__realFetch ?? window.fetch;
  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    const method = (init.method ?? 'GET').toUpperCase();
    if (method === 'POST' && (url.includes('/retry') || url.includes('/chat'))) {
      window.__captured.push({
        url,
        method,
        body: init.body ? JSON.parse(init.body) : null,
        headers: Object.fromEntries(Object.entries(init.headers ?? {})),
      });
      return new Response(
        JSON.stringify({ detail: 'the platform refused this for the sake of the test' }),
        { status: 409, headers: { 'content-type': 'application/json' } },
      );
    }
    return window.__realFetch(input, init);
  };
  true;
`;

const page = await connect();
try {
  await page.goto(`${UI}/`);
  await page.evaluate(SET_VALUE);
  if (await page.evaluate(`document.body.innerText.includes('Platform access')`)) {
    await page.evaluate(`__set('input[type=password]', ${JSON.stringify(KEY)})`);
    await page.evaluate(`__click('Continue')`);
    await page.until(`!document.body.innerText.includes('Platform access')`, 'the gate');
  }

  // --- Retry grant -------------------------------------------------------------------------
  await page.goto(`${UI}/features/${FEATURE}/repositories`);
  await page.evaluate(SET_VALUE);
  await page.evaluate(INTERCEPT);
  // The retry control lives with the repository it acts on, one expansion from its row.
  await page.until(`__click && document.body.innerText.includes('Repository workstreams')`, 'the repository table');
  await page.evaluate(`
    [...document.querySelectorAll('table[aria-label="Repository workstreams"] tbody button')]
      .forEach((button) => button.click());
    true;
  `);
  await page.until(`document.body.innerText.includes('Grant another attempt')`, 'the retry control');

  await page.evaluate(`__click('Grant another attempt')`);
  await page.until(`document.body.innerText.includes('Grant and run')`, 'the grant form');

  const disabledEmpty = await page.evaluate(`
    [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Grant and run').disabled
  `);
  console.log('retry : refuses to send with no author or reason =', disabledEmpty);

  await page.evaluate(`
    (() => {
      const byLabel = (t) => [...document.querySelectorAll('label')]
        .filter((l) => l.textContent.trim().startsWith(t))
        .map((l) => document.getElementById(l.getAttribute('for')))
        .filter(Boolean)[0];
      __set('#' + CSS.escape(byLabel('Your name').id), 'akhilesh');
      __set('#' + CSS.escape(byLabel('What changed').id), 'Repaired the runner out of band.');
      return true;
    })()
  `);
  const enabled = await page.evaluate(`
    ![...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Grant and run').disabled
  `);
  console.log('retry : enabled once both are given =', enabled);

  await page.evaluate(`__click('Grant and run')`);
  await page.until(`window.__captured.length > 0`, 'the grant request');
  const grant = await page.evaluate(`JSON.stringify(window.__captured[0])`);
  console.log('retry : sent', grant.slice(0, 230));

  await page.until(`document.querySelector('[role=alert]')`, 'the refusal to be shown');
  const shown = await page.evaluate(`document.querySelector('[role=alert]').innerText`);
  console.log('retry : showed refusal =', JSON.stringify(shown.slice(0, 120)));

  // --- Chat --------------------------------------------------------------------------------
  await page.goto(`${UI}/features/${FEATURE}/chat`);
  await page.evaluate(SET_VALUE);
  await page.evaluate(INTERCEPT);
  await page.until(`document.body.innerText.includes('Ask about this feature')`, 'the composer');

  await page.evaluate(`
    (() => {
      const label = [...document.querySelectorAll('label')]
        .find((l) => l.textContent.trim() === 'Ask about this feature');
      __set('#' + CSS.escape(label.getAttribute('for')), 'Which repositories stopped, and why?');
      const key = [...document.querySelectorAll('label')]
        .find((l) => l.textContent.trim() === 'OpenAI API key');
      if (key) __set('#' + CSS.escape(key.getAttribute('for')), 'sk-typed-by-a-person');
      return true;
    })()
  `);
  await page.evaluate(`__click('Send')`);
  await page.until(`window.__captured.length > 0`, 'the chat request');
  const chat = await page.evaluate(`JSON.stringify(window.__captured[0])`);
  console.log('chat  : sent', chat.slice(0, 260));
  console.log('chat  : key travels as a header, not in the body =',
    await page.evaluate(`
      Boolean(window.__captured[0].headers['X-OpenAI-Api-Key']) &&
      !JSON.stringify(window.__captured[0].body).includes('sk-typed-by-a-person')
    `));
} finally {
  page.close();
}
