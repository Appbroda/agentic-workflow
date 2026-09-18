// Click Resume and Cancel, capturing what each sends and what the refusal looks like.
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
    if (method === 'POST' && (url.includes('/resume') || url.includes('/cancel'))) {
      window.__captured.push({ url, body: init.body ? JSON.parse(init.body) : null });
      return new Response(JSON.stringify({ detail: 'refused for the sake of the test' }),
        { status: 409, headers: { 'content-type': 'application/json' } });
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

  for (const [action, button, dialog] of [
    ['cancel', 'Cancel feature', 'Cancel this feature?'],
    ['resume', 'Resume', 'Resume this feature?'],
  ]) {
    await page.goto(`${UI}/features/${FEATURE}`);
    await page.evaluate(SET_VALUE);
    await page.evaluate(INTERCEPT);
    await page.until(`document.body.innerText.includes(${JSON.stringify(button)})`, button);

    await page.evaluate(`__click(${JSON.stringify(button)})`);
    await page.until(`document.querySelector('[role=dialog]')`, 'the confirmation');
    const name = await page.evaluate(`document.querySelector('[role=dialog]').getAttribute('aria-label')`);
    const sentEarly = await page.evaluate(`window.__captured.length`);
    console.log(`${action}: confirms first (dialog "${name}"), nothing sent yet = ${sentEarly === 0}`);
    console.log(`${action}: focus is inside the dialog =`,
      await page.evaluate(`document.querySelector('[role=dialog]').contains(document.activeElement)`));

    await page.evaluate(`
      (() => {
        const dialog = document.querySelector('[role=dialog]');
        const go = [...dialog.querySelectorAll('button')]
          .find((b) => b.textContent.trim() === ${JSON.stringify(button)});
        go.click();
        return true;
      })()
    `);
    await page.until(`window.__captured.length > 0`, `the ${action} request`);
    console.log(`${action}: sent`, (await page.evaluate(`JSON.stringify(window.__captured[0])`)).slice(0, 150));
    await page.until(`document.querySelector('[role=alert]')`, 'the refusal');
    console.log(`${action}: showed`,
      JSON.stringify((await page.evaluate(`document.querySelector('[role=alert]').innerText`)).slice(0, 100)));
  }
} finally {
  page.close();
}
