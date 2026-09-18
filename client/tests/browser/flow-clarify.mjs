// Answer a clarification through the UI, on a feature that is really waiting.
import { connect, SET_VALUE } from './drive.mjs';

const UI = process.env.UI ?? 'http://localhost:8000/ui';
const KEY = process.env.PLATFORM_API_KEY;
const FEATURE = process.env.FEATURE ?? 'clarify-mock';

const page = await connect();
try {
  await page.goto(`${UI}/`);
  await page.evaluate(SET_VALUE);
  if (await page.evaluate(`document.body.innerText.includes('Platform access')`)) {
    await page.evaluate(`__set('input[type=password]', ${JSON.stringify(KEY)})`);
    await page.evaluate(`__click('Continue')`);
    await page.until(`!document.body.innerText.includes('Platform access')`, 'the gate');
  }

  // Wait for the list, rather than racing it: the first check read an empty dashboard.
  await page.until(`document.querySelectorAll('li').length > 3`, 'the feature list');
  const onDashboard = await page.evaluate(`document.body.innerText.includes(${JSON.stringify(FEATURE)})`);
  console.log('dash  : the waiting feature is listed =', onDashboard);
  console.log('dash  : flagged as needing a person =',
    await page.evaluate(`
      (() => {
        const row = [...document.querySelectorAll('li')].find((n) => n.innerText.includes(${JSON.stringify(FEATURE)}));
        return Boolean(row && row.innerText.includes('Needs you'));
      })()
    `));

  await page.goto(`${UI}/features/${FEATURE}`);
  await page.evaluate(SET_VALUE);
  await page.until(`document.body.innerText.includes('waiting on you')`, 'the clarification panel');

  const questions = await page.evaluate(`document.querySelectorAll('textarea[id^="q-"]').length`);
  const submitDisabled = await page.evaluate(`
    [...document.querySelectorAll('button')].find((b) => /Submit answers|Answer/.test(b.textContent))?.disabled
  `);
  console.log('ask   : questions shown =', questions, '| submit blocked while incomplete =', submitDisabled);
  console.log('ask   : rationale shown with each =',
    await page.evaluate(`document.querySelectorAll('.field__hint').length >= ${questions}`));

  // Answer every one, the way a person would.
  await page.evaluate(`
    (() => {
      [...document.querySelectorAll('textarea[id^="q-"]')].forEach((node, i) => {
        const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
        set.call(node, 'Drop that criterion; it is checked by a person outside the platform. Answer ' + (i + 1) + '.');
        node.dispatchEvent(new Event('input', { bubbles: true }));
      });
      return true;
    })()
  `);
  const enabled = await page.evaluate(`
    ![...document.querySelectorAll('button')].find((b) => /Submit answers|Answer/.test(b.textContent))?.disabled
  `);
  console.log('ask   : submit enabled once every question has an answer =', enabled);

  await page.evaluate(`
    (() => {
      const b = [...document.querySelectorAll('button')].find((x) => /Submit answers|Answer/.test(x.textContent));
      b.click();
      return true;
    })()
  `);
  await page.until(`!document.body.innerText.includes('waiting on you')`, 'the feature to continue', 60_000);
  const after = await page.evaluate(`document.body.innerText.split('\\n').slice(0, 8).join(' | ')`);
  console.log('after :', after.slice(0, 220));
} finally {
  page.close();
}
