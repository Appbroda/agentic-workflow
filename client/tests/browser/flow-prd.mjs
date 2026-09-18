// Submit a PRD through the form, by clicking, and follow where it lands.
import { connect, SET_VALUE } from './drive.mjs';

const UI = process.env.UI ?? 'http://localhost:8000/ui';
const KEY = process.env.PLATFORM_API_KEY;
const ID = process.env.FEATURE_ID ?? 'ui-click-check';

const page = await connect();
try {
  await page.goto(`${UI}/`);
  await page.evaluate(SET_VALUE);

  // The gate: the token is typed, held for the tab, and never stored anywhere else.
  if (await page.evaluate(`Boolean(document.querySelector('#\\\\:r0\\\\:, [type=password]'))`)) {
    const gated = await page.evaluate(`document.body.innerText.includes('Platform access')`);
    if (gated) {
      await page.evaluate(`__set('input[type=password]', ${JSON.stringify(KEY)})`);
      await page.evaluate(`__click('Continue')`);
      await page.until(`!document.body.innerText.includes('Platform access')`, 'the gate to close');
      console.log('gate  : token accepted, dashboard rendered');
      console.log('gate  : localStorage empty =', await page.evaluate(`window.localStorage.length === 0`));
    }
  }

  await page.goto(`${UI}/features/new`);
  await page.evaluate(SET_VALUE);
  await page.until(`document.body.innerText.includes('New feature')`, 'the form');

  // Fill it as a person would: every field by its label, through React's own value setter.
  const fill = async (label, value) => {
    const ok = await page.evaluate(`
      (() => {
        const labels = [...document.querySelectorAll('label')];
        const match = labels.find((l) => l.textContent.trim().startsWith(${JSON.stringify(label)}));
        if (!match) return false;
        const id = match.getAttribute('for');
        const node = id ? document.getElementById(id) : match.querySelector('input,textarea,select');
        if (!node) return false;
        __set('#' + CSS.escape(node.id || ''), ${JSON.stringify(value)});
        return true;
      })()
    `);
    if (!ok) throw new Error(`could not fill ${label}`);
  };

  await fill('Title', 'Clicked through the form');
  await fill('Problem statement', 'Prove that the submission form works when a person uses it.');
  await fill('Feature ID', ID);

  // The repeatable groups start with one empty entry each; fill the first of every one.
  await page.evaluate(`
    (() => {
      const byLabel = (text) => [...document.querySelectorAll('label')]
        .filter((l) => l.textContent.trim().startsWith(text))
        .map((l) => l.getAttribute('for') ? document.getElementById(l.getAttribute('for')) : l.querySelector('input,textarea,select'))
        .filter(Boolean);
      const set = (node, value) => {
        const proto = node.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype
          : node.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(proto, 'value').set.call(node, value);
        node.dispatchEvent(new Event('input', { bubbles: true }));
        node.dispatchEvent(new Event('change', { bubbles: true }));
      };
      const first = (text, value) => { const n = byLabel(text)[0]; if (n) set(n, value); };
      first('Goal', 'A person can submit a PRD and land on its workspace.');
      first('Persona', 'Operator');
      first('Need', 'submit a feature without writing JSON');
      first('Benefit', 'the console is usable');
      first('ID', 'US-1');
      first('Description', 'The form submits and navigates to the workspace.');
      // Every acceptance-criteria box, in both the story and the requirement.
      byLabel('Acceptance criteria').forEach((n, i) => set(n, 'Criterion ' + (i + 1)));
      // Repository fields.
      first('Repository ID', 'admanager-server');
      first('Name', 'Ad Manager Server');
      first('Repository URL', 'https://github.com/cryn3t/admanager_console-2.0.git');
      first('Default branch', 'master');
      return true;
    })()
  `);

  // Requirement ID and description share labels with the story; set the remaining empties.
  await page.evaluate(`
    (() => {
      const empties = [...document.querySelectorAll('input[type=text], input:not([type]), textarea')]
        .filter((n) => !n.value.trim() && n.type !== 'password');
      empties.forEach((n, i) => {
        const proto = n.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(proto, 'value').set.call(n, 'REQ-' + (i + 1));
        n.dispatchEvent(new Event('input', { bubbles: true }));
      });
      return empties.length;
    })()
  `);

  const before = await page.evaluate(`location.pathname`);
  await page.evaluate(`__click('Submit feature')`);
  await page.until(
    `location.pathname !== ${JSON.stringify(before)} || document.querySelector('[role=alert]')`,
    'the form to submit',
    30_000,
  );

  const landed = await page.evaluate(`location.pathname`);
  const alert = await page.evaluate(`document.querySelector('[role=alert]')?.innerText ?? ''`);
  console.log('submit: path now', landed);
  if (alert) console.log('submit: reported', alert.slice(0, 200));
  if (landed.includes('/features/')) {
    await page.until(`document.body.innerText.includes('Progress')`, 'the workspace', 30_000);
    const text = await page.evaluate(`__text()`);
    console.log('submit: workspace shows', text.split('\n').slice(0, 6).join(' | ').slice(0, 200));
  }
} finally {
  page.close();
}
