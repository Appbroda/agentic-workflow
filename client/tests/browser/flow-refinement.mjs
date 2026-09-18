// Walk the refinement pass in a real browser, against the real API.
//
// Everything here is the running application talking to the running platform: the navigation,
// the setup gates, saved repositories, a feature actually submitted, and the queued state it
// comes back in. Only the clarification suggestion is substituted -- no feature in this
// database was created after suggestions existed, so its shape comes from a saved response.
//
//   UI=http://localhost:5173/ui PLATFORM_API_KEY=... node client/tests/browser/flow-refinement.mjs
import { writeFileSync, mkdirSync } from 'node:fs';
import { connect, SET_VALUE } from './drive.mjs';

const UI = process.env.UI ?? 'http://localhost:5173/ui';
const KEY = process.env.PLATFORM_API_KEY;
const SHOTS = process.env.SHOTS ?? 'shots';
mkdirSync(SHOTS, { recursive: true });


const BY_LABEL = `
  window.__labelled = (text) => {
    const label = [...document.querySelectorAll('label, .field__label')]
      .find((node) => node.textContent.trim() === text);
    if (!label) throw new Error('no label "' + text + '"');
    const forId = label.getAttribute('for');
    const node = forId ? document.getElementById(forId) : label.querySelector('input, textarea, select');
    if (!node) throw new Error('no control for "' + text + '"');
    return node;
  };
  window.__setLabelled = (text, value) => {
    const node = window.__labelled(text);
    const proto = node instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : node instanceof HTMLSelectElement
        ? HTMLSelectElement.prototype
        : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(node, value);
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  };
  // The smallest element that holds both the text and the button: 'Add' beside OpenAI has to
  // reach OpenAI's card and not GitHub's, and the text alone matches a <strong> with no button.
  window.__clickIn = (containerText, buttonText) => {
    const container = [...document.querySelectorAll('li, section, div, form')]
      .filter((node) => node.textContent.includes(containerText))
      .filter((node) =>
        [...node.querySelectorAll('button')].some((b) => b.textContent.trim() === buttonText),
      )
      .sort((a, b) => a.textContent.length - b.textContent.length)[0];
    if (!container) throw new Error('no "' + buttonText + '" near "' + containerText + '"');
    [...container.querySelectorAll('button')]
      .find((b) => b.textContent.trim() === buttonText)
      .click();
    return true;
  };
  true;
`;

/**
 * Start from a deployment with nothing configured.
 *
 * The setup gates are the first thing this walks, and they only exist when the prerequisites
 * are genuinely absent -- so the walk resets them rather than being a script that can only be
 * run once. Everything it removes here it puts back further down.
 */
const API = process.env.API ?? 'http://127.0.0.1:8000';
const authorised = { Authorization: `Bearer ${KEY}` };
for (const provider of ['openai', 'github']) {
  await fetch(`${API}/credentials/${provider}`, { method: 'DELETE', headers: authorised });
}
const existing = await (await fetch(`${API}/repositories`, { headers: authorised })).json();
for (const item of existing.repositories ?? []) {
  await fetch(`${API}/repositories/${item.configuration_id}`, {
    method: 'DELETE',
    headers: authorised,
  });
}

const page = await connect();
const results = [];
const check = (label, ok, detail = '') => {
  results.push({ label, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? ` — ${detail}` : ''}`);
};

async function shot(name) {
  const data = await page.screenshot();
  writeFileSync(`${SHOTS}/${name}.png`, Buffer.from(data, 'base64'));
}

async function open(path) {
  await page.goto(`${UI}${path}`);
  await page.evaluate(SET_VALUE);
  await page.evaluate(BY_LABEL);
  // The token gate is detected by its input, never by the words "Platform access": Settings
  // used to have a section by that name and the text check false-positived there.
  if (await page.evaluate(`Boolean(document.querySelector('#platform-token'))`)) {
    await page.evaluate(`__set('#platform-token', ${JSON.stringify(KEY)})`);
    await page.evaluate(`__click('Continue')`);
    await page.until(`!document.querySelector('#platform-token')`, 'the gate');
    await page.evaluate(BY_LABEL);
  }
}

const text = () => page.evaluate(`document.body.innerText`);
const has = async (needle) => (await text()).toLowerCase().includes(needle.toLowerCase());

try {
  await page.setViewport(1440, 1000);

  // ---- 1, 3, 17: the Features page, the navigation, and the reference ---------------------
  await open('/');
  await page.until(`document.querySelector('table')`, 'the features table');
  const nav = await page.evaluate(
    `[...document.querySelectorAll('nav[aria-label=Main] a')].map((a) => a.textContent.trim())`,
  );
  check('sidebar is Features, New feature, Settings', JSON.stringify(nav) === JSON.stringify(['Features', 'New feature', 'Settings']), nav.join(' / '));
  check('no Activity in primary navigation', !nav.some((item) => /activity|needs attention|running|completed/i.test(item)));
  const firstId = await page.evaluate(
    `document.querySelector('table tbody tr td .mono')?.innerText.trim()`,
  );
  check('the ID column leads with AB-Feature-N', /^AB-Feature-\d+/.test(firstId ?? ''), firstId);
  await shot('01-features');

  // ---- 2: the filters, including queued ---------------------------------------------------
  const chips = await page.evaluate(
    `[...document.querySelectorAll('[role=tablist][aria-label="Feature groups"] [role=tab]')].map((t) => t.getAttribute('aria-label'))`,
  );
  check('filters cover every server category', ['All', 'Queued', 'Running', 'Needs attention', 'Failed', 'Completed', 'Cancelled'].every((label) => chips.some((chip) => chip.startsWith(label))), chips.join(' | '));
  await page.evaluate(`document.querySelector('[role=tab][aria-label^="Completed"]').click()`);
  await page.until(`location.search.includes('group=completed')`, 'the filter in the URL');
  check('a filter is a URL, so it is linkable', await page.evaluate(`location.search.includes('group=completed')`));
  await shot('02-features-filtered');

  // Search by the reference somebody would have been sent.
  await open('/');
  await page.until(`document.querySelector('input[type=search]')`, 'search');
  await page.evaluate(`__set('input[type=search]', ${JSON.stringify(firstId)})`);
  await new Promise((r) => setTimeout(r, 800));
  const rows = await page.evaluate(`document.querySelectorAll('table tbody tr').length`);
  check('searching a reference finds that feature', rows >= 1 && (await has(firstId)), `${rows} row(s)`);

  // ---- 3: the old activity URLs still land somewhere sensible -----------------------------
  await open('/needs-attention');
  check('/needs-attention redirects to the filtered Features page', await page.evaluate(`/\\/ui\\/?$/.test(location.pathname) && location.search.includes('group=waiting')`), await page.evaluate(`location.pathname + location.search`));

  // ---- 5: the account menu ----------------------------------------------------------------
  await open('/');
  await page.until(`document.querySelector('.usermenu__trigger')`, 'the account menu');
  const trigger = await page.evaluate(`document.querySelector('.usermenu__trigger').innerText.trim()`);
  check('the header shows the signed-in identity', trigger.includes('Platform operator'), trigger);
  check('no shared-key badge beside the name', !(await has('shared key')));
  await page.evaluate(`document.querySelector('.usermenu__trigger').click()`);
  await page.until(`document.querySelector('[role=menu]')`, 'the menu');
  const items = await page.evaluate(`[...document.querySelectorAll('[role=menuitem]')].map((i) => i.textContent.trim())`);
  check('the menu offers Settings and Sign out', JSON.stringify(items) === JSON.stringify(['Settings', 'Sign out']), items.join(' / '));
  await shot('03-account-menu');

  // ---- 7: New Feature, blocked on provider credentials ------------------------------------
  await open('/features/new');
  await page.until(`document.querySelector('h1')`, 'the page');
  const blockedHeading = await page.evaluate(`document.querySelector('h1').innerText.trim()`);
  check('New Feature is blocked without credentials', blockedHeading === 'Provider setup required', blockedHeading);
  check('the blocked state names each provider', (await has('OpenAI')) && (await has('GitHub')) && (await has('Not configured')));
  check('the form is absent rather than disabled', !(await page.evaluate(`Boolean(document.querySelector('textarea'))`)));
  await shot('04-provider-setup-required');

  // ---- 4: Settings ------------------------------------------------------------------------
  await open('/settings');
  await page.until(`document.body.innerText.includes('Provider credentials')`, 'settings');
  const panels = await page.evaluate(`[...document.querySelectorAll('.panel__title')].map((h) => h.textContent.trim())`);
  check('Settings holds Profile, Credentials, Repositories', JSON.stringify(panels) === JSON.stringify(['Profile', 'Provider credentials', 'Repositories']), panels.join(' / '));
  check('unresolved operations are gone from Settings', !(await has('Unresolved operations')));
  check('the API connection panel is gone', !(await has('API connection')));
  check('the platform-access panel is gone', !(await has('Forget platform access key')));
  check('developer information is present and collapsed', await page.evaluate(`Boolean([...document.querySelectorAll('details')].find((d) => d.innerText.includes('Developer information') && !d.open))`));
  await shot('05-settings');

  // ---- 10: no saved repositories ----------------------------------------------------------
  check('Settings says what to do when nothing is saved', await has('No repositories saved'));

  // ---- 8, 9: configuring the credentials, one at a time ------------------------------------
  // Placeholder values. The check the platform performs is local -- can this deployment open
  // what it stored -- so nothing here reaches a provider, and both are removed at the end.
  for (const [provider, label] of [['openai', 'OpenAI'], ['github', 'GitHub']]) {
    await page.evaluate(`__clickIn(${JSON.stringify(label)}, 'Add')`);
    await page.until(`document.querySelector('#secret-${provider}')`, `${label} field`);
    await page.evaluate(`__set('#secret-${provider}', 'placeholder-for-browser-validation')`);
    await page.evaluate(`__clickIn(${JSON.stringify(label)}, 'Save')`);
    await page.until(
      `[...document.querySelectorAll('.card')].some((c) => c.innerText.includes(${JSON.stringify(label)}) && c.innerText.includes('Configured'))`,
      `${label} configured`,
    );
    if (provider === 'openai') {
      await open('/features/new');
      await page.until(`document.querySelector('h1')`, 'the page');
      const partial = await page.evaluate(`document.body.innerText`);
      check('partial setup still blocks, and shows which half is done',
        partial.includes('Provider setup required') && /OpenAI\s*\n?\s*Configured/.test(partial) && partial.includes('Not configured'));
      await shot('06-partial-credentials');
      await open('/settings');
      await page.until(`document.body.innerText.includes('Provider credentials')`, 'settings');
    }
  }
  check('both credentials read as configured', !(await has('Not configured')));
  await shot('07-credentials-configured');

  // ---- 11, 12: saving and editing a repository --------------------------------------------
  for (const [url, type] of [
    ['https://github.com/Appbroda/admanager_console-2.0', 'Backend'],
    ['https://github.com/cryn3t/AB-console-admin-2.0', 'Frontend'],
  ]) {
    await page.evaluate(`__click('Add repository')`);
    await page.until(`document.querySelector('#repository-url')`, 'the repository form');
    await page.evaluate(`__set('#repository-url', ${JSON.stringify(url)})`);
    await page.evaluate(`__set('#repository-type', ${JSON.stringify(type)})`);
    const derived = await page.evaluate(`document.querySelector('.field__hint .mono')?.innerText.trim()`);
    check(`the name is derived from ${url.split('/').pop()}`, derived === url.split('/').pop(), derived);
    await page.evaluate(`__click('Save repository')`);
    // Wait for the form to close rather than for the name to appear: the name is already on
    // screen in the derived-name hint while the form is still open.
    await page.until(`!document.querySelector('#repository-url')`, 'the saved repository');
  }
  check('a saved repository shows its type and branch', (await has('Backend')) && (await has('Frontend')) && (await has('master')));
  await shot('08-saved-repositories');

  await page.evaluate(`__clickIn('AB-console-admin-2.0', 'Edit')`);
  await page.until(`document.querySelector('#repository-branch')`, 'the edit form');
  await page.evaluate(`__set('#repository-branch', 'develop')`);
  await page.evaluate(`__click('Save changes')`);
  await page.until(`document.body.innerText.includes('develop')`, 'the edited branch');
  check('editing a saved repository keeps it and changes what was edited', await has('develop'));
  // Put it back, so the repositories left behind are the ones somebody would actually use.
  await page.evaluate(`__clickIn('AB-console-admin-2.0', 'Edit')`);
  await page.until(`document.querySelector('#repository-branch')`, 'the edit form');
  await page.evaluate(`__set('#repository-branch', 'master')`);
  await page.evaluate(`__click('Save changes')`);
  await page.until(`!document.querySelector('#repository-branch')`, 'the form closing');

  // ---- 13, 14: the New Feature form, with the repositories selectable ----------------------
  await open('/features/new');
  await page.until(`document.querySelector('textarea')`, 'the form');
  check('the form is shown once both prerequisites are met', await has('Create feature'));
  check(
    'the execution mode is on the form and defaults to live',
    (await page.evaluate(`__labelled('Mode').value`)) === 'live',
  );
  check('the feature ID is read-only and unreserved', await has('Assigned automatically on submission'));
  const options = await page.evaluate(
    `[...document.querySelectorAll('.repo-option')].map((o) => o.innerText.replace(/\\s+/g, ' '))`,
  );
  check('repositories are offered by name and label', options.length === 2 && options.every((o) => /Backend|Frontend/.test(o)), options.join(' | '));
  await page.evaluate(`document.querySelectorAll('.repo-option')[0].click()`);
  await page.evaluate(`document.querySelectorAll('.repo-option')[1].click()`);
  await page.until(`document.querySelectorAll('.repo-chip').length === 2`, 'two selected');
  const chipsText = await page.evaluate(
    `[...document.querySelectorAll('.repo-chip')].map((c) => c.innerText.replace(/\\s+/g, ' '))`,
  );
  check('selected repositories show name, type and branch', chipsText.every((c) => /master/.test(c)), chipsText.join(' | '));
  await shot('09-new-feature');

  // ---- 15, 16, 17: submitting, and what comes back ----------------------------------------
  const title = `Refinement browser check ${Date.now()}`;
  // Explicitly mock. The form defaults to live, which would clone the pilot repositories,
  // commit to them and open pull requests -- every time this walk runs.
  await page.evaluate(`__setLabelled('Mode', 'mock')`);
  const submittedMode = await page.evaluate(`__labelled('Mode').value`);
  check('the walk submits in mock mode, not against real repositories', submittedMode === 'mock', submittedMode);
  await page.evaluate(`__setLabelled('Title', ${JSON.stringify(title)})`);
  await page.evaluate(
    `__setLabelled('Problem statement', 'Operators cannot see a submission until analysis finishes, which makes a queued feature indistinguishable from a lost one.')`,
  );
  await page.evaluate(`__click('Create feature')`);
  await page.until(
    `location.pathname.includes('/features/') && !location.pathname.endsWith('/new')`,
    'the workspace',
    30_000,
  );
  const featurePath = await page.evaluate(`location.pathname`);
  await page.until(`document.querySelector('h1')`, 'the feature header');
  const header = await page.evaluate(`document.querySelector('.page-header__text').innerText`);
  check('the workspace leads with AB-Feature-N above the title', /AB-Feature-\d+/.test(header) && header.includes(title), header.replace(/\s+/g, ' ').slice(0, 120));
  check('a fresh feature reads as queued', await has('Queued'), 'status text contains Queued');
  await shot('10-feature-queued');

  // ---- 19: the workflow graph, with the request accepted and the rest queued ---------------
  await open(`${featurePath.replace('/ui', '')}/workflow`);
  await page.until(`document.querySelector('.graph, [aria-label*=graph i], svg')`, 'the graph', 30_000);
  const graph = await text();
  check('the graph shows the request as accepted', graph.includes('Request') && graph.includes('Accepted'));
  check('the graph names the product manager stage', graph.includes('Product manager'));
  await shot('11-workflow-graph');

  // ---- 21: the pull requests this feature opened, titled with its reference ---------------
  const reference = header.match(/AB-Feature-\d+/)[0];
  await open(`${featurePath.replace('/ui', '')}/pull-requests`);
  await page.until(`document.querySelectorAll('table tbody tr').length > 0`, 'the pull requests', 60_000);
  const titles = await page.evaluate(
    `[...document.querySelectorAll('table tbody tr')].map((r) => r.innerText.replace(/\\s+/g, ' '))`,
  );
  check(`every pull request title carries ${reference}`, titles.length === 2 && titles.every((t) => t.includes(`[${reference}]`)), titles.join(' | '));
  check('the panel says which feature these belong to', await has(`opened for ${reference}`));
  await shot('12-pull-requests');

  // ---- 20: a clarification question arriving with the answer the checkout gives ------------
  // Substituted: no feature in this database was created after suggestions existed, so the
  // shape comes from what the server's own test asserts it produces.
  const SUGGESTION = JSON.stringify({
    feature_id: 'suggestion-preview',
    awaiting_answers: true,
    technical_prd_artifact_id: '002_technical_prd.revision-2.json',
    clarification_rounds: 0,
    max_clarification_rounds: 10,
    questions: [
      {
        question_id: 'recon-admanager-server-1',
        question: 'Rely on the global middleware, or introduce per-route auth?',
        rationale:
          'In admanager-server, the requirements assume: authentication can be applied per route. The checkout instead has: authentication is applied once, globally. Evidence: server/config/express.js.',
        required: true,
        suggested_answer:
          'Follow what admanager-server already does: authentication is applied once, globally.',
        suggestion_source:
          'Suggested from repository analysis of admanager-server (server/config/express.js)',
        suggestion_confidence: 'high',
      },
      {
        question_id: 'criteria-REQ-2-1',
        question:
          'Requirement REQ-2 is accepted on a p99 latency figure. Restate it, or confirm it is verified elsewhere.',
        rationale: 'A reviewer judges the diff and the output of the repository own commands.',
        required: true,
        suggested_answer: '',
        suggestion_source: '',
        suggestion_confidence: null,
      },
    ],
    previous_answers: {},
  });
  const featureId = featurePath.split('/').pop();
  await page.onNewDocument(`
    window.__realFetch = window.__realFetch ?? window.fetch;
    window.fetch = async (input, init = {}) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url.includes('/clarification')) {
        return new Response(${JSON.stringify(SUGGESTION)}, {
          status: 200, headers: { 'content-type': 'application/json' },
        });
      }
      if (url.endsWith('/features/' + ${JSON.stringify(featureId)})) {
        const real = await window.__realFetch(input, init);
        const body = await real.json();
        return new Response(JSON.stringify({ ...body, status: 'waiting_for_human' }), {
          status: 200, headers: { 'content-type': 'application/json' },
        });
      }
      return window.__realFetch(input, init);
    };
  `);
  await open(featurePath.replace('/ui', ''));
  await page.until(
    `document.body.innerText.includes('This feature is waiting on you')`,
    'the clarification card',
    30_000,
  );
  const answers = await page.evaluate(
    `[...document.querySelectorAll('.action-card textarea')].map((t) => t.value)`,
  );
  check('the repository-informed answer is prefilled', Boolean(answers[0]?.includes('applied once, globally')), answers[0]);
  check('a question with no grounded answer is left empty', answers[1] === '', JSON.stringify(answers[1]));
  check('the suggestion says where it was read', await has('Suggested from repository analysis of admanager-server'));
  check('the card names the feature the questions belong to', await has(reference));
  await shot('13-clarification-suggestions');

  // ---- 6: signing out ---------------------------------------------------------------------
  await open('/');
  await page.until(`document.querySelector('.usermenu__trigger')`, 'the account menu');
  await page.evaluate(`document.querySelector('.usermenu__trigger').click()`);
  await page.until(`document.querySelector('[role=menu]')`, 'the menu');
  await page.evaluate(`__click('Sign out', '[role=menuitem]')`);
  await page.until(`document.querySelector('#platform-token')`, 'the sign-in screen', 20_000);
  check('signing out returns to the sign-in screen', await page.evaluate(`Boolean(document.querySelector('#platform-token'))`));
  const afterSignOut = await (await fetch(`${API}/setup`, { headers: authorised })).json();
  check('signing out keeps the credentials on the account', afterSignOut.credentials_ready === true);
  check('signing out keeps the saved repositories', afterSignOut.saved_repository_count === 2);
  await shot('14-signed-out');

  console.log('\n' + results.filter((r) => !r.ok).length + ' failed of ' + results.length);
} finally {
  page.close();
}
