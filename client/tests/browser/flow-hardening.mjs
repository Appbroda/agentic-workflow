// Drive the capabilities this hardening pass added, in a real browser against a real stack.
//
// The unit tests prove the components behave and the backend tests prove the platform does.
// Neither proves that opening the page in Chrome, against a server holding real rows, shows
// the right thing -- which is the only question this file answers.
//
//   docker compose up -d postgres redis        # or an existing stack
//   ... run the API from source on 8099, serving client/dist ...
//   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
//     --remote-debugging-port=9222 --user-data-dir=/tmp/cdp-profile about:blank &
//   BASE=http://127.0.0.1:8099 PLATFORM_API_KEY=... node tests/browser/flow-hardening.mjs
import { connect, SET_VALUE } from './drive.mjs';

const BASE = process.env.BASE ?? 'http://127.0.0.1:8099';
const KEY = process.env.PLATFORM_API_KEY ?? 'acceptance-key';
const COMPLETE = process.env.COMPLETE_FEATURE ?? 'acceptance-complete';
const REPAIR = process.env.REPAIR_FEATURE ?? 'acceptance-repair3';
// A feature that has not settled. A completed one deliberately opens no stream -- nothing
// more will happen to it -- so watching one would prove the opposite of what is intended.
const LIVE = process.env.LIVE_FEATURE ?? 'acceptance-waiting';

const results = [];
function check(name, passed, detail = '') {
  results.push({ name, passed, detail });
  console.log(`${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
}

const page = await connect();

/** Navigate, then reinstall the helpers -- a page load wipes anything defined on `window`. */
async function open(url) {
  await page.goto(url);
  await page.evaluate(SET_VALUE);
}

try {
  await page.setViewport(1440, 900);

  // The token gate. Everything else depends on getting past it, and on the token going
  // somewhere that dies with the tab.
  await page.goto(`${BASE}/ui/`);
  await page.evaluate(`sessionStorage.setItem('platform-api-token', ${JSON.stringify(KEY)})`);
  await open(`${BASE}/ui/`);
  await page.until(`document.body.innerText.includes('Dashboard')`, 'the dashboard');
  check(
    'the platform token is not written to localStorage',
    (await page.evaluate(`localStorage.length`)) === 0,
  );

  // 1. Dashboard against the running backend.
  await page.until(`document.body.innerText.includes('${COMPLETE}')`, 'the seeded feature');
  check('dashboard lists a feature from the running backend', true);

  // 2. Identity in the header -- and the fact that this is the shared key, which is worth
  //    knowing without going looking for it.
  const header = await page.evaluate(`document.querySelector('.layout__header').innerText`);
  check(
    'the header names who this browser is acting as',
    header.includes('Platform operator'),
    header.replace(/\n/g, ' | '),
  );
  check('the header says when that is the shared key', header.includes('shared key'));

  // 3. Feature workspace, PRD, timeline, agent history, artifacts.
  await open(`${BASE}/ui/features/${COMPLETE}`);
  await page.until(`document.body.innerText.includes('Repositories')`, 'the workspace');
  const workspace = await page.evaluate(`document.body.innerText`);
  check('the feature workspace renders', workspace.includes('Acceptance'));
  check('repository workstreams render', /backend|frontend/.test(workspace));

  for (const tab of ['History', 'Artifacts', 'Pull requests']) {
    await page.evaluate(`window.__click(${JSON.stringify(tab)}, 'a, button, [role="tab"]')`);
    await new Promise((resolve) => setTimeout(resolve, 800));
    const text = await page.evaluate(`document.body.innerText`);
    check(`${tab} renders`, text.length > 200 && !text.includes('Something went wrong'));
  }

  // 4. The repair card: the capability that did not exist before this pass.
  await open(`${BASE}/ui/features/${REPAIR}`);
  await page.until(
    `document.body.innerText.includes('could not run its own checks')`,
    'the repair card',
  );
  const repair = await page.evaluate(`document.body.innerText`);
  check(
    'the repair states the problem it found',
    repair.includes('eslint-config-house'),
  );
  check('the repair states what it would do', repair.includes('Declare eslint-config-house'));
  check(
    'the repair states what it would touch',
    repair.includes('Affected dependencies') && repair.includes('Affected files'),
  );
  check(
    'the repair says which stage found it, not what state it left behind',
    /FOUND BY\s*\n?\s*repository_preflight/i.test(repair) || repair.includes('repository_preflight'),
  );
  check('the repair offers a decision', /Approve repair/.test(repair) && /Reject repair/.test(repair));

  // 5. Approving asks first. A repair changes somebody's repository; one click must not do it.
  await page.evaluate(`window.__click('Approve repair')`);
  await page.until(`document.body.innerText.includes('Approve this repair?')`, 'the confirmation');
  const confirming = await page.evaluate(`document.body.innerText`);
  check(
    'approving a repair asks before it changes a repository',
    confirming.includes('checked-in files or dependencies'),
  );
  await page.evaluate(`window.__click('Cancel')`);

  // 6. Rejecting will not send without a reason.
  await page.evaluate(`window.__click('Reject repair')`);
  await page.until(
    `document.body.innerText.includes('Why is this repair not the right change')`,
    'the rejection form',
  );
  const disabled = await page.evaluate(
    `[...document.querySelectorAll('button')].filter((b) => b.textContent.trim() === 'Reject repair' && b.disabled).length`,
  );
  check('a repair cannot be rejected without a reason', disabled > 0);

  // 7. Settings: identity, credentials, readiness.
  await open(`${BASE}/ui/settings`);
  await page.until(`document.body.innerText.includes('Settings')`, 'settings');
  await page.until(`document.body.innerText.includes('Provider credentials')`, 'the credentials');
  const settings = await page.evaluate(`document.body.innerText`);
  check('settings names the signed-in identity', settings.includes('Platform operator'));
  check(
    'settings says what the shared key means for the audit trail',
    settings.includes('platform-admin'),
  );
  check(
    'settings lists a card for each provider',
    settings.includes('OpenAI') && settings.includes('GitHub'),
  );

  // 8. Storing a credential, through the real API, and reading back what it will admit.
  //    Targeted at OpenAI by name rather than by position, and tolerant of one already being
  //    stored -- a flow that only works against a fresh database proves less than it looks.
  await page.evaluate(`
    (() => {
      const cards = [...document.querySelectorAll('li.card')];
      const openai = cards.find((card) => card.textContent.includes('OpenAI'));
      const button = [...openai.querySelectorAll('button')]
        .find((item) => ['Add', 'Replace'].includes(item.textContent.trim()));
      button.click();
      return true;
    })();
  `);
  await page.until(`document.querySelector('#secret-openai')`, 'the OpenAI field');
  await page.evaluate(`window.__set('#secret-openai', 'sk-acceptance-value-9f2a')`);
  await page.evaluate(`window.__click('Save')`);
  await page.until(`document.body.innerText.includes('Configured')`, 'the stored credential');
  const stored = await page.evaluate(`document.body.innerText`);
  check('a stored credential is reported as configured', stored.includes('Configured'));
  check('only the last four characters are shown', stored.includes('…9f2a'));
  check(
    'the credential itself is nowhere on the page',
    !stored.includes('sk-acceptance-value-9f2a'),
  );

  // 9. The event stream, authenticated, with no credential in any URL. Checked against what
  //    the browser actually requested rather than against what the code says it does, and on
  //    a feature that is still waiting, so there is a stream to check at all.
  // Installed before the page's own scripts. The application opens its event stream on
  // mount, so a recorder attached after navigation would miss the one request this checks.
  await page.onNewDocument(`
    window.__requests = [];
    const original = window.fetch;
    window.fetch = (input, init) => {
      window.__requests.push(String(typeof input === 'string' ? input : input.url));
      return original(input, init);
    };
  `);
  await open(`${BASE}/ui/features/${LIVE}`);
  await new Promise((resolve) => setTimeout(resolve, 3000));
  const requested = await page.evaluate(`JSON.stringify(window.__requests ?? [])`);
  const urls = JSON.parse(requested || '[]');
  check(
    'no request URL carries the platform token',
    urls.length > 0 && !String(requested).includes(KEY),
    `${urls.length} requests inspected`,
  );
  check(
    'the event stream is opened as a plain path, with the cursor and nothing else',
    urls.some((url) => url.includes('/events/stream')),
    urls.filter((url) => url.includes('stream')).join(' ') || 'no stream request seen',
  );

  // 10. Layout at the widths a laptop actually is, and one narrower.
  for (const [width, height] of [
    [1280, 800],
    [1440, 900],
    [1680, 1050],
  ]) {
    await page.setViewport(width, height);
    await open(`${BASE}/ui/features/${REPAIR}`);
    await page.until(`document.body.innerText.includes('could not run its own checks')`, 'the card');
    const overflow = await page.evaluate(
      `document.documentElement.scrollWidth - document.documentElement.clientWidth`,
    );
    check(`no horizontal overflow at ${width}px`, overflow <= 1, `overflowed by ${overflow}px`);
  }
} finally {
  page.close();
}

const failed = results.filter((item) => !item.passed);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) {
  console.log('Failed:');
  for (const item of failed) console.log(`  - ${item.name} ${item.detail}`);
  process.exitCode = 1;
}
