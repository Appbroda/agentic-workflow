// Screenshot the main views. Every check so far read text content; none looked at the page.
import { connect } from './drive.mjs';
import { writeFileSync } from 'node:fs';

const UI = process.env.UI ?? 'http://localhost:8000/ui';
const KEY = process.env.PLATFORM_API_KEY;
const F = 'adunit-deactivate-live-086';

const VIEWS = [
  ['dashboard', `${UI}/`, 1440],
  ['needs-attention', `${UI}/needs-attention`, 1440],
  ['workspace', `${UI}/features/${F}`, 1440],
  ['repositories', `${UI}/features/${F}/repositories`, 1440],
  ['repository', `${UI}/features/${F}/repositories/admanager-server`, 1440],
  ['validation', `${UI}/features/${F}/repositories/admanager-server?view=validation`, 1440],
  ['prd', `${UI}/features/${F}/prd`, 1440],
  ['plan', `${UI}/features/${F}/plan`, 1440],
  ['history', `${UI}/features/${F}/history`, 1440],
  ['pull-requests', `${UI}/features/${F}/pull-requests`, 1440],
  ['chat', `${UI}/features/${F}?chat=open`, 1440],
  ['artifacts', `${UI}/features/${F}/artifacts?artifact=007_review.admanager-server.attempt-5.json`, 1440],
  ['new-feature', `${UI}/features/new`, 1440],
  ['laptop-dashboard', `${UI}/`, 1366],
  ['laptop-workspace', `${UI}/features/${F}`, 1366],
  ['narrow-workspace', `${UI}/features/${F}`, 1180],
];

const page = await connect();
try {
  for (const [name, url, width] of VIEWS) {
    await page.setViewport(width, 1100);
    await page.goto(url);
    // The token field, not the words "Platform access" -- Settings has a section by that name.
    if (await page.evaluate(`Boolean(document.querySelector('#platform-token'))`)) {
      await page.evaluate(`
        (() => {
          const n = document.querySelector('input[type=password]');
          Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(n, ${JSON.stringify(KEY)});
          n.dispatchEvent(new Event('input', { bubbles: true }));
          [...document.querySelectorAll('button')].find((b)=>b.textContent.trim()==='Continue').click();
          return true;
        })()
      `);
      await new Promise((r) => setTimeout(r, 2500));
      await page.goto(url);
    }
    await new Promise((r) => setTimeout(r, 2000));
    const data = await page.screenshot();
    writeFileSync(`shot-${name}.png`, Buffer.from(data, 'base64'));
    console.log('captured', name, width);
  }
} finally {
  page.close();
}
