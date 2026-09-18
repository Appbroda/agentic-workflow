// Drive a real Chrome over the DevTools Protocol.
//
// The rendering checks so far proved pages display correctly. They cannot prove that pressing
// a button does the right thing, which is the part of this application that changes state.
import WebSocket from 'ws';

const PORT = process.env.CDP_PORT ?? '9222';

async function targets() {
  const response = await fetch(`http://127.0.0.1:${PORT}/json/list`);
  return response.json();
}

export async function connect() {
  const pages = (await targets()).filter((t) => t.type === 'page');
  const socket = new WebSocket(pages[0].webSocketDebuggerUrl, { maxPayload: 256 * 1024 * 1024 });
  await new Promise((resolve, reject) => {
    socket.once('open', resolve);
    socket.once('error', reject);
  });

  let id = 0;
  const pending = new Map();
  socket.on('message', (raw) => {
    const message = JSON.parse(raw.toString());
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    }
  });

  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const next = ++id;
      pending.set(next, { resolve, reject });
      socket.send(JSON.stringify({ id: next, method, params }));
      setTimeout(() => {
        if (pending.has(next)) {
          pending.delete(next);
          reject(new Error(`${method} timed out`));
        }
      }, 60_000);
    });

  await send('Page.enable');
  await send('Runtime.enable');

  const evaluate = async (expression) => {
    const result = await send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description ?? 'evaluate failed');
    }
    return result.result.value;
  };

  return {
    close: () => socket.close(),
    evaluate,
    /**
     * Run an expression before the next page's own scripts.
     *
     * Anything installed after a navigation misses whatever the application did on load,
     * which for this app is most of what is worth watching: the event stream opens on mount.
     */
    onNewDocument: (expression) =>
      send('Page.addScriptToEvaluateOnNewDocument', { source: expression }),
    /** Emulate a viewport, so a layout can be looked at at more than one width. */
    setViewport: (width, height) =>
      send('Emulation.setDeviceMetricsOverride', {
        width,
        height,
        deviceScaleFactor: 1,
        mobile: false,
      }),
    screenshot: async () => (await send('Page.captureScreenshot', { format: 'png' })).data,
    goto: async (url) => {
      await send('Page.navigate', { url });
      await new Promise((r) => setTimeout(r, 2500));
    },
    /** Wait until an expression is truthy, so a step never races the render after it. */
    until: async (expression, label, timeoutMs = 20_000) => {
      const deadline = Date.now() + timeoutMs;
      for (;;) {
        if (await evaluate(`Boolean(${expression})`)) return;
        if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
        await new Promise((r) => setTimeout(r, 400));
      }
    },
  };
}

/** Set a React-controlled input the way a keystroke does, so React sees the change. */
export const SET_VALUE = `
  window.__set = (selector, value) => {
    const node = document.querySelector(selector);
    if (!node) throw new Error('no element for ' + selector);
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
  window.__click = (text, role) => {
    const nodes = [...document.querySelectorAll(role ?? 'button, a, [role="tab"]')];
    const node = nodes.find((n) => (n.textContent ?? '').trim() === text);
    if (!node) throw new Error('no clickable "' + text + '"');
    node.click();
    return true;
  };
  window.__text = () => document.body.innerText;
  true;
`;
