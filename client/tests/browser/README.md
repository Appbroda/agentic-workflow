# Browser flows

The unit tests prove components behave and the rendering checks prove pages display. Neither
proves that **pressing a button does the right thing**, which is the part of this application
that changes state. These scripts drive a real Chrome over the DevTools Protocol and click.

They are not part of `npm test`: they need a browser and a running stack. Run them when the
mutating paths change.

```sh
# 1. a stack, serving the built client
docker compose up -d api                      # http://localhost:8000/ui/

# 2. a Chrome with the protocol open
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --remote-debugging-port=9222 \
  --user-data-dir=/tmp/cdp-profile about:blank &

# 3. the flows
export PLATFORM_API_KEY=...   # the same key the server has
node tests/browser/flow-prd.mjs        # fill the form, submit, land on the workspace
node tests/browser/flow-actions.mjs    # grant a retry; send a chat message
node tests/browser/flow-lifecycle.mjs  # cancel and resume, through their confirmations
node tests/browser/flow-clarify.mjs    # answer a clarification, on a feature really waiting
node tests/browser/shots.mjs           # screenshot the main views, at three widths
```

`shots.mjs` writes `shot-*.png` beside itself, at 1440, 1366 and 1180. Look at them. A page can
carry every correct string and still be laid out badly — that is how Dismiss buttons ended up a
foot from the text they dismissed, how a filter label came to sit flush against its select, and
how a table's last two columns ended up off the right-hand edge behind a 90-character branch
name.

`flow-clarify.mjs` needs a feature that is paused. Mock mode reaches one without a provider
call or a clone: give a requirement an acceptance criterion the platform cannot check, such as
"confirmed by manual verification", and it stops to ask about it.

`flow-prd.mjs` really creates a feature — pass `FEATURE_ID` to name it, and use a mock-mode
repository so nothing is cloned or pushed.

`flow-actions.mjs` and `flow-lifecycle.mjs` replace `window.fetch` for the mutating call only.
The request the client builds is inspected exactly as it would leave the browser — path, body,
headers — and a 409 is fed back so the refusal rendering is exercised. Nothing reaches the
platform, so running them costs no attempts and changes no feature. The server side of each
path is covered by its own tests and by the live schema check.

What they establish, which nothing else does:

- the token gate accepts a key and `localStorage` stays empty;
- the PRD form submits and navigates to the new feature's workspace;
- the retry grant will not send without an author and a reason, sends the right body, and shows
  the platform's refusal;
- a chat message carries the provider key as a header and never in the body;
- cancel and resume confirm first, send nothing until confirmed, move focus into the dialog,
  and report what the platform decided;
- a paused feature is listed on the dashboard and flagged as needing a person, its questions
  show their rationale, the submit stays blocked until every one is answered, and the panel
  clears once the platform has accepted them.
