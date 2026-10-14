#!/usr/bin/env node
/**
 * Print the real, framework-resolved routes one Express route module registers.
 *
 * Deliberately narrow: this requires only the one file named on the command line -- a route
 * module such as `server/routes/serviceRequest.route.js`, exporting an `express.Router()` --
 * never the application's own entry point. A route module built the conventional way
 * (`module.exports = router`) has no side effects of its own: no `app.listen()`, no database
 * connection, no outbound call. What mounts it under a prefix lives in a different file
 * (`index.route.js`'s `router.use('/service-requests', serviceRequestRoutes)`), which this
 * script never touches -- that prefix is resolved separately, by reading source text, not by
 * requiring it. This script answers one question only: what does this module itself register,
 * exactly as Express itself would answer it, with none of a regex's ambiguity about chained
 * `.route().post()` calls versus `.post()` shorthand versus any other registration shape.
 *
 * Output: a JSON array of `{method, path}` on stdout, nothing else. Any failure -- the file does
 * not exist, requiring it throws, its export is not router-shaped -- is one JSON object
 * `{"error": "..."}` on stderr and a non-zero exit, never a thrown stack trace on stdout. The
 * caller treats either shape as "this file could not be dumped" and falls back to static
 * evidence; it never has to tell a crash apart from a clean refusal.
 */

'use strict';

function fail(message) {
  process.stderr.write(JSON.stringify({ error: String(message) }));
  process.exit(1);
}

const path = require('path');

const targetPath = process.argv[2];
if (!targetPath) {
  fail('usage: express_route_module.js <path-to-route-module>');
}

// Many repositories write routes in ESM/modern syntax and transpile at require-time via
// `@babel/register` (the same hook their own `babel-jest` tests already go through) rather than
// a prebuilt `dist/`. Resolved from the target file's own directory upward -- the standard Node
// module-resolution walk -- never from this script's own location, which has no node_modules of
// its own. A repository with no `@babel/register` installed, or whose route module is already
// plain CommonJS, simply requires it unchanged; this is a transpilation hook only, not a second
// bootstrap, so it adds no side effect of its own -- it is a prerequisite the file needs before
// `require` can even parse it, not a second module the file depends on at runtime.
try {
  const registerPath = require.resolve('@babel/register', { paths: [path.dirname(targetPath)] });
  // eslint-disable-next-line import/no-dynamic-require, global-require
  require(registerPath)();
} catch (error) {
  // No `@babel/register` here -- proceed and let the plain `require` below decide.
}

let moduleExport;
try {
  // eslint-disable-next-line import/no-dynamic-require, global-require
  moduleExport = require(targetPath);
} catch (error) {
  fail(`could not require ${targetPath}: ${error && error.message ? error.message : error}`);
}

/** An Express `Router()` (a callable function) and a nested sub-router both expose `.stack`. */
function routerStackOf(candidate) {
  if (candidate && typeof candidate === 'function' && Array.isArray(candidate.stack)) {
    return candidate.stack;
  }
  if (candidate && Array.isArray(candidate.stack)) {
    return candidate.stack;
  }
  return null;
}

// A route module conventionally exports the router directly, but a small number export it
// under a named key (`module.exports = { router }`) -- tried in this order, the direct export
// first, since that is what `module.exports = router` (the shape this whole feature exists to
// read) actually produces.
let stack = routerStackOf(moduleExport);
if (!stack && moduleExport && typeof moduleExport === 'object') {
  for (const value of Object.values(moduleExport)) {
    stack = routerStackOf(value);
    if (stack) break;
  }
}
if (!stack) {
  fail(`${targetPath} does not export an Express Router (no .stack found)`);
}

const routes = [];
for (const layer of stack) {
  if (!layer || !layer.route) continue;
  const route = layer.route;
  const path = route.path;
  const methods = route.methods && typeof route.methods === 'object' ? Object.keys(route.methods) : [];
  for (const method of methods) {
    if (method === '_all') continue;
    routes.push({ method: method.toUpperCase(), path });
  }
}

process.stdout.write(JSON.stringify(routes));
