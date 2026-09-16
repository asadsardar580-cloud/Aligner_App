// Renders <App/> once in Node. Effects do not run under SSR, so three.js never
// initialises — but the component FUNCTION BODY does execute, which is exactly
// where a temporal-dead-zone error throws. This is the check that a passing
// `npm run build` does not give you.
import React from "react";
import { renderToString } from "react-dom/server";
import App from "./src/App.jsx";
import ErrorBoundary from "./src/ErrorBoundary.jsx";

let failed = false;

function check(label, fn) {
  try {
    fn();
  } catch (e) {
    console.log(`RENDER FAILED (${label}) — ${e.name}: ${e.message}`);
    failed = true;
  }
}

check("app", () => {
  const html = renderToString(React.createElement(App));
  console.log(`RENDER OK — ${html.length} bytes of markup, no exception`);
});

// The boundary is what stands between a TDZ throw and a white rectangle, so it
// is rendered here in the SAME nesting main.jsx uses. A boundary that itself
// throws is the worst possible failure and nothing else in the stack sees it.
check("boundary wraps app", () => {
  const html = renderToString(
    React.createElement(ErrorBoundary, null, React.createElement(App)));
  console.log(`BOUNDARY PASS-THROUGH OK — ${html.length} bytes`);
});

// And the fallback itself — driven DIRECTLY, not by throwing inside a child.
// Legacy `renderToString` does not recover through an error boundary; it
// rethrows, and boundary recovery is a client/streaming behaviour. So the way
// to cover the fallback in a headless gate is to put the class in its error
// state by hand and render what it returns. That still exercises the two things
// that can rot: getDerivedStateFromError's shape, and the recovery markup.
check("boundary fallback", () => {
  const err = new Error("smoke: deliberate render throw");
  const inst = new ErrorBoundary({ children: null });
  inst.state = ErrorBoundary.getDerivedStateFromError(err);
  if (inst.state.error !== err) throw new Error("getDerivedStateFromError lost the error");
  const html = renderToString(inst.render());
  for (const needle of ["Your case is safe", "error-boundary-reload",
                        "smoke: deliberate render throw"]) {
    if (!html.includes(needle)) throw new Error(`fallback markup lacks "${needle}"`);
  }
  console.log(`BOUNDARY FALLBACK OK — ${html.length} bytes, recovery text and `
              + `reload control present`);
});

process.exitCode = failed ? 1 : 0;
