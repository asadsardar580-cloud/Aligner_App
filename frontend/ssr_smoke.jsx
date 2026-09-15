// Renders <App/> once in Node. Effects do not run under SSR, so three.js never
// initialises — but the component FUNCTION BODY does execute, which is exactly
// where a temporal-dead-zone error throws. This is the check that a passing
// `npm run build` does not give you.
import React from "react";
import { renderToString } from "react-dom/server";
import App from "./src/App.jsx";

try {
  const html = renderToString(React.createElement(App));
  console.log(`RENDER OK — ${html.length} bytes of markup, no exception`);
} catch (e) {
  console.log(`RENDER FAILED — ${e.name}: ${e.message}`);
  process.exitCode = 1;
}
