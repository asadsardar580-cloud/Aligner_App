import { Component } from "react";

/**
 * The last line of defence against a blank page.
 *
 * WHY THIS EXISTS, SPECIFICALLY. This application has white-screened five times
 * during development, every one of them the same way: a `useEffect`/`useCallback`
 * dependency array naming a `const` declared further down the component. A
 * dependency array is evaluated DURING RENDER, so the reference throws
 * `Cannot access 'X' before initialization` — and `npm run build` reports a
 * clean bundle every time, because the code is syntactically perfect.
 *
 * Without a boundary, React unmounts the whole tree on such a throw and the
 * clinician gets a white rectangle. No message, no way back, and — this is the
 * part that actually matters — no indication that their case is fine.
 *
 * SO THE RECOVERY MESSAGE LEADS, NOT THE ERROR. The scan, the occlusal frame,
 * the cuts and every committed prescription live in the session on the server;
 * a render crash loses none of it. Someone whose screen just went blank needs
 * to know that before they need a stack trace.
 *
 * This is a class component because `getDerivedStateFromError` and
 * `componentDidCatch` have no hook equivalent. React has never shipped one.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, info: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    this.setState({ info });
    // Keep it in the console too: the boundary swallows the throw, so without
    // this the stack would be gone from the one place a developer looks first.
    console.error("[Aligner] render error caught by boundary:", error, info);
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    const stack = (info?.componentStack || "").trim().split("\n").slice(0, 8).join("\n");

    return (
      <div style={S.wrap} data-testid="error-boundary">
        <div style={S.card}>
          <div style={S.head}>The viewport stopped rendering</div>

          {/* Recovery first. The case is the thing they care about. */}
          <p style={S.lead}>
            <strong>Your case is safe.</strong> The scan, the occlusal reference, every cut
            and every committed movement are held in the session on the backend — a
            display fault does not touch them. Reloading restores the case from the
            server.
          </p>

          <button style={S.reload} onClick={() => window.location.reload()}
                  data-testid="error-boundary-reload">
            Reload and restore the case
          </button>

          <div style={S.detailHead}>What went wrong</div>
          <pre style={S.pre} data-testid="error-boundary-message">
            {String(error?.message || error)}
          </pre>

          {stack && (
            <details style={S.details}>
              <summary style={S.summary}>Component stack</summary>
              <pre style={S.pre}>{stack}</pre>
            </details>
          )}

          <p style={S.foot}>
            If this repeats on the same action, the backend is still running and the
            case can be exported from a fresh session. Report the message above —
            it names the component that threw.
          </p>
        </div>
      </div>
    );
  }
}

const S = {
  wrap: { position: "fixed", inset: 0, background: "#0d0f12", color: "#e7ebee",
          font: "13px/1.6 system-ui, sans-serif", display: "flex",
          alignItems: "center", justifyContent: "center", padding: 24, zIndex: 9999 },
  card: { maxWidth: 680, width: "100%", background: "#161a20",
          border: "1px solid #2c313a", borderRadius: 8, padding: "22px 24px" },
  head: { fontSize: 17, fontWeight: 600, marginBottom: 12, color: "#ffa53c" },
  lead: { margin: "0 0 16px", color: "#c3cad3" },
  reload: { background: "linear-gradient(#3fc6d4,#2a8b96)", border: "none",
            color: "#06232a", fontWeight: 700, fontSize: 13, padding: "9px 16px",
            borderRadius: 5, cursor: "pointer", marginBottom: 20 },
  detailHead: { fontSize: 10.5, letterSpacing: 0.6, color: "#8b93a0",
                textTransform: "uppercase", marginBottom: 6 },
  pre: { background: "#101215", border: "1px solid #21252b", borderRadius: 4,
         padding: "9px 11px", fontSize: 11.5, lineHeight: 1.5, color: "#e6194b",
         whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0,
         maxHeight: 220, overflow: "auto" },
  details: { marginTop: 12 },
  summary: { cursor: "pointer", fontSize: 11.5, color: "#8b93a0" },
  foot: { marginTop: 18, paddingTop: 12, borderTop: "1px solid #21252b",
          fontSize: 11.5, color: "#6b727d" },
};
