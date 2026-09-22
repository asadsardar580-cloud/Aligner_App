import { Check, Circle, Lock, AlertTriangle } from "lucide-react";
import { color, space, radius, type, transition, motion } from "./theme.js";

/**
 * The clinical workflow, as a rail.
 *
 * WHY THIS EXISTS AND WHAT IT IS NOT. The seven accordion panels already
 * describe the workflow, but only in the order they happen to be stacked -
 * nothing said which step the case is ON, which are finished, or which cannot
 * be reached yet. A clinician opening a restored case saw seven closed
 * headings and had to open each one to find out.
 *
 * FOUR STATES, and the fourth is the one that matters. `done`, `current`,
 * `blocked` and `pending` are distinct, because "you have not done this yet"
 * and "you CANNOT do this yet, and here is why" are different facts - the same
 * distinction ValidationPanel draws between a failing check and one that was
 * never computed, and CLAUDE.md s.14 draws between NOT_CHECKED and CLEAR. A
 * blocked step carries its reason; a pending one does not pretend to have one.
 *
 * PURELY PRESENTATIONAL. It takes a plain array and an onSelect callback, and
 * holds no state and no refs. This app has white-screened five times from a
 * dependency array naming a `const` declared below it (CLAUDE.md s.13, s.17),
 * so a new component on the render path earns its place by having nothing in
 * it that can participate in that.
 */

const STATE_STYLE = {
  done: { fg: color.ok, bg: color.okSurface, ring: color.okBorder, Icon: Check },
  current: { fg: color.accent, bg: "#13272b", ring: color.accent, Icon: Circle },
  blocked: { fg: color.warn, bg: color.warnSurface, ring: color.warnBorder,
             Icon: AlertTriangle },
  pending: { fg: color.textFaint, bg: color.surfaceSunken, ring: color.border,
             Icon: Lock },
};

export default function WorkflowRail({ steps, onSelect }) {
  if (!steps?.length) return null;
  const done = steps.filter((s) => s.state === "done").length;

  return (
    <nav style={S.wrap} aria-label="Treatment workflow" data-testid="workflow-rail">
      <div style={S.head}>
        <span>WORKFLOW</span>
        <span style={S.progress}>{done}/{steps.length}</span>
      </div>
      {/* A single track behind the dots, so the rail reads as one sequence
          rather than seven unrelated rows. */}
      <div style={S.track}>
        <span style={{ ...S.trackFill,
                       height: `${(done / steps.length) * 100}%`,
                       transition: transition("height", motion.slow) }} />
      </div>
      <ol style={S.list}>
        {steps.map((s) => {
          const st = STATE_STYLE[s.state] || STATE_STYLE.pending;
          const { Icon } = st;
          const reachable = s.state !== "pending" && s.state !== "blocked";
          return (
            <li key={s.id} style={S.row}>
              <button
                type="button"
                onClick={() => onSelect?.(s.id)}
                title={s.reason || s.title}
                aria-current={s.state === "current" ? "step" : undefined}
                style={{
                  ...S.btn,
                  color: s.state === "pending" ? color.textFaint : color.text,
                  cursor: onSelect ? "pointer" : "default",
                  transition: transition("background-color", motion.fast),
                }}
              >
                <span style={{ ...S.dot, color: st.fg, background: st.bg,
                               borderColor: st.ring }}>
                  <Icon size={11} strokeWidth={3}
                        fill={s.state === "current" ? st.fg : "none"} aria-hidden />
                </span>
                <span style={S.label}>
                  <span style={S.title}>{s.title}</span>
                  {/* A blocked step says WHY. A pending one has nothing
                      honest to say, so it says nothing. */}
                  {s.state === "blocked" && s.reason && (
                    <span style={S.reason}>{s.reason}</span>
                  )}
                  {s.state === "done" && s.detail && (
                    <span style={S.detail}>{s.detail}</span>
                  )}
                </span>
                {!reachable && s.state === "blocked" && (
                  <span style={S.badge}>blocked</span>
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

const S = {
  wrap: {
    position: "relative",
    border: `1px solid ${color.border}`, borderRadius: radius.lg,
    background: color.surface, padding: `${space.sm}px ${space.sm}px ${space.xs}px`,
    marginBottom: space.sm,
  },
  head: {
    display: "flex", justifyContent: "space-between", alignItems: "baseline",
    fontSize: type.size.xs, letterSpacing: 0.7, color: color.textMuted,
    textTransform: "uppercase", padding: `0 ${space.xs}px ${space.xs}px`,
  },
  progress: { fontFamily: type.numeric, color: color.accent, letterSpacing: 0 },
  track: {
    position: "absolute", left: 17, top: 34, bottom: 14, width: 2,
    background: color.border, borderRadius: radius.pill, overflow: "hidden",
  },
  trackFill: {
    display: "block", width: "100%", background: color.accentQuiet,
    borderRadius: radius.pill,
  },
  list: { listStyle: "none", margin: 0, padding: 0, position: "relative" },
  row: { margin: 0 },
  btn: {
    display: "flex", alignItems: "flex-start", gap: space.sm, width: "100%",
    padding: `${space.xs}px ${space.xs}px`, background: "transparent",
    border: "none", borderRadius: radius.sm, font: "inherit",
    fontSize: type.size.base, textAlign: "left",
  },
  dot: {
    display: "grid", placeItems: "center", width: 18, height: 18, flexShrink: 0,
    borderRadius: radius.pill, border: "1px solid", zIndex: 1,
  },
  label: { display: "flex", flexDirection: "column", gap: 1, flex: 1, minWidth: 0 },
  title: { lineHeight: 1.3 },
  reason: { fontSize: type.size.xs, color: color.warnText, lineHeight: 1.35 },
  detail: { fontSize: type.size.xs, color: color.textMuted,
            fontFamily: type.numeric, lineHeight: 1.35 },
  badge: {
    flexShrink: 0, alignSelf: "center", fontSize: type.size.xs,
    color: color.warnText, background: color.warnSurface,
    border: `1px solid ${color.warnBorder}`, borderRadius: radius.sm,
    padding: "1px 5px", textTransform: "uppercase", letterSpacing: 0.4,
  },
};
