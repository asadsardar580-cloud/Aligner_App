/**
 * Case validation readout.
 *
 * THE RULE THIS COMPONENT EXISTS TO ENFORCE: a tick mark is a claim, and the
 * software may only make it about something it actually computed.
 *
 * There are three states, not two. "Pass", "needs review", and "not
 * determined" — and the third is rendered in neutral grey with an em dash,
 * never green. The antagonist check in particular is skipped whenever the
 * opposing arch was never loaded, and showing that as a green tick would tell a
 * clinician the bite was checked when nothing looked at it.
 *
 * Nothing here is a clinical guarantee. A passing row means a geometric or
 * bookkeeping condition held, not that a movement is safe.
 */

export const PASS = "pass";
export const REVIEW = "review";
export const FAIL = "fail";
export const UNKNOWN = "unknown";   // not computed — NEVER shown as success

const ICON = { [PASS]: "✓", [REVIEW]: "⚠", [FAIL]: "✕", [UNKNOWN]: "—" };
const COLOUR = {
  [PASS]: "#3cb44b",
  [REVIEW]: "#ffa53c",
  [FAIL]: "#e6194b",
  [UNKNOWN]: "#6b727d",
};

export default function ValidationPanel({ checks }) {
  if (!checks?.length) return null;
  const counts = checks.reduce((a, c) => ({ ...a, [c.state]: (a[c.state] || 0) + 1 }), {});

  return (
    <div style={S.wrap}>
      <div style={S.head}>
        CASE VALIDATION
        <span style={S.counts}>
          {counts[PASS] ? <span style={{ color: COLOUR[PASS] }}>{counts[PASS]} ok</span> : null}
          {counts[REVIEW] ? <span style={{ color: COLOUR[REVIEW] }}> · {counts[REVIEW]} review</span> : null}
          {counts[FAIL] ? <span style={{ color: COLOUR[FAIL] }}> · {counts[FAIL]} blocked</span> : null}
          {counts[UNKNOWN] ? <span style={{ color: COLOUR[UNKNOWN] }}> · {counts[UNKNOWN]} not run</span> : null}
        </span>
      </div>

      {checks.map((c) => (
        <div key={c.id} style={S.row} title={c.detail || ""}>
          <span style={{ ...S.icon, color: COLOUR[c.state] }}>{ICON[c.state]}</span>
          <span style={S.label}>
            {c.label}
            {c.detail ? <span style={S.detail}>{c.detail}</span> : null}
            {c.basis ? <span style={S.basis}>{c.basis}</span> : null}
          </span>
        </div>
      ))}

      <div style={S.foot}>
        Geometric and bookkeeping checks only. Passing rows are not a statement of
        clinical safety, and a grey dash means the check did not run — not that it passed.
      </div>
    </div>
  );
}

const S = {
  wrap: { background: "#101215", border: "1px solid #2c313a", borderRadius: 5,
          padding: "9px 10px", marginTop: 10, fontSize: 11.5, lineHeight: 1.45 },
  head: { display: "flex", justifyContent: "space-between", alignItems: "baseline",
          color: "#8b93a0", letterSpacing: 0.6, fontSize: 10, marginBottom: 7 },
  counts: { fontSize: 10 },
  row: { display: "flex", gap: 7, alignItems: "flex-start", padding: "2px 0" },
  icon: { width: 11, flexShrink: 0, fontWeight: 700 },
  label: { color: "#c3cad3" },
  detail: { color: "#8b93a0", display: "block", fontSize: 10.5 },
  basis: { color: "#6b727d", display: "block", fontSize: 9.5, fontStyle: "italic" },
  foot: { color: "#6b727d", fontSize: 9.5, marginTop: 8, paddingTop: 6,
          borderTop: "1px solid #21252b", lineHeight: 1.4 },
};
