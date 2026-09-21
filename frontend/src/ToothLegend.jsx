import { useMemo } from "react";
import { colorForFDI, GINGIVA_HEX, UNKNOWN_HEX } from "./fdiPalette.js";
import { color, space, radius, type, transition } from "./theme.js";

/**
 * Which colour is which tooth.
 *
 * WHY THIS IS WORTH A COMPONENT. Thirty-two colours on a cast are only useful
 * if the clinician can turn one back into an FDI number, and until now there
 * was nowhere to look it up - the colours were assigned by array position
 * inside `App.jsx` and never rendered anywhere else. It also makes the palette
 * INSPECTABLE: a duplicate or an invisible colour is obvious in a legend and
 * invisible in a viewport, which is how seventeen teeth came to share a
 * colour without anyone noticing.
 *
 * IT SHOWS WHAT SEGMENTATION FOUND, NOT THE WHOLE DENTITION. Listing all 32
 * teeth on a scan that has twelve would invite a clinician to look for teeth
 * that were never labelled. Absent teeth are simply not rows.
 *
 * PRESENTATION ONLY. It reads labels and draws swatches; it changes nothing.
 */
export default function ToothLegend({
  counts,
  diagnostics,
  selectedFDI = null,
  onSelect = null,
}) {
  // THE PROP IS A SUMMARY, NOT THE LABEL ARRAY, and that is deliberate. The
  // per-vertex labels are 94,848 integers on a real scan and live in a REF in
  // `App.jsx`, never in React state - the same rule that keeps three.js
  // transforms and stage scrubbing off the render path. A component that took
  // the raw array would drag it into a dependency list and into every
  // reconciliation. `counts` is one entry per labelled tooth: at most 32.
  const rows = useMemo(() => {
    const entries = counts instanceof Map ? [...counts.entries()]
      : Array.isArray(counts) ? counts : [];
    const flags = new Map();
    for (const t of diagnostics?.teeth || []) flags.set(t.label, t);
    return entries
      .filter(([fdi]) => Number(fdi) > 0)     // 0 is gingiva, not a tooth
      .map(([fdi, n]) => [Number(fdi), n])
      .sort((a, b) => a[0] - b[0])
      .map(([fdi, n]) => ({ fdi, vertices: n, diag: flags.get(fdi) || null }));
  }, [counts, diagnostics]);

  if (!rows.length) return null;

  const suspect = rows.filter(
    (r) => r.diag && (!r.diag.plausible_size || !r.diag.connected));

  return (
    <div>
      <div style={S.head}>
        {rows.length} labelled {rows.length === 1 ? "tooth" : "teeth"}
      </div>
      <div style={S.grid} role="list">
        {rows.map(({ fdi, vertices, diag }) => {
          const hex = colorForFDI(fdi);
          const isSel = selectedFDI === fdi;
          // A REVIEW FLAG IS A BADGE, NOT A COLOUR CHANGE. Recolouring the
          // swatch would break the one thing the legend is for.
          const flagged = diag && (!diag.plausible_size || !diag.connected);
          return (
            <button
              key={fdi}
              role="listitem"
              type="button"
              data-testid={`legend-${fdi}`}
              onClick={onSelect ? () => onSelect(fdi) : undefined}
              title={
                diag
                  ? `FDI ${fdi} — ${vertices} vertices, `
                    + `${diag.bbox_diagonal_mm} mm across, `
                    + `${diag.components} `
                    + `${diag.components === 1 ? "piece" : "pieces"}`
                    + (flagged ? " — needs review" : "")
                  : `FDI ${fdi} — ${vertices} vertices`
              }
              style={{
                ...S.cell,
                cursor: onSelect ? "pointer" : "default",
                borderColor: isSel ? color.selection : color.border,
                background: isSel
                  ? "rgba(63,169,255,0.12)" : "rgba(255,255,255,0.02)",
                transition: transition("border-color, background"),
              }}
            >
              <span style={{ ...S.swatch, background: hex }} aria-hidden />
              <span style={S.fdi}>{fdi}</span>
              {flagged && <span style={S.flag} aria-hidden>!</span>}
            </button>
          );
        })}
      </div>

      <div style={S.foot}>
        <span style={{ ...S.swatch, background: GINGIVA_HEX }} aria-hidden />
        <span>gingiva</span>
        <span style={{ ...S.swatch, background: UNKNOWN_HEX, marginLeft: space.sm }}
              aria-hidden />
        <span>unlabelled</span>
      </div>

      {suspect.length > 0 && (
        <div style={S.review}>
          <b>{suspect.length} needs review:</b>{" "}
          {suspect.map((r) => `FDI ${r.fdi}`).join(", ")}
          <div style={S.reviewWhy}>
            {/* THE REASON, NOT JUST THE FLAG. A badge with no reason is a
                badge a clinician learns to ignore. */}
            {suspect.map((r) => (
              <div key={r.fdi}>
                {r.fdi}: {r.diag.bbox_diagonal_mm} mm across
                {r.diag.components > 1
                  ? `, ${r.diag.components} disconnected pieces `
                    + `(largest ${Math.round(
                        (r.diag.largest_component_fraction || 0) * 100)}%)`
                  : ""}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const S = {
  head: {
    color: color.textMuted, fontSize: type.size.sm,
    marginBottom: space.sm,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(52px, 1fr))",
    gap: space.xs,
  },
  cell: {
    display: "flex", alignItems: "center", gap: space.xs,
    padding: `${space.xs}px ${space.xs + 2}px`,
    border: `1px solid ${color.border}`,
    borderRadius: radius.sm,
    color: color.text, fontSize: type.size.sm,
    fontFamily: "inherit", textAlign: "left",
  },
  swatch: {
    width: 10, height: 10, borderRadius: 2, flexShrink: 0,
    boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.45)",
  },
  fdi: { fontVariantNumeric: "tabular-nums" },
  flag: { color: color.warn, fontWeight: type.weight.bold, marginLeft: "auto" },
  foot: {
    display: "flex", alignItems: "center", gap: space.xs,
    marginTop: space.sm, color: color.textFaint, fontSize: type.size.xs,
  },
  review: {
    marginTop: space.sm, padding: `${space.sm}px ${space.sm + 2}px`,
    borderRadius: radius.sm, background: color.warnSurface,
    border: `1px solid ${color.warnBorder}`, color: color.warnText,
    fontSize: type.size.xs, lineHeight: 1.5,
  },
  reviewWhy: { marginTop: space.xs, color: color.warn },
};
