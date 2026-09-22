import { colorForFDI } from "./fdiPalette.js";
import { color, space, radius, type } from "./theme.js";

/**
 * Everything the software knows about ONE tooth, in one place.
 *
 * WHAT IT REPLACES. The same facts existed - the FDI came from the status
 * line, the review reason from the legend, the root length from a line under
 * the cut button, the prescription from six inputs, the segmentation geometry
 * from nowhere at all - and a clinician deciding whether to trust a label had
 * to assemble them. This is the contextual panel the workflow asks for: it
 * appears when a tooth is selected and is absent otherwise.
 *
 * THREE RULES IT KEEPS, all of them learned elsewhere in this project:
 *
 *  1. A MEASUREMENT THAT WAS NOT TAKEN IS RENDERED AS AN EM DASH, never as a
 *     zero and never as a green tick (CLAUDE.md s.14). `null` and `undefined`
 *     both take that path; `0` does not, because a genuine zero is a result.
 *  2. THE COLOUR COMES FROM `colorForFDI`, the same pure function the viewport
 *     uses, so the swatch here and the tooth on screen cannot drift. A second
 *     palette is how seventeen teeth came to share one colour (s.24.5).
 *  3. IT TAKES SUMMARIES. The per-vertex label array is 94,848 integers on a
 *     real scan and stays in a ref; this receives one row of
 *     `segmentation_diagnostics` and one tooth record.
 *
 * Nothing here is a clinical claim. Every geometric row is a plausibility
 * measurement - a tooth is one connected lump of a plausible size - and no
 * segmentation model in this app has been scored against an independent
 * annotation.
 */

const DASH = "—";

function fmt(value, digits = 2, unit = "") {
  if (value === null || value === undefined) return DASH;
  if (typeof value === "number" && !Number.isFinite(value)) return DASH;
  if (typeof value !== "number") return String(value);
  return `${value.toFixed(digits)}${unit}`;
}

function Row({ label, value, tone, title }) {
  return (
    <div style={S.row} title={title}>
      <span style={S.key}>{label}</span>
      <span style={{ ...S.val, color: tone || color.text }}>{value}</span>
    </div>
  );
}

export default function ToothInspector({
  fdi, vertexCount, faceCount, diagnostics, review, rootLengthMm,
  rootLengthSource, prescription, moved, stage,
}) {
  if (fdi == null && !vertexCount) return null;

  const swatch = fdi == null ? color.textFaint : colorForFDI(fdi);
  const frac = diagnostics?.largest_component_fraction;
  const comps = diagnostics?.components;
  // ONE CONNECTED LUMP is the plausibility claim. Below 0.95 the label is in
  // more than one piece and the selection took only the largest - a fact the
  // clinician has to know before cutting, because the rest of the tooth is
  // still on the cast.
  const split = frac != null && frac < 0.95;
  const bigBox = diagnostics?.bbox_diagonal_mm != null
    && diagnostics.bbox_diagonal_mm > 22.0;

  return (
    <section style={S.wrap} data-testid="tooth-inspector">
      <header style={S.head}>
        <span style={{ ...S.swatch, background: swatch }} aria-hidden />
        <span style={S.fdi}>{fdi == null ? "Unlabelled region" : `FDI ${fdi}`}</span>
        {moved && <span style={S.moved}>moved</span>}
      </header>

      <div style={S.group}>
        <Row label="Vertices" value={vertexCount == null ? DASH
          : vertexCount.toLocaleString()} />
        <Row label="Faces" value={faceCount == null ? DASH
          : faceCount.toLocaleString()} />
        <Row label="Box diagonal"
             value={fmt(diagnostics?.bbox_diagonal_mm, 2, " mm")}
             tone={bigBox ? color.warnText : undefined}
             title="A crown's bounding-box diagonal. A tooth spanning much more
                    than a crown is usually two teeth merged into one label." />
        <Row label="Components" value={comps == null ? DASH : comps}
             tone={comps > 1 ? color.warnText : undefined} />
        <Row label="Largest piece"
             value={frac == null ? DASH : `${(frac * 100).toFixed(1)}%`}
             tone={split ? color.warnText : undefined}
             title="The fraction of the label in its largest connected piece.
                    Selection takes that piece and discards the rest." />
      </div>

      {/* THE REVIEW VERDICT, with its reasons - not just a score. A clinician
          can disagree with a factor; they cannot disagree with a number. */}
      {review && (
        <div style={{
          ...S.verdict,
          background: review.blocks_auto_cut ? color.warnSurface : color.surfaceSunken,
          borderColor: review.blocks_auto_cut ? color.warnBorder : color.border,
        }}>
          <div style={S.verdictHead}>
            <span>{review.blocks_auto_cut ? "REVIEW REQUIRED" : "Segmentation review"}</span>
            <span style={S.verdictScore}>
              {fmt(review.confidence, 3)}
              {review.auto_cut_threshold != null
                && ` / ${fmt(review.auto_cut_threshold, 2)}`}
            </span>
          </div>
          {Array.isArray(review.failed_factors) && review.failed_factors.length > 0 && (
            <ul style={S.factors}>
              {review.failed_factors.map((f) => (
                <li key={typeof f === "string" ? f : f.name}>
                  {typeof f === "string" ? f : (f.name || JSON.stringify(f))}
                </li>
              ))}
            </ul>
          )}
          <div style={S.disclaimer}>
            Weighted agreement between independently checkable facts. It is
            NOT a model probability and not a measured accuracy.
          </div>
        </div>
      )}

      {(rootLengthMm != null || prescription) && (
        <div style={S.group}>
          <Row label="Root length" value={fmt(rootLengthMm, 1, " mm")}
               title="C_res is extrapolated along the long axis by exactly this
                      distance, so it is derived per tooth and never defaulted." />
          {rootLengthSource && (
            <Row label="Derived from" value={rootLengthSource}
                 tone={color.textMuted} />
          )}
        </div>
      )}

      {prescription && (
        <div style={S.group}>
          <div style={S.groupHead}>
            PRESCRIPTION{stage ? ` · stage ${stage}` : ""}
          </div>
          <Row label="Tip" value={fmt(prescription.tip_deg, 1, "°")} />
          <Row label="Torque" value={fmt(prescription.torque_deg, 1, "°")} />
          <Row label="Rotation" value={fmt(prescription.rotation_deg, 1, "°")} />
          <Row label="Mesiodistal" value={fmt(prescription.d_md, 2, " mm")} />
          <Row label="Buccolingual" value={fmt(prescription.d_bl, 2, " mm")} />
          <Row label="Occlusoapical" value={fmt(prescription.d_oa, 2, " mm")} />
        </div>
      )}
    </section>
  );
}

const S = {
  wrap: {
    border: `1px solid ${color.border}`, borderRadius: radius.lg,
    background: color.surface, marginBottom: space.sm, overflow: "hidden",
  },
  head: {
    display: "flex", alignItems: "center", gap: space.sm,
    padding: `${space.sm}px ${space.md - 4}px`,
    borderBottom: `1px solid ${color.border}`, background: color.surfaceRaised,
  },
  swatch: {
    width: 14, height: 14, borderRadius: radius.sm, flexShrink: 0,
    boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.45)",
  },
  fdi: {
    flex: 1, fontSize: type.size.md, fontWeight: type.weight.medium,
    color: color.text, letterSpacing: 0.2,
  },
  moved: {
    fontSize: type.size.xs, color: color.accent, background: "#13272b",
    border: `1px solid ${color.accentQuiet}`, borderRadius: radius.sm,
    padding: "1px 5px", textTransform: "uppercase", letterSpacing: 0.4,
  },
  group: {
    padding: `${space.sm}px ${space.md - 4}px`,
    borderBottom: `1px solid ${color.border}`,
  },
  groupHead: {
    fontSize: type.size.xs, letterSpacing: 0.7, color: color.textMuted,
    textTransform: "uppercase", marginBottom: space.xs,
  },
  row: {
    display: "flex", justifyContent: "space-between", alignItems: "baseline",
    gap: space.sm, fontSize: type.size.base, lineHeight: 1.7,
  },
  key: { color: color.textMuted },
  val: { fontFamily: type.numeric, fontVariantNumeric: "tabular-nums" },
  verdict: {
    margin: `${space.sm}px ${space.md - 4}px`, padding: space.sm,
    border: "1px solid", borderRadius: radius.md,
  },
  verdictHead: {
    display: "flex", justifyContent: "space-between", alignItems: "baseline",
    fontSize: type.size.xs, letterSpacing: 0.6, textTransform: "uppercase",
    color: color.warnText,
  },
  verdictScore: { fontFamily: type.numeric, letterSpacing: 0 },
  factors: {
    margin: `${space.xs}px 0 0`, paddingLeft: space.md,
    fontSize: type.size.sm, color: color.text, lineHeight: 1.5,
  },
  disclaimer: {
    marginTop: space.xs, fontSize: type.size.xs, color: color.textMuted,
    lineHeight: 1.45,
  },
};
