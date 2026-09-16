import { SHAPES, LIMITS } from "./AttachmentPlacementTool";

/**
 * Attachment placement controls.
 *
 * Every shape shows WHAT IT IS FOR, not just its name. A clinician choosing
 * between a vertical rectangle and a horizontal bevel is choosing between
 * resisting rotation and resisting extrusion; the label alone does not say
 * that, and a list of four names is a quiz rather than a control.
 *
 * The sliders clamp to the same band the server enforces
 * (attachments.py MIN_DIM_MM / MAX_DIM_MM). Offering a size the backend refuses
 * reads to a clinician as a broken button, not as a validation message.
 */
export default function AttachmentPanel({
  enabled, active, onToggle, settings, onChange, placed, onClear, targetLabel,
}) {
  const s = settings;
  const set = (patch) => onChange({ ...s, ...patch });
  const setDim = (k, v) => onChange({ ...s, dimensions: { ...s.dimensions, [k]: v } });

  return (
    <div>
      <button
        onClick={onToggle}
        disabled={!enabled}
        data-testid="attachment-mode-toggle"
        style={{ ...S.toggle,
                 background: active ? "linear-gradient(#00d0b0,#00907a)" : "#21252b",
                 borderColor: active ? "#00d0b0" : "#2c313a" }}>
        {active ? "Placing — click the crown surface" : "Place attachment"}
      </button>

      {!enabled && (
        <div style={S.hint}>Cut and select a tooth first — an attachment is bonded
          to a crown, so there has to be one.</div>
      )}

      {active && (
        <>
          <div style={S.hint} data-testid="attachment-target">
            Target: {targetLabel || "none"}
          </div>

          <label style={S.label}>Type</label>
          {Object.entries(SHAPES).map(([key, meta]) => (
            <button
              key={key}
              data-testid={`attachment-shape-${key}`}
              onClick={() => set({ shape: key, dimensions: {
                md: meta.md, oa: meta.oa, bl: meta.bl } })}
              style={{ ...S.shape,
                       borderColor: s.shape === key ? "#00d0b0" : "#2c313a",
                       background: s.shape === key ? "#12211f" : "#181b20" }}>
              <div style={S.shapeName}>{meta.label}</div>
              {/* The mechanical reason, because that is the actual choice. */}
              <div style={S.shapeWhy}>{meta.purpose}</div>
            </button>
          ))}

          <Slider label="Height (occlusoapical)" unit="mm" value={s.dimensions.oa}
                  min={LIMITS.oa[0]} max={LIMITS.oa[1]} step={0.1}
                  testid="attachment-height" onChange={(v) => setDim("oa", v)} />
          <Slider label="Width (mesiodistal)" unit="mm" value={s.dimensions.md}
                  min={LIMITS.md[0]} max={LIMITS.md[1]} step={0.1}
                  testid="attachment-width" onChange={(v) => setDim("md", v)} />
          <Slider label="Depth (off the surface)" unit="mm" value={s.dimensions.bl}
                  min={LIMITS.bl[0]} max={LIMITS.bl[1]} step={0.05}
                  testid="attachment-depth" onChange={(v) => setDim("bl", v)} />
          <Slider label="Rotation about the normal" unit="deg" value={s.rotation_deg}
                  min={0} max={360} step={1}
                  testid="attachment-rotation" onChange={(v) => set({ rotation_deg: v })} />
        </>
      )}

      {placed?.length > 0 && (
        <div style={S.placed} data-testid="attachment-list">
          <div style={S.placedHead}>
            {placed.length} placed
            <button onClick={onClear} style={S.clear} data-testid="attachment-clear">
              clear all
            </button>
          </div>
          {placed.map((a) => (
            <div key={a.attachment_id} style={S.row}>
              {SHAPES[a.shape]?.label || a.shape}
              <span style={S.dims}>
                {a.dimensions_mm.md} x {a.dimensions_mm.oa} x {a.dimensions_mm.bl} mm
              </span>
            </div>
          ))}
          {/* Removing one from a fused solid is not a subtraction of the same
              block - the union has already merged the surfaces. */}
          <div style={S.hint}>Clearing restores the crown as it was cut; re-place
            whichever attachments are still wanted.</div>
        </div>
      )}
    </div>
  );
}

function Slider({ label, unit, value, min, max, step, onChange, testid }) {
  return (
    <div style={{ marginTop: 8 }}>
      <label style={S.label}>
        {label}
        <span style={S.value} data-testid={`${testid}-value`}>
          {Number(value).toFixed(step < 0.1 ? 2 : 1)} {unit}
        </span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value}
             data-testid={testid}
             onChange={(e) => onChange(parseFloat(e.target.value))}
             style={{ width: "100%" }} />
    </div>
  );
}

const S = {
  toggle: { width: "100%", padding: "8px 10px", marginTop: 8, borderRadius: 5,
            border: "1px solid #2c313a", color: "#e7ebee", cursor: "pointer",
            fontSize: 12, fontWeight: 600 },
  label: { display: "flex", justifyContent: "space-between", fontSize: 10.5,
           color: "#8b93a0", letterSpacing: 0.4, marginTop: 10 },
  value: { color: "#c3cad3" },
  shape: { display: "block", width: "100%", textAlign: "left", marginTop: 6,
           padding: "6px 8px", borderRadius: 4, border: "1px solid #2c313a",
           color: "#e7ebee", cursor: "pointer" },
  shapeName: { fontSize: 12, fontWeight: 600 },
  shapeWhy: { fontSize: 10, color: "#8b93a0", lineHeight: 1.35, marginTop: 2 },
  placed: { marginTop: 12, paddingTop: 8, borderTop: "1px solid #21252b" },
  placedHead: { display: "flex", justifyContent: "space-between",
                fontSize: 10.5, color: "#8b93a0", letterSpacing: 0.4 },
  clear: { background: "none", border: "none", color: "#e6194b", cursor: "pointer",
           fontSize: 10.5, padding: 0 },
  row: { display: "flex", justifyContent: "space-between", fontSize: 11.5,
         color: "#c3cad3", marginTop: 4 },
  dims: { color: "#6b727d", fontSize: 10.5 },
  hint: { fontSize: 10.5, color: "#6b727d", lineHeight: 1.4, marginTop: 6 },
};
