import { useEffect, useRef, useState, useCallback } from "react";
import * as Slider from "@radix-ui/react-slider";
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from "lucide-react";

/**
 * The staging timeline — scrub the case from T0 to the planned setup.
 *
 * WHAT DRIVES WHAT, and this is the whole performance design:
 *
 * React renders the CHROME only — the buttons, the stage label, the per-tooth
 * readout. The scrub itself writes through a ref into three.js inside a
 * requestAnimationFrame loop and never calls setState per frame. A 14-crown
 * case poses 14 matrices per frame; routing that through React state would
 * re-render the whole sidebar 60 times a second and re-run every ClinicalInput,
 * which is how a viewport drops frames on a machine that can easily draw the
 * geometry.
 *
 * This is also why framer-motion was uninstalled rather than used here: its
 * job on this screen would have been scrub transitions, and scrub transitions
 * are exactly the thing that must not go through React. It measured +39.4 KB
 * gzip of the +51.5 KB the four libraries cost together, which put the suite
 * 24.8% over a 20% budget; Radix (+11.0 KB) and lucide (+1.7 KB) stayed.
 *
 * STAGE k IS NOT AN INTERPOLATED MATRIX. It is deltaFromClinical evaluated at
 * clinical x k/N — absolute from T0, per tooth. Interpolating the 4x4
 * component-wise would not produce a rotation at all: the 3x3 block of
 * (1-t)*I + t*R is not orthonormal for any t strictly between 0 and 1, so every
 * intermediate stage would shear and scale the crown. Interpolating the six
 * clinical parameters and rebuilding the matrix keeps every stage rigid by
 * construction, which verify-kinematics.mjs asserts.
 */
export default function StagingTimeline({ totalStages, stage, playing, perTooth,
                                          onStage, onPlayPause }) {
  if (!totalStages) return null;

  const pct = totalStages > 0 ? (stage / totalStages) * 100 : 0;
  const atEnd = stage >= totalStages;

  return (
    <div style={S.bar}>
      <div style={S.controls}>
        <button style={S.btn} title="First stage (T0)"
                onClick={() => onStage(0)}><SkipBack size={15} /></button>
        <button style={S.btn} title="Previous stage" disabled={stage <= 0}
                onClick={() => onStage(stage - 1)}><ChevronLeft size={15} /></button>
        <button style={{ ...S.btn, ...S.play }} title={playing ? "Pause" : "Play"}
                onClick={onPlayPause}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button style={S.btn} title="Next stage" disabled={atEnd}
                onClick={() => onStage(stage + 1)}><ChevronRight size={15} /></button>
        <button style={S.btn} title="Last stage (planned setup)"
                onClick={() => onStage(totalStages)}><SkipForward size={15} /></button>
      </div>

      <div style={S.track}>
        <Slider.Root style={S.sliderRoot} min={0} max={totalStages} step={1}
                     value={[stage]} onValueChange={([v]) => onStage(v)}
                     aria-label="Treatment stage">
          <Slider.Track style={S.sliderTrack}>
            <Slider.Range style={{ ...S.sliderRange, width: `${pct}%` }} />
          </Slider.Track>
          <Slider.Thumb style={S.sliderThumb} />
        </Slider.Root>
        <div style={S.ticks}>
          <span>T0</span>
          <span style={S.stageLabel}>
            Stage {stage} / {totalStages}
            {stage === 0 && " — as scanned"}
            {atEnd && " — planned setup"}
          </span>
          <span>{totalStages}</span>
        </div>
      </div>

      <div style={S.legend}>
        {perTooth.length === 0 && <span style={S.muted}>no movement prescribed</span>}
        {perTooth.map((t) => (
          <span key={t.tid} style={S.chip}
                title={t.occlusion?.collides
                  ? `${t.stages} stages, ${t.driver}-driven — occlusal interference, `
                    + `${t.occlusion.max_penetration_mm}mm into the antagonist`
                  : `${t.stages} stages, ${t.driver}-driven`}>
            {/* Amber marks a tooth whose trajectory runs into the opposing arch.
                Informational: the clinician decides, exactly as with the
                reference bands and the staging counts. */}
            {t.occlusion?.collides && <span style={S.clash} aria-label="occlusal interference">▲</span>}
            <b style={{ color: t.occlusion?.collides ? "#e8a06a"
                              : t.binds ? "#3fc6d4" : "#8b93a0" }}>
              {t.fdi != null ? t.fdi : t.tid.slice(0, 4)}
            </b>
            <span style={S.muted}>
              {t.stages}× · {CHANNEL_LABEL[t.channel] || "—"}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

const CHANNEL_LABEL = {
  tip_deg: "tip", torque_deg: "torque", rotation_deg: "rotation",
  d_md: "mesiodistal", d_bl: "buccolingual", d_oa: "intrusion/extrusion",
};

/**
 * Playback clock. Lives outside React state so a running animation does not
 * re-render the tree; the caller gets one callback per stage change, not per
 * frame.
 */
export function useStagePlayback(totalStages, stage, setStage, msPerStage = 280) {
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);
  const last = useRef(0);
  const cur = useRef(stage);
  cur.current = stage;

  useEffect(() => {
    if (!playing || !totalStages) return undefined;
    last.current = performance.now();
    const tick = (now) => {
      if (now - last.current >= msPerStage) {
        last.current = now;
        const next = cur.current + 1;
        if (next > totalStages) { setPlaying(false); return; }
        setStage(next);
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, totalStages, msPerStage, setStage]);

  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p && cur.current >= totalStages) setStage(0);   // replay from T0
      return !p;
    });
  }, [totalStages, setStage]);

  return [playing, toggle, setPlaying];
}

const S = {
  bar: {
    position: "absolute", left: 0, right: 0, bottom: 0,
    display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 16,
    alignItems: "center", padding: "10px 16px",
    background: "rgba(20,23,28,0.92)", borderTop: "1px solid #2c313a",
    backdropFilter: "blur(6px)", fontSize: 12, color: "#e7ebee",
  },
  controls: { display: "flex", gap: 4, alignItems: "center" },
  btn: {
    display: "grid", placeItems: "center", width: 28, height: 28,
    background: "#22262d", color: "#c9d1d9", border: "1px solid #2c313a",
    borderRadius: 5, cursor: "pointer",
  },
  play: { background: "linear-gradient(#3fc6d4,#2a8b96)", color: "#0d0f12", width: 34 },
  track: { display: "flex", flexDirection: "column", gap: 3 },
  sliderRoot: {
    position: "relative", display: "flex", alignItems: "center",
    userSelect: "none", touchAction: "none", height: 18, width: "100%",
  },
  sliderTrack: {
    position: "relative", flexGrow: 1, height: 4,
    background: "#2c313a", borderRadius: 2,
  },
  sliderRange: { position: "absolute", height: "100%", background: "#3fc6d4", borderRadius: 2 },
  sliderThumb: {
    display: "block", width: 14, height: 14, borderRadius: "50%",
    background: "#e7ebee", border: "2px solid #3fc6d4", cursor: "grab",
  },
  ticks: { display: "flex", justifyContent: "space-between", color: "#6c7480", fontSize: 10 },
  stageLabel: { color: "#e7ebee", fontVariantNumeric: "tabular-nums" },
  legend: { display: "flex", gap: 8, flexWrap: "wrap", maxWidth: 320, justifyContent: "flex-end" },
  chip: { display: "flex", gap: 4, alignItems: "baseline", fontSize: 10 },
  muted: { color: "#6c7480" },
  clash: { color: "#e8a06a", fontSize: 8, lineHeight: 1 },
};
