import { useEffect, useRef, useState, useCallback } from "react";

/**
 * Playback clock for the staging timeline.
 *
 * Lives outside React state so a running animation does not re-render the tree;
 * the caller gets one callback per stage change, not per frame. See
 * StagingTimeline.jsx for why the scrub must never go through React state.
 *
 * Extracted from StagingTimeline.jsx so that file exports only a component.
 * Mixing a hook and a component in one module breaks Vite's Fast Refresh —
 * editing either one forces a full reload instead of a hot update.
 */
export function useStagePlayback(totalStages, stage, setStage, msPerStage = 280) {
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);
  const last = useRef(0);

  // `cur` is a latest-value mirror of `stage`, so the rAF tick below can read
  // the current stage without the effect re-subscribing on every stage change
  // (which would cancel and restart the animation frame 30-odd times a run).
  //
  // The mirror is written in an EFFECT, not during render. React may render a
  // component and then discard that render; writing a ref during render would
  // publish a value for a render that never committed. After commit is also
  // exactly when the stage is real on screen, and the tick only advances every
  // msPerStage (280ms, ~17 frames), so the write always lands first.
  const cur = useRef(stage);
  useEffect(() => { cur.current = stage; }, [stage]);

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
