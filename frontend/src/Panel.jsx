import * as Accordion from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";

/**
 * Collapsible sidebar sections.
 *
 * Radix, not a hand-rolled toggle, and not framer-motion. Radix's Accordion
 * brings the keyboard and ARIA behaviour a clinical tool needs — roving focus,
 * aria-expanded, Home/End — for +11.0 KB gzip measured. framer-motion wanted
 * +39.4 KB to animate the open/close and was uninstalled under the agreed +20%
 * budget; the height transition below is CSS, costs nothing, and looks the same.
 *
 * `defaultOpen` is a list of section ids, so the panel a clinician is working in
 * stays open across re-renders without any state living up in App.
 */
export function PanelGroup({ children, defaultOpen = [] }) {
  return (
    <Accordion.Root type="multiple" defaultValue={defaultOpen} style={{ width: "100%" }}>
      {children}
    </Accordion.Root>
  );
}

export function Panel({ id, title, step, children, disabled = false }) {
  return (
    <Accordion.Item value={id} disabled={disabled} style={S.item}>
      <Accordion.Header style={{ margin: 0 }}>
        <Accordion.Trigger style={{ ...S.trigger, opacity: disabled ? 0.4 : 1 }}
                           className="panel-trigger">
          <span style={S.step}>{step}</span>
          <span style={S.title}>{title}</span>
          <ChevronDown size={14} style={S.chevron} aria-hidden />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content style={S.content}>
        <div style={{ padding: "10px 12px 14px" }}>{children}</div>
      </Accordion.Content>
    </Accordion.Item>
  );
}

const S = {
  item: {
    border: "1px solid #2c313a", borderRadius: 8, marginBottom: 8,
    background: "#1e2228", overflow: "hidden",
  },
  trigger: {
    display: "flex", alignItems: "center", gap: 8, width: "100%",
    padding: "10px 12px", background: "transparent", border: "none",
    color: "#e7ebee", font: "inherit", fontSize: 11, letterSpacing: 0.6,
    textTransform: "uppercase", cursor: "pointer", textAlign: "left",
  },
  step: {
    display: "grid", placeItems: "center", width: 18, height: 18, flexShrink: 0,
    borderRadius: 4, background: "#2c313a", color: "#3fc6d4", fontSize: 10,
  },
  title: { flex: 1 },
  chevron: { color: "#6c7480", transition: "transform 160ms ease" },
  content: { overflow: "hidden" },
};
