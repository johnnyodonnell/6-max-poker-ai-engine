import type { BotDecision } from "../engine/neural";
import "./DebugPanel.css";

interface DebugPanelProps {
  enabled: boolean;
  onToggle: () => void;
  decisions: BotDecision[];
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

// The debug view: a toggle, and (when on) the bots' decisions this hand —
// who decided on which street, the legal options with the net's probability
// of each, and what was picked. The bots' cards are shown face up on the
// table while it's on (see App).
export default function DebugPanel({ enabled, onToggle, decisions }: DebugPanelProps) {
  return (
    <div className="debug">
      <button className={`debug-toggle ${enabled ? "on" : ""}`} onClick={onToggle}>
        Debug: {enabled ? "on" : "off"}
      </button>
      {enabled && (
        <div className="debug-panel">
          <div className="debug-title">
            Bots' decisions{decisions.length ? ` — hand #${decisions[0].hand}` : ""}
          </div>
          {decisions.length === 0 && <div className="debug-empty">None yet this hand.</div>}
          {decisions.map((d, i) => (
            <div key={i} className="debug-decision">
              <div className="debug-head">
                <span className="debug-name">{d.name}</span>
                <span className="debug-dim"> · </span>
                <span className="debug-street">{d.street}</span>
              </div>
              {d.fallback ? (
                <div className="debug-fallback">The net couldn't run ({d.fallback}); played randomly.</div>
              ) : (
                <table className="debug-options">
                  <tbody>
                    {d.options.map((o, j) => (
                      <tr key={j} className={o.prob >= 0.005 ? "" : "debug-zero"}>
                        <td className="debug-opt">{o.label}</td>
                        <td className="debug-bar">
                          <div style={{ width: pct(o.prob) }} />
                        </td>
                        <td className="debug-prob">{pct(o.prob)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="debug-chosen">→ {d.chosen}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
