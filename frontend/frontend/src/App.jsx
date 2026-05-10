import { useEffect, useMemo, useState, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, ShieldAlert, Brain, AlertTriangle,
  Database, Wifi, WifiOff, ChevronUp, ChevronDown,
  Minus, Zap, Target, BarChart2, ChevronRight,
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, Cell,
  CartesianGrid, Tooltip, XAxis, YAxis,
} from "recharts";

// ─── constants ────────────────────────────────────────────────────────────────
const API_URL = "http://localhost:8000";
const WS_URL  = "ws://localhost:8000/ws";

// ─── design tokens ────────────────────────────────────────────────────────────
const T = {
  bg:        "#ECEAE4",
  surface:   "#FFFFFF",
  surfaceAlt:"#F7F5F0",
  border:    "#D4D1CA",
  borderFt:  "#C2BEB5",
  text:      "#0A0908",
  textMid:   "#332F2A",
  textDim:   "#5C5850",
  green:     "#0F5132",
  greenBg:   "#D8EDDF",
  greenLine: "#198754",
  red:       "#7F1010",
  redBg:     "#FDDDDD",
  redLine:   "#C82020",
  amber:     "#6B2D0A",
  amberBg:   "#FEF0C7",
  blue:      "#143285",
  blueBg:    "#E8EEFF",
  purple:    "#4C1D95",
  purpleBg:  "#EDE9FE",
  mono:      "'IBM Plex Mono', 'Courier New', monospace",
  sans:      "'DM Sans', 'Helvetica Neue', sans-serif",
};

// ─── severity map ─────────────────────────────────────────────────────────────
const SEV = {
  CRITICAL: { color: T.red,    bg: T.redBg,    label: "CRIT"   },
  HIGH:     { color: T.amber,  bg: T.amberBg,  label: "HIGH"   },
  MEDIUM:   { color: T.blue,   bg: T.blueBg,   label: "MED"    },
  LOW:      { color: T.green,  bg: T.greenBg,  label: "LOW"    },
};
const sev = (s) => SEV[s?.toUpperCase()] || SEV.LOW;

// ─── safe parsers ─────────────────────────────────────────────────────────────
const safeNum  = (v, fb = 0)  => { const n = Number(v); return isFinite(n) ? n : fb; };
const safeJSON = (v)           => { try { return typeof v === "string" ? JSON.parse(v) : (Array.isArray(v) ? v : []); } catch { return []; } };

// ─── chart tooltip ────────────────────────────────────────────────────────────
function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 6, padding: "8px 12px",
      fontFamily: T.mono, fontSize: 11, color: T.text,
      boxShadow: "0 4px 16px rgba(0,0,0,0.10)",
    }}>
      <div style={{ color: T.textDim, marginBottom: 4 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.name === "anomaly" ? T.red : T.green, fontWeight: 600 }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
}

// ─── spark bars ───────────────────────────────────────────────────────────────
function SparkBar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ display: "flex", alignItems: "flex-end", height: 26, gap: 1.5 }}>
      {Array.from({ length: 12 }).map((_, i) => {
        const threshold = (i / 12) * 100;
        const active = pct >= threshold;
        return (
          <motion.div
            key={i}
            animate={{ height: active ? 7 + (i / 12) * 18 : 3, opacity: active ? 1 : 0.10 }}
            transition={{ duration: 0.3, delay: i * 0.018 }}
            style={{ width: 3, borderRadius: 2, background: active ? color : T.borderFt }}
          />
        );
      })}
    </div>
  );
}

// ─── ticker bar ───────────────────────────────────────────────────────────────
function TickerBar({ alerts }) {
  const items = alerts.slice(0, 24);
  if (!items.length) return null;
  return (
    <div style={{ background: T.text, overflow: "hidden", height: 30, display: "flex", alignItems: "center" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 40, animation: "ticker 28s linear infinite", whiteSpace: "nowrap" }}>
        {[...items, ...items].map((a, i) => (
          <span key={i} style={{
            fontFamily: T.mono, fontSize: 11,
            color: a.is_anomaly ? "#FCA5A5" : "#86EFAC",
            display: "flex", alignItems: "center", gap: 6,
          }}>
            {a.is_anomaly ? "▲" : "▼"}
            {a.severity || "LOW"}
            {a.attack_category ? ` · ${a.attack_category}` : ""}
            &nbsp;·&nbsp;
            xgb {(safeNum(a.confidence) * 100).toFixed(0)}%
          </span>
        ))}
      </div>
      <style>{`@keyframes ticker { from{transform:translateX(0)} to{transform:translateX(-50%)} }`}</style>
    </div>
  );
}

// ─── metric card ──────────────────────────────────────────────────────────────
function MetricCard({ title, value, subValue, icon, accent, delta, highlight }) {
  return (
    <div style={{
      background: highlight ? T.text : T.surface,
      border: `1px solid ${highlight ? "transparent" : T.border}`,
      borderRadius: 12, padding: "16px 18px",
      display: "flex", flexDirection: "column", gap: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ fontSize: 10, fontFamily: T.sans, fontWeight: 700, color: highlight ? "#8A8680" : T.textDim, letterSpacing: ".06em", textTransform: "uppercase" }}>
          {title}
        </div>
        <div style={{ color: highlight ? "#555" : T.textDim, opacity: 0.7 }}>{icon}</div>
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
        <div style={{ fontSize: 28, fontFamily: T.mono, fontWeight: 700, color: highlight ? "#FFFFFF" : (accent || T.text), lineHeight: 1 }}>
          {value}
        </div>
        {delta != null && (
          <div style={{ fontSize: 11, fontFamily: T.mono, fontWeight: 700, color: delta > 0 ? T.red : T.green, marginBottom: 2, display: "flex", alignItems: "center", gap: 1 }}>
            {delta > 0 ? <ChevronUp size={11} /> : delta < 0 ? <ChevronDown size={11} /> : <Minus size={11} />}
            {Math.abs(delta)}
          </div>
        )}
      </div>
      {subValue && (
        <div style={{ fontSize: 10, fontFamily: T.mono, color: highlight ? "#5C5850" : T.textDim, marginTop: -2 }}>
          {subValue}
        </div>
      )}
    </div>
  );
}

// ─── section header ───────────────────────────────────────────────────────────
function SectionHead({ title, icon, badge, accent }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 18px", borderBottom: `1px solid ${T.border}`, background: T.surface }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
        <span style={{ color: accent || T.textDim }}>{icon}</span>
        <span style={{ fontFamily: T.sans, fontWeight: 700, fontSize: 13, color: T.text }}>{title}</span>
      </div>
      {badge != null && (
        <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600, background: T.bg, border: `1px solid ${T.border}`, borderRadius: 6, padding: "2px 8px", color: T.textMid }}>
          {badge}
        </span>
      )}
    </div>
  );
}

// ─── live clock ───────────────────────────────────────────────────────────────
function LiveClock() {
  const [t, setT] = useState(() => new Date().toLocaleTimeString());
  useEffect(() => {
    const id = setInterval(() => setT(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);
  return <span style={{ fontFamily: T.mono, fontSize: 12, fontWeight: 600, color: T.textMid, letterSpacing: ".04em" }}>{t}</span>;
}

// ─── SHAP features panel ──────────────────────────────────────────────────────
// features: array of { feature, importance }
function ShapPanel({ features, title }) {
  if (!features?.length) return (
    <div style={{ padding: "20px 18px", fontFamily: T.mono, fontSize: 11, color: T.textDim }}>
      No feature data yet…
    </div>
  );
  const maxImp = Math.max(...features.map(f => Math.abs(safeNum(f.importance))));
  return (
    <div style={{ padding: "12px 18px 14px" }}>
      {features.slice(0, 8).map((f, i) => {
        const imp  = safeNum(f.importance);
        const pct  = maxImp > 0 ? Math.abs(imp) / maxImp * 100 : 0;
        const color = imp >= 0 ? T.red : T.blue; // positive = pushes toward anomaly
        return (
          <div key={i} style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600, color: T.textMid }}>
                {f.feature}
              </span>
              <span style={{ fontFamily: T.mono, fontSize: 11, color: T.textDim }}>
                {imp >= 0 ? "+" : ""}{imp.toFixed(5)}
              </span>
            </div>
            <div style={{ height: 5, background: T.border, borderRadius: 99, overflow: "hidden" }}>
              <motion.div
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.5, delay: i * 0.04 }}
                style={{ height: "100%", background: color, borderRadius: 99 }}
              />
            </div>
          </div>
        );
      })}
      <div style={{ marginTop: 8, display: "flex", gap: 14, fontFamily: T.mono, fontSize: 10, color: T.textDim }}>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 8, height: 3, background: T.red, display: "inline-block", borderRadius: 1 }} />
          pushes anomaly
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 8, height: 3, background: T.blue, display: "inline-block", borderRadius: 1 }} />
          pushes normal
        </span>
      </div>
    </div>
  );
}

// ─── attack category bar chart ────────────────────────────────────────────────
function AttackCatChart({ data }) {
  if (!data?.length) return (
    <div style={{ padding: "20px 18px", fontFamily: T.mono, fontSize: 11, color: T.textDim }}>
      Waiting for attack categories…
    </div>
  );
  const COLORS = [T.red, T.amber, T.blue, T.purple, T.green, "#0E7490", "#BE185D"];
  return (
    <div style={{ padding: "12px 18px 14px" }}>
      {data.map((d, i) => {
        const pct = data[0].count > 0 ? (d.count / data[0].count) * 100 : 0;
        return (
          <div key={d.cat} style={{ marginBottom: 9 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600, color: T.textMid }}>{d.cat}</span>
              <span style={{ fontFamily: T.mono, fontSize: 11, color: T.textDim }}>{d.count}</span>
            </div>
            <div style={{ height: 5, background: T.border, borderRadius: 99, overflow: "hidden" }}>
              <motion.div
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.5, delay: i * 0.04 }}
                style={{ height: "100%", background: COLORS[i % COLORS.length], borderRadius: 99 }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── live alert row ───────────────────────────────────────────────────────────
function AlertRow({ alert, onClick, selected }) {
  const s     = sev(alert.severity);
  const conf  = safeNum(alert.confidence);
  const score = safeNum(alert.anomaly_score);
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onClick}
      style={{
        display: "grid",
        gridTemplateColumns: "76px 68px 1fr 130px 90px 110px 72px",
        alignItems: "center",
        padding: "0 18px",
        height: 42,
        borderBottom: `1px solid ${T.border}`,
        background: selected ? T.blueBg : alert.is_anomaly ? `${T.redBg}CC` : T.surface,
        borderLeft: `3px solid ${alert.is_anomaly ? T.red : T.green}`,
        fontFamily: T.mono,
        fontSize: 11,
        cursor: "pointer",
        transition: "background .1s",
      }}
    >
      <span style={{ color: T.textDim }}>{alert.timestamp}</span>

      <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontWeight: 700, color: alert.is_anomaly ? T.red : T.green }}>
        <span style={{ width: 5, height: 5, borderRadius: "50%", background: alert.is_anomaly ? T.red : T.green }} />
        {alert.is_anomaly ? "THREAT" : "NORMAL"}
      </span>

      {/* src → dst with protocol */}
      <span style={{ color: T.textMid, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {alert.src_ip || "—"}
        <span style={{ color: T.textDim }}> → </span>
        {alert.dst_ip || "—"}
        {alert.protocol ? <span style={{ color: T.textDim }}> · {alert.protocol}</span> : ""}
      </span>

      {/* attack category */}
      <span style={{ color: T.textDim, fontSize: 10, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {alert.attack_category || "—"}
      </span>

      {/* anomaly score sparkbar */}
      <SparkBar value={score} max={1} color={alert.is_anomaly ? T.red : T.green} />

      {/* confidence bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ flex: 1, height: 4, background: T.border, borderRadius: 99, overflow: "hidden" }}>
          <motion.div
            animate={{ width: `${(conf * 100).toFixed(0)}%` }}
            transition={{ duration: 0.4 }}
            style={{ height: "100%", borderRadius: 99, background: conf > 0.7 ? T.red : conf > 0.4 ? "#C57A0A" : T.green }}
          />
        </div>
        <span style={{ fontSize: 11, color: T.textMid, width: 30, textAlign: "right", fontWeight: 700 }}>
          {(conf * 100).toFixed(0)}%
        </span>
      </div>

      {/* severity badge */}
      <div style={{ textAlign: "right" }}>
        <span style={{ display: "inline-block", fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4, letterSpacing: ".05em", background: s.bg, color: s.color }}>
          {s.label}
        </span>
      </div>
    </motion.div>
  );
}

// ─── live alert detail drawer ─────────────────────────────────────────────────
function AlertDrawer({ alert, onClose }) {
  if (!alert) return null;
  const features = safeJSON(alert.top_features);
  const s = sev(alert.severity);
  const fields = [
    ["src ip",        alert.src_ip       || "—"],
    ["dst ip",        alert.dst_ip       || "—"],
    ["src port",      alert.src_port     ?? "—"],
    ["dst port",      alert.dst_port     ?? "—"],
    ["protocol",      alert.protocol     || "—"],
    ["service",       alert.service      || "—"],
    ["state",         alert.state        || "—"],
    ["duration",      alert.duration != null ? `${safeNum(alert.duration).toFixed(4)}s` : "—"],
    ["src bytes",     alert.src_bytes    ?? "—"],
    ["dst bytes",     alert.dst_bytes    ?? "—"],
    ["src packets",   alert.src_packets  ?? "—"],
    ["dst packets",   alert.dst_packets  ?? "—"],
    ["src load",      alert.src_load != null ? safeNum(alert.src_load).toFixed(2) : "—"],
    ["dst load",      alert.dst_load != null ? safeNum(alert.dst_load).toFixed(2) : "—"],
    ["attack cat",    alert.attack_category || "—"],
    ["anomaly score", safeNum(alert.anomaly_score).toFixed(6)],
    ["threshold",     safeNum(alert.threshold).toFixed(6)],
    ["xgb prob",      (safeNum(alert.confidence) * 100).toFixed(2) + "%"],
  ];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      style={{
        background: T.surface, border: `1px solid ${T.border}`,
        borderRadius: 12, overflow: "hidden", marginTop: 12,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "11px 18px", borderBottom: `1px solid ${T.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: T.sans, fontWeight: 700, fontSize: 13, color: T.text }}>Packet Inspector</span>
          <span style={{ display: "inline-block", fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4, background: s.bg, color: s.color }}>
            {s.label}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{ background: "none", border: "none", cursor: "pointer", color: T.textDim, fontSize: 18, lineHeight: 1, padding: "0 4px" }}
        >×</button>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", padding: "4px 0" }}>
        {fields.map(([k, v], i) => (
          <div key={k} style={{ padding: "7px 18px", borderBottom: `1px solid ${T.border}`, borderRight: i % 2 === 0 ? `1px solid ${T.border}` : "none" }}>
            <div style={{ fontSize: 10, color: T.textDim, letterSpacing: ".04em", fontFamily: T.sans, marginBottom: 2 }}>{k}</div>
            <div style={{ fontSize: 12, fontWeight: 600, fontFamily: T.mono, color: T.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{String(v)}</div>
          </div>
        ))}
      </div>
      {features.length > 0 && (
        <>
          <div style={{ padding: "10px 18px 4px", borderTop: `1px solid ${T.border}` }}>
            <span style={{ fontFamily: T.sans, fontWeight: 700, fontSize: 12, color: T.textMid }}>SHAP Top Features</span>
          </div>
          <ShapPanel features={features} />
        </>
      )}
    </motion.div>
  );
}

// ─── historical row ───────────────────────────────────────────────────────────
function HistRow({ alert, index, onSelect, selected }) {
  const s     = sev(alert.severity);
  const conf  = safeNum(alert.xgb_probability);
  const score = safeNum(alert.anomaly_score);
  const thr   = safeNum(alert.threshold);
  const features = safeJSON(alert.top_features);
  return (
    <>
      <div
        onClick={onSelect}
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 80px 86px 86px 76px 64px",
          alignItems: "center",
          padding: "0 18px",
          height: 38,
          borderBottom: `1px solid ${T.border}`,
          background: selected ? T.blueBg : alert.is_anomaly ? `${T.redBg}88` : T.surface,
          borderLeft: `3px solid ${alert.is_anomaly ? T.red : T.green}`,
          fontFamily: T.mono, fontSize: 11,
          cursor: "pointer", transition: "background .1s",
        }}
      >
        <span style={{ color: T.textMid, fontWeight: 500 }}>
          {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : "Stored Alert"}
        </span>
        <span style={{ color: T.textMid, textAlign: "right", fontWeight: 600 }}>{score.toFixed(4)}</span>
        <span style={{ color: T.textDim, textAlign: "right" }}>{thr.toFixed(4)}</span>
        <span style={{ textAlign: "right", color: T.textMid, fontWeight: 600 }}>{(conf * 100).toFixed(1)}%</span>
        <div style={{ textAlign: "right" }}>
          <span style={{ display: "inline-block", fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: s.bg, color: s.color }}>{s.label}</span>
        </div>
        <div style={{ textAlign: "right", color: T.textDim }}>
          {features.length > 0 && <ChevronRight size={12} style={{ transform: selected ? "rotate(90deg)" : "none", transition: "transform .2s" }} />}
        </div>
      </div>
      {selected && features.length > 0 && (
        <div style={{ background: T.surfaceAlt, borderBottom: `1px solid ${T.border}`, borderLeft: `3px solid ${T.blue}`, padding: "4px 0 4px 6px" }}>
          <div style={{ padding: "6px 18px 2px", fontFamily: T.sans, fontSize: 10, fontWeight: 700, color: T.textDim, letterSpacing: ".06em", textTransform: "uppercase" }}>
            SHAP Features
          </div>
          <ShapPanel features={features} />
        </div>
      )}
    </>
  );
}

// ─── anomaly score vs threshold gauge ────────────────────────────────────────
function ScoreGauge({ score, threshold }) {
  const hasData = score > 0 || threshold > 0;
  const maxVal  = Math.max(score, threshold, 1) * 1.2;
  const sPct    = Math.min(100, (score     / maxVal) * 100);
  const tPct    = Math.min(100, (threshold / maxVal) * 100);
  const over    = score > threshold;
  return (
    <div style={{ padding: "14px 18px" }}>
      {!hasData ? (
        <div style={{ fontFamily: T.mono, fontSize: 11, color: T.textDim }}>Awaiting data…</div>
      ) : (
        <>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontFamily: T.mono, fontSize: 11, color: T.textMid, fontWeight: 600 }}>
              Score: <span style={{ color: over ? T.red : T.green }}>{score.toFixed(5)}</span>
            </span>
            <span style={{ fontFamily: T.mono, fontSize: 11, color: T.textDim }}>
              Threshold: {threshold.toFixed(5)}
            </span>
          </div>
          {/* score bar */}
          <div style={{ position: "relative", height: 12, background: T.border, borderRadius: 99, overflow: "hidden", marginBottom: 6 }}>
            <motion.div
              animate={{ width: `${sPct}%` }}
              transition={{ duration: 0.4 }}
              style={{ height: "100%", background: over ? T.red : T.green, borderRadius: 99 }}
            />
          </div>
          {/* threshold marker line */}
          <div style={{ position: "relative", height: 6, marginBottom: 8 }}>
            <div style={{ position: "absolute", left: `${tPct}%`, top: 0, width: 2, height: 14, background: T.amber, borderRadius: 1, transform: "translateX(-50%) translateY(-4px)" }} />
          </div>
          <div style={{ display: "flex", gap: 14, fontFamily: T.mono, fontSize: 10, color: T.textDim }}>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 8, height: 3, background: over ? T.red : T.green, display: "inline-block", borderRadius: 1 }} />
              anomaly score
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 3, height: 10, background: T.amber, display: "inline-block", borderRadius: 1 }} />
              threshold
            </span>
          </div>
        </>
      )}
    </div>
  );
}

// ─── main app ─────────────────────────────────────────────────────────────────
export default function App() {

  // ── original state ─────────────────────────────────────────────────────────
  const [alerts,           setAlerts]           = useState([]);
  const [historicalAlerts, setHistoricalAlerts] = useState([]);
  const [trafficData,      setTrafficData]      = useState([]);
  const [wsStatus,         setWsStatus]         = useState("CONNECTING");

  // ── UI state ───────────────────────────────────────────────────────────────
  const [selectedLive, setSelectedLive]   = useState(null);  // live row for drawer
  const [selectedHist, setSelectedHist]   = useState(null);  // hist row index

  // ── running counters (refs — never reset) ──────────────────────────────────
  const packetCount = useRef(0);
  const threatCount = useRef(0);
  const critCount   = useRef(0);
  const apsAccum    = useRef(0);

  // ── snapshot state — flushed every 1 s ────────────────────────────────────
  const [liveCounters, setLiveCounters] = useState({ packets: 0, threats: 0, critical: 0, alertsPerSec: 0, prevAps: 0 });

  useEffect(() => {
    const id = setInterval(() => {
      const aps = apsAccum.current;
      apsAccum.current = 0;
      setLiveCounters(prev => ({
        packets:      packetCount.current,
        threats:      threatCount.current,
        critical:     critCount.current,
        alertsPerSec: aps,
        prevAps:      prev.alertsPerSec,
      }));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── LOAD HISTORICAL ALERTS (original logic) ────────────────────────────────
  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        const res = await axios.get(`${API_URL}/alerts`);
        setHistoricalAlerts(res.data || []);
      } catch (err) {
        console.log(err);
      }
    };
    fetchHistorical();
    const interval = setInterval(fetchHistorical, 5000);
    return () => clearInterval(interval);
  }, []);

  // ── WEBSOCKET STREAM (original logic, extended packet fields) ─────────────
  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socket.onopen  = () => setWsStatus("LIVE");
    socket.onerror = () => setWsStatus("ERROR");
    socket.onclose = () => setWsStatus("DISCONNECTED");
    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type !== "traffic_update") return;

      const d = msg.data;
      const packet = {
        id:               crypto.randomUUID(),
        timestamp:        new Date().toLocaleTimeString(),
        // core fields (original)
        severity:         d.severity      || "LOW",
        is_anomaly:       Boolean(d.is_anomaly),
        anomaly_score:    safeNum(d.anomaly_score),
        threshold:        safeNum(d.threshold),
        confidence:       safeNum(d.xgb_probability),
        top_features:     d.top_features  || [],
        // packet fields from UNSW-NB15
        src_ip:           d.src_ip        || null,
        dst_ip:           d.dst_ip        || null,
        src_port:         d.src_port      ?? null,
        dst_port:         d.dst_port      ?? null,
        protocol:         d.protocol      || null,
        service:          d.service       || null,
        state:            d.state         || null,
        duration:         d.duration      ?? null,
        src_bytes:        d.src_bytes     ?? null,
        dst_bytes:        d.dst_bytes     ?? null,
        src_packets:      d.src_packets   ?? null,
        dst_packets:      d.dst_packets   ?? null,
        src_load:         d.src_load      ?? null,
        dst_load:         d.dst_load      ?? null,
        attack_category:  d.attack_category || null,
      };

      // running counters
      packetCount.current += 1;
      apsAccum.current    += 1;
      if (packet.is_anomaly)                             threatCount.current += 1;
      if (packet.severity?.toUpperCase() === "CRITICAL") critCount.current   += 1;

      // original state updates
      setAlerts(prev => [packet, ...prev.slice(0, 99)]);
      setTrafficData(prev => [...prev, {
        time:    packet.timestamp,
        normal:  packet.is_anomaly ? 0 : 1,
        anomaly: packet.is_anomaly ? 1 : 0,
      }].slice(-60));
    };
    return () => socket.close();
  }, []);

  // ── derived metrics ────────────────────────────────────────────────────────
  const combinedData = useMemo(() => [...alerts, ...historicalAlerts], [alerts, historicalAlerts]);

  const avgConfidence = useMemo(() => {
    const total = combinedData.length;
    if (!total) return 0;
    return combinedData.reduce((sum, a) => sum + safeNum(a.xgb_probability || a.confidence), 0) / total;
  }, [combinedData]);

  const avgAnomalyScore = useMemo(() => {
    const total = combinedData.length;
    if (!total) return 0;
    return combinedData.reduce((sum, a) => sum + safeNum(a.anomaly_score), 0) / total;
  }, [combinedData]);

  const avgThreshold = useMemo(() => {
    // only historical alerts have threshold from DB; live packets also carry it
    const withThresh = combinedData.filter(a => safeNum(a.threshold) > 0);
    if (!withThresh.length) return 0;
    return withThresh.reduce((sum, a) => sum + safeNum(a.threshold), 0) / withThresh.length;
  }, [combinedData]);

  // top SHAP features aggregated across all data — sum absolute importance per feature name
  const aggregatedShap = useMemo(() => {
    const acc = {};
    combinedData.forEach(a => {
      const feats = safeJSON(a.top_features);
      feats.forEach(f => {
        if (!f?.feature) return;
        acc[f.feature] = (acc[f.feature] || 0) + Math.abs(safeNum(f.importance));
      });
    });
    return Object.entries(acc)
      .map(([feature, importance]) => ({ feature, importance }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 8);
  }, [combinedData]);

  // attack category distribution from live alerts
  const attackCatData = useMemo(() => {
    const acc = {};
    alerts.forEach(a => {
      if (a.is_anomaly && a.attack_category) {
        acc[a.attack_category] = (acc[a.attack_category] || 0) + 1;
      }
    });
    return Object.entries(acc)
      .map(([cat, count]) => ({ cat, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 7);
  }, [alerts]);

  // latest packet for score vs threshold gauge
  const latestPacket = alerts[0] || null;

  // WS status colors
  const wsColor = wsStatus === "LIVE" ? T.green : wsStatus === "ERROR" ? T.red : T.amber;
  const wsBg    = wsStatus === "LIVE" ? T.greenBg : wsStatus === "ERROR" ? T.redBg : T.amberBg;
  const WsIcon  = wsStatus === "LIVE" ? Wifi : WifiOff;
  const apsDelta = liveCounters.alertsPerSec - liveCounters.prevAps;

  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.sans, color: T.text }}>

      {/* ── topbar ── */}
      <div style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: "0 26px", height: 54, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 30, height: 30, borderRadius: 7, background: T.text, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <ShieldAlert size={15} color="#fff" />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 14, letterSpacing: ".03em", color: T.text }}>
              NETSAGE <span style={{ fontWeight: 400, color: T.textMid }}>IDS</span>
            </div>
            <div style={{ fontSize: 9, color: T.textDim, letterSpacing: ".09em", fontFamily: T.mono }}>ML-BASED INTRUSION DETECTION · UNSW-NB15</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <LiveClock />
          <div style={{ display: "flex", alignItems: "center", gap: 6, background: wsBg, border: `1px solid ${wsColor}44`, borderRadius: 8, padding: "4px 12px", fontFamily: T.mono, fontSize: 11, fontWeight: 700, color: wsColor }}>
            {wsStatus === "LIVE" && <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.green, animation: "pulse 1.5s infinite" }} />}
            <WsIcon size={12} />
            {wsStatus}
          </div>
        </div>
      </div>

      {/* ── ticker ── */}
      <TickerBar alerts={alerts} />

      {/* ── metrics row — 7 cards ── */}
      <div style={{ padding: "18px 26px 0", display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 10 }}>
        <MetricCard title="Packets" value={liveCounters.packets.toLocaleString()} subValue="running total" icon={<Activity size={14} />} highlight />
        <MetricCard title="Alerts/s" value={liveCounters.alertsPerSec} subValue={`${apsDelta >= 0 ? "+" : ""}${apsDelta} vs prev`} icon={<Zap size={14} />} accent={liveCounters.alertsPerSec > 5 ? T.red : T.blue} delta={apsDelta} />
        <MetricCard title="Threats" value={liveCounters.threats.toLocaleString()} subValue="continuous" icon={<ShieldAlert size={14} />} accent={liveCounters.threats > 0 ? T.red : T.text} />
        <MetricCard title="Critical" value={liveCounters.critical.toLocaleString()} subValue="continuous" icon={<AlertTriangle size={14} />} accent={liveCounters.critical > 0 ? T.amber : T.text} />
        <MetricCard title="Avg XGB" value={`${(avgConfidence * 100).toFixed(1)}%`} subValue={`${combinedData.length} samples`} icon={<Brain size={14} />} accent={T.blue} />
        <MetricCard title="Avg Score" value={avgAnomalyScore.toFixed(4)} subValue="autoencoder" icon={<Target size={14} />} accent={avgAnomalyScore > avgThreshold ? T.red : T.green} />
        <MetricCard title="Avg Threshold" value={avgThreshold > 0 ? avgThreshold.toFixed(4) : "—"} subValue="model boundary" icon={<BarChart2 size={14} />} accent={T.textMid} />
      </div>

      {/* ── traffic chart ── */}
      <div style={{ padding: "14px 26px 0" }}>
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Continuous Traffic Activity" icon={<Activity size={13} />} badge={`${trafficData.length} pts`} />
          <div style={{ padding: "12px 18px 16px" }}>
            <div style={{ display: "flex", gap: 18, marginBottom: 10 }}>
              {[["Normal", T.greenLine], ["Anomaly", T.redLine]].map(([l, c]) => (
                <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: T.mono, fontSize: 11, color: T.textMid, fontWeight: 600 }}>
                  <span style={{ width: 18, height: 2, background: c, display: "inline-block", borderRadius: 1 }} /> {l}
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={trafficData} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                <defs>
                  <linearGradient id="gN" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={T.greenLine} stopOpacity={0.16} />
                    <stop offset="95%" stopColor={T.greenLine} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={T.redLine} stopOpacity={0.20} />
                    <stop offset="95%" stopColor={T.redLine} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={T.border} vertical={false} />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: T.textDim, fontFamily: T.mono }} tickLine={false} axisLine={{ stroke: T.border }} interval="preserveStartEnd" />
                <YAxis tick={{ fontSize: 10, fill: T.textDim, fontFamily: T.mono }} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTip />} />
                <Area type="monotone" dataKey="normal"  stackId="1" stroke={T.greenLine} fill="url(#gN)" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="anomaly" stackId="1" stroke={T.redLine}   fill="url(#gA)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── analysis row: score gauge + attack cats + aggregated SHAP ── */}
      <div style={{ padding: "14px 26px 0", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

        {/* score vs threshold */}
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Anomaly Score vs Threshold" icon={<Target size={13} />} accent={T.red} />
          <ScoreGauge
            score={safeNum(latestPacket?.anomaly_score)}
            threshold={safeNum(latestPacket?.threshold || avgThreshold)}
          />
        </div>

        {/* attack categories — from live is_anomaly packets */}
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Attack Category Distribution" icon={<BarChart2 size={13} />} badge={attackCatData.length} accent={T.amber} />
          <AttackCatChart data={attackCatData} />
        </div>

        {/* aggregated SHAP — across all data */}
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Top SHAP Features (Aggregated)" icon={<Brain size={13} />} badge={aggregatedShap.length} accent={T.blue} />
          <ShapPanel features={aggregatedShap} />
        </div>
      </div>

      {/* ── live feed ── */}
      <div style={{ padding: "14px 26px 0" }}>
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Live Detection Feed" icon={<Activity size={13} />} badge={alerts.length} accent={T.green} />
          {/* col headers */}
          <div style={{
            display: "grid", gridTemplateColumns: "76px 68px 1fr 130px 90px 110px 72px",
            padding: "0 18px", height: 26, alignItems: "center",
            background: T.bg, borderBottom: `1px solid ${T.borderFt}`,
            fontFamily: T.mono, fontSize: 9, fontWeight: 700, color: T.textDim, letterSpacing: ".07em", textTransform: "uppercase",
          }}>
            <span>Time</span><span>Status</span><span>Src → Dst · Proto</span>
            <span>Attack Cat</span><span>Score</span><span style={{ textAlign: "right" }}>XGB Conf</span><span style={{ textAlign: "right" }}>Sev</span>
          </div>
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            <AnimatePresence initial={false}>
              {alerts.map(alert => (
                <AlertRow
                  key={alert.id}
                  alert={alert}
                  selected={selectedLive?.id === alert.id}
                  onClick={() => setSelectedLive(prev => prev?.id === alert.id ? null : alert)}
                />
              ))}
            </AnimatePresence>
            {alerts.length === 0 && (
              <div style={{ padding: "28px 18px", textAlign: "center", fontFamily: T.mono, fontSize: 12, color: T.textDim }}>
                Waiting for stream…
              </div>
            )}
          </div>
          {/* packet drawer inline */}
          <AnimatePresence>
            {selectedLive && (
              <div style={{ padding: "0 18px 16px" }}>
                <AlertDrawer alert={selectedLive} onClose={() => setSelectedLive(null)} />
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── historical alerts ── */}
      <div style={{ padding: "14px 26px 28px" }}>
        <div style={{ background: T.surface, borderRadius: 12, border: `1px solid ${T.border}`, overflow: "hidden" }}>
          <SectionHead title="Historical Database Alerts" icon={<Database size={13} />} badge={historicalAlerts.length} accent={T.blue} />
          {/* col headers */}
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 80px 86px 86px 76px 64px",
            padding: "0 18px", height: 26, alignItems: "center",
            background: T.bg, borderBottom: `1px solid ${T.borderFt}`,
            fontFamily: T.mono, fontSize: 9, fontWeight: 700, color: T.textDim, letterSpacing: ".07em", textTransform: "uppercase",
          }}>
            <span>Timestamp</span>
            <span style={{ textAlign: "right" }}>Score</span>
            <span style={{ textAlign: "right" }}>Threshold</span>
            <span style={{ textAlign: "right" }}>XGB Conf</span>
            <span style={{ textAlign: "right" }}>Sev</span>
            <span style={{ textAlign: "right" }}>SHAP</span>
          </div>
          <div style={{ maxHeight: 400, overflowY: "auto" }}>
            {historicalAlerts.map((alert, index) => (
              <HistRow
                key={index}
                alert={alert}
                index={index}
                selected={selectedHist === index}
                onSelect={() => setSelectedHist(prev => prev === index ? null : index)}
              />
            ))}
            {historicalAlerts.length === 0 && (
              <div style={{ padding: "28px 18px", textAlign: "center", fontFamily: T.mono, fontSize: 12, color: T.textDim }}>
                No stored alerts yet…
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        ::-webkit-scrollbar { width: 4px }
        ::-webkit-scrollbar-track { background: ${T.bg} }
        ::-webkit-scrollbar-thumb { background: ${T.borderFt}; border-radius: 99px }
      `}</style>
    </div>
  );
}