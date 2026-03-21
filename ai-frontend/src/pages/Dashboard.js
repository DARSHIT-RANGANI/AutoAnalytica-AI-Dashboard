import React, { useState, useEffect, useRef, createContext, useContext } from "react";
import { useLocation } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const API = "http://127.0.0.1:8000";

// ══════════════════════════════════════════════════════════════════════════════
// THEME SYSTEM
// ══════════════════════════════════════════════════════════════════════════════
const LIGHT = {
  bg:"#fdfaff", surface:"#f5eeff", card:"#ffffff", border:"#e4d0f8", borderHi:"#c080e0",
  primary:"#9b40d0", violet:"#b060e0", cyan:"#e0409a", teal:"#c030a0", amber:"#d070b8",
  rose:"#e8409a", hiText:"#1e0838", midText:"#7030a0", loText:"#c090d8",
  coralTint:"#fff0f8", violetTint:"#f8f0ff", tealTint:"#fff5f8", isDark:false,
};
const DARK = {
  bg:"#0d0b18", surface:"#13111f", card:"#1a1730", border:"#2a2448", borderHi:"#3d3570",
  primary:"#9b7ff0", violet:"#b49dfc", cyan:"#22c5c5", teal:"#14d4bb", amber:"#e89a5a",
  rose:"#e86892", hiText:"#ede9ff", midText:"#9088c0", loText:"#3a3360",
  coralTint:"#1e1228", violetTint:"#1c1838", tealTint:"#0f1e1c", isDark:true,
};
const ThemeCtx = createContext(LIGHT);
const useTheme  = () => useContext(ThemeCtx);

// ══════════════════════════════════════════════════════════════════════════════
// FIX A — resolvePerf
// Backend always populates `performance`. `metrics` may be empty {} which is
// truthy — use it only when it contains actual keys.
// ══════════════════════════════════════════════════════════════════════════════
function resolvePerf(result) {
  if (!result) return {};
  const m = result.metrics;
  if (m && typeof m === "object" && Object.keys(m).length > 0) return m;
  return result.performance || result.model_metrics || {};
}

// ══════════════════════════════════════════════════════════════════════════════
// FIX B — resolveScore
// The backend does NOT return prediction.score.
// Top-level test_score, or perf.accuracy / perf.R2 are the real sources.
// ══════════════════════════════════════════════════════════════════════════════
function resolveScore(result, perf) {
  if (result?.test_score   != null) return result.test_score;
  if (perf?.accuracy       != null) return perf.accuracy;
  if (perf?.R2             != null) return perf.R2;
  if (perf?.r2             != null) return perf.r2;
  return null;
}

// ══════════════════════════════════════════════════════════════════════════════
// FIX C — resolveCvScores
// Backend stores per-model CV scores inside perf.all_model_scores, not at a
// top-level cv_scores key.  Extract them here into the [{name, value}] shape
// the BarChart needs.
// ══════════════════════════════════════════════════════════════════════════════
function resolveCvScores(result, perf, problemType) {
  // 1. Try dedicated top-level key (may be present in some API versions)
  const raw = result?.cv_scores;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const entries = Object.entries(raw);
    if (entries.length > 0) {
      return entries.map(([k, v]) => ({
        name:  String(k),
        value: typeof v === "number" && !isNaN(v) ? parseFloat(v.toFixed(4)) : 0,
      }));
    }
  }
  // 2. Derive from perf.all_model_scores (the canonical source)
  const allScores = perf?.all_model_scores;
  if (!allScores || typeof allScores !== "object") return [];
  return Object.entries(allScores)
    .map(([name, s]) => {
      const cv = problemType === "regression"
        ? (s?.cv_r2_mean ?? null)
        : (s?.cv_mean ?? s?.f1_macro ?? null);
      return { name: String(name), value: typeof cv === "number" && !isNaN(cv) ? parseFloat(cv.toFixed(4)) : null };
    })
    .filter(e => e.value !== null);
}

// ══════════════════════════════════════════════════════════════════════════════
// FIX D — resolveDatasetDimensions
// Backend returns dataset_diagnostics.n_rows / n_cols — NOT dataset_analysis.
// ══════════════════════════════════════════════════════════════════════════════
function resolveDatasetDimensions(result, perf) {
  // Primary source
  const diag = result?.dataset_diagnostics;
  if (diag) {
    return {
      rows: diag.n_rows ?? null,
      cols: diag.n_cols ?? null,
    };
  }
  // Fallback: reconstruct from perf
  const nTrain = perf?.n_train;
  const nTest  = perf?.n_test;
  const rows   = (typeof nTrain === "number" && typeof nTest === "number") ? nTrain + nTest : null;
  return { rows, cols: null };
}

// ══════════════════════════════════════════════════════════════════════════════
// Safe helpers
// ══════════════════════════════════════════════════════════════════════════════
const safeNumber = (val, decimals = 4) =>
  typeof val === "number" && !isNaN(val) ? val.toFixed(decimals) : "N/A";

const safeArray = (arr) => (Array.isArray(arr) ? arr : []);

const safeString = (val) =>
  val !== null && val !== undefined && typeof val !== "object" ? String(val) : "—";

const fmt = s => String(s || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

const pct = v => (v !== null && v !== undefined) ? `${(v * 100).toFixed(1)}%` : "—";

// ══════════════════════════════════════════════════════════════════════════════
// UI ATOMS
// ══════════════════════════════════════════════════════════════════════════════
function NeuralGrid({ isDark }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext("2d"); let frame, t = 0;
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize(); window.addEventListener("resize", resize);
    const nodes = Array.from({ length: 55 }, () => ({
      x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.8 + 0.6, pulse: Math.random() * Math.PI * 2,
      type: Math.random() > 0.7 ? "coral" : Math.random() > 0.5 ? "teal" : "violet",
    }));
    const draw = () => {
      t += 0.008; ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y, d = Math.sqrt(dx*dx+dy*dy);
        if (d < 145) { ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = isDark ? `rgba(155,127,240,${(1-d/145)*0.22})` : `rgba(107,79,200,${(1-d/145)*0.12})`;
          ctx.lineWidth = 0.8; ctx.stroke(); }
      }
      nodes.forEach(n => {
        const p = Math.sin(t * 1.5 + n.pulse) * 0.5 + 0.5;
        const colors = isDark
          ? { violet:`rgba(155,127,240,${0.45+p*0.4})`, coral:`rgba(232,104,146,${0.4+p*0.38})`, teal:`rgba(20,212,187,${0.4+p*0.38})` }
          : { violet:`rgba(107,79,200,${0.25+p*0.3})`,  coral:`rgba(229,97,60,${0.2+p*0.28})`,   teal:`rgba(30,158,158,${0.2+p*0.28})` };
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r + p * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = colors[n.type]; ctx.fill();
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(frame); window.removeEventListener("resize", resize); };
  }, [isDark]);
  return <canvas ref={ref} style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none", opacity:isDark?0.65:0.62 }} />;
}

function DataStreams() {
  const C = useTheme();
  return (
    <div style={{ position:"fixed", inset:0, zIndex:0, pointerEvents:"none", overflow:"hidden" }}>
      {[...Array(6)].map((_, i) => (
        <div key={i} style={{
          position:"absolute", left:`${10+i*16}%`, top:0, bottom:0, width:1,
          background:`linear-gradient(180deg, transparent, ${C.primary}18, ${C.cyan}22, transparent)`,
          animation:`streamFlow ${3+i*0.7}s ease-in-out infinite`, animationDelay:`${i*0.5}s`, opacity:0.5,
        }} />
      ))}
    </div>
  );
}

function GlassCard({ children, style = {}, accent, hover = true }) {
  const C = useTheme(); const accentColor = accent || C.primary;
  const [h, setH] = useState(false); const [pos, setPos] = useState({ x:0, y:0 }); const ref = useRef(null);
  const onMove = e => { if (!ref.current) return; const r = ref.current.getBoundingClientRect(); setPos({ x:e.clientX-r.left, y:e.clientY-r.top }); };
  return (
    <div ref={ref} onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)} onMouseMove={onMove}
      style={{
        position:"relative", background:`linear-gradient(135deg, ${C.card}f0, ${C.surface}cc)`,
        border:`1px solid ${h&&hover?accentColor+"55":C.border}`, borderRadius:20,
        backdropFilter:"blur(20px)", WebkitBackdropFilter:"blur(20px)",
        transition:"border-color 0.3s, transform 0.2s, box-shadow 0.3s",
        transform:h&&hover?"translateY(-2px)":"none",
        boxShadow:h&&hover ? (C.isDark?`0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px ${accentColor}33, inset 0 1px 0 ${accentColor}18`:`0 20px 60px rgba(107,79,200,0.12), 0 0 0 1px ${accentColor}22, inset 0 1px 0 rgba(255,255,255,0.9)`)
          : (C.isDark?`0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)`:`0 4px 24px rgba(107,79,200,0.07), 0 1px 4px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8)`),
        overflow:"hidden", ...style,
      }}>
      {h&&hover&&<div style={{ position:"absolute", width:300, height:300, borderRadius:"50%", background:`radial-gradient(circle, ${accentColor}${C.isDark?"18":"0c"} 0%, transparent 70%)`, left:pos.x-150, top:pos.y-150, pointerEvents:"none", zIndex:0 }} />}
      <div style={{ position:"relative", zIndex:1 }}>{children}</div>
    </div>
  );
}

function Panel({ title, subtitle, icon, children, accent, badge }) {
  const C = useTheme(); const accentColor = accent || C.primary;
  return (
    <GlassCard accent={accent} hover={false} style={{ padding:"28px 30px 32px" }}>
      <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg, transparent, ${C.rose}60, ${accentColor}, ${accentColor}88, transparent)` }} />
      {title && (
        <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:subtitle?4:24 }}>
          <div style={{ width:36, height:36, borderRadius:10, flexShrink:0, background:`${accentColor}14`, border:`1px solid ${accentColor}28`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:17, boxShadow:`0 2px 8px ${accentColor}15` }}>{icon}</div>
          <div style={{ flex:1 }}>
            <div style={{ display:"flex", alignItems:"center", gap:10 }}>
              <h2 style={{ color:C.hiText, fontSize:13, fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace" }}>{title}</h2>
              {badge && <span style={{ padding:"2px 8px", borderRadius:20, fontSize:10, fontWeight:700, background:`${accentColor}14`, color:accentColor, border:`1px solid ${accentColor}28`, letterSpacing:"0.08em" }}>{badge}</span>}
            </div>
          </div>
          <div style={{ flex:1, height:1, background:`linear-gradient(90deg, ${accentColor}35, transparent)` }} />
        </div>
      )}
      {subtitle && <p style={{ color:C.midText, fontSize:12, marginBottom:22, marginLeft:48 }}>{subtitle}</p>}
      {children}
    </GlassCard>
  );
}

function AnimatedValue({ value, decimals = 0, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  const target = parseFloat(value) || 0;
  useEffect(() => {
    const duration = 1200, startTime = Date.now();
    const tick = () => {
      const elapsed = Date.now() - startTime, progress = Math.min(elapsed / duration, 1), eased = 1 - Math.pow(1-progress, 3);
      setDisplay(target * eased); if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [target]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

function KpiCard({ label, value, accent, icon, sub, animated = false, decimals = 0, suffix = "" }) {
  const C = useTheme(); const accentColor = accent || C.primary;
  const len = String(value).length; const fs = len>10?16:len>7?22:len>4?28:36;
  return (
    <GlassCard accent={accent} style={{ padding:"20px 18px" }}>
      <div style={{ position:"absolute", top:-20, right:-20, width:80, height:80, borderRadius:"50%", background:`radial-gradient(circle, ${accentColor}18 0%, transparent 70%)`, filter:"blur(12px)" }} />
      <p style={{ color:C.midText, fontSize:10, letterSpacing:"0.14em", textTransform:"uppercase", fontWeight:700, marginBottom:10, display:"flex", alignItems:"center", gap:6 }}>
        <span style={{ width:22, height:22, borderRadius:6, fontSize:11, background:`${accentColor}14`, border:`1px solid ${accentColor}22`, display:"flex", alignItems:"center", justifyContent:"center", boxShadow:`0 1px 4px ${accentColor}12` }}>{icon}</span>
        {label}
      </p>
      <p style={{ color:accentColor, fontWeight:800, lineHeight:1.1, fontFamily:"'Space Grotesk', monospace", fontSize:fs, wordBreak:"break-all" }}>
        {animated && !isNaN(parseFloat(value)) ? <AnimatedValue value={parseFloat(value)} decimals={decimals} suffix={suffix} /> : value}
      </p>
      {sub && <p style={{ color:`${accentColor}80`, fontSize:11, marginTop:8 }}>{sub}</p>}
    </GlassCard>
  );
}

function ProgressBar({ value = 0, color, height = 4 }) {
  const C = useTheme(); const barColor = color || C.primary; const [w, setW] = useState(0);
  useEffect(() => { setTimeout(() => setW(Math.max(0, Math.min(1, value)) * 100), 100); }, [value]);
  return (
    <div style={{ height, background:C.border, borderRadius:height, overflow:"hidden" }}>
      <div style={{ width:`${w}%`, height:"100%", borderRadius:height, background:`linear-gradient(90deg, ${C.rose}50, ${barColor})`, transition:"width 1s cubic-bezier(0.4,0,0.2,1)", position:"relative", overflow:"hidden" }}>
        <div style={{ position:"absolute", inset:0, background:"linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)", animation:"shimmerBar 1.8s ease infinite" }} />
      </div>
    </div>
  );
}

function Toast({ text, ok, onClose }) {
  const C = useTheme(); if (!text) return null; const color = ok ? C.teal : C.rose;
  return (
    <div style={{ position:"fixed", top:80, right:24, zIndex:9998, padding:"14px 20px", background:C.surface, border:`1px solid ${color}35`, borderLeft:`3px solid ${color}`, borderRadius:14, fontSize:13, fontWeight:600, display:"flex", alignItems:"center", gap:10, boxShadow:`0 8px 32px rgba(107,79,200,0.12), 0 0 0 1px ${color}12`, animation:"slideInRight 0.3s cubic-bezier(0.34,1.56,0.64,1)", minWidth:280, maxWidth:360 }}>
      <span style={{ width:24, height:24, borderRadius:"50%", background:`${color}16`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, flexShrink:0, color }}>{ok?"✓":"✕"}</span>
      <span style={{ flex:1, color:C.hiText, fontSize:13 }}>{text}</span>
      <button onClick={onClose} style={{ background:"none", border:"none", color:C.midText, cursor:"pointer", fontSize:18, lineHeight:1, padding:0 }}>×</button>
    </div>
  );
}

function InlineBadge({ text, color }) {
  return <span style={{ padding:"3px 10px", borderRadius:20, fontSize:10, fontWeight:700, background:`${color}14`, color:color, border:`1px solid ${color}28`, letterSpacing:"0.08em", fontFamily:"'Space Grotesk', monospace" }}>{text}</span>;
}

function AccuracyBadge({ accuracy }) {
  const C = useTheme();
  if (accuracy === null || accuracy === undefined) return <span style={{ color:C.loText }}>—</span>;
  const color = accuracy >= 90 ? C.teal : accuracy >= 75 ? C.primary : accuracy >= 60 ? C.amber : C.rose;
  return <InlineBadge text={`${accuracy}%`} color={color} />;
}

function EmptyState({ emoji, title, sub }) {
  const C = useTheme();
  return (
    <div style={{ textAlign:"center", padding:"40px 20px" }}>
      <p style={{ fontSize:44, marginBottom:14 }}>{emoji}</p>
      <p style={{ color:C.hiText, fontWeight:700, fontSize:15, fontFamily:"'Space Grotesk', sans-serif", marginBottom:8 }}>{title}</p>
      <p style={{ color:C.midText, fontSize:13, lineHeight:1.7 }}>{sub}</p>
    </div>
  );
}

function LoadingSpinner() {
  const C = useTheme();
  return (
    <div style={{ display:"flex", alignItems:"center", gap:12, padding:"60px 0", justifyContent:"center" }}>
      <div style={{ position:"relative", width:44, height:44 }}>
        {[C.primary, C.rose, C.cyan].map((col, i) => (
          <div key={i} style={{ position:"absolute", inset:i*6, border:"1.5px solid transparent", borderTopColor:col, borderRadius:"50%", animation:`orbit ${0.8+i*0.4}s linear infinite` }} />
        ))}
      </div>
      <p style={{ color:C.midText, fontSize:13, fontFamily:"'Space Grotesk', monospace", letterSpacing:"0.1em" }}>Loading dashboard…</p>
    </div>
  );
}

function DetailRow({ label, value, color }) {
  const C = useTheme();
  return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:12, padding:"11px 16px", background:`linear-gradient(135deg, ${C.surface}, ${C.violetTint}40)`, borderRadius:10, border:`1px solid ${C.border}` }}>
      <span style={{ color:C.midText, fontSize:12, fontWeight:600 }}>{label}</span>
      <span style={{ color:color||C.hiText, fontSize:12, fontWeight:700, fontFamily:"'Space Grotesk', monospace", textAlign:"right", wordBreak:"break-word", maxWidth:"60%" }}>{value}</span>
    </div>
  );
}

function CustomTooltip({ active, payload, label, accentColor }) {
  const C = useTheme();
  if (!active || !payload || !payload.length) return null;
  return (
    <div style={{ background:C.card, border:`1px solid ${accentColor}40`, borderRadius:10, padding:"10px 14px", boxShadow:`0 4px 20px ${accentColor}18` }}>
      <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:4 }}>{label}</p>
      <p style={{ color:accentColor, fontWeight:800, fontSize:15, fontFamily:"'Space Grotesk', monospace" }}>
        {typeof payload[0].value === "number" ? payload[0].value.toFixed(4) : String(payload[0].value)}
      </p>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN DASHBOARD
// ══════════════════════════════════════════════════════════════════════════════
export default function Dashboard() {
  const location    = useLocation();
  const routeResult = location.state?.result ?? null;

  const [isDark, setIsDark] = useState(() => {
    try { return localStorage.getItem("aa_theme") === "dark"; } catch { return false; }
  });
  const theme = isDark ? DARK : LIGHT;
  const C     = theme;

  useEffect(() => {
    try { localStorage.setItem("aa_theme", isDark ? "dark" : "light"); } catch {}
    document.body.style.background = isDark ? DARK.bg : LIGHT.bg;
  }, [isDark]);

  const [stats,       setStats]       = useState(null);
  const [dashboards,  setDashboards]  = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState("");
  const [toast,       setToast]       = useState({ text:"", ok:true });
  const [viewingHtml, setViewingHtml] = useState(null);

  const notify = (text, ok = true) => { setToast({ text, ok }); setTimeout(() => setToast({ text:"", ok:true }), 4500); };

  const fetchAll = async () => {
    setLoading(true); setError("");
    try {
      const [statsRes, reportsRes] = await Promise.all([
        fetch(`${API}/dashboard/stats`),
        fetch(`${API}/reports/list`),
      ]);
      if (!statsRes.ok) throw new Error(`Stats HTTP ${statsRes.status}`);
      const statsData = await statsRes.json();
      setStats(statsData);
      if (reportsRes.ok) {
        const repData = await reportsRes.json();
        const htmlOnly = (repData.reports || []).filter(r => r.report_filename && r.report_filename.startsWith("dashboard_"));
        setDashboards(htmlOnly);
      }
    } catch (err) { setError("Could not load dashboard data. Is the backend running?"); console.error(err); }
    finally { setLoading(false); }
  };

  useEffect(() => { fetchAll(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Stats ───────────────────────────────────────────────────────────────────
  const datasetsUploaded = stats?.datasets_uploaded ?? 0;
  const modelsTrained    = stats?.models_trained    ?? 0;
  const reportsGenerated = stats?.reports_generated ?? 0;
  const bestAccuracy     = stats?.best_accuracy     ?? null;
  const recentTrainings  = stats?.recent_trainings  ?? [];

  // ══════════════════════════════════════════════════════════════════════════
  // FIX — derive all analytics data from the actual backend response shape
  // ══════════════════════════════════════════════════════════════════════════
  const hasResult   = routeResult !== null;
  const perf        = resolvePerf(routeResult);
  const problemType = routeResult?.problem_type ?? null;

  // FIX B: best model — backend uses `best_model`, not `best_model_name`
  const bestModel = routeResult?.best_model_name ?? routeResult?.best_model ?? routeResult?.model_name ?? null;

  // FIX B: score — from actual perf keys, not prediction.score
  const score = resolveScore(routeResult, perf);

  // FIX D: rows / cols from dataset_diagnostics
  const { rows: dataRows, cols: dataCols } = resolveDatasetDimensions(routeResult, perf);

  // FIX C: cv scores chart data
  const cvScores = resolveCvScores(routeResult, perf, problemType);

  // Numeric / categorical columns — backend may or may not return these;
  // dataset_diagnostics doesn't include them directly, so we fall back gracefully.
  const numericCols = safeArray(
    routeResult?.dataset_analysis?.numeric_columns ??
    routeResult?.numeric_columns ??
    null
  );
  const catCols = safeArray(
    routeResult?.dataset_analysis?.categorical_columns ??
    routeResult?.categorical_columns ??
    null
  );

  // Stage logs, risk flags, top predictions — all safe-array guarded
  const stageLogs = safeArray(routeResult?.pipeline_stage_logs);
  const riskFlags = safeArray(
    routeResult?.meta_insight_full?.risk_flags ??
    routeResult?.risk_flags ??
    null
  );
  const topPreds  = safeArray(routeResult?.meta_insight_full?.top_model_predictions);

  // Retrain decision — supports object (new) or bool (old)
  const retrainDecision = routeResult?.retrain_decision;
  const shouldRetrain   = typeof retrainDecision === "object" && retrainDecision !== null
    ? !!retrainDecision.should_retrain : !!retrainDecision;
  const retrainSeverity = typeof retrainDecision === "object" && retrainDecision !== null
    ? safeString(retrainDecision.severity) : "—";
  const retrainReason   = typeof retrainDecision === "object" && retrainDecision !== null
    ? safeString(retrainDecision.reason) : "—";

  // meta_insight — string or object
  const metaInsightRec = (() => {
    const mi = routeResult?.meta_insight;
    if (!mi) return "—";
    if (typeof mi === "string") return mi;
    if (typeof mi === "object") return safeString(mi.recommendation ?? mi.text ?? mi.summary ?? mi.insight);
    return "—";
  })();
  const metaInsightConf   = (() => { const mi = routeResult?.meta_insight; if (!mi||typeof mi!=="object") return null; return typeof mi.confidence==="number"?mi.confidence:null; })();
  const metaInsightSource = (() => { const mi = routeResult?.meta_insight; if (!mi||typeof mi!=="object") return null; return mi.source!=null?safeString(mi.source):null; })();

  // FIX — friendly overview KPI values (all primitive strings — never objects)
  const overviewKpis = [
    { label:"Total Rows",    value: dataRows != null ? String(dataRows) : "N/A", icon:"⇅", accent:C.primary },
    { label:"Total Columns", value: dataCols != null ? String(dataCols) : "N/A", icon:"⇄", accent:C.cyan    },
    { label:"Problem Type",  value: fmt(safeString(problemType)),                 icon:"⊕", accent:C.teal    },
    { label:"Best Model",    value: safeString(bestModel),                        icon:"🏅", accent:C.violet },
    {
      label: problemType === "regression" ? "R² Score" : "Accuracy",
      value: score != null ? (problemType === "regression" ? safeNumber(score, 4) : pct(score)) : "N/A",
      icon:"◈", accent:C.amber,
    },
  ];

  // FIX — additional detail rows for Overview panel
  const overviewDetails = [
    { label:"Rows",              value: dataRows  != null ? String(dataRows)  : "N/A",           color:C.primary },
    { label:"Columns",           value: dataCols  != null ? String(dataCols)  : "N/A",           color:C.cyan    },
    { label:"Train Samples",     value: perf?.n_train != null ? String(perf.n_train) : "N/A",    color:C.teal    },
    { label:"Test Samples",      value: perf?.n_test  != null ? String(perf.n_test)  : "N/A",    color:C.violet  },
    { label:"Problem Type",      value: fmt(safeString(problemType)),                              color:C.teal    },
    { label:"Best Model",        value: safeString(bestModel),                                     color:C.violet  },
    { label:"Scale Tier",        value: safeString(routeResult?.scale_tier_name ?? routeResult?.scale_tier), color:C.primary },
    {
      label: problemType === "regression" ? "Test R²" : "Test Accuracy",
      value: score != null ? (problemType === "regression" ? safeNumber(score, 4) : pct(score)) : "N/A",
      color: C.amber,
    },
    {
      label:"CV Score",
      value: perf?.cv_score_mean != null
        ? (problemType === "regression" ? safeNumber(perf.cv_score_mean, 4) : pct(perf.cv_score_mean))
        : "N/A",
      color:C.cyan,
    },
    { label:"Confidence Label",  value: safeString(routeResult?.confidence_label),                color:C.rose    },
    { label:"Overfitting",       value: routeResult?.overfitting != null ? (routeResult.overfitting ? "Detected ⚠️" : "None ✓") : "N/A", color:routeResult?.overfitting?C.rose:C.teal },
  ].filter(r => r.value !== "N/A" && r.value !== "—");

  return (
    <ThemeCtx.Provider value={theme}>
      <>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
          *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
          html, body { background:${C.bg}; color:${C.hiText}; font-family:'DM Sans',sans-serif; overflow-x:hidden; min-height:100vh; transition: background 0.4s ease, color 0.3s ease; }
          @keyframes streamFlow  { 0%,100%{opacity:.1;transform:scaleY(.3) translateY(-20%)} 50%{opacity:.45;transform:scaleY(1) translateY(0)} }
          @keyframes orbit       { to{transform:rotate(360deg)} }
          @keyframes shimmerBar  { 0%{transform:translateX(-100%)} 100%{transform:translateX(300%)} }
          @keyframes slideInRight{ from{opacity:0;transform:translateX(20px)} to{opacity:1;transform:translateX(0)} }
          @keyframes fadeSlideUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
          @keyframes titleGlow   { 0%{background-position:0% center} 100%{background-position:300% center} }
          @keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:.5} }
          .reveal { animation:fadeSlideUp .55s cubic-bezier(.34,1.1,.64,1) both; }
          .r1{animation-delay:.05s} .r2{animation-delay:.12s} .r3{animation-delay:.20s}
          .r4{animation-delay:.28s} .r5{animation-delay:.36s} .r6{animation-delay:.44s}
          .r7{animation-delay:.52s} .r8{animation-delay:.60s}
          ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:${C.bg}}
          ::-webkit-scrollbar-thumb{background:${C.border};border-radius:4px} ::-webkit-scrollbar-thumb:hover{background:${C.borderHi}}
          .row-hover:hover { background:${C.isDark?`${C.primary}08`:`${C.primary}05`} !important; }
        `}</style>

        {/* Backgrounds */}
        {isDark ? (
          <>
            <div style={{ position:"fixed",inset:0,zIndex:0, backgroundImage:`url("https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=1800&auto=format&fit=crop&q=80")`, backgroundSize:"cover",backgroundPosition:"center" }} />
            <div style={{ position:"fixed",inset:0,zIndex:1, background:`linear-gradient(160deg, rgba(8,6,18,0.52) 0%, rgba(13,8,30,0.46) 35%, rgba(10,5,22,0.50) 70%, rgba(5,3,14,0.55) 100%)` }} />
            <div style={{ position:"fixed",inset:0,zIndex:2,pointerEvents:"none", background:`radial-gradient(ellipse 55% 45% at 10% 35%, rgba(155,127,240,0.18) 0%, transparent 60%), radial-gradient(ellipse 45% 35% at 90% 15%, rgba(20,212,187,0.14) 0%, transparent 55%)` }} />
          </>
        ) : (
          <>
            <div style={{ position:"fixed",inset:0,zIndex:0, backgroundImage:`url("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1800&auto=format&fit=crop&q=85")`, backgroundSize:"cover",backgroundPosition:"center 40%" }} />
            <div style={{ position:"fixed",inset:0,zIndex:1, background:`linear-gradient(160deg, rgba(255,252,255,0.55) 0%, rgba(252,245,255,0.48) 35%, rgba(255,248,255,0.52) 70%, rgba(250,245,255,0.58) 100%)` }} />
            <div style={{ position:"fixed",inset:0,zIndex:2,pointerEvents:"none", background:`radial-gradient(ellipse 55% 45% at 15% 30%, rgba(155,64,208,0.06) 0%, transparent 60%)` }} />
          </>
        )}

        <NeuralGrid isDark={isDark} />
        <DataStreams />

        <div style={{ position:"relative", zIndex:2, minHeight:"100vh" }}>

          {/* HEADER */}
          <header style={{
            padding:"0 40px", height:68, display:"flex", alignItems:"center", justifyContent:"space-between",
            background:isDark?`rgba(13,11,24,0.92)`:`rgba(255,252,255,0.92)`,
            backdropFilter:"blur(30px)", WebkitBackdropFilter:"blur(30px)",
            borderBottom:`1px solid ${C.border}`,
            boxShadow:isDark?"0 1px 0 rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.4)":"0 1px 0 rgba(255,255,255,0.9), 0 4px 20px rgba(160,60,200,0.10)",
            position:"sticky", top:0, zIndex:100,
          }}>
            <div style={{ display:"flex", alignItems:"center", gap:14 }}>
              <div style={{ width:38, height:38, borderRadius:12, background:`linear-gradient(135deg, ${C.primary}, ${C.rose})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18, boxShadow:`0 4px 16px ${C.primary}35` }}>⚡</div>
              <div>
                <div style={{ fontSize:17, fontWeight:800, fontFamily:"'Space Grotesk',sans-serif", letterSpacing:"0.02em" }}>
                  <span style={{ display:"inline-block", background:`linear-gradient(90deg, ${C.hiText}, ${C.primary}, ${C.rose}, ${C.hiText})`, backgroundSize:"300% auto", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text", animation:"titleGlow 7s linear infinite" }}>AutoAnalytica</span>
                </div>
                <div style={{ color:C.loText, fontSize:10, letterSpacing:"0.16em", textTransform:"uppercase" }}>Analytics Dashboard</div>
              </div>
            </div>
            <div style={{ display:"flex", alignItems:"center", gap:12 }}>
              <button onClick={fetchAll} style={{ display:"flex", alignItems:"center", gap:7, padding:"7px 16px", borderRadius:10, cursor:"pointer", border:`1px solid ${C.border}`, background:C.isDark?`${C.primary}10`:`${C.primary}08`, color:C.primary, fontSize:11, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", transition:"all 0.2s" }}>🔄 Refresh</button>
              <button onClick={() => setIsDark(d => !d)} style={{ display:"flex", alignItems:"center", gap:8, padding:"7px 14px", borderRadius:24, cursor:"pointer", border:`1px solid ${C.border}`, background:isDark?`linear-gradient(135deg, rgba(155,127,240,0.15), rgba(20,212,187,0.08))`:`linear-gradient(135deg, rgba(107,79,200,0.08), rgba(212,84,122,0.05))`, transition:"all 0.3s cubic-bezier(0.34,1.56,0.64,1)" }}>
                <div style={{ width:40, height:22, borderRadius:11, position:"relative", background:isDark?`linear-gradient(135deg, ${C.primary}, ${C.teal})`:`linear-gradient(135deg, ${C.primary}60, ${C.rose}80)`, transition:"background 0.35s" }}>
                  <div style={{ position:"absolute", top:3, left:isDark?21:3, width:16, height:16, borderRadius:"50%", background:"#fff", transition:"left 0.3s cubic-bezier(0.34,1.56,0.64,1)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:9 }}>{isDark?"🌙":"☀️"}</div>
                </div>
                <span style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace" }}>{isDark?"DARK":"LIGHT"}</span>
              </button>
              <div style={{ padding:"6px 14px", borderRadius:20, background:`${C.teal}12`, border:`1px solid ${C.teal}30`, display:"flex", alignItems:"center", gap:7 }}>
                <div style={{ width:6, height:6, borderRadius:"50%", background:C.teal, boxShadow:`0 0 8px ${C.teal}`, animation:"pulse 2s ease infinite" }} />
                <span style={{ color:C.teal, fontSize:11, fontWeight:700, letterSpacing:"0.1em" }}>LIVE DATA</span>
              </div>
            </div>
          </header>

          {/* MAIN */}
          <main style={{ maxWidth:1100, margin:"0 auto", padding:"44px 28px 60px", display:"flex", flexDirection:"column", gap:28 }}>

            {/* HERO */}
            <div className="reveal" style={{ textAlign:"center", padding:"28px 0 8px", position:"relative", zIndex:10, isolation:"isolate" }}>
              <div style={{ position:"absolute", inset:"-40px -80px", background:`radial-gradient(ellipse 70% 80% at 50% 50%, ${C.bg}99 0%, transparent 70%)`, pointerEvents:"none", zIndex:-1 }} />
              <div style={{ display:"flex", alignItems:"center", gap:14, justifyContent:"center", marginBottom:20 }}>
                <div style={{ height:1, width:48, background:`linear-gradient(90deg, transparent, ${C.rose}60)` }} />
                <span style={{ color:C.rose, fontSize:9, letterSpacing:"0.3em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", fontWeight:700 }}>MongoDB · Live Analytics · Real-Time</span>
                <div style={{ height:1, width:48, background:`linear-gradient(90deg, ${C.rose}60, transparent)` }} />
              </div>
              <h1 style={{ fontSize:52, fontWeight:800, fontFamily:"'Space Grotesk', sans-serif", letterSpacing:"-0.02em", lineHeight:1.1, marginBottom:14, isolation:"isolate", filter:isDark?`drop-shadow(0 0 36px ${C.primary}60) drop-shadow(0 2px 12px rgba(0,0,0,0.7))`:`drop-shadow(0 0 40px ${C.primary}30)` }}>
                <span style={{ display:"inline-block", background:isDark?`linear-gradient(90deg, ${C.hiText}, ${C.primary}, ${C.rose}, ${C.hiText})`:`linear-gradient(90deg, ${C.primary}, ${C.rose}, ${C.violet}, ${C.primary})`, backgroundSize:"300% auto", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text", animation:"titleGlow 6s linear infinite" }}>Analytics Dashboard</span>
              </h1>
              <p style={{ color:isDark?C.hiText:C.midText, opacity:isDark?0.55:0.85, fontSize:13, letterSpacing:"0.22em", textTransform:"uppercase", fontWeight:600 }}>
                Datasets &nbsp;·&nbsp; Models &nbsp;·&nbsp; Reports &nbsp;·&nbsp; Insights
              </p>
            </div>

            {/* ERROR */}
            {error && (
              <div className="reveal">
                <GlassCard accent={C.rose} hover={false} style={{ padding:"20px 24px" }}>
                  <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg, transparent, ${C.rose}, transparent)` }} />
                  <div style={{ display:"flex", alignItems:"center", gap:14 }}>
                    <div style={{ width:40, height:40, borderRadius:12, background:`${C.rose}12`, border:`1px solid ${C.rose}25`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20 }}>⚠️</div>
                    <div style={{ flex:1 }}>
                      <p style={{ color:C.rose, fontSize:12, fontWeight:700, letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:4 }}>Connection Error</p>
                      <p style={{ color:C.midText, fontSize:13 }}>{error}</p>
                    </div>
                    <button onClick={fetchAll} style={{ padding:"8px 20px", borderRadius:10, cursor:"pointer", border:`1px solid ${C.rose}40`, background:`${C.rose}10`, color:C.rose, fontSize:12, fontWeight:700, fontFamily:"'Space Grotesk', monospace", letterSpacing:"0.1em", textTransform:"uppercase" }}>Retry</button>
                  </div>
                </GlassCard>
              </div>
            )}

            {/* LOADING */}
            {loading && !error && <GlassCard hover={false} style={{ padding:"20px" }}><LoadingSpinner /></GlassCard>}

            {/* KPI CARDS */}
            {!loading && !error && stats && (
              <div className="reveal r1" style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(200px, 1fr))", gap:14 }}>
                <KpiCard label="Datasets Uploaded" value={String(datasetsUploaded)} accent={C.primary} icon="📁" sub="Total files in MongoDB"  animated decimals={0} />
                <KpiCard label="Models Trained"    value={String(modelsTrained)}    accent={C.cyan}    icon="🤖" sub="Saved .pkl artefacts"   animated decimals={0} />
                <KpiCard label="Reports Generated" value={String(reportsGenerated)} accent={C.teal}    icon="📄" sub="PDF + HTML reports"     animated decimals={0} />
                <KpiCard label="Best Accuracy"     value={bestAccuracy !== null ? `${bestAccuracy}%` : "—"} accent={C.amber} icon="🎯" sub={bestAccuracy !== null ? "Highest across all models" : "Train a model first"} animated={bestAccuracy !== null} decimals={2} suffix="%" />
              </div>
            )}

            {/* RECENT TRAININGS */}
            {!loading && !error && (
              <div className="reveal r2">
                <Panel title="Recent Model Trainings" subtitle="Last 5 trained models — pulled live from MongoDB" icon="🕒" accent={C.violet} badge="LIVE">
                  {recentTrainings.length === 0 ? (
                    <EmptyState emoji="🤖" title="No models trained yet" sub="Upload a dataset and train your first model to see results here." />
                  ) : (
                    <div style={{ display:"flex", flexDirection:"column", gap:0 }}>
                      <div style={{ display:"grid", gridTemplateColumns:"1.8fr 1.6fr 1fr 1fr 1fr 1.2fr", gap:8, padding:"8px 16px 12px", borderBottom:`1px solid ${C.border}` }}>
                        {["Model","Dataset","Target","Type","Accuracy","Trained At"].map((h, i) => (
                          <span key={i} style={{ color:C.loText, fontSize:9, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace" }}>{h}</span>
                        ))}
                      </div>
                      {recentTrainings.map((m, i) => (
                        <div key={m.model_id || i} className="row-hover" style={{ display:"grid", gridTemplateColumns:"1.8fr 1.6fr 1fr 1fr 1fr 1.2fr", gap:8, padding:"14px 16px", borderBottom:i<recentTrainings.length-1?`1px solid ${C.border}40`:"none", alignItems:"center", borderRadius:8, transition:"background 0.15s" }}>
                          <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                            {i === 0 && <span style={{ fontSize:14 }}>👑</span>}
                            <span style={{ color:C.hiText, fontWeight:700, fontSize:13 }}>{m.model_name || "—"}</span>
                          </div>
                          <span style={{ color:C.midText, fontSize:11, fontFamily:"'Space Grotesk', monospace", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{m.dataset?(m.dataset.length>22?`…${m.dataset.slice(-18)}`:m.dataset):"—"}</span>
                          <span style={{ color:C.midText, fontSize:12 }}>{m.target || "—"}</span>
                          <InlineBadge text={m.problem_type || "?"} color={m.problem_type==="classification"?C.primary:C.amber} />
                          <AccuracyBadge accuracy={m.accuracy} />
                          <span style={{ color:C.loText, fontSize:10, fontFamily:"'Space Grotesk', monospace" }}>{m.trained_at || "—"}</span>
                        </div>
                      ))}
                      {recentTrainings.some(m => m.accuracy !== null) && (
                        <div style={{ padding:"18px 16px 4px", borderTop:`1px solid ${C.border}40`, marginTop:4 }}>
                          <p style={{ color:C.loText, fontSize:9, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12 }}>Accuracy Comparison</p>
                          <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                            {recentTrainings.filter(m => m.accuracy !== null).map((m, i) => (
                              <div key={i} style={{ display:"flex", alignItems:"center", gap:10 }}>
                                <span style={{ width:140, fontSize:11, color:C.midText, fontWeight:600, flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{m.model_name}</span>
                                <div style={{ flex:1 }}><ProgressBar value={m.accuracy/100} color={i===0?C.amber:C.primary} height={5} /></div>
                                <span style={{ width:50, fontSize:11, fontWeight:700, color:i===0?C.amber:C.primary, fontFamily:"'Space Grotesk', monospace", textAlign:"right", flexShrink:0 }}>{m.accuracy}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </Panel>
              </div>
            )}

            {/* SYSTEM OVERVIEW CHIPS */}
            {!loading && !error && stats && (
              <div className="reveal r3">
                <GlassCard accent={C.cyan} hover={false} style={{ padding:"24px 28px" }}>
                  <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg, transparent, ${C.cyan}60, ${C.primary}, transparent)` }} />
                  <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:16 }}>🧮 System Overview</p>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:10 }}>
                    {[
                      { icon:"📁", val:datasetsUploaded,       label:"Datasets",          color:C.primary },
                      { icon:"🤖", val:modelsTrained,          label:"Models",             color:C.cyan    },
                      { icon:"📄", val:reportsGenerated,       label:"Reports",            color:C.teal    },
                      { icon:"🎯", val:bestAccuracy !== null ? `${bestAccuracy}%` : "N/A", label:"Best Acc", color:C.amber },
                      { icon:"📊", val:dashboards.length,      label:"Visual Dashboards",  color:C.violet  },
                      { icon:"🕒", val:recentTrainings.length, label:"Recent Trainings",   color:C.rose    },
                    ].map((s, i) => (
                      <div key={i} style={{ display:"flex", alignItems:"center", gap:8, padding:"8px 18px", borderRadius:30, background:C.isDark?`${s.color}14`:`${s.color}10`, border:`1px solid ${s.color}35`, backdropFilter:"blur(10px)" }}>
                        <span style={{ fontSize:14 }}>{s.icon}</span>
                        <span style={{ fontSize:16, fontWeight:800, color:s.color, fontFamily:"'Space Grotesk', sans-serif" }}>{s.val}</span>
                        <span style={{ fontSize:10, color:C.midText, fontWeight:600, letterSpacing:"0.08em", textTransform:"uppercase" }}>{s.label}</span>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </div>
            )}

            {/* VISUAL DASHBOARD REPORTS */}
            {!loading && !error && (
              <div className="reveal r4">
                <Panel title="Visual Dashboard Reports" subtitle="Auto-generated HTML analytics from your uploaded datasets" icon="📈" accent={C.rose} badge={dashboards.length>0?`${dashboards.length} REPORT${dashboards.length!==1?"S":""}`:undefined}>
                  {dashboards.length === 0 ? (
                    <EmptyState emoji="📊" title="No visual dashboards yet" sub="Go to the Upload page, upload a dataset, and click 📊 Dashboard to generate one." />
                  ) : (
                    <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                      {dashboards.map((r, i) => (
                        <GlassCard key={r._id || i} accent={C.violet} style={{ padding:"14px 18px" }}>
                          <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:12 }}>
                            <div style={{ display:"flex", alignItems:"center", gap:12, flex:1, minWidth:0 }}>
                              <div style={{ width:36, height:36, borderRadius:10, flexShrink:0, background:`linear-gradient(135deg, ${C.violet}20, ${C.primary}10)`, border:`1px solid ${C.violet}30`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:17 }}>📊</div>
                              <div style={{ minWidth:0 }}>
                                <p style={{ color:C.hiText, fontWeight:700, fontSize:13, fontFamily:"'Space Grotesk', monospace", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{r.report_filename}</p>
                                <p style={{ color:C.loText, fontSize:10, marginTop:2 }}>Dataset: {r.dataset_filename||"—"} · {r.created_at||""}</p>
                              </div>
                            </div>
                            <div style={{ display:"flex", gap:8, flexShrink:0 }}>
                              <button onClick={()=>setViewingHtml(viewingHtml===r.report_filename?null:r.report_filename)} style={{ display:"flex", alignItems:"center", gap:6, padding:"7px 14px", borderRadius:10, cursor:"pointer", border:`1px solid ${viewingHtml===r.report_filename?C.teal+"60":C.violet+"40"}`, background:viewingHtml===r.report_filename?`${C.teal}14`:`${C.violet}10`, color:viewingHtml===r.report_filename?C.teal:C.violet, fontSize:11, fontWeight:700, fontFamily:"'Space Grotesk', monospace", letterSpacing:"0.08em", textTransform:"uppercase", transition:"all 0.2s" }}>
                                {viewingHtml===r.report_filename?"✕ Close":"👁 View"}
                              </button>
                              <a href={`${API}/static/reports/${r.report_filename}`} target="_blank" rel="noreferrer" style={{ display:"flex", alignItems:"center", gap:6, padding:"7px 14px", borderRadius:10, cursor:"pointer", border:`1px solid ${C.border}`, background:C.isDark?`${C.primary}08`:`${C.primary}06`, color:C.midText, textDecoration:"none", fontSize:11, fontWeight:700, fontFamily:"'Space Grotesk', monospace", letterSpacing:"0.08em", textTransform:"uppercase" }}>↗ Open</a>
                            </div>
                          </div>
                          {viewingHtml===r.report_filename && (
                            <div className="reveal" style={{ marginTop:14, borderRadius:12, overflow:"hidden", border:`1px solid ${C.border}` }}>
                              <iframe src={`${API}/static/reports/${r.report_filename}`} title={r.report_filename} width="100%" style={{ height:"75vh", display:"block", border:"none" }} />
                            </div>
                          )}
                        </GlassCard>
                      ))}
                    </div>
                  )}
                </Panel>
              </div>
            )}

            {/* QUICK ACTIONS */}
            {!loading && !error && (
              <div className="reveal r5">
                <GlassCard accent={C.primary} hover={false} style={{ padding:"24px 28px" }}>
                  <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:`linear-gradient(90deg, transparent, ${C.rose}60, ${C.primary}, ${C.primary}88, transparent)` }} />
                  <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:16 }}>⚡ Quick Actions</p>
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(200px, 1fr))", gap:12 }}>
                    {[
                      { icon:"⬆️", label:"Upload Dataset",   sub:"CSV, XLSX, XLS",        href:"/upload",  color:C.primary },
                      { icon:"🧠", label:"Train a Model",    sub:"AutoML pipeline",        href:"/upload",  color:C.cyan    },
                      { icon:"📊", label:"View Reports",     sub:"PDF + HTML reports",     href:"/reports", color:C.teal    },
                      { icon:"🤖", label:"Browse Models",    sub:"Trained .pkl artefacts", href:"/models",  color:C.violet  },
                    ].map((action, i) => (
                      <a key={i} href={action.href} style={{ display:"flex", alignItems:"center", gap:12, padding:"14px 16px", borderRadius:14, textDecoration:"none", background:C.isDark?`${action.color}10`:`${action.color}08`, border:`1px solid ${action.color}30`, transition:"all 0.2s" }}>
                        <div style={{ width:36, height:36, borderRadius:10, flexShrink:0, background:`${action.color}18`, border:`1px solid ${action.color}30`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:18 }}>{action.icon}</div>
                        <div>
                          <p style={{ color:C.hiText, fontWeight:700, fontSize:13, fontFamily:"'Space Grotesk', sans-serif" }}>{action.label}</p>
                          <p style={{ color:C.loText, fontSize:10, marginTop:2 }}>{action.sub}</p>
                        </div>
                        <span style={{ marginLeft:"auto", color:action.color, fontSize:14 }}>→</span>
                      </a>
                    ))}
                  </div>
                </GlassCard>
              </div>
            )}

            {/* ════════════════════════════════════════════════════════════════
                ANALYTICS — only shown when routeResult is present
            ════════════════════════════════════════════════════════════════ */}
            {hasResult && (
              <>

                {/* ── SECTION 1 · OVERVIEW ─── */}
                <div className="reveal r6">
                  <Panel title="Overview" subtitle="Dataset dimensions · problem type · best model summary" icon="📊" accent={C.primary} badge="OVERVIEW">
                    {/* KPI tiles */}
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(140px, 1fr))", gap:12, marginBottom:22 }}>
                      {overviewKpis.map((kpi, i) => (
                        <KpiCard key={i} label={kpi.label} value={kpi.value} accent={kpi.accent} icon={kpi.icon} />
                      ))}
                    </div>
                    {/* Detail rows — all values are primitives */}
                    <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                      {overviewDetails.map((row, i) => <DetailRow key={i} {...row} />)}
                    </div>
                  </Panel>
                </div>

                {/* ── SECTION 2 · NUMERIC ANALYSIS ─── */}
                <div className="reveal r6">
                  <Panel title="Numeric Analysis" subtitle="Numeric columns detected · CV score chart" icon="🔢" accent={C.cyan} badge={numericCols.length>0?`${numericCols.length} COLS`:undefined}>
                    {/* Column chips */}
                    {numericCols.length > 0 ? (
                      <div style={{ display:"flex", flexWrap:"wrap", gap:8, marginBottom:20 }}>
                        {numericCols.map((col, i) => (
                          <span key={i} style={{ padding:"5px 14px", borderRadius:22, fontSize:12, fontWeight:500, background:C.isDark?`${C.cyan}14`:`${C.cyan}10`, border:`1px solid ${C.cyan}30`, color:C.cyan }}>{safeString(col)}</span>
                        ))}
                      </div>
                    ) : (
                      <div style={{ padding:"12px 16px", marginBottom:16, borderRadius:10, background:`${C.cyan}08`, border:`1px solid ${C.cyan}20` }}>
                        <p style={{ color:C.loText, fontSize:12 }}>Numeric column list is not included in this response. The chart below shows CV scores per model.</p>
                      </div>
                    )}

                    {/* FIX C: CV score chart from all_model_scores */}
                    {cvScores.length > 0 ? (
                      <div>
                        <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:14 }}>CV Score by Model</p>
                        <div style={{ width:"100%", height:220 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={cvScores} margin={{ top:4, right:8, left:-12, bottom:4 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                              <XAxis dataKey="name" tick={{ fill:C.midText, fontSize:10, fontFamily:"'Space Grotesk', monospace" }} axisLine={{ stroke:C.border }} tickLine={false} />
                              <YAxis tick={{ fill:C.midText, fontSize:10, fontFamily:"'Space Grotesk', monospace" }} axisLine={false} tickLine={false} domain={[0,1]} />
                              <Tooltip content={<CustomTooltip accentColor={C.cyan} />} cursor={{ fill:`${C.cyan}10` }} />
                              <Bar dataKey="value" fill={C.cyan} radius={[6,6,0,0]} maxBarSize={48} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        {/* Per-model CV detail rows */}
                        <div style={{ display:"flex", flexDirection:"column", gap:6, marginTop:16 }}>
                          {cvScores.map((entry, i) => (
                            <div key={i} style={{ display:"flex", alignItems:"center", gap:10 }}>
                              <span style={{ width:160, fontSize:11, color:C.midText, fontWeight:600, flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{entry.name}</span>
                              <div style={{ flex:1 }}><ProgressBar value={entry.value} color={i===0?C.amber:C.cyan} height={5} /></div>
                              <span style={{ width:58, fontSize:11, fontWeight:700, color:i===0?C.amber:C.cyan, fontFamily:"'Space Grotesk', monospace", textAlign:"right", flexShrink:0 }}>{(entry.value*100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div style={{ padding:"14px 18px", borderRadius:12, background:`${C.cyan}08`, border:`1px solid ${C.cyan}20` }}>
                        <p style={{ color:C.loText, fontSize:12 }}>CV score data not available. Train a model to populate this chart.</p>
                      </div>
                    )}
                  </Panel>
                </div>

                {/* ── SECTION 3 · CATEGORICAL ANALYSIS ─── */}
                <div className="reveal r6">
                  <Panel title="Categorical Analysis" subtitle="Categorical columns · top model predictions" icon="🔤" accent={C.violet} badge={catCols.length>0?`${catCols.length} COLS`:undefined}>
                    {catCols.length > 0 ? (
                      <div style={{ display:"flex", flexWrap:"wrap", gap:8, marginBottom:20 }}>
                        {catCols.map((col, i) => (
                          <span key={i} style={{ padding:"5px 14px", borderRadius:22, fontSize:12, fontWeight:500, background:C.isDark?`${C.violet}14`:`${C.violet}10`, border:`1px solid ${C.violet}30`, color:C.violet }}>{safeString(col)}</span>
                        ))}
                      </div>
                    ) : (
                      <p style={{ color:C.loText, fontSize:12, marginBottom:topPreds.length>0?20:0 }}>Categorical column list not included in this response.</p>
                    )}

                    {topPreds.length > 0 && (
                      <div>
                        <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12 }}>Top Model Predictions</p>
                        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                          {topPreds.map((pred, i) => {
                            const predLabel = typeof pred==="object"&&pred!==null ? safeString(pred.label??pred.name??pred.value) : safeString(pred);
                            const predConf  = typeof pred==="object"&&pred!==null&&typeof pred.confidence==="number" ? pred.confidence : null;
                            return (
                              <div key={i} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:12, padding:"10px 16px", borderRadius:10, background:`linear-gradient(135deg, ${C.surface}, ${C.violetTint}40)`, border:`1px solid ${C.border}` }}>
                                <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                                  <div style={{ width:22, height:22, borderRadius:6, flexShrink:0, background:`${C.violet}14`, border:`1px solid ${C.violet}28`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:9, fontWeight:800, color:C.violet, fontFamily:"'Space Grotesk', monospace" }}>{i+1}</div>
                                  <span style={{ color:C.hiText, fontSize:13, fontWeight:600 }}>{predLabel}</span>
                                </div>
                                {predConf !== null && <span style={{ color:C.violet, fontSize:12, fontWeight:700, fontFamily:"'Space Grotesk', monospace" }}>{(predConf*100).toFixed(1)}%</span>}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* FIX: show leaderboard from all_model_scores when no catCols / topPreds */}
                    {catCols.length === 0 && topPreds.length === 0 && (
                      (() => {
                        const allScores = perf?.all_model_scores;
                        if (!allScores || typeof allScores !== "object") {
                          return <EmptyState emoji="🔤" title="No categorical data" sub="Categorical columns and top predictions will appear here after training." />;
                        }
                        const rows = Object.entries(allScores).map(([name, s]) => {
                          const testScore = problemType === "regression" ? (s?.R2??s?.r2??null) : (s?.accuracy??null);
                          return { name, testScore, f1: s?.f1_macro??null };
                        }).sort((a, b) => (b.testScore??-Infinity) - (a.testScore??-Infinity));
                        if (rows.length === 0) return <EmptyState emoji="🔤" title="No categorical data" sub="Categorical columns and top predictions will appear here after training." />;
                        return (
                          <div>
                            <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12 }}>All Model Test Scores</p>
                            <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
                              {rows.map((row, i) => (
                                <div key={i} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", gap:12, padding:"10px 16px", borderRadius:10, background:`linear-gradient(135deg, ${C.surface}, ${C.violetTint}40)`, border:`1px solid ${C.border}` }}>
                                  <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                                    <span style={{ color:C.violet, fontSize:11, fontWeight:700, fontFamily:"'Space Grotesk', monospace", width:18 }}>{i+1}</span>
                                    <span style={{ color:C.hiText, fontSize:13, fontWeight:600 }}>{row.name}</span>
                                  </div>
                                  <div style={{ display:"flex", gap:12, alignItems:"center" }}>
                                    {row.f1 !== null && <span style={{ color:C.midText, fontSize:11 }}>F1 {(row.f1*100).toFixed(1)}%</span>}
                                    {row.testScore !== null && (
                                      <span style={{ color:i===0?C.amber:C.violet, fontSize:12, fontWeight:700, fontFamily:"'Space Grotesk', monospace" }}>
                                        {problemType==="regression" ? safeNumber(row.testScore,4) : pct(row.testScore)}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })()
                    )}
                  </Panel>
                </div>

                {/* ── SECTION 4 · PIPELINE STAGE LOGS ─── */}
                <div className="reveal r7">
                  <Panel title="Pipeline Stage Logs" subtitle="Execution timeline — stage · status · elapsed time" icon="📅" accent={C.teal} badge={stageLogs.length>0?`${stageLogs.length} STAGES`:undefined}>
                    {stageLogs.length === 0 ? (
                      <EmptyState emoji="📅" title="No pipeline logs" sub="Stage execution data will appear here after a model training run." />
                    ) : (
                      <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                        <div style={{ display:"grid", gridTemplateColumns:"1.6fr 1fr 0.8fr", gap:8, padding:"6px 14px 10px", borderBottom:`1px solid ${C.border}` }}>
                          {["Stage","Status","Elapsed"].map((h, i) => (
                            <span key={i} style={{ color:C.loText, fontSize:9, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace" }}>{h}</span>
                          ))}
                        </div>
                        {stageLogs.map((stage, i) => {
                          const stageName   = safeString(stage?.stage ?? `Stage ${i+1}`);
                          const stageStatus = safeString(stage?.status ?? "—");
                          const elapsedRaw  = stage?.elapsed_s;
                          const stageTime   = typeof elapsedRaw === "number" && !isNaN(elapsedRaw) ? `${elapsedRaw.toFixed(2)}s` : "—";
                          const isDone  = stageStatus.toLowerCase() === "done"  || stageStatus.toLowerCase() === "complete";
                          const isError = stageStatus.toLowerCase() === "error" || stageStatus.toLowerCase() === "failed";
                          const rowColor = isError ? C.rose : isDone ? C.teal : C.amber;
                          return (
                            <div key={i} className="row-hover" style={{ display:"grid", gridTemplateColumns:"1.6fr 1fr 0.8fr", gap:8, padding:"11px 14px", borderRadius:10, transition:"background 0.15s", background:C.isDark?`${rowColor}08`:`${rowColor}05`, border:`1px solid ${rowColor}18` }}>
                              <span style={{ color:C.hiText, fontSize:12, fontWeight:600 }}>{stageName}</span>
                              <div><InlineBadge text={stageStatus.toUpperCase()} color={rowColor} /></div>
                              <span style={{ color:C.loText, fontSize:11, fontFamily:"'Space Grotesk', monospace" }}>{stageTime}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </Panel>
                </div>

                {/* ── SECTION 5 · DATA QUALITY ─── */}
                <div className="reveal r8">
                  <Panel title="Data Quality" subtitle="Risk flags · retrain decision · AI insight" icon="🩺" accent={C.rose} badge={riskFlags.length>0?`${riskFlags.length} RISK${riskFlags.length!==1?"S":""}`:undefined}>

                    {/* Risk flags */}
                    {riskFlags.length > 0 ? (
                      <div style={{ marginBottom:20 }}>
                        <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12, display:"flex", alignItems:"center", gap:8 }}>
                          <span style={{ fontSize:13 }}>⚠️</span>Risk Flags ({riskFlags.length})
                        </p>
                        <div style={{ display:"flex", flexWrap:"wrap", gap:10 }}>
                          {riskFlags.map((risk, i) => {
                            const riskLabel = typeof risk==="object"&&risk!==null ? safeString(risk.label??risk.name??risk.type??risk.message) : safeString(risk);
                            return (
                              <div key={i} style={{ display:"flex", alignItems:"center", gap:8, padding:"8px 14px", borderRadius:24, background:`${C.rose}12`, border:`1px solid ${C.rose}30` }}>
                                <span style={{ fontSize:13 }}>⚠️</span>
                                <span style={{ color:C.rose, fontSize:12, fontWeight:700, fontFamily:"'Space Grotesk', monospace" }}>{riskLabel}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div style={{ padding:"12px 16px", marginBottom:20, borderRadius:10, background:`${C.teal}10`, border:`1px solid ${C.teal}28`, display:"flex", alignItems:"center", gap:10 }}>
                        <span style={{ fontSize:16 }}>✓</span>
                        <p style={{ color:C.teal, fontSize:12, fontWeight:600 }}>No risk flags detected — data quality looks healthy.</p>
                      </div>
                    )}

                    {/* FIX: extra data quality rows from dataset_diagnostics */}
                    {routeResult?.dataset_diagnostics && (() => {
                      const diag = routeResult.dataset_diagnostics;
                      const qualRows = [
                        { label:"Total Rows",            value: diag.n_rows   != null ? String(diag.n_rows)   : null, color:C.primary },
                        { label:"Total Columns",         value: diag.n_cols   != null ? String(diag.n_cols)   : null, color:C.cyan    },
                        { label:"Overall Missing %",     value: diag.overall_missing_pct != null ? `${diag.overall_missing_pct}%` : null, color: diag.overall_missing_pct > 5 ? C.amber : C.teal },
                        { label:"Class Imbalance Ratio", value: diag.imbalance_ratio != null ? `${diag.imbalance_ratio}×` : null, color: diag.imbalance_ratio > 3 ? C.amber : C.teal },
                        { label:"Num Classes",           value: diag.n_classes != null ? String(diag.n_classes) : null, color:C.violet },
                      ].filter(r => r.value !== null);
                      if (qualRows.length === 0) return null;
                      return (
                        <div style={{ marginBottom:20 }}>
                          <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12 }}>Dataset Diagnostics</p>
                          <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                            {qualRows.map((row, i) => <DetailRow key={i} {...row} />)}
                          </div>
                        </div>
                      );
                    })()}

                    {/* Retrain decision */}
                    <div style={{ marginBottom:16 }}>
                      <p style={{ color:C.midText, fontSize:10, fontWeight:700, letterSpacing:"0.14em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:12, display:"flex", alignItems:"center", gap:8 }}>
                        <span style={{ fontSize:13 }}>🔄</span>Retrain Decision
                      </p>
                      {routeResult?.retrain_decision != null ? (
                        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                          <DetailRow label="Should Retrain" value={shouldRetrain?"YES — Retraining Recommended":"NO — Model is Stable"} color={shouldRetrain?C.amber:C.teal} />
                          {retrainSeverity !== "—" && <DetailRow label="Severity" value={retrainSeverity} color={C.amber} />}
                          {retrainReason   !== "—" && <DetailRow label="Reason"   value={retrainReason}   color={C.midText} />}
                        </div>
                      ) : (
                        <p style={{ color:C.loText, fontSize:12 }}>No retrain decision available for this run.</p>
                      )}
                    </div>

                    {/* Leakage + overfitting banners */}
                    {routeResult?.leakage_detected && (
                      <div style={{ padding:"12px 16px", marginBottom:10, borderRadius:10, background:`${C.rose}10`, border:`1px solid ${C.rose}30`, display:"flex", alignItems:"center", gap:10 }}>
                        <span style={{ fontSize:16 }}>🔍</span>
                        <div>
                          <p style={{ color:C.rose, fontSize:11, fontWeight:700, letterSpacing:"0.08em", textTransform:"uppercase" }}>Data Leakage Detected</p>
                          {Array.isArray(routeResult.removed_features) && routeResult.removed_features.length > 0 && (
                            <p style={{ color:C.midText, fontSize:12, marginTop:2 }}>Removed: {routeResult.removed_features.join(", ")}</p>
                          )}
                        </div>
                      </div>
                    )}

                    {routeResult?.overfitting && (
                      <div style={{ padding:"12px 16px", marginBottom:10, borderRadius:10, background:`${C.amber}10`, border:`1px solid ${C.amber}30`, display:"flex", alignItems:"center", gap:10 }}>
                        <span style={{ fontSize:16 }}>📈</span>
                        <div>
                          <p style={{ color:C.amber, fontSize:11, fontWeight:700, letterSpacing:"0.08em", textTransform:"uppercase" }}>Overfitting Detected</p>
                          <p style={{ color:C.midText, fontSize:12, marginTop:2 }}>Training score significantly exceeds CV score. Consider regularisation.</p>
                        </div>
                      </div>
                    )}

                    {/* AI Insight */}
                    {metaInsightRec !== "—" && (
                      <div style={{ padding:"18px 20px", marginTop:8, borderRadius:14, background:`linear-gradient(135deg, ${C.violetTint}80, ${C.coralTint}60)`, border:`1px solid ${C.primary}25`, borderLeft:`4px solid ${C.primary}` }}>
                        <p style={{ color:C.primary, fontSize:10, fontWeight:700, letterSpacing:"0.16em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace", marginBottom:10, display:"flex", alignItems:"center", gap:8 }}>
                          <span style={{ width:22, height:22, borderRadius:6, fontSize:11, background:`${C.primary}14`, border:`1px solid ${C.primary}28`, display:"flex", alignItems:"center", justifyContent:"center" }}>🤖</span>
                          AI Insight
                          {metaInsightSource && <span style={{ color:C.loText, fontSize:9, fontWeight:600, marginLeft:4 }}>· {metaInsightSource}</span>}
                        </p>
                        <p style={{ color:C.hiText, fontSize:13, lineHeight:1.85, fontStyle:"italic" }}>"{metaInsightRec}"</p>
                        {metaInsightConf !== null && (
                          <p style={{ color:C.midText, fontSize:11, marginTop:10 }}>Confidence: <span style={{ color:C.primary, fontWeight:700 }}>{(metaInsightConf*100).toFixed(1)}%</span></p>
                        )}
                      </div>
                    )}
                  </Panel>
                </div>

              </>
            )}

            {/* FOOTER */}
            <div style={{ textAlign:"center", padding:"8px 0", display:"flex", alignItems:"center", justifyContent:"center", gap:16 }}>
              <div style={{ flex:1, height:1, background:`linear-gradient(90deg, transparent, ${C.border})` }} />
              <p style={{ color:C.loText, fontSize:10, letterSpacing:"0.22em", textTransform:"uppercase", fontFamily:"'Space Grotesk', monospace" }}>AutoAnalytica AI Platform · Powered by Machine Intelligence</p>
              <div style={{ flex:1, height:1, background:`linear-gradient(90deg, ${C.border}, transparent)` }} />
            </div>

          </main>
        </div>

        <Toast text={toast.text} ok={toast.ok} onClose={() => setToast({ text:"" })} />
      </>
    </ThemeCtx.Provider>
  );
}