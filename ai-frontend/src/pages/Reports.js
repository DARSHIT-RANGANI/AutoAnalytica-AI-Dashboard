import React, { useState, useEffect, useRef, createContext, useContext } from "react";

const API = "http://127.0.0.1:8000";

// ══════════════════════════════════════════════════════════════════════════════
// THEME SYSTEM — identical to Upload.js
// ══════════════════════════════════════════════════════════════════════════════
const LIGHT = {
  bg:         "#fdfaff",
  surface:    "#f5eeff",
  card:       "#ffffff",
  border:     "#e4d0f8",
  borderHi:   "#c080e0",
  primary:    "#9b40d0",
  violet:     "#b060e0",
  cyan:       "#e0409a",
  teal:       "#c030a0",
  amber:      "#d070b8",
  rose:       "#e8409a",
  hiText:     "#1e0838",
  midText:    "#7030a0",
  loText:     "#c090d8",
  coralTint:  "#fff0f8",
  violetTint: "#f8f0ff",
  tealTint:   "#fff5f8",
  isDark:     false,
};

const DARK = {
  bg:         "#0d0b18",
  surface:    "#13111f",
  card:       "#1a1730",
  border:     "#2a2448",
  borderHi:   "#3d3570",
  primary:    "#9b7ff0",
  violet:     "#b49dfc",
  cyan:       "#22c5c5",
  teal:       "#14d4bb",
  amber:      "#e89a5a",
  rose:       "#e86892",
  hiText:     "#ede9ff",
  midText:    "#9088c0",
  loText:     "#3a3360",
  coralTint:  "#1e1228",
  violetTint: "#1c1838",
  tealTint:   "#0f1e1c",
  isDark:     true,
};

const ThemeCtx = createContext(LIGHT);
const useTheme = () => useContext(ThemeCtx);

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt = s => String(s || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

// ── NeuralGrid background ─────────────────────────────────────────────────────
function NeuralGrid({ isDark }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let frame, t = 0;
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    const nodes = Array.from({ length: 55 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 1.8 + 0.6,
      pulse: Math.random() * Math.PI * 2,
      type: Math.random() > 0.7 ? "coral" : Math.random() > 0.5 ? "teal" : "violet",
    }));
    const draw = () => {
      t += 0.008;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < 145) {
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = isDark
              ? `rgba(155,127,240,${(1 - d / 145) * 0.22})`
              : `rgba(107,79,200,${(1 - d / 145) * 0.12})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }
      nodes.forEach(n => {
        const p = Math.sin(t * 1.5 + n.pulse) * 0.5 + 0.5;
        const colors = isDark ? {
          violet: `rgba(155,127,240,${0.45 + p * 0.4})`,
          coral:  `rgba(232,104,146,${0.4  + p * 0.38})`,
          teal:   `rgba(20,212,187,${0.4   + p * 0.38})`,
        } : {
          violet: `rgba(107,79,200,${0.25 + p * 0.3})`,
          coral:  `rgba(229,97,60,${0.2   + p * 0.28})`,
          teal:   `rgba(30,158,158,${0.2  + p * 0.28})`,
        };
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + p * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = colors[n.type];
        ctx.fill();
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(frame); window.removeEventListener("resize", resize); };
  }, [isDark]);
  return (
    <canvas ref={ref} style={{
      position: "fixed", inset: 0, zIndex: 0,
      pointerEvents: "none", opacity: isDark ? 0.65 : 0.62,
    }} />
  );
}

// ── DataStreams ────────────────────────────────────────────────────────────────
function DataStreams() {
  const C = useTheme();
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none", overflow: "hidden" }}>
      {[...Array(6)].map((_, i) => (
        <div key={i} style={{
          position: "absolute", left: `${10 + i * 16}%`, top: 0, bottom: 0, width: 1,
          background: `linear-gradient(180deg, transparent, ${C.primary}18, ${C.cyan}22, transparent)`,
          animation: `streamFlow ${3 + i * 0.7}s ease-in-out infinite`,
          animationDelay: `${i * 0.5}s`, opacity: 0.5,
        }} />
      ))}
    </div>
  );
}

// ── GlassCard ─────────────────────────────────────────────────────────────────
function GlassCard({ children, style = {}, accent, hover = true }) {
  const C = useTheme();
  const accentColor = accent || C.primary;
  const [h, setH]     = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const ref = useRef(null);
  const onMove = e => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    setPos({ x: e.clientX - r.left, y: e.clientY - r.top });
  };
  return (
    <div ref={ref}
      onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)} onMouseMove={onMove}
      style={{
        position: "relative",
        background: `linear-gradient(135deg, ${C.card}f0, ${C.surface}cc)`,
        border: `1px solid ${h && hover ? accentColor + "55" : C.border}`,
        borderRadius: 20, backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
        transition: "border-color 0.3s, transform 0.2s, box-shadow 0.3s",
        transform: h && hover ? "translateY(-2px)" : "none",
        boxShadow: h && hover
          ? C.isDark
            ? `0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px ${accentColor}33, inset 0 1px 0 ${accentColor}18`
            : `0 20px 60px rgba(107,79,200,0.12), 0 0 0 1px ${accentColor}22, inset 0 1px 0 rgba(255,255,255,0.9)`
          : C.isDark
            ? `0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)`
            : `0 4px 24px rgba(107,79,200,0.07), 0 1px 4px rgba(0,0,0,0.04), inset 0 1px 0 rgba(255,255,255,0.8)`,
        overflow: "hidden", ...style,
      }}>
      {h && hover && (
        <div style={{
          position: "absolute", width: 300, height: 300, borderRadius: "50%",
          background: `radial-gradient(circle, ${accentColor}${C.isDark ? "18" : "0c"} 0%, transparent 70%)`,
          left: pos.x - 150, top: pos.y - 150, pointerEvents: "none", zIndex: 0,
        }} />
      )}
      <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────────
function Panel({ title, subtitle, icon, children, accent, badge }) {
  const C = useTheme();
  const accentColor = accent || C.primary;
  return (
    <GlassCard accent={accent} hover={false} style={{ padding: "28px 30px 32px" }}>
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, transparent, ${C.rose}60, ${accentColor}, ${accentColor}88, transparent)`,
      }} />
      {title && (
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: subtitle ? 4 : 24 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, flexShrink: 0,
            background: `${accentColor}14`, border: `1px solid ${accentColor}28`,
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 17,
            boxShadow: `0 2px 8px ${accentColor}15`,
          }}>{icon}</div>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h2 style={{
                color: C.hiText, fontSize: 13, fontWeight: 700, letterSpacing: "0.12em",
                textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace",
              }}>{title}</h2>
              {badge && (
                <span style={{
                  padding: "2px 8px", borderRadius: 20, fontSize: 10, fontWeight: 700,
                  background: `${accentColor}14`, color: accentColor,
                  border: `1px solid ${accentColor}28`, letterSpacing: "0.08em",
                }}>{badge}</span>
              )}
            </div>
          </div>
          <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${accentColor}35, transparent)` }} />
        </div>
      )}
      {subtitle && <p style={{ color: C.midText, fontSize: 12, marginBottom: 22, marginLeft: 48 }}>{subtitle}</p>}
      {children}
    </GlassCard>
  );
}

// ── StepIndicator — identical to Upload.js ────────────────────────────────────
function StepIndicator({ steps }) {
  const C = useTheme();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
      {steps.map((s, i) => (
        <React.Fragment key={i}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <div style={{
              width: 28, height: 28, borderRadius: "50%",
              border: `2px solid ${s.done ? C.teal : s.active ? C.primary : C.border}`,
              background: s.done ? `${C.teal}18` : s.active ? `${C.primary}12` : C.surface,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 11, fontWeight: 800,
              color: s.done ? C.teal : s.active ? C.primary : C.midText,
              transition: "all 0.4s",
              boxShadow: s.active ? `0 0 16px ${C.primary}35` : s.done ? `0 0 12px ${C.teal}28` : "none",
            }}>{s.done ? "✓" : i + 1}</div>
            <span style={{
              fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase",
              fontWeight: 700, fontFamily: "'Space Grotesk', monospace",
              color: s.done ? C.teal : s.active ? C.primary : C.loText, transition: "color 0.4s",
            }}>{s.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div style={{
              flex: 1, height: 1, margin: "0 6px", marginBottom: 18,
              background: steps[i].done
                ? `linear-gradient(90deg, ${C.teal}60, ${C.teal}25)`
                : C.border,
              transition: "background 0.4s",
            }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function Toast({ text, ok, onClose }) {
  const C = useTheme();
  if (!text) return null;
  const color = ok ? C.teal : C.rose;
  return (
    <div style={{
      position: "fixed", top: 80, right: 24, zIndex: 9998, padding: "14px 20px",
      background: C.surface, border: `1px solid ${color}35`, borderLeft: `3px solid ${color}`,
      borderRadius: 14, fontSize: 13, fontWeight: 600,
      display: "flex", alignItems: "center", gap: 10,
      boxShadow: `0 8px 32px rgba(107,79,200,0.12), 0 0 0 1px ${color}12`,
      animation: "slideInRight 0.3s cubic-bezier(0.34,1.56,0.64,1)",
      minWidth: 280, maxWidth: 360,
    }}>
      <span style={{ width: 24, height: 24, borderRadius: "50%", background: `${color}16`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, flexShrink: 0, color }}>{ok ? "✓" : "✕"}</span>
      <span style={{ flex: 1, color: C.hiText }}>{text}</span>
      <button onClick={onClose} style={{ background: "none", border: "none", color: C.midText, cursor: "pointer", fontSize: 18, lineHeight: 1, padding: 0 }}>×</button>
    </div>
  );
}

// ── ConfirmDialog ─────────────────────────────────────────────────────────────
function ConfirmDialog({ show, title, message, onConfirm, onCancel }) {
  const C = useTheme();
  if (!show) return null;
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9999,
      background: "rgba(0,0,0,0.6)", backdropFilter: "blur(8px)",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <GlassCard accent={C.rose} hover={false} style={{ padding: "32px 36px", maxWidth: 420, width: "90%", margin: "0 auto" }}>
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${C.rose}, transparent)` }} />
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>🗑️</div>
          <h3 style={{ color: C.hiText, fontSize: 17, fontWeight: 700, fontFamily: "'Space Grotesk', sans-serif", marginBottom: 8 }}>{title}</h3>
          <p style={{ color: C.midText, fontSize: 13, lineHeight: 1.7 }}>{message}</p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <button onClick={onCancel} style={{
            padding: "11px", borderRadius: 12, cursor: "pointer",
            border: `1px solid ${C.border}`, background: C.surface,
            color: C.midText, fontSize: 12, fontWeight: 700,
            fontFamily: "'Space Grotesk', monospace", letterSpacing: "0.08em", textTransform: "uppercase",
          }}>Cancel</button>
          <button onClick={onConfirm} style={{
            padding: "11px", borderRadius: 12, cursor: "pointer",
            border: `1px solid ${C.rose}50`, background: `${C.rose}14`,
            color: C.rose, fontSize: 12, fontWeight: 700,
            fontFamily: "'Space Grotesk', monospace", letterSpacing: "0.08em", textTransform: "uppercase",
          }}>Delete</button>
        </div>
      </GlassCard>
    </div>
  );
}

// ── LoadingSpinner ────────────────────────────────────────────────────────────
function LoadingSpinner({ message = "Loading reports…" }) {
  const C = useTheme();
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "60px 0", justifyContent: "center" }}>
      <div style={{ position: "relative", width: 44, height: 44 }}>
        {[C.primary, C.rose, C.cyan].map((col, i) => (
          <div key={i} style={{
            position: "absolute", inset: i * 6,
            border: "1.5px solid transparent", borderTopColor: col,
            borderRadius: "50%", animation: `orbit ${0.8 + i * 0.4}s linear infinite`,
          }} />
        ))}
      </div>
      <p style={{ color: C.midText, fontSize: 13, fontFamily: "'Space Grotesk', monospace", letterSpacing: "0.1em" }}>
        {message}
      </p>
    </div>
  );
}

// ── EmptyState ────────────────────────────────────────────────────────────────
function EmptyState({ emoji, title, sub }) {
  const C = useTheme();
  return (
    <div style={{ textAlign: "center", padding: "48px 20px" }}>
      <p style={{ fontSize: 48, marginBottom: 16 }}>{emoji}</p>
      <p style={{ color: C.hiText, fontWeight: 700, fontSize: 15, fontFamily: "'Space Grotesk', sans-serif", marginBottom: 8 }}>{title}</p>
      <p style={{ color: C.midText, fontSize: 13, lineHeight: 1.7 }}>{sub}</p>
    </div>
  );
}

// ── TypeBadge ─────────────────────────────────────────────────────────────────
function TypeBadge({ filename }) {
  const C = useTheme();
  const isPDF  = filename?.endsWith(".pdf");
  const isHTML = filename?.endsWith(".html");
  const color  = isPDF ? C.rose : isHTML ? C.violet : C.amber;
  const label  = isPDF ? "PDF" : isHTML ? "HTML" : "FILE";
  const icon   = isPDF ? "📄" : isHTML ? "🌐" : "📎";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "3px 10px", borderRadius: 20, fontSize: 10, fontWeight: 700,
      background: `${color}14`, color, border: `1px solid ${color}28`,
      letterSpacing: "0.08em", fontFamily: "'Space Grotesk', monospace",
    }}>{icon} {label}</span>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN REPORTS COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
export default function Reports() {

  const [isDark, setIsDark] = useState(() => {
    try { return localStorage.getItem("aa_theme") === "dark"; }
    catch { return false; }
  });

  const theme = isDark ? DARK : LIGHT;
  const C     = theme;

  useEffect(() => {
    try { localStorage.setItem("aa_theme", isDark ? "dark" : "light"); }
    catch {}
    document.body.style.background = isDark ? DARK.bg : LIGHT.bg;
  }, [isDark]);

  // ── State ──────────────────────────────────────────────────────────────────
  const [reports,      setReports]      = useState([]);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState("");
  const [toast,        setToast]        = useState({ text: "", ok: true });
  const [viewingId,    setViewingId]    = useState(null);   // _id of report being previewed
  const [deleting,     setDeleting]     = useState(null);   // _id being deleted
  const [confirmId,    setConfirmId]    = useState(null);   // _id to confirm delete
  const [search,       setSearch]       = useState("");
  const [filterType,   setFilterType]   = useState("all"); // "all" | "pdf" | "html"

  // Step indicator — all done since we're on the reports page
  const steps = [
    { label: "Upload",    done: true,  active: false },
    { label: "Configure", done: true,  active: false },
    { label: "Train",     done: true,  active: false },
    { label: "Predict",   done: false, active: true  },
  ];

  const notify = (text, ok = true) => {
    setToast({ text, ok });
    setTimeout(() => setToast({ text: "", ok: true }), 4500);
  };

  // ── Fetch ──────────────────────────────────────────────────────────────────
  const fetchReports = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/reports/list`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setReports(data.reports || []);
    } catch (err) {
      setError("Could not load reports. Is the backend running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchReports(); }, []);

  // ── Delete ─────────────────────────────────────────────────────────────────
  const handleDelete = async (reportId) => {
    setConfirmId(null);
    setDeleting(reportId);
    try {
      const res = await fetch(`${API}/reports/delete/${reportId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setReports(r => r.filter(x => x._id !== reportId));
      if (viewingId === reportId) setViewingId(null);
      notify("Report deleted successfully ✓");
    } catch (err) {
      notify("Failed to delete report.", false);
      console.error(err);
    } finally {
      setDeleting(null);
    }
  };

  // ── Filtered list ──────────────────────────────────────────────────────────
  const filtered = reports.filter(r => {
    const name = (r.report_filename || "").toLowerCase();
    const ds   = (r.dataset_filename || "").toLowerCase();
    const matchSearch = !search || name.includes(search.toLowerCase()) || ds.includes(search.toLowerCase());
    const matchType   = filterType === "all"
      || (filterType === "pdf"  && name.endsWith(".pdf"))
      || (filterType === "html" && name.endsWith(".html"));
    return matchSearch && matchType;
  });

  const pdfCount  = reports.filter(r => r.report_filename?.endsWith(".pdf")).length;
  const htmlCount = reports.filter(r => r.report_filename?.endsWith(".html")).length;

  // ── Viewing report ─────────────────────────────────────────────────────────
  const viewingReport = reports.find(r => r._id === viewingId) || null;

  return (
    <ThemeCtx.Provider value={theme}>
      <>
        {/* ── Global CSS ── */}
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
          *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
          html, body {
            background:${C.bg}; color:${C.hiText};
            font-family:'DM Sans',sans-serif;
            overflow-x:hidden; min-height:100vh;
            transition: background 0.4s ease, color 0.3s ease;
          }
          @keyframes streamFlow  { 0%,100%{opacity:.1;transform:scaleY(.3) translateY(-20%)} 50%{opacity:.45;transform:scaleY(1) translateY(0)} }
          @keyframes orbit       { to{transform:rotate(360deg)} }
          @keyframes shimmerBar  { 0%{transform:translateX(-100%)} 100%{transform:translateX(300%)} }
          @keyframes slideInRight{ from{opacity:0;transform:translateX(20px)} to{opacity:1;transform:translateX(0)} }
          @keyframes fadeSlideUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
          @keyframes titleGlow   { 0%{background-position:0% center} 100%{background-position:300% center} }
          @keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:.5} }
          .reveal { animation:fadeSlideUp .55s cubic-bezier(.34,1.1,.64,1) both; }
          .r1{animation-delay:.05s} .r2{animation-delay:.12s} .r3{animation-delay:.20s}
          .r4{animation-delay:.28s} .r5{animation-delay:.36s}
          .report-row:hover { background:${C.isDark ? `${C.primary}08` : `${C.primary}05`} !important; }
          ::-webkit-scrollbar{width:4px}
          ::-webkit-scrollbar-track{background:${C.bg}}
          ::-webkit-scrollbar-thumb{background:${C.border};border-radius:4px}
          input::placeholder{color:${C.loText}}
        `}</style>

        {/* ── Backgrounds ── */}
        {isDark ? (
          <>
            <div style={{ position:"fixed",inset:0,zIndex:0,backgroundImage:`url("https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=1800&auto=format&fit=crop&q=80")`,backgroundSize:"cover",backgroundPosition:"center" }} />
            <div style={{ position:"fixed",inset:0,zIndex:1,background:`linear-gradient(160deg, rgba(8,6,18,0.52) 0%, rgba(13,8,30,0.46) 35%, rgba(10,5,22,0.50) 70%, rgba(5,3,14,0.55) 100%)` }} />
            <div style={{ position:"fixed",inset:0,zIndex:2,pointerEvents:"none",background:`radial-gradient(ellipse 55% 45% at 10% 35%, rgba(155,127,240,0.18) 0%, transparent 60%), radial-gradient(ellipse 45% 35% at 90% 15%, rgba(20,212,187,0.14) 0%, transparent 55%)` }} />
          </>
        ) : (
          <>
            <div style={{ position:"fixed",inset:0,zIndex:0,backgroundImage:`url("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1800&auto=format&fit=crop&q=85")`,backgroundSize:"cover",backgroundPosition:"center 40%" }} />
            <div style={{ position:"fixed",inset:0,zIndex:1,background:`linear-gradient(160deg, rgba(255,252,255,0.55) 0%, rgba(252,245,255,0.48) 35%, rgba(255,248,255,0.52) 70%, rgba(250,245,255,0.58) 100%)` }} />
            <div style={{ position:"fixed",inset:0,zIndex:2,pointerEvents:"none",background:`radial-gradient(ellipse 55% 45% at 15% 30%, rgba(155,64,208,0.06) 0%, transparent 60%)` }} />
          </>
        )}

        <NeuralGrid isDark={isDark} />
        <DataStreams />

        <div style={{ position: "relative", zIndex: 2, minHeight: "100vh" }}>

          {/* ════════════════════════════════════════════════════════════════
              HEADER — identical layout to Upload.js
          ════════════════════════════════════════════════════════════════ */}
          <header style={{
            padding: "0 40px", height: 68,
            display: "flex", alignItems: "center", justifyContent: "space-between",
            background: isDark ? `rgba(13,11,24,0.92)` : `rgba(255,252,255,0.92)`,
            backdropFilter: "blur(30px)", WebkitBackdropFilter: "blur(30px)",
            borderBottom: `1px solid ${C.border}`,
            boxShadow: isDark
              ? "0 1px 0 rgba(255,255,255,0.04), 0 4px 20px rgba(0,0,0,0.4)"
              : "0 1px 0 rgba(255,255,255,0.9), 0 4px 20px rgba(160,60,200,0.10)",
            position: "sticky", top: 0, zIndex: 100,
          }}>

            {/* ── Logo ── */}
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{
                width: 38, height: 38, borderRadius: 12,
                background: `linear-gradient(135deg, ${C.primary}, ${C.rose})`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 18, boxShadow: `0 4px 16px ${C.primary}35`,
              }}>⚡</div>
              <div>
                <div style={{ fontSize: 17, fontWeight: 800, fontFamily: "'Space Grotesk',sans-serif", letterSpacing: "0.02em" }}>
                  <span style={{
                    display: "inline-block",
                    background: `linear-gradient(90deg, ${C.hiText}, ${C.primary}, ${C.rose}, ${C.hiText})`,
                    backgroundSize: "300% auto", WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent", backgroundClip: "text",
                    animation: "titleGlow 7s linear infinite",
                  }}>AutoAnalytica</span>
                </div>
                <div style={{ color: C.loText, fontSize: 10, letterSpacing: "0.16em", textTransform: "uppercase" }}>
                  ML Platform v2.0
                </div>
              </div>
            </div>

            {/* ── Step Indicator — centre ── */}
            <div style={{ flex: 1, maxWidth: 520, padding: "0 40px" }}>
              <StepIndicator steps={steps} />
            </div>

            {/* ── Right controls ── */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>

              {/* Theme toggle — identical to Upload.js */}
              <button onClick={() => setIsDark(d => !d)} title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
                style={{
                  display: "flex", alignItems: "center", gap: 8, padding: "7px 14px",
                  borderRadius: 24, cursor: "pointer", border: `1px solid ${C.border}`,
                  background: isDark
                    ? `linear-gradient(135deg, rgba(155,127,240,0.15), rgba(20,212,187,0.08))`
                    : `linear-gradient(135deg, rgba(107,79,200,0.08), rgba(212,84,122,0.05))`,
                  boxShadow: isDark
                    ? "0 2px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)"
                    : "0 2px 8px rgba(107,79,200,0.10), inset 0 1px 0 rgba(255,255,255,0.8)",
                  transition: "all 0.3s cubic-bezier(0.34,1.56,0.64,1)",
                }}>
                <div style={{
                  width: 40, height: 22, borderRadius: 11, position: "relative",
                  background: isDark
                    ? `linear-gradient(135deg, ${C.primary}, ${C.teal})`
                    : `linear-gradient(135deg, ${C.primary}60, ${C.rose}80)`,
                  transition: "background 0.35s",
                }}>
                  <div style={{
                    position: "absolute", top: 3, left: isDark ? 21 : 3,
                    width: 16, height: 16, borderRadius: "50%", background: "#fff",
                    boxShadow: isDark ? `0 0 6px ${C.primary}60` : "0 1px 4px rgba(0,0,0,0.2)",
                    transition: "left 0.3s cubic-bezier(0.34,1.56,0.64,1)",
                    display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9,
                  }}>{isDark ? "🌙" : "☀️"}</div>
                </div>
                <span style={{
                  color: C.midText, fontSize: 10, fontWeight: 700,
                  letterSpacing: "0.1em", textTransform: "uppercase",
                  fontFamily: "'Space Grotesk', monospace",
                }}>{isDark ? "DARK" : "LIGHT"}</span>
              </button>

              {/* MODEL READY badge */}
              <div style={{
                padding: "6px 14px", borderRadius: 20,
                background: `${C.teal}12`, border: `1px solid ${C.teal}30`,
                display: "flex", alignItems: "center", gap: 7,
                boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              }}>
                <div style={{
                  width: 6, height: 6, borderRadius: "50%", background: C.teal,
                  boxShadow: `0 0 8px ${C.teal}`, animation: "pulse 2s ease infinite",
                }} />
                <span style={{ color: C.teal, fontSize: 11, fontWeight: 700, letterSpacing: "0.1em" }}>
                  MODEL READY
                </span>
              </div>

              {/* 📄 Report button */}
              <button
                onClick={() => notify("Open the Upload page to generate a new report.", true)}
                style={{
                  display: "flex", alignItems: "center", gap: 7,
                  padding: "7px 16px", borderRadius: 10, cursor: "pointer",
                  border: `1px solid ${C.primary}40`,
                  background: `linear-gradient(135deg, ${C.primary}12, ${C.rose}08)`,
                  boxShadow: "0 2px 8px rgba(107,79,200,0.12), inset 0 1px 0 rgba(255,255,255,0.8)",
                  fontFamily: "'Space Grotesk', monospace", color: C.primary,
                  fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase",
                  transition: "all 0.2s",
                }}>
                <span style={{ fontSize: 14 }}>📄</span> Report
              </button>

            </div>
          </header>

          {/* ════════════════════════════════════════════════════════════════
              MAIN CONTENT
          ════════════════════════════════════════════════════════════════ */}
          <main style={{ maxWidth: 1100, margin: "0 auto", padding: "44px 28px 60px", display: "flex", flexDirection: "column", gap: 28 }}>

            {/* ── HERO ── */}
            <div className="reveal" style={{ textAlign: "center", padding: "28px 0 8px", position: "relative", zIndex: 10, isolation: "isolate" }}>
              <div style={{ position: "absolute", inset: "-40px -80px", background: `radial-gradient(ellipse 70% 80% at 50% 50%, ${C.bg}99 0%, transparent 70%)`, pointerEvents: "none", zIndex: -1 }} />
              <div style={{ display: "flex", alignItems: "center", gap: 14, justifyContent: "center", marginBottom: 20 }}>
                <div style={{ height: 1, width: 48, background: `linear-gradient(90deg, transparent, ${C.rose}60)` }} />
                <span style={{ color: C.rose, fontSize: 9, letterSpacing: "0.3em", textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace", fontWeight: 700 }}>
                  MongoDB · PDF · HTML · Auto-generated
                </span>
                <div style={{ height: 1, width: 48, background: `linear-gradient(90deg, ${C.rose}60, transparent)` }} />
              </div>
              <h1 style={{
                fontSize: 52, fontWeight: 800, fontFamily: "'Space Grotesk', sans-serif",
                letterSpacing: "-0.02em", lineHeight: 1.1, marginBottom: 14,
                isolation: "isolate",
                filter: isDark
                  ? `drop-shadow(0 0 36px ${C.primary}60) drop-shadow(0 2px 12px rgba(0,0,0,0.7))`
                  : `drop-shadow(0 0 40px ${C.primary}30) drop-shadow(0 2px 8px rgba(0,0,0,0.08))`,
              }}>
                <span style={{
                  display: "inline-block",
                  background: isDark
                    ? `linear-gradient(90deg, ${C.hiText}, ${C.primary}, ${C.rose}, ${C.hiText})`
                    : `linear-gradient(90deg, ${C.primary}, ${C.rose}, ${C.violet}, ${C.primary})`,
                  backgroundSize: "300% auto", WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent", backgroundClip: "text",
                  animation: "titleGlow 6s linear infinite",
                }}>Generated Reports</span>
              </h1>
              <p style={{ color: isDark ? C.hiText : C.midText, opacity: isDark ? 0.55 : 0.85, fontSize: 13, letterSpacing: "0.22em", textTransform: "uppercase", fontWeight: 600 }}>
                View &nbsp;·&nbsp; Download &nbsp;·&nbsp; Delete &nbsp;·&nbsp; Preview
              </p>

              {/* Stat chips */}
              <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap", marginTop: 20 }}>
                {[
                  { icon: "📄", val: String(reports.length), label: "Total Reports",  color: C.primary },
                  { icon: "📋", val: String(pdfCount),       label: "PDF Reports",    color: C.rose    },
                  { icon: "🌐", val: String(htmlCount),      label: "HTML Reports",   color: C.violet  },
                ].map((s, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 7, padding: "7px 16px", borderRadius: 30,
                    background: C.isDark ? `${s.color}14` : `${s.color}10`,
                    border: `1px solid ${s.color}35`, backdropFilter: "blur(10px)",
                    boxShadow: `0 2px 10px ${s.color}18`,
                  }}>
                    <span style={{ fontSize: 14 }}>{s.icon}</span>
                    <span style={{ fontSize: 15, fontWeight: 800, color: s.color, fontFamily: "'Space Grotesk', sans-serif" }}>{s.val}</span>
                    <span style={{ fontSize: 10, color: C.midText, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" }}>{s.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── ERROR STATE ── */}
            {error && (
              <div className="reveal">
                <GlassCard accent={C.rose} hover={false} style={{ padding: "20px 24px" }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${C.rose}, transparent)` }} />
                  <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    <div style={{ width: 40, height: 40, borderRadius: 12, background: `${C.rose}12`, border: `1px solid ${C.rose}25`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20 }}>⚠️</div>
                    <div style={{ flex: 1 }}>
                      <p style={{ color: C.rose, fontSize: 12, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 4 }}>Connection Error</p>
                      <p style={{ color: C.midText, fontSize: 13 }}>{error}</p>
                    </div>
                    <button onClick={fetchReports} style={{
                      padding: "8px 20px", borderRadius: 10, cursor: "pointer",
                      border: `1px solid ${C.rose}40`, background: `${C.rose}10`,
                      color: C.rose, fontSize: 12, fontWeight: 700,
                      fontFamily: "'Space Grotesk', monospace", letterSpacing: "0.1em", textTransform: "uppercase",
                    }}>Retry</button>
                  </div>
                </GlassCard>
              </div>
            )}

            {/* ── SEARCH + FILTER BAR ── */}
            {!loading && !error && reports.length > 0 && (
              <div className="reveal r1">
                <GlassCard hover={false} style={{ padding: "16px 22px" }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>

                    {/* Search input */}
                    <div style={{ flex: 1, minWidth: 200, position: "relative" }}>
                      <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", fontSize: 14, pointerEvents: "none", color: C.loText }}>🔍</span>
                      <input
                        type="text"
                        placeholder="Search reports by name or dataset…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        style={{
                          width: "100%", padding: "10px 13px 10px 36px",
                          background: C.surface, border: `1px solid ${C.border}`,
                          borderRadius: 10, color: C.hiText, fontSize: 13,
                          fontFamily: "'DM Sans', sans-serif", outline: "none",
                          boxShadow: `0 1px 3px rgba(0,0,0,0.05)`,
                          transition: "border-color 0.2s",
                        }}
                        onFocus={e => e.target.style.borderColor = C.primary + "60"}
                        onBlur={e => e.target.style.borderColor = C.border}
                      />
                    </div>

                    {/* Type filters */}
                    <div style={{ display: "flex", gap: 6 }}>
                      {[
                        { key: "all",  label: "All",  count: reports.length },
                        { key: "pdf",  label: "PDF",  count: pdfCount       },
                        { key: "html", label: "HTML", count: htmlCount       },
                      ].map(f => (
                        <button key={f.key} onClick={() => setFilterType(f.key)} style={{
                          padding: "8px 16px", borderRadius: 10, cursor: "pointer",
                          border: `1px solid ${filterType === f.key ? C.primary + "55" : C.border}`,
                          background: filterType === f.key ? `${C.primary}12` : C.surface,
                          color: filterType === f.key ? C.primary : C.midText,
                          fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
                          fontFamily: "'Space Grotesk', monospace", transition: "all 0.2s",
                          display: "flex", alignItems: "center", gap: 6,
                        }}>
                          {f.label}
                          <span style={{
                            padding: "1px 6px", borderRadius: 10, fontSize: 9,
                            background: filterType === f.key ? `${C.primary}20` : C.border,
                            color: filterType === f.key ? C.primary : C.loText, fontWeight: 800,
                          }}>{f.count}</span>
                        </button>
                      ))}
                    </div>

                    {/* Refresh */}
                    <button onClick={fetchReports} style={{
                      padding: "9px 16px", borderRadius: 10, cursor: "pointer",
                      border: `1px solid ${C.border}`, background: C.surface,
                      color: C.midText, fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
                      textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace",
                      display: "flex", alignItems: "center", gap: 6,
                    }}>🔄 Refresh</button>
                  </div>
                </GlassCard>
              </div>
            )}

            {/* ── REPORTS LIST ── */}
            <div className="reveal r2">
              <Panel
                title="All Reports"
                subtitle="PDF and HTML reports saved from training and dataset analysis"
                icon="📑"
                accent={C.primary}
                badge={reports.length > 0 ? `${filtered.length} / ${reports.length}` : undefined}
              >
                {loading ? (
                  <LoadingSpinner />
                ) : reports.length === 0 ? (
                  <EmptyState
                    emoji="📑"
                    title="No reports yet"
                    sub="Train a model or generate a dashboard to create your first report. Reports appear here automatically."
                  />
                ) : filtered.length === 0 ? (
                  <EmptyState
                    emoji="🔍"
                    title="No matches found"
                    sub={`No reports match "${search}". Try a different search term.`}
                  />
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {filtered.map((r, i) => {
                      const isPDF     = r.report_filename?.endsWith(".pdf");
                      const isHTML    = r.report_filename?.endsWith(".html");
                      const isViewing = viewingId === r._id;
                      const isDeleting = deleting === r._id;
                      const fileColor  = isPDF ? C.rose : isHTML ? C.violet : C.amber;

                      return (
                        <div key={r._id || i}>
                          <GlassCard accent={fileColor} style={{ padding: "16px 20px" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>

                              {/* File icon */}
                              <div style={{
                                width: 44, height: 44, borderRadius: 12, flexShrink: 0,
                                background: `linear-gradient(135deg, ${fileColor}20, ${fileColor}08)`,
                                border: `1px solid ${fileColor}30`,
                                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22,
                                boxShadow: `0 2px 8px ${fileColor}15`,
                              }}>{isPDF ? "📄" : isHTML ? "🌐" : "📎"}</div>

                              {/* Info */}
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, flexWrap: "wrap" }}>
                                  <TypeBadge filename={r.report_filename} />
                                  <span style={{ color: C.hiText, fontWeight: 700, fontSize: 13, fontFamily: "'Space Grotesk', monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 340 }}>
                                    {r.report_filename || "—"}
                                  </span>
                                </div>
                                <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
                                  <span style={{ color: C.loText, fontSize: 11 }}>
                                    📁 Dataset: <span style={{ color: C.midText, fontFamily: "'Space Grotesk', monospace" }}>{r.dataset_filename || "—"}</span>
                                  </span>
                                  <span style={{ color: C.loText, fontSize: 11 }}>
                                    🕒 {r.created_at || "—"}
                                  </span>
                                </div>
                              </div>

                              {/* Actions */}
                              <div style={{ display: "flex", gap: 8, flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>

                                {/* View/Preview (HTML or PDF) */}
                                {isHTML && (
                                  <button
                                    onClick={() => setViewingId(isViewing ? null : r._id)}
                                    style={{
                                      display: "flex", alignItems: "center", gap: 6,
                                      padding: "7px 14px", borderRadius: 10, cursor: "pointer",
                                      border: `1px solid ${isViewing ? C.teal + "60" : C.violet + "40"}`,
                                      background: isViewing ? `${C.teal}14` : `${C.violet}10`,
                                      color: isViewing ? C.teal : C.violet,
                                      fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
                                      textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace",
                                      transition: "all 0.2s",
                                    }}>
                                    {isViewing ? "✕ Close" : "👁 Preview"}
                                  </button>
                                )}

                                {/* Open in new tab — /static/reports/ for HTML, /static/reports/ for PDF */}
                                <a
                                  href={`${API}/static/reports/${r.report_filename}`}
                                  target="_blank"
                                  rel="noreferrer"
                                  style={{
                                    display: "flex", alignItems: "center", gap: 6,
                                    padding: "7px 14px", borderRadius: 10, cursor: "pointer",
                                    border: `1px solid ${fileColor}35`,
                                    background: `${fileColor}10`,
                                    color: fileColor, textDecoration: "none",
                                    fontSize: 11, fontWeight: 700, letterSpacing: "0.08em",
                                    textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace",
                                    transition: "all 0.2s",
                                  }}>
                                  {isPDF ? "⬇ Download" : "↗ Open"}
                                </a>

                                {/* Delete */}
                                <button
                                  onClick={() => setConfirmId(r._id)}
                                  disabled={isDeleting}
                                  style={{
                                    display: "flex", alignItems: "center", gap: 6,
                                    padding: "7px 14px", borderRadius: 10, cursor: isDeleting ? "not-allowed" : "pointer",
                                    border: `1px solid ${C.rose}35`,
                                    background: `${C.rose}08`,
                                    color: C.rose, fontSize: 11, fontWeight: 700,
                                    letterSpacing: "0.08em", textTransform: "uppercase",
                                    fontFamily: "'Space Grotesk', monospace",
                                    opacity: isDeleting ? 0.5 : 1, transition: "all 0.2s",
                                  }}>
                                  {isDeleting ? "…" : "🗑 Delete"}
                                </button>
                              </div>
                            </div>

                            {/* Inline iframe preview for HTML reports */}
                            {isViewing && isHTML && (
                              <div className="reveal" style={{ marginTop: 14, borderRadius: 12, overflow: "hidden", border: `1px solid ${C.border}` }}>
                                <div style={{
                                  padding: "10px 16px", background: C.surface,
                                  borderBottom: `1px solid ${C.border}`,
                                  display: "flex", alignItems: "center", justifyContent: "space-between",
                                }}>
                                  <span style={{ color: C.midText, fontSize: 11, fontFamily: "'Space Grotesk', monospace" }}>
                                    🌐 {r.report_filename}
                                  </span>
                                  <a
                                    href={`${API}/static/reports/${r.report_filename}`}
                                    target="_blank" rel="noreferrer"
                                    style={{ color: C.primary, fontSize: 11, fontWeight: 700, textDecoration: "none", fontFamily: "'Space Grotesk', monospace" }}>
                                    ↗ Open full screen
                                  </a>
                                </div>
                                {/* ✅ Correct URL: /static/reports/ not /reports/ */}
                                <iframe
                                  src={`${API}/static/reports/${r.report_filename}`}
                                  title={r.report_filename}
                                  width="100%"
                                  style={{ height: "75vh", display: "block", border: "none" }}
                                />
                              </div>
                            )}
                          </GlassCard>
                        </div>
                      );
                    })}
                  </div>
                )}
              </Panel>
            </div>

            {/* ── FOOTER ── */}
            <div style={{ textAlign: "center", padding: "8px 0", display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}>
              <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, transparent, ${C.border})` }} />
              <p style={{ color: C.loText, fontSize: 10, letterSpacing: "0.22em", textTransform: "uppercase", fontFamily: "'Space Grotesk', monospace" }}>
                AutoAnalytica AI Platform · Powered by Machine Intelligence
              </p>
              <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${C.border}, transparent)` }} />
            </div>

          </main>
        </div>

        {/* ── CONFIRM DELETE DIALOG ── */}
        <ConfirmDialog
          show={!!confirmId}
          title="Delete Report?"
          message="This will permanently delete the report file from disk and remove its record from MongoDB. This action cannot be undone."
          onConfirm={() => handleDelete(confirmId)}
          onCancel={() => setConfirmId(null)}
        />

        <Toast text={toast.text} ok={toast.ok} onClose={() => setToast({ text: "" })} />
      </>
    </ThemeCtx.Provider>
  );
}