import React, { useState, useEffect, useRef, createContext, useContext } from "react";
import axios from "axios";
import {
  extractAgentFields, getDecisionMeta, getUrgencyMeta,
  getQualityLabelMeta, getRiskMeta, getImproveMeta, sendFeedback,
} from "../services/api";

// ══════════════════════════════════════════════════════════════════════════════
// THEME SYSTEM — Light & Dark palettes
// ══════════════════════════════════════════════════════════════════════════════
const LIGHT = {
  bg:"#fdfaff",surface:"#f5eeff",card:"#ffffff",border:"#e4d0f8",borderHi:"#c080e0",
  primary:"#9b40d0",violet:"#b060e0",cyan:"#e0409a",teal:"#c030a0",amber:"#d070b8",
  rose:"#e8409a",hiText:"#1e0838",midText:"#7030a0",loText:"#c090d8",
  coralTint:"#fff0f8",violetTint:"#f8f0ff",tealTint:"#fff5f8",isDark:false,
};
const DARK = {
  bg:"#0d0b18",surface:"#13111f",card:"#1a1730",border:"#2a2448",borderHi:"#3d3570",
  primary:"#9b7ff0",violet:"#b49dfc",cyan:"#22c5c5",teal:"#14d4bb",amber:"#e89a5a",
  rose:"#e86892",hiText:"#ede9ff",midText:"#9088c0",loText:"#3a3360",
  coralTint:"#1e1228",violetTint:"#1c1838",tealTint:"#0f1e1c",isDark:true,
};
const ThemeCtx = createContext(LIGHT);
const useTheme = () => useContext(ThemeCtx);
let C = LIGHT;

const fmt = s => String(s||"").replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
const pct = v => (v!==null&&v!==undefined)?`${(v*100).toFixed(1)}%`:"—";
const displayPrediction = val => {
  if(val===null||val===undefined)return"—";
  if(typeof val==="number")return val%1===0?val.toLocaleString():val.toLocaleString(undefined,{maximumFractionDigits:4});
  return String(val);
};
const getScoreLabel=(score,numClasses)=>{
  if(score===null||score===undefined)return null;
  const baseline=1/(numClasses||2),ratio=score/baseline;
  if(ratio>=3.0)return"Excellent";if(ratio>=2.0)return"Good";
  if(ratio>=1.3)return"Moderate";return"Needs improvement";
};
const scoreColor=(score,numClasses,theme)=>{
  const t=theme||C,label=getScoreLabel(score,numClasses);
  if(!label)return t.midText;
  if(label==="Excellent")return t.teal;if(label==="Good")return t.cyan;
  if(label==="Moderate")return t.amber;return t.rose;
};

// ══════════════════════════════════════════════════════════════════════════════
// FIX — Safe perf resolver
// Backend always returns `performance`. The old code used
//   perf = trainResponse?.metrics || trainResponse?.performance
// which breaks when `metrics` is present but EMPTY ({} is truthy!).
// ══════════════════════════════════════════════════════════════════════════════
function resolvePerf(trainResponse) {
  if (!trainResponse) return {};
  // Prefer `metrics` only when it is a non-empty object
  const m = trainResponse.metrics;
  if (m && typeof m === "object" && Object.keys(m).length > 0) return m;
  // Fall through to canonical `performance` (always set by run_automl)
  return trainResponse.performance || trainResponse.model_metrics || {};
}

// ══════════════════════════════════════════════════════════════════════════════
// FIX — SHAP data normaliser
// Backend returns sample_explanation as an OBJECT:
//   { shap_values: {feat: val, ...}, feature_values: {...}, base_value, available }
// Frontend renderer expects an ARRAY:
//   [ {feature, shap_value}, … ]
// ══════════════════════════════════════════════════════════════════════════════
function normaliseShap(raw) {
  if (!raw) return [];
  // Already an array (legacy shape)
  if (Array.isArray(raw)) return raw;
  // Object shape returned by run_automl v5.6
  if (raw.available === false) return [];
  const sv = raw.shap_values;
  if (!sv || typeof sv !== "object") return [];
  return Object.entries(sv).map(([feature, shap_value]) => ({ feature, shap_value }));
}

function NeuralGrid({isDark}){
  const ref=useRef(null);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas)return;
    const ctx=canvas.getContext("2d");let frame,t=0;
    const resize=()=>{canvas.width=window.innerWidth;canvas.height=window.innerHeight;};
    resize();window.addEventListener("resize",resize);
    const nodes=Array.from({length:55},()=>({
      x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,
      vx:(Math.random()-0.5)*0.4,vy:(Math.random()-0.5)*0.4,
      r:Math.random()*1.8+0.6,pulse:Math.random()*Math.PI*2,
      type:Math.random()>0.7?"coral":Math.random()>0.5?"teal":"violet",
    }));
    const draw=()=>{
      t+=0.008;ctx.clearRect(0,0,canvas.width,canvas.height);
      for(let i=0;i<nodes.length;i++){for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<145){ctx.beginPath();ctx.moveTo(nodes[i].x,nodes[i].y);ctx.lineTo(nodes[j].x,nodes[j].y);
          ctx.strokeStyle=isDark?`rgba(155,127,240,${(1-d/145)*0.22})`:`rgba(107,79,200,${(1-d/145)*0.12})`;
          ctx.lineWidth=0.8;ctx.stroke();}}}
      nodes.forEach(n=>{
        const p=Math.sin(t*1.5+n.pulse)*0.5+0.5;
        const colors=isDark?{violet:`rgba(155,127,240,${0.45+p*0.4})`,coral:`rgba(232,104,146,${0.4+p*0.38})`,teal:`rgba(20,212,187,${0.4+p*0.38})`}:{violet:`rgba(107,79,200,${0.25+p*0.3})`,coral:`rgba(229,97,60,${0.2+p*0.28})`,teal:`rgba(30,158,158,${0.2+p*0.28})`};
        ctx.beginPath();ctx.arc(n.x,n.y,n.r+p*1.2,0,Math.PI*2);ctx.fillStyle=colors[n.type];ctx.fill();
        n.x+=n.vx;n.y+=n.vy;
        if(n.x<0||n.x>canvas.width)n.vx*=-1;if(n.y<0||n.y>canvas.height)n.vy*=-1;
      });
      frame=requestAnimationFrame(draw);
    };
    draw();
    return()=>{cancelAnimationFrame(frame);window.removeEventListener("resize",resize);};
  },[isDark]);
  return <canvas ref={ref} style={{position:"fixed",inset:0,zIndex:0,pointerEvents:"none",opacity:isDark?0.65:0.62}}/>;
}

function DataStreams(){
  const C=useTheme();
  return(
    <div style={{position:"fixed",inset:0,zIndex:0,pointerEvents:"none",overflow:"hidden"}}>
      {[...Array(6)].map((_,i)=>(
        <div key={i} style={{position:"absolute",left:`${10+i*16}%`,top:0,bottom:0,width:1,
          background:`linear-gradient(180deg,transparent,${C.primary}18,${C.cyan}22,transparent)`,
          animation:`streamFlow ${3+i*0.7}s ease-in-out infinite`,animationDelay:`${i*0.5}s`,opacity:0.5}}/>
      ))}
    </div>
  );
}

function GlassCard({children,style={},accent,hover=true}){
  const C=useTheme();const accentColor=accent||C.primary;
  const [h,setH]=useState(false);const [pos,setPos]=useState({x:0,y:0});const ref=useRef(null);
  const onMove=e=>{if(!ref.current)return;const r=ref.current.getBoundingClientRect();setPos({x:e.clientX-r.left,y:e.clientY-r.top});};
  return(
    <div ref={ref} onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)} onMouseMove={onMove}
      style={{position:"relative",background:`linear-gradient(135deg,${C.card}f0,${C.surface}cc)`,
        border:`1px solid ${h&&hover?accentColor+"55":C.border}`,borderRadius:20,
        backdropFilter:"blur(20px)",WebkitBackdropFilter:"blur(20px)",
        transition:"border-color 0.3s,transform 0.2s,box-shadow 0.3s",
        transform:h&&hover?"translateY(-2px)":"none",
        boxShadow:h&&hover?C.isDark?`0 20px 60px rgba(0,0,0,0.5),0 0 0 1px ${accentColor}33,inset 0 1px 0 ${accentColor}18`:`0 20px 60px rgba(107,79,200,0.12),0 0 0 1px ${accentColor}22,inset 0 1px 0 rgba(255,255,255,0.9)`:C.isDark?`0 4px 24px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04)`:`0 4px 24px rgba(107,79,200,0.07),0 1px 4px rgba(0,0,0,0.04),inset 0 1px 0 rgba(255,255,255,0.8)`,
        overflow:"hidden",...style}}>
      {h&&hover&&(<div style={{position:"absolute",width:300,height:300,borderRadius:"50%",background:`radial-gradient(circle,${accentColor}${C.isDark?"18":"0c"} 0%,transparent 70%)`,left:pos.x-150,top:pos.y-150,pointerEvents:"none",zIndex:0}}/>)}
      <div style={{position:"relative",zIndex:1}}>{children}</div>
    </div>
  );
}

function Panel({title,subtitle,icon,children,accent,badge}){
  const C=useTheme();const accentColor=accent||C.primary;
  return(
    <GlassCard accent={accent} hover={false} style={{padding:"28px 30px 32px"}}>
      <div style={{position:"absolute",top:0,left:0,right:0,height:2,background:`linear-gradient(90deg,transparent,${C.rose}60,${accentColor},${accentColor}88,transparent)`}}/>
      {title&&(
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:subtitle?4:24}}>
          <div style={{width:36,height:36,borderRadius:10,flexShrink:0,background:`${accentColor}14`,border:`1px solid ${accentColor}28`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:17,boxShadow:`0 2px 8px ${accentColor}15`}}>{icon}</div>
          <div style={{flex:1}}>
            <div style={{display:"flex",alignItems:"center",gap:10}}>
              <h2 style={{color:C.hiText,fontSize:13,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace"}}>{title}</h2>
              {badge&&(<span style={{padding:"2px 8px",borderRadius:20,fontSize:10,fontWeight:700,background:`${accentColor}14`,color:accentColor,border:`1px solid ${accentColor}28`,letterSpacing:"0.08em"}}>{badge}</span>)}
            </div>
          </div>
          <div style={{flex:1,height:1,background:`linear-gradient(90deg,${accentColor}35,transparent)`}}/>
        </div>
      )}
      {subtitle&&<p style={{color:C.midText,fontSize:12,marginBottom:22,marginLeft:48}}>{subtitle}</p>}
      {children}
    </GlassCard>
  );
}

function AnimatedValue({value,decimals=0,suffix=""}){
  const [display,setDisplay]=useState(0);const target=parseFloat(value)||0;
  useEffect(()=>{
    const duration=1200,startTime=Date.now();let animVal=0;
    const tick=()=>{const elapsed=Date.now()-startTime,progress=Math.min(elapsed/duration,1),eased=1-Math.pow(1-progress,3);animVal=target*eased;setDisplay(animVal);if(progress<1)requestAnimationFrame(tick);};
    requestAnimationFrame(tick);
  },[target]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

function KpiCard({label,value,accent,icon,sub,animated=false,decimals=1,suffix=""}){
  const C=useTheme();const accentColor=accent||C.primary;
  const len=String(value).length;const fs=len>10?16:len>7?22:len>4?28:36;
  return(
    <GlassCard accent={accent} style={{padding:"20px 18px"}}>
      <div style={{position:"absolute",top:-20,right:-20,width:80,height:80,borderRadius:"50%",background:`radial-gradient(circle,${accentColor}18 0%,transparent 70%)`,filter:"blur(12px)"}}/>
      <p style={{color:C.midText,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",fontWeight:700,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
        <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:`${accentColor}14`,border:`1px solid ${accentColor}22`,display:"flex",alignItems:"center",justifyContent:"center",boxShadow:`0 1px 4px ${accentColor}12`}}>{icon}</span>
        {label}
      </p>
      <p style={{color:accentColor,fontWeight:800,lineHeight:1.1,fontFamily:"'Space Grotesk',monospace",fontSize:fs,wordBreak:"break-all"}}>
        {animated&&!isNaN(parseFloat(value))?<AnimatedValue value={parseFloat(value)} decimals={decimals} suffix={suffix}/>:value}
      </p>
      {sub&&<p style={{color:`${accentColor}80`,fontSize:11,marginTop:8}}>{sub}</p>}
    </GlassCard>
  );
}

function Btn({onClick,children,color,disabled=false,size="md",variant="outline"}){
  const C=useTheme();const btnColor=color||C.primary;
  const [h,setH]=useState(false);const [active,setActive]=useState(false);
  const pad=size==="lg"?"14px 28px":size==="sm"?"8px 16px":"11px 20px";
  const fs=size==="lg"?13:size==="sm"?11:12;
  return(
    <button onClick={onClick} disabled={disabled} onMouseEnter={()=>setH(true)} onMouseLeave={()=>{setH(false);setActive(false);}} onMouseDown={()=>setActive(true)} onMouseUp={()=>setActive(false)}
      style={{padding:pad,fontSize:fs,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",cursor:disabled?"not-allowed":"pointer",border:`1px solid ${h?btnColor+"80":btnColor+"40"}`,borderRadius:12,fontFamily:"'Space Grotesk',monospace",background:h?(variant==="solid"?btnColor:`${btnColor}14`):(variant==="solid"?`${btnColor}cc`:C.isDark?`${btnColor}10`:`${btnColor}08`),color:h?(variant==="solid"?(C.isDark?C.bg:"#fff"):C.hiText):btnColor,boxShadow:h?`0 4px 20px ${btnColor}28,inset 0 1px 0 ${btnColor}25`:C.isDark?"0 1px 4px rgba(0,0,0,0.3)":"0 1px 4px rgba(0,0,0,0.05)",transform:active?"scale(0.97)":"none",transition:"all 0.18s",opacity:disabled?0.35:1,display:"flex",alignItems:"center",justifyContent:"center",gap:8,width:"100%",position:"relative",overflow:"hidden"}}>
      {h&&!disabled&&(<div style={{position:"absolute",inset:0,background:`linear-gradient(105deg,transparent 30%,rgba(255,255,255,0.3) 50%,transparent 70%)`,animation:"btnShine 0.6s ease forwards"}}/>)}
      {children}
    </button>
  );
}

function Input({label,placeholder,value,onChange,accent}){
  const C=useTheme();const accentColor=accent||C.teal;const [f,setF]=useState(false);
  return(
    <div>
      {label&&(<p style={{color:f?accentColor:C.midText,fontSize:10,letterSpacing:"0.12em",textTransform:"uppercase",fontWeight:700,marginBottom:6,transition:"color 0.2s"}}>{label}</p>)}
      <div style={{position:"relative"}}>
        <input type="text" placeholder={placeholder} value={value} onChange={onChange} onFocus={()=>setF(true)} onBlur={()=>setF(false)}
          style={{width:"100%",padding:"10px 13px",boxSizing:"border-box",background:f?`${accentColor}07`:C.surface,border:`1px solid ${f?accentColor+"60":C.border}`,borderRadius:10,color:C.hiText,fontSize:13,fontFamily:"'Space Grotesk',sans-serif",outline:"none",transition:"all 0.2s",boxShadow:f?`0 0 0 3px ${accentColor}10,0 2px 8px rgba(0,0,0,0.04)`:C.isDark?"0 1px 3px rgba(0,0,0,0.3)":"0 1px 3px rgba(0,0,0,0.05)"}}/>
        {f&&(<div style={{position:"absolute",bottom:0,left:"10%",right:"10%",height:1,background:`linear-gradient(90deg,transparent,${accentColor},${C.rose}50,transparent)`}}/>)}
      </div>
    </div>
  );
}

function ProgressBar({value=0,color,height=4}){
  const C=useTheme();const barColor=color||C.primary;const [w,setW]=useState(0);
  useEffect(()=>{setTimeout(()=>setW(Math.max(0,Math.min(1,value))*100),100);},[value]);
  return(
    <div style={{height,background:C.border,borderRadius:height,overflow:"hidden",boxShadow:"inset 0 1px 2px rgba(0,0,0,0.04)"}}>
      <div style={{width:`${w}%`,height:"100%",borderRadius:height,background:`linear-gradient(90deg,${C.rose}50,${barColor})`,transition:"width 1s cubic-bezier(0.4,0,0.2,1)",position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",inset:0,background:"linear-gradient(90deg,transparent,rgba(255,255,255,0.35),transparent)",animation:"shimmerBar 1.8s ease infinite"}}/>
      </div>
    </div>
  );
}

function ConfusionMatrix({cm,labels}){
  const C=useTheme();
  if(!cm||!Array.isArray(cm)||cm.length===0)return null;
  const size=cm.length,total=cm.flat().reduce((a,b)=>a+b,0);
  if(size===2){
    const[[tn,fp],[fn,tp]]=cm;
    const cells=[{s:"TN",l:"True Neg",desc:"Predicted No · Was No",v:tn,a:C.teal},{s:"FP",l:"False Pos",desc:"Predicted Yes · Was No",v:fp,a:C.rose},{s:"FN",l:"False Neg",desc:"Predicted No · Was Yes",v:fn,a:C.amber},{s:"TP",l:"True Pos",desc:"Predicted Yes · Was Yes",v:tp,a:C.cyan}];
    return(<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>{cells.map(c=>(<GlassCard key={c.s} accent={c.a} style={{padding:"14px 16px"}}><div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}><span style={{color:c.a,fontSize:16,fontWeight:800,fontFamily:"'Space Grotesk',monospace"}}>{c.s}</span><span style={{color:C.midText,fontSize:9,letterSpacing:"0.1em",textTransform:"uppercase",fontWeight:600}}>{c.l}</span></div><p style={{color:C.hiText,fontSize:32,fontWeight:800,fontFamily:"'Space Grotesk',monospace",lineHeight:1}}>{c.v??"—"}</p><p style={{color:C.loText,fontSize:10,marginTop:7}}>{c.desc}</p></GlassCard>))}</div>);
  }
  const classLabels=labels||cm.map((_,i)=>`Class ${i}`);
  return(<div><p style={{color:C.midText,fontSize:11,marginBottom:10}}>{size}×{size} matrix · {total} test samples · diagonal = correct predictions</p><div style={{overflowX:"auto"}}><table style={{borderCollapse:"collapse",fontSize:11,width:"100%"}}><thead><tr><th style={{padding:"6px 10px",color:C.loText,fontWeight:600,textAlign:"left",whiteSpace:"nowrap"}}>Actual ↓ / Pred →</th>{classLabels.map((l,i)=>(<th key={i} style={{padding:"6px 10px",color:C.midText,fontWeight:700,textAlign:"center",whiteSpace:"nowrap"}}>{l}</th>))}</tr></thead><tbody>{cm.map((row,ri)=>(<tr key={ri}><td style={{padding:"6px 10px",color:C.midText,fontWeight:700,whiteSpace:"nowrap",borderRight:`1px solid ${C.border}`}}>{classLabels[ri]}</td>{row.map((val,ci)=>{const isDiag=ri===ci,intensity=total>0?val/total:0;const bg=isDiag?`rgba(30,158,158,${0.08+intensity*0.45})`:val>0?`rgba(212,84,122,${0.05+intensity*0.38})`:"transparent";return(<td key={ci} style={{padding:"6px 10px",textAlign:"center",background:bg,color:isDiag?C.teal:val>0?C.rose:C.loText,fontWeight:isDiag?800:500,fontFamily:"'Space Grotesk',monospace",border:`1px solid ${C.border}22`,minWidth:44}}>{val}</td>);})}</tr>))}</tbody></table></div></div>);
}

function StepIndicator({steps}){
  const C=useTheme();
  return(
    <div style={{display:"flex",alignItems:"center",gap:0}}>
      {steps.map((s,i)=>(
        <React.Fragment key={i}>
          <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:4}}>
            <div style={{width:28,height:28,borderRadius:"50%",border:`2px solid ${s.done?C.teal:s.active?C.primary:C.border}`,background:s.done?`${C.teal}18`:s.active?`${C.primary}12`:C.surface,display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,fontWeight:800,color:s.done?C.teal:s.active?C.primary:C.midText,transition:"all 0.4s",boxShadow:s.active?`0 0 16px ${C.primary}35`:s.done?`0 0 12px ${C.teal}28`:"none"}}>{s.done?"✓":i+1}</div>
            <span style={{fontSize:9,letterSpacing:"0.1em",textTransform:"uppercase",fontWeight:700,fontFamily:"'Space Grotesk',monospace",color:s.done?C.teal:s.active?C.primary:C.loText,transition:"color 0.4s"}}>{s.label}</span>
          </div>
          {i<steps.length-1&&(<div style={{flex:1,height:1,margin:"0 6px",marginBottom:18,background:steps[i].done?`linear-gradient(90deg,${C.teal}60,${C.teal}25)`:C.border,transition:"background 0.4s"}}/>)}
        </React.Fragment>
      ))}
    </div>
  );
}

function Toast({text,ok,onClose}){
  const C=useTheme();if(!text)return null;
  const color=ok?C.teal:C.rose;
  return(<div style={{position:"fixed",top:80,right:24,zIndex:9998,padding:"14px 20px",background:C.surface,border:`1px solid ${color}35`,borderLeft:`3px solid ${color}`,borderRadius:14,fontSize:13,fontWeight:600,display:"flex",alignItems:"center",gap:10,boxShadow:`0 8px 32px rgba(107,79,200,0.12),0 0 0 1px ${color}12`,animation:"slideInRight 0.3s cubic-bezier(0.34,1.56,0.64,1)",minWidth:280,maxWidth:360}}>
    <span style={{width:24,height:24,borderRadius:"50%",background:`${color}16`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,flexShrink:0,color}}>{ok?"✓":"✕"}</span>
    <span style={{flex:1,color:C.hiText,fontSize:13}}>{text}</span>
    <button onClick={onClose} style={{background:"none",border:"none",color:C.midText,cursor:"pointer",fontSize:18,lineHeight:1,padding:0}}>×</button>
  </div>);
}

function LoadingOverlay({show,message}){
  const C=useTheme();if(!show)return null;
  return(<div style={{position:"fixed",inset:0,zIndex:9999,background:C.isDark?`rgba(13,11,24,0.92)`:`rgba(248,246,242,0.88)`,backdropFilter:"blur(20px)",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:28}}>
    <div style={{position:"relative",width:80,height:80}}>
      {[C.primary,C.rose,C.cyan].map((col,i)=>(<div key={i} style={{position:"absolute",inset:i*12,border:"1.5px solid transparent",borderTopColor:col,borderRadius:"50%",animation:`orbit ${0.8+i*0.4}s linear infinite`,boxShadow:`0 0 8px ${col}30`}}/>))}
      <div style={{position:"absolute",left:"50%",top:"50%",transform:"translate(-50%,-50%)",width:10,height:10,borderRadius:"50%",background:`radial-gradient(circle,${C.primary},${C.rose})`,boxShadow:`0 0 16px ${C.primary}60`,animation:"corePulse 1s ease infinite"}}/>
    </div>
    <div style={{textAlign:"center"}}>
      <p style={{color:C.hiText,fontFamily:"'Space Grotesk',monospace",fontSize:11,letterSpacing:"0.25em",textTransform:"uppercase",marginBottom:8}}>Processing</p>
      <p style={{color:C.midText,fontSize:13,animation:"flicker 2s ease infinite"}}>{message}</p>
    </div>
    <div style={{position:"absolute",left:0,right:0,height:1,background:`linear-gradient(90deg,transparent,${C.primary}50,${C.rose}35,transparent)`,animation:"scanV 2.5s ease-in-out infinite"}}/>
  </div>);
}

// ── Safe string extractor for meta_insight (may be object or string) ─────────
function safeMetaInsightText(val) {
  if (!val) return null;
  if (typeof val === "string") return val;
  if (typeof val === "object") {
    return val.recommendation ?? val.text ?? val.summary ?? val.insight ?? null;
  }
  return null;
}

// ── Safe number formatter ─────────────────────────────────────────────────────
function safeNum(v, decimals = 4) {
  const n = parseFloat(v);
  if (isNaN(n)) return "N/A";
  return n.toFixed(decimals);
}

async function generateHTMLReport({trainResponse,dashboardData,columns,target,file,apiBase}){
  const perf=resolvePerf(trainResponse);
  const modelType=trainResponse?.problem_type||"—";
  const modelName=trainResponse?.best_model_name||trainResponse?.best_model||trainResponse?.model_name||"—";
  const accuracy=perf?.accuracy??null;const roc_auc=perf?.roc_auc??null;
  // eslint-disable-next-line no-unused-vars
  const mae=perf?.MAE??perf?.mae??null;const rmse=perf?.RMSE??perf?.rmse??null;
  const r2=perf?.R2??perf?.r2??null;
  const cm=Array.isArray(perf?.confusion_matrix)?perf.confusion_matrix:null;
  const cmLabels=perf?.confusion_matrix_labels||null;
  const quality=trainResponse?.model_quality||{};
  const allScores=perf?.all_model_scores||{};
  const baseline=perf?.baseline_accuracy;
  const nTrain=perf?.n_train??"—";const nTest=perf?.n_test??"—";
  const nTotal=(nTrain!=="—"&&nTest!=="—")?(Number(nTrain)+Number(nTest)):"—";
  const nFeat=columns.length-1;const nClasses=perf?.n_classes??"—";
  const tier=trainResponse?.scale_tier_name||trainResponse?.scale_tier||"—";
  const now=new Date().toLocaleString();
  const cvScoreMean=perf?.cv_score_mean??null;const cvScoreStd=perf?.cv_score_std??null;
  const p=v=>(v!==null&&v!==undefined)?`${(v*100).toFixed(2)}%`:"—";
  const num=v=>(v!==null&&v!==undefined)?Number(v).toFixed(4):"—";
  const fmtL=s=>String(s||"").replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
  const mnLen=modelName.length;
  const mnFs=mnLen>20?"13px":mnLen>16?"15px":mnLen>12?"17px":mnLen>8?"20px":"26px";
  const leaderboard=Object.entries(allScores).map(([n,s])=>({
    name:n,
    // FIX: use cv_mean (actual backend key) instead of cv_accuracy_mean
    score:modelType==="classification"?(s?.accuracy??null):(s?.R2??s?.r2??null),
    cv:modelType==="classification"?(s?.cv_mean??s?.f1_macro??null):(s?.cv_r2_mean??null)
  })).sort((a,b)=>(b.score??-Infinity)-(a.score??-Infinity));
  const chartEmbeds=[];
  if(dashboardData?.charts?.length&&apiBase){try{const res=await fetch(`${apiBase}/static/reports/${dashboardData.charts[0]}`);if(res.ok){const raw=await res.text();const titleRe=/<button[^>]+class="tab[^"]*"[^>]*>([\s\S]*?)<\/button>/gi;const titles=[];let m;while((m=titleRe.exec(raw))!==null)titles.push(m[1].replace(/<[^>]+>/g,"").replace(/[^\w\s\-·/]/gu,"").trim()||`Chart ${titles.length+1}`);const imgRe=/src="(data:image\/png;base64,[^"]+)"/gi;let idx=0;while((m=imgRe.exec(raw))!==null){chartEmbeds.push({title:titles[idx]||`Chart ${idx+1}`,src:m[1]});idx++;}}}catch(e){console.error("Dashboard fetch error:",e);}}
  const hasCharts=chartEmbeds.length>0;
  let cmHTML="";
  if(cm){const size=cm.length;const lbls=cmLabels||cm.map((_,i)=>`Class ${i}`);if(size===2){const[[tn,fp],[fn,tp]]=cm;cmHTML=`<div class="cm2-grid"><div class="cm2-cell tn"><div class="cm2-abbr">TN</div><div class="cm2-count">${tn}</div><div class="cm2-name">True Negative</div><div class="cm2-desc">Predicted No · Was No</div></div><div class="cm2-cell fp"><div class="cm2-abbr">FP</div><div class="cm2-count">${fp}</div><div class="cm2-name">False Positive</div><div class="cm2-desc">Predicted Yes · Was No</div></div><div class="cm2-cell fn"><div class="cm2-abbr">FN</div><div class="cm2-count">${fn}</div><div class="cm2-name">False Negative</div><div class="cm2-desc">Predicted No · Was Yes</div></div><div class="cm2-cell tp"><div class="cm2-abbr">TP</div><div class="cm2-count">${tp}</div><div class="cm2-name">True Positive</div><div class="cm2-desc">Predicted Yes · Was Yes</div></div></div>`;}else{const total=cm.flat().reduce((a,b)=>a+b,0);cmHTML=`<div style="overflow-x:auto"><table class="cm-table"><thead><tr><th class="cm-corner">Actual ↓ / Pred →</th>${lbls.map(l=>`<th>${l}</th>`).join("")}</tr></thead><tbody>${cm.map((row,ri)=>`<tr><th class="cm-row-head">${lbls[ri]}</th>${row.map((val,ci)=>{const isDiag=ri===ci,intens=total>0?val/total:0;const bg=isDiag?`rgba(16,185,129,${0.12+intens*0.5})`:val>0?`rgba(239,68,68,${0.07+intens*0.4})`:"transparent";const col=isDiag?"#065f46":val>0?"#991b1b":"#94a3b8";return`<td style="background:${bg};color:${col};font-weight:${isDiag?700:400}">${val}</td>`;}).join("")}</tr>`).join("")}</tbody></table><p class="cm-note">${size}×${size} · ${total} test samples · diagonal = correct predictions</p></div>`;}}
  const bestScore=leaderboard[0]?.score??0;
  const lbRows=leaderboard.map((m,i)=>{const rel=bestScore>0?((m.score??0)/bestScore*100).toFixed(1):0;const rankIcon=i===0?"🥇":i===1?"🥈":i===2?"🥉":`${i+1}`;return`<tr class="${i===0?"lb-best":i%2===0?"lb-even":""}"><td class="lb-rank">${rankIcon}</td><td class="lb-name">${m.name}${i===0?'<span class="best-badge">BEST</span>':""}</td><td class="lb-score">${m.score!==null?Number(m.score).toFixed(4):"—"}</td><td class="lb-cv">${m.cv!==null&&m.cv!==undefined?Number(m.cv).toFixed(4):"—"}</td><td class="lb-bar-cell"><div class="lb-bar-track"><div class="lb-bar-fill ${i===0?"lb-bar-gold":"lb-bar-blue"}" style="width:${rel}%"></div></div><span class="lb-bar-pct">${rel}%</span></td></tr>`;}).join("");
  const qRating=quality?.rating?.toLowerCase()||"";
  const qStyles={excellent:"background:#d1fae5;color:#065f46;border:1px solid #6ee7b7",good:"background:#dbeafe;color:#1e40af;border:1px solid #93c5fd",fair:"background:#fef3c7;color:#92400e;border:1px solid #fcd34d",poor:"background:#fee2e2;color:#991b1b;border:1px solid #fca5a5"};
  const qStyle=qStyles[qRating]||qStyles.good;
  const chartPages=hasCharts?chartEmbeds.map((c,i)=>`<div class="chart-page"><div class="chart-pg-header"><div class="chart-pg-num">${i+1}</div><div class="chart-pg-title">${c.title}</div><div class="chart-pg-tag">Matplotlib + Seaborn · Auto-generated</div></div><div class="chart-pg-body"><img src="${c.src}" alt="${c.title}" class="chart-img"/></div></div>`).join(""):`<div class="no-charts"><div class="no-charts-icon">📊</div><div class="no-charts-title">Dashboard Charts Not Included</div><div class="no-charts-sub">Click "📊 Dashboard" in the app, then re-download the report to embed all charts.</div></div>`;
  const featureChips=columns.filter(c=>c!==target).map(c=>`<span class="feat-chip">${fmtL(c)}</span>`).join("");
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/><title>AutoAnalytica Report — ${modelName}</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/><style>*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}:root{--indigo:#4f46e5;--violet:#7c3aed;--teal:#0d9488;--green:#10b981;--amber:#f59e0b;--red:#ef4444;--s50:#f8fafc;--s100:#f1f5f9;--s200:#e2e8f0;--s300:#cbd5e1;--s400:#94a3b8;--s500:#64748b;--s600:#475569;--s700:#334155;--s800:#1e293b;--s900:#0f172a;--font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;--r:12px;--sh:0 1px 3px rgba(0,0,0,0.08),0 4px 16px rgba(0,0,0,0.06);--sh-lg:0 4px 6px rgba(0,0,0,0.05),0 10px 40px rgba(0,0,0,0.10)}html{background:var(--s100)}body{font-family:var(--font);color:var(--s800);background:var(--s100);font-size:14px;line-height:1.6}.wrap{max-width:960px;margin:0 auto;background:#fff;box-shadow:var(--sh-lg)}.cover{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 28%,#312e81 52%,#4f46e5 72%,#7c3aed 88%,#a855f7 100%);padding:64px 72px 56px;position:relative;overflow:hidden;page-break-after:always}.cv-eyebrow{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.22);border-radius:100px;padding:6px 16px;font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:rgba(255,255,255,0.85);margin-bottom:28px;font-family:var(--mono)}.cv-dot{width:7px;height:7px;border-radius:50%;background:#34d399}.cv-title{font-size:42px;font-weight:900;color:#fff;letter-spacing:-.03em;line-height:1.05;margin-bottom:10px}.cv-sub{font-size:11px;color:rgba(255,255,255,.50);letter-spacing:.22em;text-transform:uppercase;font-family:var(--mono);margin-bottom:48px}.cv-divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.20),transparent);margin-bottom:32px}.cv-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:24px}.cv-grid label{display:block;font-size:9px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:rgba(255,255,255,.40);margin-bottom:5px;font-family:var(--mono)}.cv-grid span{font-size:13px;font-weight:600;color:#fff;font-family:var(--mono)}.cv-chip{display:inline-flex;align-items:center;gap:6px;background:rgba(52,211,153,.18);border:1px solid rgba(52,211,153,.35);border-radius:8px;padding:2px 10px;font-size:12px;font-weight:700;color:#6ee7b7;font-family:var(--mono)}.sec{padding:40px 48px;border-bottom:1px solid var(--s100);page-break-inside:avoid}.sec-head{display:flex;align-items:center;gap:14px;margin-bottom:28px}.sec-icon{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}.sec-title{font-size:11px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--s700);font-family:var(--mono)}.sec-rule{flex:1;height:1px;background:linear-gradient(90deg,var(--s200),transparent)}.sec-badge{padding:3px 10px;border-radius:100px;font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-family:var(--mono)}.ov-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:24px}.ov-card{background:var(--s50);border:1px solid var(--s200);border-radius:10px;padding:16px 14px;text-align:center}.ov-val{font-size:24px;font-weight:800;color:var(--s800);font-family:var(--mono)}.ov-label{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--s400);margin-top:4px;font-family:var(--mono)}.meta-t{width:100%;border-collapse:collapse;margin-bottom:20px;font-size:13px}.meta-t tr{border-bottom:1px solid var(--s100)}.meta-t td{padding:10px 14px}.meta-t td:first-child{color:var(--s500);font-weight:500;width:40%;font-size:12px}.meta-t td:last-child{color:var(--s800);font-weight:600;font-family:var(--mono);font-size:12px;text-align:right}.feat-wrap{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}.feat-chip{background:linear-gradient(135deg,#ede9fe,#e0e7ff);border:1px solid #c4b5fd;border-radius:100px;padding:4px 12px;font-size:11px;color:#5b21b6;font-weight:500}.feat-label{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--s400);font-family:var(--mono);margin-bottom:10px}.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:28px;page-break-inside:avoid}.kpi-tile{background:#fff;border:1px solid var(--s200);border-radius:var(--r);padding:18px 20px;position:relative;overflow:visible;box-shadow:var(--sh);word-break:break-word;page-break-inside:avoid}.kpi-tile::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--ta,var(--indigo));border-radius:var(--r) var(--r) 0 0}.kpi-tile-icon{font-size:18px;margin-bottom:10px;display:block}.kpi-tile-label{font-size:9px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--s400);font-family:var(--mono);margin-bottom:6px}.kpi-tile-val{font-weight:800;color:var(--ta,var(--indigo));font-family:var(--mono);line-height:1.2;word-break:break-word}.kpi-tile-sub{font-size:10px;color:var(--s400);margin-top:6px}.insight{background:linear-gradient(135deg,#f5f3ff,#faf5ff);border:1px solid #e9d5ff;border-left:4px solid var(--violet);border-radius:10px;padding:18px 22px;margin:20px 0;page-break-inside:avoid}.insight-title{font-size:9px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--violet);font-family:var(--mono);margin-bottom:8px}.insight p{color:var(--s600);font-size:13px;line-height:1.8}.qbanner{display:flex;align-items:center;gap:18px;border:1px solid var(--s200);border-radius:var(--r);padding:16px 22px;margin-bottom:20px;background:#fff;box-shadow:var(--sh);page-break-inside:avoid}.qrating{padding:6px 16px;border-radius:100px;font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-family:var(--mono);flex-shrink:0}.qtext{color:var(--s600);font-size:12px;line-height:1.7}.cm2-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;page-break-inside:avoid}.cm2-cell{border-radius:10px;padding:18px 20px;border:1px solid var(--s200);position:relative;overflow:hidden}.cm2-cell::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}.cm2-cell.tn{background:#f0fdf4}.cm2-cell.tn::before{background:#10b981}.cm2-cell.fp{background:#fff1f2}.cm2-cell.fp::before{background:#f43f5e}.cm2-cell.fn{background:#fff7ed}.cm2-cell.fn::before{background:#f59e0b}.cm2-cell.tp{background:#eff6ff}.cm2-cell.tp::before{background:#3b82f6}.cm2-abbr{font-size:13px;font-weight:800;font-family:var(--mono);margin-bottom:6px}.cm2-cell.tn .cm2-abbr{color:#065f46}.cm2-cell.fp .cm2-abbr{color:#be123c}.cm2-cell.fn .cm2-abbr{color:#92400e}.cm2-cell.tp .cm2-abbr{color:#1d4ed8}.cm2-count{font-size:40px;font-weight:900;font-family:var(--mono);line-height:1;margin-bottom:6px}.cm2-cell.tn .cm2-count{color:#10b981}.cm2-cell.fp .cm2-count{color:#f43f5e}.cm2-cell.fn .cm2-count{color:#f59e0b}.cm2-cell.tp .cm2-count{color:#3b82f6}.cm2-name{font-size:11px;font-weight:700;color:var(--s700);margin-bottom:2px}.cm2-desc{font-size:10px;color:var(--s400)}.cm-table{border-collapse:collapse;width:100%;font-size:12px}.cm-table th{padding:8px 14px;background:var(--s50);color:var(--s600);font-weight:700;text-align:center;font-family:var(--mono)}.cm-table td{padding:8px 14px;text-align:center;border:1px solid var(--s100);font-family:var(--mono);font-size:13px}.cm-corner{padding:8px 12px;background:var(--s50);color:var(--s400);font-weight:600;font-size:10px;text-align:left}.cm-row-head{background:var(--s50);color:var(--s600);font-weight:700;text-align:left;border-right:2px solid var(--s200)}.cm-note{font-size:11px;color:var(--s400);margin-top:8px}.lb-t{width:100%;border-collapse:collapse;font-size:13px}.lb-t thead tr{background:var(--s800)}.lb-t thead th{padding:12px 16px;color:var(--s300);font-size:9px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;font-family:var(--mono);text-align:left}.lb-t tbody tr{border-bottom:1px solid var(--s100)}.lb-even td{background:var(--s50)}.lb-best td{background:linear-gradient(135deg,#fefce8,#fff7ed)}.lb-t td{padding:12px 16px}.lb-rank{text-align:center;font-size:15px;width:52px}.lb-name{color:var(--s800);font-weight:600}.lb-score{text-align:right;font-weight:800;font-family:var(--mono);color:var(--indigo);font-size:14px}.lb-cv{text-align:right;color:var(--s500);font-family:var(--mono);font-size:12px}.lb-bar-cell{width:160px;display:flex;align-items:center;gap:8px;justify-content:flex-end}.lb-bar-track{flex:1;height:6px;background:var(--s200);border-radius:3px;overflow:hidden}.lb-bar-fill{height:100%;border-radius:3px}.lb-bar-gold{background:linear-gradient(90deg,#f59e0b,#ef4444)}.lb-bar-blue{background:linear-gradient(90deg,#818cf8,#4f46e5)}.lb-bar-pct{font-size:10px;font-family:var(--mono);color:var(--s400);width:36px;text-align:right}.best-badge{display:inline-block;background:linear-gradient(135deg,#fef3c7,#fde68a);color:#92400e;border:1px solid #fcd34d;border-radius:6px;padding:1px 7px;font-size:8px;font-weight:800;letter-spacing:.1em;font-family:var(--mono);margin-left:8px}.cfg-t{width:100%;border-collapse:collapse;font-size:12px}.cfg-t tr{border-bottom:1px solid var(--s100)}.cfg-t tr:nth-child(even) td{background:var(--s50)}.cfg-t td{padding:11px 16px}.cfg-t td:first-child{color:var(--s500);font-weight:500;width:38%}.cfg-t td:last-child{color:var(--s800);font-weight:600;font-family:var(--mono);font-size:11px;text-align:right}.cfg-icon{width:22px;height:22px;border-radius:5px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;background:var(--s100);margin-right:8px;vertical-align:middle}.charts-intro{padding:40px 48px 24px;border-bottom:1px solid var(--s100);page-break-before:always;page-break-inside:avoid}.chart-page{page-break-before:always;page-break-inside:avoid;background:#fff;border-bottom:1px solid var(--s100)}.chart-pg-header{display:flex;align-items:center;gap:14px;padding:20px 48px 16px;background:var(--s800)}.chart-pg-num{width:30px;height:30px;background:linear-gradient(135deg,var(--indigo),var(--violet));border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#fff;font-family:var(--mono)}.chart-pg-title{font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--s200);font-family:var(--mono);flex:1}.chart-pg-tag{font-size:9px;color:var(--s500);font-family:var(--mono)}.chart-pg-body{padding:0;line-height:0}.chart-img{width:100%;display:block;max-height:calc(297mm - 130px);object-fit:contain}.no-charts{padding:64px 48px;text-align:center}.no-charts-icon{font-size:52px;margin-bottom:16px}.no-charts-title{font-size:18px;font-weight:700;color:var(--s700);margin-bottom:8px}.no-charts-sub{font-size:13px;color:var(--s400);max-width:400px;margin:0 auto}.footer{background:linear-gradient(135deg,var(--s900),#1e1b4b);color:rgba(255,255,255,.40);padding:24px 48px;display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:.18em;text-transform:uppercase;font-family:var(--mono)}.footer-brand{color:rgba(255,255,255,.65);font-weight:600}.print-fab{position:fixed;bottom:28px;right:28px;z-index:9999;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;border:none;border-radius:100px;padding:14px 28px;font-size:13px;font-weight:700;font-family:var(--mono);cursor:pointer;box-shadow:0 8px 32px rgba(79,70,229,.45);text-transform:uppercase;display:flex;align-items:center;gap:10px}*{-webkit-print-color-adjust:exact!important;print-color-adjust:exact!important}@media print{@page{margin:.45in .35in .35in .35in;size:A4 portrait}html,body{background:#fff!important}.wrap{box-shadow:none!important;max-width:100%!important}.print-fab{display:none!important}.cover{page-break-after:always!important;padding-top:44px!important}.sec{page-break-inside:avoid!important;padding-top:32px!important}.kpi-row,.kpi-tile,.qbanner,.insight{page-break-inside:avoid!important}.charts-intro{page-break-before:always!important;page-break-inside:avoid!important;padding-top:32px!important}.chart-page{page-break-before:always!important;page-break-inside:avoid!important}.chart-img{width:100%!important;height:auto!important;max-height:calc(297mm - 130px)!important}.lb-t tr{page-break-inside:avoid!important}}</style></head><body><div class="wrap"><div class="cover"><div class="cv-eyebrow"><span class="cv-dot"></span> AutoAnalytica ML Report</div><div class="cv-title">Model Analysis<br>Report</div><div class="cv-sub">Machine Intelligence Platform · Version 5.5</div><div class="cv-divider"></div><div class="cv-grid"><div><label>Generated</label><span>${now}</span></div><div><label>Dataset</label><span>${file?.name||"Unknown"}</span></div><div><label>Target Column</label><span>${fmtL(target)}</span></div><div><label>Best Model</label><span class="cv-chip">● ${modelName}</span></div><div><label>Problem Type</label><span>${fmtL(modelType)}</span></div><div><label>Scale Tier</label><span>${tier}</span></div><div><label>Charts Included</label><span>${hasCharts?chartEmbeds.length+" chart"+(chartEmbeds.length!==1?"s":""):"None"}</span></div><div><label>File Size</label><span>${file?(file.size/1024).toFixed(1)+" KB":"—"}</span></div></div></div><div class="sec"><div class="sec-head"><div class="sec-icon" style="background:#eff6ff">📊</div><div class="sec-title">Dataset Overview</div><div class="sec-rule"></div><span class="sec-badge" style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe">${nTotal} rows</span></div><div class="ov-grid"><div class="ov-card"><div class="ov-val">${nTotal}</div><div class="ov-label">Total Rows</div></div><div class="ov-card"><div class="ov-val">${nFeat}</div><div class="ov-label">Features</div></div><div class="ov-card"><div class="ov-val">${columns.length}</div><div class="ov-label">Columns</div></div><div class="ov-card"><div class="ov-val">${nTrain}</div><div class="ov-label">Train Rows</div></div><div class="ov-card"><div class="ov-val">${nTest}</div><div class="ov-label">Test Rows</div></div><div class="ov-card"><div class="ov-val" style="font-size:${fmtL(modelType).length>10?"13px":"18px"}">${fmtL(modelType)}</div><div class="ov-label">Problem Type</div></div></div><table class="meta-t"><tr><td>File Name</td><td>${file?.name||"—"}</td></tr><tr><td>Target Column</td><td>${fmtL(target)}</td></tr><tr><td>Train / Test Split</td><td>80% / 20% — stratified random</td></tr><tr><td>Scale Tier</td><td>${tier}</td></tr>${modelType==="classification"?`<tr><td>Number of Classes</td><td>${nClasses}</td></tr>`:""} ${baseline!==undefined&&baseline!==null?`<tr><td>Majority-Class Baseline</td><td>${p(baseline)}</td></tr>`:""}</table><div class="feat-label">All Features (${nFeat})</div><div class="feat-wrap">${featureChips}</div></div><div class="sec"><div class="sec-head"><div class="sec-icon" style="background:#f5f3ff">🧠</div><div class="sec-title">Model Intelligence</div><div class="sec-rule"></div><span class="sec-badge" style="background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe">${fmtL(modelType)}</span></div><div class="kpi-row"><div class="kpi-tile" style="--ta:#4f46e5"><span class="kpi-tile-icon">🏅</span><div class="kpi-tile-label">Best Model</div><div class="kpi-tile-val" style="font-size:${mnFs}">${modelName}</div></div>${modelType==="classification"&&accuracy!==null?`<div class="kpi-tile" style="--ta:#10b981"><span class="kpi-tile-icon">✓</span><div class="kpi-tile-label">Test Accuracy</div><div class="kpi-tile-val" style="font-size:28px">${p(accuracy)}</div></div>`:""}${modelType==="classification"&&roc_auc!==null?`<div class="kpi-tile" style="--ta:#f59e0b"><span class="kpi-tile-icon">◎</span><div class="kpi-tile-label">ROC AUC</div><div class="kpi-tile-val" style="font-size:28px">${num(roc_auc)}</div></div>`:""}${modelType==="regression"&&r2!==null?`<div class="kpi-tile" style="--ta:#0d9488"><span class="kpi-tile-icon">◈</span><div class="kpi-tile-label">R² Score</div><div class="kpi-tile-val" style="font-size:28px">${num(r2)}</div></div>`:""}${cvScoreMean!==null&&cvScoreMean!==undefined?`<div class="kpi-tile" style="--ta:#6366f1"><span class="kpi-tile-icon">⊙</span><div class="kpi-tile-label">CV Score</div><div class="kpi-tile-val" style="font-size:28px">${modelType==="classification"?p(cvScoreMean):num(cvScoreMean)}</div></div>`:""}</div><table class="meta-t"><tr><td>Model</td><td>${modelName}</td></tr><tr><td>Problem Type</td><td>${fmtL(modelType)}</td></tr>${modelType==="classification"?`<tr><td>Test Accuracy</td><td>${p(accuracy)}</td></tr><tr><td>CV Score (mean±std)</td><td>${cvScoreMean!==null?(modelType==="classification"?p(cvScoreMean):num(cvScoreMean)):"—"} ± ${cvScoreStd!==null&&cvScoreStd!==undefined?(modelType==="classification"?(cvScoreStd*100).toFixed(3)+"%":num(cvScoreStd)):"—"}</td></tr>${baseline!==null?`<tr><td>Baseline Accuracy</td><td>${p(baseline)}</td></tr><tr><td>Improvement</td><td>${accuracy!==null&&baseline!==null?"+"+ ((accuracy-baseline)*100).toFixed(2)+"pp above baseline":"—"}</td></tr>`:""}`:``}${quality?.rating?`<tr><td>Quality Rating</td><td>${quality.rating}</td></tr>`:""}</table>${quality?.rating?`<div class="qbanner"><div class="qrating" style="${qStyle}">${quality.rating}</div><div>${quality.summary?`<div class="qtext"><strong>${quality.summary}</strong></div>`:""}${quality.details?`<div class="qtext">${quality.details}</div>`:""}</div></div>`:""}<div class="insight"><div class="insight-title">💡 Automated Insight</div><p>${modelType==="classification"?`The <strong>${modelName}</strong> model classifies records.${accuracy!==null?` Test accuracy: <strong>${p(accuracy)}</strong>.`:""}${baseline!==null&&accuracy!==null?` <strong>${((accuracy-baseline)*100).toFixed(1)}pp above</strong> the naive baseline of ${p(baseline)}.`:""}`:` The <strong>${modelName}</strong> model predicts <strong>${fmtL(target)}</strong>.${r2!==null?` R²: <strong>${num(r2)}</strong> — accounts for <strong>${(Math.max(0,r2)*100).toFixed(1)}%</strong> of variance.`:""}`}</p></div>${cm?`<div style="margin-top:20px"><div class="feat-label">Confusion Matrix</div>${cmHTML}</div>`:""}</div>${leaderboard.length>0?`<div class="sec"><div class="sec-head"><div class="sec-icon" style="background:#fffbeb">🏆</div><div class="sec-title">Model Leaderboard</div><div class="sec-rule"></div><span class="sec-badge" style="background:#fffbeb;color:#92400e;border:1px solid #fcd34d">${leaderboard.length} models</span></div><table class="lb-t"><thead><tr><th style="width:52px">#</th><th>Model</th><th>Test Score</th><th>CV Score</th><th style="width:180px">Relative</th></tr></thead><tbody>${lbRows}</tbody></table></div>`:""}<div class="sec"><div class="sec-head"><div class="sec-icon" style="background:#f0fdf4">⚙️</div><div class="sec-title">Technical Configuration</div><div class="sec-rule"></div></div><table class="cfg-t"><tr><td><span class="cfg-icon">🔢</span>Pipeline Version</td><td>AutoML v5.5 + Learning AI System</td></tr><tr><td><span class="cfg-icon">📏</span>Scale Tier</td><td>${tier} — ${nTotal} rows</td></tr><tr><td><span class="cfg-icon">🔄</span>Cross-Validation</td><td>3-Fold ${modelType==="classification"?"StratifiedKFold":"KFold"} · GridSearchCV ≤30 combos</td></tr><tr><td><span class="cfg-icon">🤖</span>RL Agent</td><td>DQN Multi-Armed Bandit · epsilon-greedy</td></tr><tr><td><span class="cfg-icon">🧠</span>Meta-Model</td><td>GBR Quality Scorer + k-NN Insight Retriever</td></tr><tr><td><span class="cfg-icon">🕒</span>Generated</td><td>${now}</td></tr></table></div><div class="charts-intro"><div class="sec-head" style="margin-bottom:0"><div class="sec-icon" style="background:#faf5ff">📈</div><div class="sec-title">Analytics Dashboard</div><div class="sec-rule"></div><span class="sec-badge" style="background:#faf5ff;color:#7e22ce;border:1px solid #e9d5ff">${hasCharts?chartEmbeds.length+" charts":"Not generated"}</span></div></div>${chartPages}<div class="footer"><span class="footer-brand">AutoAnalytica AI Platform · v5.5</span><span>${now}</span></div></div><script>function doPrint(){var btn=document.querySelector('.print-fab');if(btn)btn.style.display='none';setTimeout(function(){window.print();if(btn)setTimeout(function(){btn.style.display='';},1000);},600);}</script><button class="print-fab" onclick="doPrint()">🖨️ &nbsp;Save as PDF</button></body></html>`;
}

function DownloadReportBtn({trainResponse,dashboardData,columns,target,file,disabled}){
  const C=useTheme();const [h,setH]=useState(false);const [generating,setGenerating]=useState(false);
  const handleDownload=async()=>{
    if(!trainResponse||generating)return;setGenerating(true);
    try{
      const html=await generateHTMLReport({trainResponse,dashboardData,columns,target,file,apiBase:"http://127.0.0.1:8000"});
      const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([html],{type:"text/html"}));a.download=`AutoAnalytica_Report_${(target||"report").replace(/\s+/g,"_")}.html`;document.body.appendChild(a);a.click();document.body.removeChild(a);
    }catch(err){console.error("Report generation failed:",err);}
    finally{setGenerating(false);}
  };
  return(<button onClick={handleDownload} disabled={disabled||generating} onMouseEnter={()=>setH(true)} onMouseLeave={()=>setH(false)} style={{display:"flex",alignItems:"center",gap:10,padding:"12px 22px",borderRadius:12,cursor:(disabled||generating)?"not-allowed":"pointer",border:`1px solid ${h?C.primary+"70":C.primary+"35"}`,background:h?`linear-gradient(135deg,${C.primary}18,${C.rose}0a)`:`linear-gradient(135deg,${C.primary}08,${C.rose}05)`,boxShadow:h?`0 4px 20px ${C.primary}20`:"0 1px 4px rgba(107,79,200,0.06)",transition:"all 0.2s",opacity:(disabled||generating)?0.4:1,fontFamily:"'Space Grotesk',monospace",width:"100%"}}>
    <div style={{width:32,height:32,borderRadius:8,flexShrink:0,background:generating?`${C.amber}14`:`linear-gradient(135deg,${C.violetTint},${C.coralTint})`,border:`1px solid ${C.primary}22`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:15,animation:generating?"pulse 1s ease infinite":"none"}}>{generating?"⏳":"📄"}</div>
    <div style={{textAlign:"left"}}><div style={{color:h?C.primary:C.hiText,fontSize:12,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase"}}>{generating?"Building Report…":"Download Full Report"}</div><div style={{color:C.midText,fontSize:10,marginTop:1}}>Dataset · Metrics · Leaderboard · Config</div></div>
    <div style={{marginLeft:"auto",fontSize:16,color:generating?C.amber:h?C.primary:C.loText}}>{generating?"⏳":"→"}</div>
  </button>);
}

const INSIGHTS=[
  {num:"01",emoji:"🏗️",tag:"ARCHITECTURE",title:"AutoML Pipeline Architecture",subtitle:"5-Stage Pipeline",accent:"primary",short:"A full walkthrough of the 5-stage AutoML pipeline.",content:[{heading:"Stage 1 — Ingestion & Type Inference",body:"Raw CSVs parsed and every column classified. Date columns decomposed into 5 features."},{heading:"Stage 2 — 26-Step Universal Cleaning",body:"Missing values imputed. Outliers clipped at IQR×3. 26 steps verified in code."},{heading:"Stage 3 — Feature Engineering",body:"59 transforms across 8 categories. StandardScaler inside Pipeline — no leakage."},{heading:"Stage 4 — Adaptive Model Selection",body:"Tier 1 (<10k rows): 18 cls / 16 reg models, 3-fold CV + GridSearchCV. Tier-aware scaling."},{heading:"Stage 5 — SHAP Explainability",body:"TreeExplainer for tree models, LinearExplainer for linear, KernelExplainer as fallback."}]},
  {num:"02",emoji:"⚡",tag:"PERFORMANCE",title:"Make Training 20× Faster",subtitle:"Speed Optimization Guide",accent:"cyan",short:"5 concrete optimizations that dramatically reduce training time.",content:[{heading:"1 — Replace SVC(rbf) with LinearSVC",body:"LinearSVC scales linearly vs O(n²) for RBF kernel. 440× speedup on 1k rows."},{heading:"2 — Reduce n_estimators During CV",body:"CV is for ranking models. Drop RF 300→100, retrain winner at full power after."},{heading:"3 — 5-Fold → 3-Fold CV",body:"1.67× speedup with only marginal variance increase on datasets above 500 rows."},{heading:"4 — n_jobs=-1",body:"All ensemble models support parallel training. ~4× on 8-core machines."},{heading:"5 — Tier-Based Subsampling",body:"Tier 3: CV runs on stratified 50k sample. 10–20× wall-clock reduction."}]},
  {num:"03",emoji:"📊",tag:"DASHBOARDS",title:"Auto Dashboards Like Power BI",subtitle:"Zero-Config Analytics",accent:"rose",short:"How the dashboard engine auto-generates 5 tabs of visual analytics.",content:[{heading:"Tab 1 — Overview",body:"Dataset summary plus class/target distribution chart."},{heading:"Tab 2 — Numeric Analysis",body:"Box plots and violin plots for all numeric columns grouped by target."},{heading:"Tab 3 — Categorical Analysis",body:"Stacked bar charts showing value counts per category, coloured by target label."},{heading:"Tab 4 — Time Series",body:"Detects date columns and plots trend lines with moving averages."},{heading:"Tab 5 — Data Quality",body:"Heatmap of missing values, bar chart of missing %, and correlation matrix."}]},
  {num:"04",emoji:"🧠",tag:"ALGORITHMS",title:"34 Algorithms, 495 Combos",subtitle:"Full Algorithm Library",accent:"violet",short:"34 algorithms evaluated (18 classification · 16 regression) across 4 scale tiers.",content:[{heading:"Classification — 18 algorithms",body:"LogisticRegression · LinearSVC · SGDClassifier · KNN · DecisionTree · RandomForest · ExtraTrees · GradientBoosting · HistGradientBoosting · XGBoost · LightGBM · CatBoost · AdaBoost · Bagging · GaussianNB · BernoulliNB · MultinomialNB · QDA"},{heading:"Regression — 16 algorithms",body:"LinearRegression · Ridge · Lasso · ElasticNet · BayesianRidge · SGDRegressor · LinearSVR · KNN · DecisionTree · RandomForest · ExtraTrees · GradientBoosting · HistGradientBoosting · XGBoost · LightGBM · CatBoost"},{heading:"Hyperparameter Search",body:"495 total combos across 26 grids. GridSearchCV ≤30 combos, RandomizedSearchCV n_iter=20 otherwise."},{heading:"v5.5 Learning AI System",body:"RL Agent (DQN) · Meta-Model (GBR + k-NN) · Retrain Model (Calibrated RF) · Agent System (Planner + Executor)"}]},
];

function InsightDrawer(){
  const C=useTheme();const [drawerOpen,setDrawerOpen]=useState(false);const [expanded,setExpanded]=useState(null);
  const accentMap={primary:C.primary,cyan:C.cyan,rose:C.rose,violet:C.violet};
  return(<>
    <button onClick={()=>setDrawerOpen(o=>!o)} title="Learn More" style={{position:"fixed",right:drawerOpen?372:0,top:"50%",transform:"translateY(-50%)",zIndex:300,background:`linear-gradient(135deg,${C.primary},${C.rose})`,color:"#fff",border:"none",borderRadius:"12px 0 0 12px",padding:"14px 10px",cursor:"pointer",boxShadow:`-4px 0 20px ${C.primary}40`,display:"flex",flexDirection:"column",alignItems:"center",gap:6,transition:"right 0.35s cubic-bezier(0.34,1.1,0.64,1)",fontFamily:"'Space Grotesk',sans-serif",writingMode:"vertical-rl",fontSize:10,fontWeight:700,letterSpacing:"0.14em",textTransform:"uppercase"}}>
      <span style={{fontSize:16,writingMode:"horizontal-tb"}}>💡</span><span>Learn More</span><span style={{fontSize:10,writingMode:"horizontal-tb",opacity:0.75}}>{drawerOpen?"›":"‹"}</span>
    </button>
    {drawerOpen&&(<div onClick={()=>setDrawerOpen(false)} style={{position:"fixed",inset:0,zIndex:200,background:"rgba(0,0,0,0.35)",backdropFilter:"blur(2px)"}}/>)}
    <div style={{position:"fixed",top:0,right:0,bottom:0,width:370,zIndex:250,transform:drawerOpen?"translateX(0)":"translateX(100%)",transition:"transform 0.35s cubic-bezier(0.34,1.1,0.64,1)",display:"flex",flexDirection:"column",background:C.isDark?`linear-gradient(160deg,${C.surface}f8,${C.card}f0)`:`rgba(255,252,255,0.97)`,backdropFilter:"blur(32px)",borderLeft:`1px solid ${C.border}`,boxShadow:"-16px 0 48px rgba(0,0,0,0.25)"}}>
      <div style={{padding:"22px 22px 18px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",justifyContent:"space-between",background:`linear-gradient(90deg,${C.primary}10,${C.rose}08)`,flexShrink:0}}>
        <div><div style={{fontSize:11,fontWeight:700,letterSpacing:"0.18em",textTransform:"uppercase",color:C.primary,fontFamily:"'Space Grotesk',monospace",marginBottom:4}}>💡 Knowledge Base</div><div style={{fontSize:15,fontWeight:800,color:C.hiText}}>Learn AutoAnalytica</div></div>
        <button onClick={()=>setDrawerOpen(false)} style={{width:32,height:32,borderRadius:8,background:`${C.primary}14`,border:`1px solid ${C.border}`,color:C.midText,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>×</button>
      </div>
      <div style={{flex:1,overflowY:"auto",padding:"14px 14px 24px"}}>
        {INSIGHTS.map((card,ci)=>{const ac=accentMap[card.accent]||C.primary;const isOpen=expanded===ci;return(<div key={ci} style={{marginBottom:10}}>
          <button onClick={()=>setExpanded(isOpen?null:ci)} style={{width:"100%",textAlign:"left",padding:"14px 16px",borderRadius:isOpen?"12px 12px 0 0":12,border:`1px solid ${isOpen?ac+"55":C.border}`,borderBottom:isOpen?"none":`1px solid ${C.border}`,background:isOpen?`linear-gradient(135deg,${ac}14,${ac}08)`:C.isDark?`${C.card}cc`:"rgba(255,255,255,0.8)",cursor:"pointer",display:"flex",alignItems:"center",gap:12}}>
            <div style={{width:34,height:34,borderRadius:9,flexShrink:0,background:`${ac}22`,border:`1px solid ${ac}40`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:15}}>{card.emoji}</div>
            <div style={{flex:1}}><div style={{fontSize:9,fontWeight:700,letterSpacing:"0.16em",textTransform:"uppercase",color:ac,fontFamily:"'Space Grotesk',monospace"}}>{card.num} — {card.tag}</div><div style={{fontSize:13,fontWeight:700,color:C.hiText,marginTop:1}}>{card.title}</div></div>
            <span style={{color:C.midText,fontSize:11,transform:isOpen?"rotate(180deg)":"none",transition:"transform 0.25s",display:"inline-block"}}>▼</span>
          </button>
          {isOpen&&(<div style={{border:`1px solid ${ac}55`,borderTop:"none",borderRadius:"0 0 12px 12px",background:C.isDark?`${C.surface}dd`:"rgba(255,255,255,0.95)",padding:"0 16px 16px",animation:"fadeSlideUp 0.25s ease both"}}>
            <p style={{fontSize:12,color:C.midText,lineHeight:1.7,margin:"14px 0 14px",paddingBottom:12,borderBottom:`1px solid ${C.border}`}}>{card.short}</p>
            {card.content.map((sec,si)=>(<div key={si} style={{marginBottom:14}}><div style={{display:"flex",alignItems:"center",gap:7,marginBottom:4}}><div style={{width:16,height:16,borderRadius:4,flexShrink:0,background:`${ac}22`,border:`1px solid ${ac}40`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:8,fontWeight:800,color:ac,fontFamily:"'Space Grotesk',monospace"}}>{si+1}</div><div style={{fontSize:10,fontWeight:700,color:ac,letterSpacing:"0.12em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace"}}>{sec.heading}</div></div><div style={{fontSize:12,color:C.midText,lineHeight:1.72,paddingLeft:23}}>{sec.body}</div></div>))}
          </div>)}
        </div>);})}
      </div>
    </div>
  </>);
}

export default function Upload(){
  const API="http://127.0.0.1:8000";
  const [isDark,setIsDark]=useState(()=>{try{return localStorage.getItem("aa_theme")==="dark";}catch{return false;}});
  C=isDark?DARK:LIGHT;
  useEffect(()=>{try{localStorage.setItem("aa_theme",isDark?"dark":"light");}catch{}document.body.style.background=isDark?DARK.bg:LIGHT.bg;},[isDark]);
  const toggleTheme=()=>setIsDark(d=>!d);
  const theme=isDark?DARK:LIGHT;

  const [file,             setFile]             = useState(null);
  const [target,           setTarget]           = useState("");
  const [columns,          setColumns]          = useState([]);
  const [cleanedFile,      setCleanedFile]      = useState(null);
  const [trainResponse,    setTrainResponse]    = useState(null);
  const [dashboardData,    setDashboardData]    = useState(null);
  const [predictionInput,  setPredictionInput]  = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading,          setLoading]          = useState(false);
  const [loadMsg,          setLoadMsg]          = useState("Processing…");
  const [toast,            setToast]            = useState({text:"",ok:true});
  const [dragOver,         setDragOver]         = useState(false);
  const [activeTab,        setActiveTab]        = useState("metrics");
  const [agentFields,      setAgentFields]      = useState(null);
  const [agentRunId,       setAgentRunId]       = useState(null);
  const [feedbackSent,     setFeedbackSent]     = useState(false);

  const notify=(text,ok=true)=>{setToast({text,ok});setTimeout(()=>setToast({text:"",ok:true}),4500);};

  const handleDrop=e=>{e.preventDefault();setDragOver(false);const f=e.dataTransfer.files[0];if(f&&/\.(csv|xlsx?)/i.test(f.name))setFile(f);else notify("Please drop a CSV or Excel file.",false);};

  const handleUpload=async()=>{
    if(!file)return notify("Please select a file first.",false);
    const fd=new FormData();fd.append("file",file);
    try{setLoadMsg("Uploading & analysing dataset…");setLoading(true);
      const res=await axios.post(`${API}/upload/api/upload`,fd);
      setColumns(Object.keys(res.data?.analysis?.column_types||{}));
      setCleanedFile(res.data?.cleaned_filename||null);
      setTrainResponse(null);setDashboardData(null);setPredictionResult(null);setPredictionInput({});
      setAgentFields(null);setAgentRunId(null);setFeedbackSent(false);
      notify("Dataset uploaded and cleaned ✓");
    }catch{notify("Upload failed — is the backend running?",false);}
    finally{setLoading(false);}
  };

  const handleTrain=async()=>{
    if(!cleanedFile)return notify("Upload a dataset first.",false);
    if(!target)return notify("Select a target column.",false);
    setAgentFields(null);setAgentRunId(null);setFeedbackSent(false);
    try{
      setLoadMsg("Training models + AI agents — this may take a moment…");setLoading(true);
      const res=await axios.post(`${API}/ai/train`,{filename:cleanedFile,target_column:target});
      if(res.data?.error){notify(res.data.error,false);return;}

      // ── DEBUG: log raw response so you can verify shape in the browser console
      console.log("[AutoAnalytica] /ai/train raw response:", res.data);

      setTrainResponse(res.data||null);
      const agent=extractAgentFields(res.data||{});
      setAgentFields(agent);
      if(res.data?.agent_run_id)setAgentRunId(res.data.agent_run_id);
      const decisionLabel=agent.decision?` · AI: ${agent.decision.replace(/_/g," ").toUpperCase()}`:"";
      notify(`Model trained successfully ✓${decisionLabel}`);
    }catch(err){notify("Training failed — check console.",false);console.error("handleTrain error:",err);}
    finally{setLoading(false);}
  };

  const handleDashboard=async()=>{
    if(!cleanedFile)return notify("Upload a dataset first.",false);
    try{setLoadMsg("Generating analytics dashboard…");setLoading(true);
      const res=await axios.get(`${API}/dashboard/${cleanedFile}`);
      setDashboardData(res.data||null);notify("Dashboard generated ✓");
    }catch{notify("Dashboard generation failed.",false);}
    finally{setLoading(false);}
  };

  const handlePredict=async()=>{
    const modelFileName=trainResponse?.model_name;
    if(!modelFileName)return notify("Train a model first.",false);
    const cleanedInput={};
    Object.entries(predictionInput).forEach(([k,v])=>{const num=parseFloat(v);cleanedInput[k]=isNaN(num)?v:num;});
    try{setLoadMsg("Running prediction…");setLoading(true);
      const res=await axios.post(`${API}/ai/predict`,{model_name:modelFileName,input_data:cleanedInput});
      if(res.data?.error){notify(`Prediction error: ${res.data.error}`,false);}
      else{setPredictionResult(res.data);notify("Prediction complete ✓");}
    }catch(err){notify("Prediction failed — check console.",false);console.error("Predict exception:",err);}
    finally{setLoading(false);}
  };

  const handleFeedback=async(correct)=>{
    if(!agentRunId||feedbackSent)return;
    try{
      await sendFeedback(agentRunId,correct,null,null);
      setFeedbackSent(true);notify("Feedback sent — AI agent updated ✓",true);
    }catch(err){console.error("Feedback error:",err);notify("Feedback could not be sent.",false);}
  };

  // ══════════════════════════════════════════════════════════════════════════
  // FIX 1 — use resolvePerf() so an empty `metrics:{}` doesn't shadow
  //          the real data in `performance`
  // ══════════════════════════════════════════════════════════════════════════
  const perf      = resolvePerf(trainResponse);
  const modelType = trainResponse?.problem_type;
  const modelName = trainResponse?.best_model_name || trainResponse?.best_model || trainResponse?.model_name || null;

  // FIX 2 — read accuracy / R2 with safe fallbacks for both key variants
  const accuracy  = perf?.accuracy ?? null;
  const roc_auc   = perf?.roc_auc  ?? null;
  const mae       = perf?.MAE ?? perf?.mae ?? null;
  const rmse      = perf?.RMSE ?? perf?.rmse ?? null;
  const r2        = perf?.R2 ?? perf?.r2 ?? null;

  // FIX 3 — confusion matrix guard
  const cm        = Array.isArray(perf?.confusion_matrix) && perf.confusion_matrix.length >= 2
    ? perf.confusion_matrix : null;
  const cmSize    = cm ? cm.length : 2;
  const cmLabels  = perf?.confusion_matrix_labels || null;

  const accuracyLabel = getScoreLabel(accuracy, cmSize);
  const accColor      = scoreColor(accuracy, cmSize, theme);
  const allScores     = perf?.all_model_scores || {};

  const metaInsightText = agentFields ? safeMetaInsightText(agentFields.meta_insight) : null;

  const retrainDecisionRaw = trainResponse?.retrain_decision;
  const shouldRetrain = typeof retrainDecisionRaw === "object" && retrainDecisionRaw !== null
    ? !!retrainDecisionRaw.should_retrain
    : !!agentFields?.retrain_decision;
  const retrainSeverity = typeof retrainDecisionRaw === "object" && retrainDecisionRaw !== null
    ? (retrainDecisionRaw.severity ?? null) : null;
  const retrainReason = typeof retrainDecisionRaw === "object" && retrainDecisionRaw !== null
    ? (retrainDecisionRaw.reason ?? null) : null;

  const rawRiskFlags  = Array.isArray(trainResponse?.meta_insight_full?.risk_flags)
    ? trainResponse.meta_insight_full.risk_flags : [];
  const rawActiveRisks = Array.isArray(agentFields?.active_risks)
    ? agentFields.active_risks : [];
  const combinedRisks = [...new Set([...rawActiveRisks, ...rawRiskFlags])];

  const rlEpsilon = agentFields?.ls_epsilon ?? trainResponse?.rl_decision?.eps ?? null;

  const rawPrediction  = predictionResult?.prediction;
  const predVal = rawPrediction !== null && rawPrediction !== undefined
    ? (typeof rawPrediction === "object"
        ? (rawPrediction?.score ?? rawPrediction?.value ?? rawPrediction?.result ?? null)
        : rawPrediction)
    : null;
  const predDisplay    = displayPrediction(predVal);
  const predType       = predictionResult?.problem_type || modelType;
  const predConfidence = predictionResult?.confidence
    ?? (typeof rawPrediction === "object" ? rawPrediction?.confidence : null)
    ?? null;

  const pipelineStageLogs = Array.isArray(trainResponse?.pipeline_stage_logs)
    ? trainResponse.pipeline_stage_logs : [];

  const metaInsightConfidence = typeof trainResponse?.meta_insight === "object" && trainResponse.meta_insight !== null
    ? (trainResponse.meta_insight.confidence ?? null) : null;
  const metaInsightSource = typeof trainResponse?.meta_insight === "object" && trainResponse.meta_insight !== null
    ? (trainResponse.meta_insight.source ?? null) : null;

  // FIX 4 — leaderboard: use cv_mean (actual backend key), not cv_accuracy_mean
  const leaderboard = Object.entries(allScores).map(([n,s]) => ({
    name:  n,
    score: modelType === "classification" ? (s?.accuracy ?? null)     : (s?.R2 ?? s?.r2 ?? null),
    cv:    modelType === "classification" ? (s?.cv_mean ?? s?.f1_macro ?? null) : (s?.cv_r2_mean ?? null),
  })).sort((a,b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));

  // FIX 5 — SHAP: normalise object → array using resolveShap helper
  const shapArray = normaliseShap(trainResponse?.sample_explanation);

  const steps=[{label:"Upload",done:!!cleanedFile,active:!cleanedFile},{label:"Configure",done:!!cleanedFile&&!!target,active:!!cleanedFile&&!target},{label:"Train",done:!!trainResponse,active:!!target&&!trainResponse},{label:"Predict",done:!!predictionResult,active:!!trainResponse&&!predictionResult}];

  // FIX 6 — cv_score_mean / cv_r2_mean unified into a single variable for KPI display
  const cvScoreMean = perf?.cv_score_mean ?? perf?.cv_r2_mean ?? null;
  const cvScoreStd  = perf?.cv_score_std  ?? perf?.cv_r2_std  ?? null;
  const trainAcc    = perf?.train_accuracy ?? perf?.train_R2   ?? null;

  return(
    <ThemeCtx.Provider value={theme}>
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');
        *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
        html,body{background:${theme.bg};color:${theme.hiText};font-family:'DM Sans',sans-serif;overflow-x:hidden;min-height:100vh;transition:background 0.4s ease,color 0.3s ease}
        input[type=file]{display:none}input::placeholder{color:${theme.loText}}
        select option{background:${theme.surface};color:${theme.hiText}}
        @keyframes streamFlow{0%,100%{opacity:.1;transform:scaleY(.3) translateY(-20%)}50%{opacity:.45;transform:scaleY(1) translateY(0)}}
        @keyframes orbit{to{transform:rotate(360deg)}}@keyframes corePulse{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) scale(1.6)}}
        @keyframes flicker{0%,100%{opacity:1}50%{opacity:.5}}@keyframes scanV{0%{top:-5%}100%{top:105%}}
        @keyframes slideInRight{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
        @keyframes fadeSlideUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
        @keyframes shimmerBar{0%{transform:translateX(-100%)}100%{transform:translateX(300%)}}
        @keyframes btnShine{0%{transform:translateX(-100%)}100%{transform:translateX(200%)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}@keyframes titleGlow{0%{background-position:0% center}100%{background-position:300% center}}
        .reveal{animation:fadeSlideUp .55s cubic-bezier(.34,1.1,.64,1) both}
        .r1{animation-delay:.05s}.r2{animation-delay:.12s}.r3{animation-delay:.2s}.r4{animation-delay:.28s}.r5{animation-delay:.36s}
        .col-chip{transition:all .18s !important}.col-chip:hover{transform:translateY(-2px) !important}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:${theme.bg}}::-webkit-scrollbar-thumb{background:${theme.border};border-radius:4px}
      `}</style>

      {isDark?(
        <><div style={{position:"fixed",inset:0,zIndex:0,backgroundImage:`url("https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=1800&auto=format&fit=crop&q=80")`,backgroundSize:"cover",backgroundPosition:"center"}}/>
        <div style={{position:"fixed",inset:0,zIndex:1,background:`linear-gradient(160deg,rgba(8,6,18,0.52) 0%,rgba(13,8,30,0.46) 35%,rgba(10,5,22,0.50) 70%,rgba(5,3,14,0.55) 100%)`}}/>
        <div style={{position:"fixed",inset:0,zIndex:2,pointerEvents:"none",background:`radial-gradient(ellipse 55% 45% at 10% 35%,rgba(155,127,240,0.18) 0%,transparent 60%),radial-gradient(ellipse 45% 35% at 90% 15%,rgba(20,212,187,0.14) 0%,transparent 55%)`}}/></>
      ):(
        <><div style={{position:"fixed",inset:0,zIndex:0,backgroundImage:`url("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1800&auto=format&fit=crop&q=85")`,backgroundSize:"cover",backgroundPosition:"center 40%"}}/>
        <div style={{position:"fixed",inset:0,zIndex:1,background:`linear-gradient(160deg,rgba(255,252,255,0.55) 0%,rgba(252,245,255,0.48) 35%,rgba(255,248,255,0.52) 70%,rgba(250,245,255,0.58) 100%)`}}/>
        <div style={{position:"fixed",inset:0,zIndex:2,pointerEvents:"none",background:`radial-gradient(ellipse 55% 45% at 15% 30%,rgba(155,64,208,0.06) 0%,transparent 60%)`}}/></>
      )}

      <NeuralGrid isDark={isDark}/>
      <DataStreams/>

      <div style={{position:"relative",zIndex:2,minHeight:"100vh"}}>

        <div style={{textAlign:"center",padding:"52px 28px 48px",position:"relative",overflow:"hidden",zIndex:10,isolation:"isolate"}}>
          <div style={{position:"absolute",inset:0,background:isDark?`radial-gradient(ellipse 80% 100% at 50% 50%,${C.bg}cc 0%,transparent 75%)`:`radial-gradient(ellipse 80% 100% at 50% 50%,rgba(255,252,255,0.70) 0%,transparent 80%)`,pointerEvents:"none",zIndex:-1}}/>
          <div style={{display:"flex",alignItems:"center",gap:16,justifyContent:"center",marginBottom:18}}>
            <div style={{height:1,width:56,background:`linear-gradient(90deg,transparent,${C.rose}70)`}}/>
            <span style={{color:C.rose,fontSize:9,letterSpacing:"0.32em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace",fontWeight:700}}>AI · Machine Learning · Analytics</span>
            <div style={{height:1,width:56,background:`linear-gradient(90deg,${C.rose}70,transparent)`}}/>
          </div>
          <h1 style={{fontSize:54,fontWeight:800,fontFamily:"'Space Grotesk',sans-serif",letterSpacing:"-0.02em",lineHeight:1.1,marginBottom:16,margin:"0 0 16px",isolation:"isolate",filter:isDark?`drop-shadow(0 0 40px ${C.primary}70)`:`drop-shadow(0 0 32px ${C.primary}40)`}}>
            <span key={isDark?"dark-hero":"light-hero"} style={{display:"inline-block",background:isDark?`linear-gradient(90deg,${C.hiText},${C.primary},${C.rose},${C.hiText})`:`linear-gradient(90deg,${C.primary},${C.rose},${C.violet},${C.primary})`,backgroundSize:"300% auto",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text",animation:"titleGlow 6s linear infinite",willChange:"background-position",transform:"translateZ(0)"}}>AutoAnalytica AI Dashboard</span>
          </h1>
        </div>

        <header style={{padding:"0 40px",height:68,display:"flex",alignItems:"center",justifyContent:"space-between",background:isDark?`rgba(13,11,24,0.92)`:`rgba(255,252,255,0.92)`,backdropFilter:"blur(30px)",WebkitBackdropFilter:"blur(30px)",borderBottom:`1px solid ${C.border}`,boxShadow:isDark?"0 1px 0 rgba(255,255,255,0.04),0 4px 20px rgba(0,0,0,0.4)":"0 1px 0 rgba(255,255,255,0.9),0 4px 20px rgba(160,60,200,0.10)",position:"sticky",top:0,zIndex:100}}>
          <div style={{display:"flex",alignItems:"center",gap:14}}>
            <div style={{width:38,height:38,borderRadius:12,background:`linear-gradient(135deg,${C.primary},${C.rose})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:18,boxShadow:`0 4px 16px ${C.primary}35`}}>⚡</div>
            <div>
              <div style={{fontSize:17,fontWeight:800,fontFamily:"'Space Grotesk',sans-serif",letterSpacing:"0.02em"}}>
                <span key={isDark?"dark-header":"light-header"} style={{display:"inline-block",background:`linear-gradient(90deg,${C.hiText},${C.primary},${C.rose},${C.hiText})`,backgroundSize:"300% auto",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text",animation:"titleGlow 7s linear infinite",willChange:"background-position",transform:"translateZ(0)"}}>AutoAnalytica</span>
              </div>
              <div style={{color:C.loText,fontSize:10,letterSpacing:"0.16em",textTransform:"uppercase"}}>ML Platform v5.5</div>
            </div>
          </div>
          <div style={{flex:1,maxWidth:520,padding:"0 40px"}}><StepIndicator steps={steps}/></div>
          <button onClick={toggleTheme} title={isDark?"Switch to Light Mode":"Switch to Dark Mode"} style={{display:"flex",alignItems:"center",gap:8,padding:"7px 14px",borderRadius:24,cursor:"pointer",border:`1px solid ${theme.border}`,background:isDark?`linear-gradient(135deg,rgba(155,127,240,0.15),rgba(20,212,187,0.08))`:`linear-gradient(135deg,rgba(107,79,200,0.08),rgba(212,84,122,0.05))`,transition:"all 0.3s cubic-bezier(0.34,1.56,0.64,1)",marginRight:8,flexShrink:0}}>
            <div style={{width:40,height:22,borderRadius:11,position:"relative",background:isDark?`linear-gradient(135deg,${theme.primary},${theme.teal})`:`linear-gradient(135deg,${theme.primary}60,${theme.rose}80)`,transition:"background 0.35s",flexShrink:0}}>
              <div style={{position:"absolute",top:3,left:isDark?21:3,width:16,height:16,borderRadius:"50%",background:"#fff",transition:"left 0.3s cubic-bezier(0.34,1.56,0.64,1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:9}}>{isDark?"🌙":"☀️"}</div>
            </div>
            <span style={{color:theme.midText,fontSize:10,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace",whiteSpace:"nowrap"}}>{isDark?"DARK":"LIGHT"}</span>
          </button>
          <div style={{padding:"6px 14px",borderRadius:20,background:cleanedFile?`${C.teal}12`:`${C.primary}10`,border:`1px solid ${cleanedFile?C.teal+"30":C.primary+"25"}`,display:"flex",alignItems:"center",gap:7}}>
            <div style={{width:6,height:6,borderRadius:"50%",background:trainResponse?C.teal:cleanedFile?C.primary:C.rose,boxShadow:`0 0 8px ${trainResponse?C.teal:cleanedFile?C.primary:C.rose}`,animation:"pulse 2s ease infinite"}}/>
            <span style={{color:trainResponse?C.teal:cleanedFile?C.primary:C.rose,fontSize:11,fontWeight:700,letterSpacing:"0.1em"}}>{trainResponse?"MODEL READY":cleanedFile?"DATA LOADED":"READY"}</span>
          </div>
          {trainResponse&&(
            <button onClick={async()=>{const html=await generateHTMLReport({trainResponse,dashboardData,columns,target,file,apiBase:API});const blob=new Blob([html],{type:"text/html;charset=utf-8"});const url=URL.createObjectURL(blob);const win=window.open(url,"_blank");if(!win){const a=document.createElement("a");a.href=url;a.download=`AutoAnalytica_Report_${target}.html`;a.click();}setTimeout(()=>URL.revokeObjectURL(url),90000);}} style={{marginLeft:10,display:"flex",alignItems:"center",gap:7,padding:"7px 16px",borderRadius:10,cursor:"pointer",border:`1px solid ${C.primary}40`,background:`linear-gradient(135deg,${C.primary}12,${C.rose}08)`,fontFamily:"'Space Grotesk',monospace",color:C.primary,fontSize:11,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase"}}>
              <span style={{fontSize:14}}>📄</span> Report
            </button>
          )}
        </header>

        <main style={{maxWidth:1100,margin:"0 auto",padding:"44px 28px 60px",display:"flex",flexDirection:"column",gap:28}}>

          <div className="reveal" style={{textAlign:"center",padding:"28px 0 8px",position:"relative",zIndex:10,isolation:"isolate"}}>
            <div style={{display:"flex",gap:10,justifyContent:"center",flexWrap:"wrap",marginTop:20}}>
              {[{icon:"🧠",val:"34",label:"Algorithms",ac:C.primary},{icon:"⚡",val:"4",label:"Scale Tiers",ac:C.cyan},{icon:"🧹",val:"26",label:"Cleaning Steps",ac:C.rose},{icon:"🔍",val:"SHAP",label:"Explainability",ac:C.violet}].map((s,i)=>(
                <div key={i} style={{display:"flex",alignItems:"center",gap:7,padding:"7px 16px",borderRadius:30,background:C.isDark?`${s.ac}14`:`${s.ac}10`,border:`1px solid ${s.ac}35`,backdropFilter:"blur(10px)"}}>
                  <span style={{fontSize:14}}>{s.icon}</span><span style={{fontSize:15,fontWeight:800,color:s.ac,fontFamily:"'Space Grotesk',sans-serif"}}>{s.val}</span><span style={{fontSize:10,color:C.midText,fontWeight:600,letterSpacing:"0.08em",textTransform:"uppercase"}}>{s.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* UPLOAD */}
          <div className="reveal r1">
            <Panel title="Data Ingestion" subtitle="Upload a CSV or Excel file to begin analysis" icon="📡" accent={C.primary}>
              <div onDragOver={e=>{e.preventDefault();setDragOver(true);}} onDragLeave={()=>setDragOver(false)} onDrop={handleDrop}
                style={{border:`2px dashed ${dragOver?C.primary:file?C.teal+"55":C.border}`,borderRadius:16,padding:"32px 20px",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:10,textAlign:"center",background:dragOver?`${C.primary}07`:file?`${C.teal}06`:`${C.violetTint}40`,cursor:"pointer",transition:"all 0.25s",position:"relative",overflow:"hidden",marginBottom:16}}>
                {file&&<div style={{position:"absolute",inset:0,background:`radial-gradient(ellipse at 50% 50%,${C.teal}07,transparent 70%)`}}/>}
                <label style={{cursor:"pointer",width:"100%",position:"relative"}}>
                  <input type="file" accept=".csv,.xlsx,.xls" onChange={e=>setFile(e.target.files[0])}/>
                  <div style={{fontSize:42,marginBottom:4}}>{dragOver?"📥":file?"✅":"📂"}</div>
                  <p style={{color:file?C.teal:C.hiText,fontWeight:700,fontSize:15,fontFamily:"'Space Grotesk',sans-serif",marginBottom:4}}>{file?file.name:dragOver?"Drop it here!":"Select or Drop Dataset"}</p>
                  <p style={{color:C.midText,fontSize:12}}>{file?`${(file.size/1024).toFixed(1)} KB · Click to change${columns.length?" · "+columns.length+" columns detected":""}` :"CSV, XLSX or XLS · Click to browse or drag & drop"}</p>
                </label>
              </div>
              <Btn onClick={handleUpload} color={C.primary} size="lg"><span>⬆</span> Upload & Analyse Dataset</Btn>

              {columns.length>0&&(
                <div className="reveal" style={{marginTop:24,padding:22,background:`linear-gradient(135deg,${C.violetTint}70,${C.coralTint}50)`,border:`1px solid ${C.border}`,borderRadius:16}}>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
                    <p style={{color:C.midText,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase"}}>{columns.length} Columns Detected</p>
                    {target&&<span style={{padding:"3px 10px",borderRadius:20,background:`${C.teal}14`,color:C.teal,fontSize:11,fontWeight:700,border:`1px solid ${C.teal}28`}}>Target: {fmt(target)}</span>}
                  </div>
                  <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:20}}>
                    {columns.map((col,i)=>(
                      <button key={i} className="col-chip" onClick={()=>setTarget(col)} style={{padding:"6px 14px",borderRadius:22,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",fontSize:13,fontWeight:500,background:col===target?`linear-gradient(135deg,${C.violetTint},${C.coralTint})`:C.surface,border:`1px solid ${col===target?C.primary+"55":C.border}`,color:col===target?C.primary:C.midText,transition:"all 0.18s"}}>
                        {col===target&&<span style={{marginRight:4,color:C.rose}}>◉</span>}{fmt(col)}
                      </button>
                    ))}
                  </div>
                  <div style={{marginBottom:20}}>
                    <label style={{display:"block",color:C.midText,fontSize:10,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:8}}>
                      Target Column <span style={{color:C.rose}}>*</span>
                    </label>
                    <div style={{position:"relative"}}>
                      <select value={target} onChange={e=>setTarget(e.target.value)} style={{width:"100%",padding:"12px 40px 12px 14px",appearance:"none",background:target?`linear-gradient(135deg,${C.violetTint}80,${C.coralTint}60)`:C.surface,border:`1px solid ${target?C.primary+"55":C.border}`,borderRadius:12,color:target?C.hiText:C.midText,fontSize:14,fontFamily:"'DM Sans',sans-serif",cursor:"pointer",outline:"none"}}>
                        <option value="">— Choose the column to predict —</option>
                        {columns.map((col,i)=><option key={i} value={col}>{fmt(col)}</option>)}
                      </select>
                      <span style={{position:"absolute",right:14,top:"50%",transform:"translateY(-50%)",color:C.rose,pointerEvents:"none"}}>▾</span>
                    </div>
                  </div>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                    <Btn onClick={handleTrain} color={C.cyan} disabled={!target}>🧠 Train Model</Btn>
                    <Btn onClick={handleDashboard} color={C.violet}>📊 Dashboard</Btn>
                  </div>
                </div>
              )}
            </Panel>
          </div>

          {/* MODEL RESULTS */}
          {trainResponse&&(
            <div className="reveal r2">
              <Panel title="Model Intelligence" subtitle={`${fmt(modelType||"")} problem — best model selected automatically`} icon="🧠" accent={C.cyan} badge={modelType==="classification"?"CLASSIFICATION":"REGRESSION"}>
                <div style={{display:"flex",gap:4,marginBottom:24,padding:"4px",background:C.bg,borderRadius:12,border:`1px solid ${C.border}`}}>
                  {["metrics","details"].map(tab=>(
                    <button key={tab} onClick={()=>setActiveTab(tab)} style={{flex:1,padding:"9px",borderRadius:9,cursor:"pointer",background:activeTab===tab?`linear-gradient(135deg,${C.tealTint},${C.violetTint}80)`:"transparent",border:`1px solid ${activeTab===tab?C.cyan+"40":"transparent"}`,color:activeTab===tab?C.hiText:C.midText,fontSize:12,fontWeight:700,letterSpacing:"0.08em",textTransform:"uppercase",transition:"all 0.2s",fontFamily:"'Space Grotesk',monospace"}}>
                      {tab==="metrics"?"📊 Metrics":"ℹ️ Details"}
                    </button>
                  ))}
                </div>

                {trainResponse?.friendly_summary&&(
                  <div style={{padding:"12px 16px",marginBottom:18,borderRadius:10,background:`linear-gradient(135deg,${C.tealTint},${C.violetTint}60)`,border:`1px solid ${C.teal}25`,display:"flex",gap:10,alignItems:"flex-start"}}>
                    <span style={{fontSize:16,flexShrink:0}}>🤖</span>
                    <p style={{color:C.midText,fontSize:12,lineHeight:1.7,fontFamily:"'Space Grotesk',monospace"}}>{String(trainResponse.friendly_summary)}</p>
                  </div>
                )}

                {activeTab==="metrics"&&(<>
                  {/* ── FIX: KPI grid always renders; individual cards guard on null ── */}
                  <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(150px,1fr))",gap:12,marginBottom:24}}>
                    <KpiCard label="Problem Type" value={fmt(modelType||"—")} accent={C.primary} icon="⊕"/>
                    <KpiCard label="Best Model"   value={modelName??"—"}      accent={C.cyan}    icon="🏅"/>

                    {/* Classification KPIs */}
                    {modelType==="classification"&&accuracy!==null&&(
                      <KpiCard label="Accuracy" value={pct(accuracy)} accent={accColor} icon="✓"
                        sub={accuracyLabel+(cmSize>2?` · ${cmSize} classes`:"")} animated decimals={1} suffix="%"/>
                    )}
                    {modelType==="classification"&&accuracy===null&&(
                      <KpiCard label="Accuracy" value="—" accent={C.midText} icon="✓" sub="Not available"/>
                    )}
                    {modelType==="classification"&&roc_auc!==null&&(
                      <KpiCard label="ROC AUC" value={roc_auc.toFixed(3)} accent={C.amber} icon="◎" sub="Area under curve" animated decimals={3}/>
                    )}

                    {/* Regression KPIs */}
                    {modelType==="regression"&&r2!==null&&(
                      <KpiCard label="R² Score" value={r2.toFixed(4)} accent={scoreColor(r2,2)} icon="◈" sub="Variance explained" animated decimals={4}/>
                    )}
                    {modelType==="regression"&&r2===null&&(
                      <KpiCard label="R² Score" value="—" accent={C.midText} icon="◈" sub="Not available"/>
                    )}
                    {modelType==="regression"&&mae!==null&&(
                      <KpiCard label="MAE" value={mae.toFixed(3)} accent={C.amber} icon="△" sub="Mean abs. error" animated decimals={3}/>
                    )}
                    {modelType==="regression"&&rmse!==null&&(
                      <KpiCard label="RMSE" value={rmse.toFixed(3)} accent={C.rose} icon="▽" sub="Root mean sq. error" animated decimals={3}/>
                    )}

                    {/* Shared KPIs — FIX: use unified cvScoreMean */}
                    {trainAcc!==null&&(
                      <KpiCard label={modelType==="regression"?"Train R²":"Train Accuracy"}
                        value={modelType==="regression"?trainAcc.toFixed(4):pct(trainAcc)}
                        accent={C.teal} icon="📈" sub="On training set" animated
                        decimals={modelType==="regression"?4:1} suffix={modelType==="regression"?"":"%"}/>
                    )}
                    {cvScoreMean!==null&&(
                      <KpiCard label="CV Score"
                        value={modelType==="classification"?pct(cvScoreMean):cvScoreMean.toFixed(4)}
                        accent={C.violet} icon="⊙" sub="Cross-validation" animated
                        decimals={modelType==="classification"?1:4}
                        suffix={modelType==="classification"?"%":""}/>
                    )}
                  </div>

                  <GlassCard accent={C.primary} style={{padding:"16px 20px",marginBottom:20}} hover={false}>
                    <div style={{position:"absolute",top:0,left:0,right:0,height:2,background:`linear-gradient(90deg,transparent,${C.rose}50,${C.primary},transparent)`}}/>
                    <div style={{display:"flex",gap:12,alignItems:"flex-start"}}>
                      <div style={{width:34,height:34,borderRadius:10,flexShrink:0,background:`linear-gradient(135deg,${C.violetTint},${C.coralTint})`,border:`1px solid ${C.primary}22`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16}}>💡</div>
                      <div>
                        <p style={{color:C.primary,fontSize:11,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:6}}>What this means</p>
                        <p style={{color:C.midText,fontSize:13,lineHeight:1.75}}>
                          {modelType==="classification"
                            ?<>The model classifies data into categories.{cmSize>2&&<> ({cmSize} classes · random baseline {pct(1/cmSize)})</>} Accuracy of <strong style={{color:C.hiText}}>{accuracy!==null?pct(accuracy):"—"}</strong> means it predicts correctly that percentage of the time.</>
                            :<>R² of <strong style={{color:C.hiText}}>{r2!==null?r2.toFixed(3):"—"}</strong> means <strong style={{color:C.hiText}}>{r2!==null?(Math.max(0,r2)*100).toFixed(1)+"%":"—"}</strong> of the variance in <strong style={{color:C.hiText}}>{fmt(target)}</strong> is explained by the model.</>}
                        </p>
                      </div>
                    </div>
                  </GlassCard>

                  {cm&&(<div style={{marginBottom:20}}><p style={{color:C.midText,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>Confusion Matrix</p><ConfusionMatrix cm={cm} labels={cmLabels}/></div>)}

                  {perf.roc_curve_path&&(<div><p style={{color:C.midText,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>ROC Curve</p><div style={{border:`1px solid ${C.border}`,borderRadius:14,overflow:"hidden"}}><iframe src={`${API}/${perf.roc_curve_path}`} title="roc" width="100%" height="420" style={{display:"block",border:"none"}}/></div></div>)}
                </>)}

                {activeTab==="details"&&(
                  <div style={{display:"flex",flexDirection:"column",gap:14}}>
                    {[
                      {label:"Model Name",        value:modelName,                    color:C.cyan},
                      {label:"Problem Type",       value:fmt(modelType),               color:C.primary},
                      {label:"Target Column",      value:fmt(target),                  color:C.teal},
                      {label:"Features Used",      value:`${columns.length-1} features`,color:C.amber},
                      ...(cmSize>2?[{label:"Num Classes",value:String(cmSize),color:C.violet}]:[]),
                      {label:"Scale Tier",value:`Tier ${perf?.scale_tier??trainResponse?.scale_tier??"—"} — ${perf?.scale_tier_name??trainResponse?.scale_tier_name??""}`,color:C.violet},
                      {label:"Scaling Applied",value:perf?.scaling_applied?"Yes (StandardScaler)":"No (tree model)",color:perf?.scaling_applied?C.teal:C.midText},
                      // FIX: use cv_score_mean (actual backend key) for both problem types
                      ...(cvScoreMean!==null?[{
                        label:"CV Score (mean±std)",
                        value:`${modelType==="classification"?pct(cvScoreMean):safeNum(cvScoreMean,4)} ± ${cvScoreStd!==null?(modelType==="classification"?(cvScoreStd*100).toFixed(3)+"%":safeNum(cvScoreStd,4)):"—"}`,
                        color:C.cyan
                      }]:[]),
                      ...(trainAcc!==null?[{
                        label:modelType==="regression"?"Train R²":"Train Accuracy",
                        value:modelType==="regression"?trainAcc.toFixed(4):pct(trainAcc),
                        color:C.teal
                      }]:[]),
                      ...(perf?.problem_type_reason?[{label:"Detection Reason",value:perf.problem_type_reason,color:C.midText}]:[]),
                      ...(trainResponse?.prediction?.score!==undefined&&trainResponse?.prediction?.score!==null?[{label:"Prediction Score",value:safeNum(trainResponse.prediction.score,4),color:C.primary}]:[]),
                      ...(metaInsightSource?[{label:"Insight Source",value:String(metaInsightSource),color:C.violet}]:[]),
                      ...(metaInsightConfidence!==null?[{label:"Insight Confidence",value:pct(metaInsightConfidence),color:C.cyan}]:[]),
                    ].map((row,i)=>(
                      <div key={i} style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:12,padding:"12px 16px",background:`linear-gradient(135deg,${C.surface},${C.violetTint}40)`,borderRadius:10,border:`1px solid ${C.border}`}}>
                        <span style={{color:C.midText,fontSize:12,fontWeight:600}}>{row.label}</span>
                        <span style={{color:row.color,fontSize:12,fontWeight:700,fontFamily:"'Space Grotesk',monospace",textAlign:"right",wordBreak:"break-word",maxWidth:"60%"}}>{row.value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </Panel>
            </div>
          )}

          {/* AI DECISION PANEL */}
          {trainResponse&&agentFields&&(
            <div className="reveal r2">
              <Panel title="AI Decision Engine" subtitle="RL agent · Meta-model · Retrain system" icon="🤖" accent={C.violet} badge={agentFields.ls_agents_available?"AGENTS ACTIVE":"STANDALONE"}>
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:14,marginBottom:20}}>
                  {(()=>{const dm=getDecisionMeta(agentFields.decision);return(
                    <GlassCard accent={dm.color} style={{padding:"18px 20px"}}>
                      <div style={{position:"absolute",top:-20,right:-20,width:72,height:72,borderRadius:"50%",background:`radial-gradient(circle,${dm.color}18 0%,transparent 70%)`,filter:"blur(10px)"}}/>
                      <p style={{color:C.midText,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",fontWeight:700,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
                        <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:`${dm.color}14`,border:`1px solid ${dm.color}22`,display:"flex",alignItems:"center",justifyContent:"center"}}>⚡</span>AI Decision
                      </p>
                      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
                        <span style={{fontSize:28}}>{dm.emoji}</span>
                        <p style={{color:dm.color,fontWeight:800,fontSize:16,fontFamily:"'Space Grotesk',monospace",lineHeight:1.2}}>{dm.label}</p>
                      </div>
                      {agentFields.rl_reward!==null&&(<p style={{color:`${dm.color}80`,fontSize:11,marginTop:4}}>Reward: <span style={{color:dm.color,fontWeight:700}}>{agentFields.rl_reward>=0?"+":""}{agentFields.rl_reward?.toFixed(3)}</span></p>)}
                    </GlassCard>
                  );})()}
                  {agentFields.model_quality_score!==null&&(()=>{const qm=getQualityLabelMeta(agentFields.model_quality_label||"Good");const qpct=Math.round((agentFields.model_quality_score??0)*100);return(
                    <GlassCard accent={qm.color} style={{padding:"18px 20px"}}>
                      <p style={{color:C.midText,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",fontWeight:700,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
                        <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:`${qm.color}14`,border:`1px solid ${qm.color}22`,display:"flex",alignItems:"center",justifyContent:"center"}}>🎯</span>Quality Score
                      </p>
                      <p style={{color:qm.color,fontWeight:800,fontSize:32,fontFamily:"'Space Grotesk',monospace",lineHeight:1.1,marginBottom:10}}>{qpct}%</p>
                      <div style={{height:6,background:C.border,borderRadius:3,overflow:"hidden",marginBottom:8}}><div style={{width:`${qpct}%`,height:"100%",borderRadius:3,background:`linear-gradient(90deg,${C.primary}80,${qm.color})`,transition:"width 1s ease"}}/></div>
                      <span style={{padding:"3px 10px",borderRadius:20,fontSize:10,fontWeight:700,background:qm.bg,color:qm.color,border:`1px solid ${qm.border}`}}>{agentFields.model_quality_label||"Good"}</span>
                    </GlassCard>
                  );})()}
                  {(()=>{const um=getUrgencyMeta(agentFields.retrain_urgency||"None");const retrainProb=agentFields.retrain_probability!==null?agentFields.retrain_probability:null;const prob=retrainProb!==null?`${Math.round(retrainProb*100)}%`:null;return(
                    <GlassCard accent={shouldRetrain?C.amber:C.teal} style={{padding:"18px 20px"}}>
                      <p style={{color:C.midText,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",fontWeight:700,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
                        <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:shouldRetrain?`${C.amber}14`:`${C.teal}14`,border:`1px solid ${shouldRetrain?C.amber:C.teal}22`,display:"flex",alignItems:"center",justifyContent:"center"}}>🔄</span>Retrain
                      </p>
                      <p style={{color:shouldRetrain?C.amber:C.teal,fontWeight:800,fontSize:24,fontFamily:"'Space Grotesk',monospace",lineHeight:1.1,marginBottom:8}}>{shouldRetrain?"YES":"NO"}</p>
                      {prob&&<p style={{color:C.midText,fontSize:11,marginBottom:6}}>Probability: <span style={{color:shouldRetrain?C.amber:C.teal,fontWeight:700}}>{prob}</span></p>}
                      {retrainSeverity&&<p style={{color:C.midText,fontSize:11,marginBottom:6}}>Severity: <span style={{color:C.amber,fontWeight:700}}>{String(retrainSeverity)}</span></p>}
                      <span style={{padding:"3px 10px",borderRadius:20,fontSize:10,fontWeight:700,background:um.bg,color:um.color,border:`1px solid ${um.color}30`}}>Urgency: {um.label}</span>
                    </GlassCard>
                  );})()}
                  <GlassCard accent={C.cyan} style={{padding:"18px 20px"}}>
                    <p style={{color:C.midText,fontSize:10,letterSpacing:"0.14em",textTransform:"uppercase",fontWeight:700,marginBottom:10,display:"flex",alignItems:"center",gap:6}}>
                      <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:`${C.cyan}14`,border:`1px solid ${C.cyan}22`,display:"flex",alignItems:"center",justifyContent:"center"}}>⚙️</span>Pipeline
                    </p>
                    <p style={{color:C.cyan,fontWeight:800,fontSize:14,fontFamily:"'Space Grotesk',monospace",lineHeight:1.3,marginBottom:10}}>{(() => {
                        const tier = trainResponse?.scale_tier ?? null;
                        const name = trainResponse?.scale_tier_name ?? null;
                        if (!tier && !name) return "Pipeline (tier unknown)";
                        if (tier && name) return `Tier ${tier} — ${name}`;
                        return tier ? `Tier ${tier}` : name;
                      })()}</p>
                    {agentFields.workflow_selected&&(<div style={{marginBottom:6}}><span style={{padding:"3px 10px",borderRadius:20,fontSize:10,fontWeight:700,background:`${C.primary}14`,color:C.primary,border:`1px solid ${C.primary}28`}}>{agentFields.workflow_selected.replace(/_/g," ").toUpperCase()}</span></div>)}
                    {agentFields.pipeline_version&&(<p style={{color:C.loText,fontSize:10,marginTop:4,fontFamily:"'Space Grotesk',monospace"}}>{String(agentFields.pipeline_version)}</p>)}
                  </GlassCard>
                </div>

                {shouldRetrain&&retrainReason&&(
                  <div style={{padding:"12px 16px",marginBottom:14,borderRadius:10,background:`${C.amber}10`,border:`1px solid ${C.amber}30`,display:"flex",alignItems:"center",gap:10}}>
                    <span style={{fontSize:18}}>🔄</span>
                    <div><p style={{color:C.amber,fontSize:11,fontWeight:700,letterSpacing:"0.08em",textTransform:"uppercase"}}>Retrain Recommended</p><p style={{color:C.midText,fontSize:12,marginTop:2}}>{String(retrainReason)}</p></div>
                  </div>
                )}

                {agentFields.drift_severity>0&&(
                  <GlassCard accent={agentFields.drift_severity>0.25?C.rose:C.amber} hover={false} style={{padding:"16px 20px",marginBottom:14}}>
                    <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:10}}>
                      <div style={{display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:16}}>📈</span><p style={{color:C.midText,fontSize:11,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase"}}>Drift Analysis</p></div>
                      <span style={{padding:"3px 10px",borderRadius:20,fontSize:10,fontWeight:700,background:agentFields.drift_severity>0.25?`${C.rose}14`:`${C.amber}14`,color:agentFields.drift_severity>0.25?C.rose:C.amber,border:`1px solid ${agentFields.drift_severity>0.25?C.rose:C.amber}30`}}>{agentFields.drift_severity>0.25?"SIGNIFICANT":agentFields.drift_severity>0.10?"MODERATE":"LOW"}</span>
                    </div>
                    <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
                      {[{label:"Drift Severity",val:(agentFields.drift_severity*100).toFixed(1)+"%",color:agentFields.drift_severity>0.25?C.rose:C.amber},{label:"Rolling Drift",val:(agentFields.rolling_drift>=0?"+":"")+(agentFields.rolling_drift*100).toFixed(2)+"%",color:agentFields.rolling_drift<0?C.rose:C.teal},{label:"Decay Velocity",val:(agentFields.decay_velocity>=0?"+":"")+agentFields.decay_velocity?.toFixed(3),color:agentFields.decay_velocity<-0.01?C.rose:C.midText}].map((row,i)=>(
                        <div key={i} style={{textAlign:"center"}}><p style={{color:C.midText,fontSize:9,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:4}}>{row.label}</p><p style={{color:row.color,fontWeight:800,fontSize:16,fontFamily:"'Space Grotesk',monospace"}}>{row.val}</p></div>
                      ))}
                    </div>
                  </GlassCard>
                )}

                {(agentFields.class_imbalance!==null||agentFields.overall_missing_pct!==null)&&(
                  <div style={{padding:"14px 18px",borderRadius:12,background:`linear-gradient(135deg,${C.surface},${C.violetTint}40)`,border:`1px solid ${C.border}`,marginBottom:14}}>
                    <p style={{color:C.midText,fontSize:10,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>Dataset Diagnostics</p>
                    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(140px,1fr))",gap:10}}>
                      {agentFields.class_imbalance!==null&&(<div><p style={{color:C.midText,fontSize:10,fontWeight:600,marginBottom:3}}>Class Imbalance</p><p style={{color:C.hiText,fontWeight:700,fontSize:14,fontFamily:"'Space Grotesk',monospace"}}>{(agentFields.class_imbalance*100).toFixed(1)}% majority</p></div>)}
                      {agentFields.overall_missing_pct!==null&&(<div><p style={{color:C.midText,fontSize:10,fontWeight:600,marginBottom:3}}>Missing Data</p><p style={{color:agentFields.overall_missing_pct>5?C.amber:C.teal,fontWeight:700,fontSize:14,fontFamily:"'Space Grotesk',monospace"}}>{agentFields.overall_missing_pct?.toFixed(1)}%</p></div>)}
                      {agentFields.runs_since_retrain!==null&&(<div><p style={{color:C.midText,fontSize:10,fontWeight:600,marginBottom:3}}>Runs Since Retrain</p><p style={{color:C.hiText,fontWeight:700,fontSize:14,fontFamily:"'Space Grotesk',monospace"}}>{agentFields.runs_since_retrain}</p></div>)}
                    </div>
                  </div>
                )}

                {agentFields.baseline_triggered&&agentFields.baseline_gap!==null&&(
                  <div style={{padding:"12px 16px",borderRadius:10,background:`${C.rose}10`,border:`1px solid ${C.rose}30`,display:"flex",alignItems:"center",gap:10}}>
                    <span style={{fontSize:18}}>⚠️</span>
                    <div><p style={{color:C.rose,fontSize:11,fontWeight:700,letterSpacing:"0.08em",textTransform:"uppercase"}}>Baseline Alert</p><p style={{color:C.midText,fontSize:12,marginTop:2}}>Model is only <strong style={{color:C.rose}}>{(agentFields.baseline_gap*100).toFixed(1)}pp</strong> above the naive baseline.</p></div>
                  </div>
                )}

                {pipelineStageLogs.length > 0 && (
                  <div style={{marginTop:16,padding:"14px 18px",borderRadius:12,background:`linear-gradient(135deg,${C.surface},${C.violetTint}30)`,border:`1px solid ${C.border}`}}>
                    <p style={{color:C.midText,fontSize:10,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12,display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:13}}>🔬</span>Pipeline Stage Logs</p>
                    <div style={{display:"flex",flexDirection:"column",gap:6}}>
                      {pipelineStageLogs.map((stage, i) => {
                        const stageName   = String(stage?.stage   ?? `Stage ${i+1}`);
                        const stageStatus = String(stage?.status  ?? "—");
                        const stageTime   = stage?.elapsed_s != null ? `${Number(stage.elapsed_s).toFixed(2)}s` : null;
                        const isDone      = stageStatus.toLowerCase() === "done" || stageStatus.toLowerCase() === "complete";
                        const isError     = stageStatus.toLowerCase() === "error" || stageStatus.toLowerCase() === "failed";
                        const stageColor  = isError ? C.rose : isDone ? C.teal : C.amber;
                        return (
                          <div key={i} style={{display:"flex",alignItems:"center",gap:10,padding:"8px 12px",borderRadius:8,background:C.isDark?`${stageColor}08`:`${stageColor}05`,border:`1px solid ${stageColor}20`}}>
                            <span style={{fontSize:11,color:stageColor}}>{isError?"✕":isDone?"✓":"○"}</span>
                            <span style={{color:C.hiText,fontSize:12,fontWeight:600,flex:1}}>{stageName}</span>
                            <span style={{color:stageColor,fontSize:11,fontWeight:700,fontFamily:"'Space Grotesk',monospace"}}>{stageStatus}</span>
                            {stageTime&&<span style={{color:C.loText,fontSize:10,fontFamily:"'Space Grotesk',monospace"}}>{stageTime}</span>}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </Panel>
            </div>
          )}

          {/* LEADERBOARD */}
          {leaderboard.length>0&&(
            <div className="reveal r3">
              <Panel title="Model Leaderboard" subtitle={`${leaderboard.length} models benchmarked · ${modelType==="classification"?"Score = Accuracy":"Score = R²"} · higher is better`} icon="🏆" accent={C.amber}>
                <div style={{display:"flex",flexDirection:"column",gap:8}}>
                  {leaderboard.map((m,i)=>{const best=leaderboard[0].score??0,rel=best>0?(m.score??0)/best:0;const rc=i===0?C.amber:i===1?C.primary:C.midText;
                    return(<GlassCard key={i} accent={i===0?C.amber:C.primary} style={{padding:"14px 18px"}}>
                      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:6}}>
                        <div style={{width:30,height:30,borderRadius:"50%",flexShrink:0,background:i===0?`linear-gradient(135deg,${C.amber},${C.rose})`:C.bg,border:`1px solid ${i===0?C.amber+"50":C.border}`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,fontWeight:800,color:i===0?"#fff":C.midText,fontFamily:"'Space Grotesk',monospace"}}>{i===0?"👑":i+1}</div>
                        <span style={{color:rc,fontSize:14,fontWeight:600,flex:1}}>{m.name}</span>
                        {i===0&&<span style={{padding:"3px 10px",borderRadius:20,fontSize:10,fontWeight:700,background:`${C.amber}14`,color:C.amber,border:`1px solid ${C.amber}28`}}>BEST</span>}
                        <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:2}}>
                          <span style={{color:rc,fontFamily:"'Space Grotesk',monospace",fontSize:14,fontWeight:800}}>{m.score!==null?Number(m.score).toFixed(4):"—"}</span>
                          {m.cv!==null&&m.cv!==undefined&&<span style={{color:C.loText,fontFamily:"'Space Grotesk',monospace",fontSize:10}}>CV {Number(m.cv).toFixed(4)}</span>}
                        </div>
                      </div>
                      <ProgressBar value={rel} color={i===0?C.amber:i===1?C.primary:C.midText} height={3}/>
                    </GlassCard>);
                  })}
                </div>
              </Panel>
            </div>
          )}

          {/* FIX 5 — SHAP: render from normalised array (works for both object and array shapes) */}
          {shapArray.length>0&&(
            <div className="reveal r3">
              <Panel title="SHAP Feature Importance" subtitle="Mean absolute SHAP values — higher = more influential" icon="🔍" accent={C.violet}>
                <div style={{display:"flex",flexDirection:"column",gap:8}}>
                  {(()=>{
                    const sorted=[...shapArray]
                      .sort((a,b)=>Math.abs(b.shap_value??b.importance??0)-Math.abs(a.shap_value??a.importance??0))
                      .slice(0,12);
                    const maxVal=Math.abs(sorted[0]?.shap_value??sorted[0]?.importance??1)||1;
                    return sorted.map((item,i)=>{
                      const val=item.shap_value??item.importance??0,abs=Math.abs(val),rel=abs/maxVal,isPos=val>=0;
                      return(<div key={i} style={{display:"flex",alignItems:"center",gap:12}}>
                        <span style={{width:140,fontSize:12,color:C.midText,fontWeight:600,flexShrink:0,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{fmt(item.feature??item.name??`Feature ${i+1}`)}</span>
                        <div style={{flex:1,height:8,borderRadius:4,background:C.border,overflow:"hidden"}}><div style={{width:`${(rel*100).toFixed(1)}%`,height:"100%",borderRadius:4,background:i===0?`linear-gradient(90deg,${C.violet},${C.primary})`:isPos?C.cyan:C.rose,transition:"width 0.9s ease"}}/></div>
                        <span style={{width:64,fontSize:11,fontWeight:700,color:isPos?C.cyan:C.rose,fontFamily:"'Space Grotesk',monospace",textAlign:"right",flexShrink:0}}>{val>=0?"+":""}{abs.toFixed(4)}</span>
                      </div>);
                    });
                  })()}
                </div>
              </Panel>
            </div>
          )}

          {/* AI INSIGHTS & LEARNING STATE */}
          {trainResponse&&agentFields&&(
            <div className="reveal r4">
              <Panel title="AI Insights & Learning State" subtitle="Meta-model analysis · Risk flags · Agent learning progress" icon="💡" accent={C.rose}>
                {metaInsightText&&(
                  <div style={{padding:"20px 22px",marginBottom:20,borderRadius:14,background:`linear-gradient(135deg,${C.violetTint}80,${C.coralTint}60)`,border:`1px solid ${C.primary}25`,borderLeft:`4px solid ${C.primary}`,boxShadow:`inset 0 1px 0 rgba(255,255,255,0.7),0 2px 8px ${C.primary}08`}}>
                    
                    <p style={{color:C.primary,fontSize:10,fontWeight:700,letterSpacing:"0.16em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace",marginBottom:10,display:"flex",alignItems:"center",gap:8}}>
                      <span style={{width:22,height:22,borderRadius:6,fontSize:11,background:`${C.primary}14`,border:`1px solid ${C.primary}28`,display:"flex",alignItems:"center",justifyContent:"center"}}>🤖</span>
                      Model-Based Insight
                      
                      {metaInsightSource&&(
                        <span style={{color:C.loText,fontSize:9,fontWeight:600,marginLeft:4}}>
                          · {String(metaInsightSource)}
                        </span>
                      )}
                    </p>

                    <p style={{color:C.hiText,fontSize:13,lineHeight:1.85,fontStyle:"italic"}}>
                      "{metaInsightText}"
                    </p>

                    {metaInsightConfidence !== null &&
                    typeof metaInsightConfidence === "number" &&
                    !isNaN(metaInsightConfidence) && (
                      <p style={{color:C.midText,fontSize:11,marginTop:10}}>
                        Confidence:{" "}
                        <span style={{color:C.primary,fontWeight:700}}>
                          {(Math.min(Math.max(metaInsightConfidence, 0), 1) * 100).toFixed(1)}%
                        </span>
                      </p>
                    )}

                  </div>
                )}

                {combinedRisks.length > 0 && (
                  <div style={{marginBottom:20}}>
                    <p style={{color:C.midText,fontSize:10,fontWeight:700,letterSpacing:"0.14em",textTransform:"uppercase",marginBottom:12,display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:14}}>⚠️</span>Risk Flags Detected ({combinedRisks.length})</p>
                    <div style={{display:"flex",flexWrap:"wrap",gap:10}}>
                      {combinedRisks.map((risk, i) => {
                        const riskStr = typeof risk === "string" ? risk : String(risk ?? "");
                        let rm;
                        try { rm = getRiskMeta(riskStr); } catch { rm = { emoji:"⚠️", label:riskStr, color:C.amber }; }
                        const riskLabel = rm?.label ?? riskStr;
                        const riskEmoji = rm?.emoji ?? "⚠️";
                        const riskColor = rm?.color ?? C.amber;
                        return (
                          <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"8px 14px",borderRadius:24,background:`${riskColor}12`,border:`1px solid ${riskColor}30`}}>
                            <span style={{fontSize:14}}>{riskEmoji}</span>
                            <span style={{color:riskColor,fontSize:12,fontWeight:700,fontFamily:"'Space Grotesk',monospace"}}>{riskLabel}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {agentFields.improvements?.length>0&&(
                  <div style={{marginBottom:20}}>
                    <p style={{color:C.midText,fontSize:10,fontWeight:700,letterSpacing:"0.14em",textTransform:"uppercase",marginBottom:12,display:"flex",alignItems:"center",gap:8}}><span style={{fontSize:14}}>🔧</span>Suggested Improvements</p>
                    <div style={{display:"flex",flexDirection:"column",gap:8}}>
                      {agentFields.improvements.map((imp,i)=>{
                        let im;
                        try { im = getImproveMeta(imp); } catch { im = { emoji:"🔧", label: typeof imp === "string" ? imp : String(imp ?? "") }; }
                        const rankColors=[C.amber,C.primary,C.cyan];const rc=rankColors[i]||C.midText;return(
                        <div key={i} style={{display:"flex",alignItems:"center",gap:12,padding:"12px 16px",borderRadius:12,background:i===0?`linear-gradient(135deg,${C.amber}0e,${C.coralTint}60)`:`linear-gradient(135deg,${C.surface},${C.violetTint}30)`,border:`1px solid ${rc}${i===0?"35":"18"}`}}>
                          <div style={{width:28,height:28,borderRadius:8,flexShrink:0,background:`${rc}14`,border:`1px solid ${rc}28`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:800,color:rc,fontFamily:"'Space Grotesk',monospace"}}>{i+1}</div>
                          <span style={{fontSize:16}}>{im.emoji}</span>
                          <span style={{color:i===0?C.hiText:C.midText,fontSize:13,fontWeight:i===0?700:500,flex:1}}>{im.label}</span>
                          {i===0&&<span style={{padding:"2px 8px",borderRadius:10,fontSize:9,fontWeight:700,background:`${C.amber}18`,color:C.amber,border:`1px solid ${C.amber}30`,letterSpacing:"0.1em",flexShrink:0}}>TOP</span>}
                        </div>
                      );})}
                    </div>
                  </div>
                )}

                {agentFields.ls_agents_available&&(
                  <div style={{padding:"18px 20px",marginBottom:20,borderRadius:14,background:C.isDark?`linear-gradient(135deg,rgba(20,212,187,0.06),rgba(155,127,240,0.06))`:`linear-gradient(135deg,${C.tealTint},${C.violetTint}50)`,border:`1px solid ${C.teal}25`}}>
                    <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
                      <p style={{color:C.teal,fontSize:10,fontWeight:700,letterSpacing:"0.16em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace",display:"flex",alignItems:"center",gap:8}}>
                        <span style={{width:22,height:22,borderRadius:6,background:`${C.teal}14`,border:`1px solid ${C.teal}28`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:11}}>🧠</span>Agent Learning State
                      </p>
                      <div style={{display:"flex",alignItems:"center",gap:6,padding:"4px 12px",borderRadius:20,background:`${C.teal}12`,border:`1px solid ${C.teal}28`}}>
                        <div style={{width:6,height:6,borderRadius:"50%",background:C.teal,animation:"pulse 2s ease infinite"}}/>
                        <span style={{color:C.teal,fontSize:10,fontWeight:700,letterSpacing:"0.1em"}}>ACTIVE</span>
                      </div>
                    </div>
                    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(110px,1fr))",gap:12}}>
                      {[
                        {label:"Exploration ε",value:rlEpsilon!==null?`${(rlEpsilon*100).toFixed(1)}%`:"—",sub:"Explore vs exploit",color:C.primary,icon:"🎲",barVal:rlEpsilon??0,barColor:C.primary},
                        {label:"Training Steps",value:agentFields.ls_step_count??"—",sub:"Q-network updates",color:C.cyan,icon:"⚡",barVal:Math.min((agentFields.ls_step_count??0)/500,1),barColor:C.cyan},
                        {label:"Memory Buffer",value:agentFields.ls_buffer_size??"—",sub:"Stored experiences",color:C.violet,icon:"💾",barVal:Math.min((agentFields.ls_buffer_size??0)/500,1),barColor:C.violet},
                        {label:"Avg Reward",value:agentFields.ls_avg_reward!==null&&agentFields.ls_avg_reward!==undefined?(agentFields.ls_avg_reward>=0?"+":"")+agentFields.ls_avg_reward.toFixed(3):"—",sub:"Last 100 runs",color:agentFields.ls_avg_reward!==null&&agentFields.ls_avg_reward>=0?C.teal:C.rose,icon:"🏆",barVal:Math.min(Math.max((agentFields.ls_avg_reward??0)+1,0)/2,1),barColor:agentFields.ls_avg_reward!==null&&agentFields.ls_avg_reward>=0?C.teal:C.rose},
                      ].map((stat,i)=>(
                        <div key={i} style={{padding:"14px 14px",borderRadius:10,background:C.isDark?`${stat.color}0a`:`${stat.color}07`,border:`1px solid ${stat.color}20`}}>
                          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:8}}><span style={{fontSize:12}}>{stat.icon}</span><p style={{color:C.midText,fontSize:9,fontWeight:700,letterSpacing:"0.14em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace"}}>{stat.label}</p></div>
                          <p style={{color:stat.color,fontWeight:800,fontSize:20,fontFamily:"'Space Grotesk',monospace",lineHeight:1,marginBottom:8}}>{stat.value}</p>
                          <div style={{height:3,background:C.border,borderRadius:2,overflow:"hidden",marginBottom:5}}><div style={{width:`${(stat.barVal*100).toFixed(1)}%`,height:"100%",borderRadius:2,background:`linear-gradient(90deg,${stat.barColor}60,${stat.barColor})`,transition:"width 1s ease"}}/></div>
                          <p style={{color:C.loText,fontSize:9}}>{stat.sub}</p>
                        </div>
                      ))}
                    </div>
                    <div style={{marginTop:14,display:"flex",alignItems:"center",gap:10}}>
                      <div style={{display:"flex",alignItems:"center",gap:6,padding:"5px 12px",borderRadius:20,background:agentFields.ls_network_fitted?`${C.teal}12`:`${C.amber}12`,border:`1px solid ${agentFields.ls_network_fitted?C.teal:C.amber}28`}}>
                        <span style={{fontSize:11}}>{agentFields.ls_network_fitted?"✓":"○"}</span>
                        <span style={{color:agentFields.ls_network_fitted?C.teal:C.amber,fontSize:10,fontWeight:700}}>Q-Network {agentFields.ls_network_fitted?"Trained":"Bootstrapping"}</span>
                      </div>
                      {agentFields.agent_run_id&&(<p style={{color:C.loText,fontSize:10,fontFamily:"'Space Grotesk',monospace"}}>Run: {String(agentFields.agent_run_id).slice(-12)}</p>)}
                    </div>
                  </div>
                )}

                {agentRunId&&!feedbackSent&&(
                  <div style={{padding:"16px 18px",borderRadius:12,marginBottom:4,background:`linear-gradient(135deg,${C.surface},${C.violetTint}40)`,border:`1px solid ${C.border}`}}>
                    <p style={{color:C.midText,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:4}}>Was the prediction correct?</p>
                    <p style={{color:C.loText,fontSize:12,marginBottom:14}}>Your feedback improves the RL agent for future runs.</p>
                    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                      <Btn onClick={()=>handleFeedback(true)} color={C.teal} size="sm"><span>✓</span> Yes, correct</Btn>
                      <Btn onClick={()=>handleFeedback(false)} color={C.rose} size="sm"><span>✕</span> No, incorrect</Btn>
                    </div>
                  </div>
                )}

                {feedbackSent&&(
                  <div style={{padding:"12px 16px",borderRadius:10,background:`${C.teal}10`,border:`1px solid ${C.teal}28`,display:"flex",alignItems:"center",gap:10}}>
                    <span style={{fontSize:16}}>✓</span>
                    <p style={{color:C.teal,fontSize:12,fontWeight:600}}>Feedback sent — the AI agent has been updated.</p>
                  </div>
                )}
              </Panel>
            </div>
          )}

          {/* DOWNLOAD REPORT */}
          {trainResponse&&(
            <div className="reveal r4">
              <GlassCard accent={C.primary} hover={false} style={{padding:"28px 30px"}}>
                <div style={{position:"absolute",top:0,left:0,right:0,height:3,background:`linear-gradient(90deg,transparent,${C.primary}60,${C.rose}80,${C.cyan}60,transparent)`}}/>
                <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",gap:24,flexWrap:"wrap"}}>
                  <div style={{flex:1,minWidth:260}}>
                    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
                      <div style={{width:36,height:36,borderRadius:10,flexShrink:0,background:`linear-gradient(135deg,${C.violetTint},${C.coralTint})`,border:`1px solid ${C.primary}22`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:17}}>📄</div>
                      <div><h3 style={{color:C.hiText,fontSize:13,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace"}}>Download Full Analysis Report</h3><p style={{color:C.midText,fontSize:11,marginTop:2}}>Opens in new tab · Print or save as PDF</p></div>
                    </div>
                    <p style={{color:C.midText,fontSize:12,lineHeight:1.75,marginBottom:16}}>A complete, print-ready report covering your entire analysis session.</p>
                    <DownloadReportBtn trainResponse={trainResponse} dashboardData={dashboardData} columns={columns} target={target} file={file} disabled={!trainResponse}/>
                  </div>
                  <div style={{minWidth:200,maxWidth:240,background:`linear-gradient(145deg,${C.surface},${C.violetTint}50)`,border:`1px solid ${C.border}`,borderRadius:14,padding:"18px 20px"}}>
                    <p style={{color:C.midText,fontSize:9,fontWeight:700,letterSpacing:"0.18em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace",marginBottom:14}}>Report Preview</p>
                    {[
                      {label:"Best Model",val:modelName||"—",color:C.primary},
                      {label:"Problem",val:fmt(trainResponse?.problem_type||"—"),color:C.cyan},
                      {label:"Target",val:fmt(target),color:C.teal},
                      {label:"Features",val:`${columns.length-1}`,color:C.amber},
                      {label:"Charts",val:dashboardData?.charts?.length>0?`${dashboardData.charts.length} included`:"Generate first",color:dashboardData?.charts?.length>0?C.violet:C.loText},
                      ...(accuracy!==null&&accuracy!==undefined?[{label:"Accuracy",val:`${(accuracy*100).toFixed(1)}%`,color:C.teal}]:r2!==null&&r2!==undefined?[{label:"R² Score",val:Number(r2).toFixed(4),color:C.teal}]:[]),
                      {label:"Quality",val:trainResponse?.model_quality?.rating||"—",color:C.rose},
                      ...(agentFields?.decision?[{label:"AI Decision",val:String(agentFields.decision).replace(/_/g," ").toUpperCase(),color:C.violet}]:[]),
                    ].map((row,i)=>(
                      <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"7px 0",borderBottom:`1px solid ${C.border}40`}}>
                        <span style={{color:C.midText,fontSize:10,fontWeight:600}}>{row.label}</span>
                        <span style={{color:row.color,fontSize:11,fontWeight:700,fontFamily:"'Space Grotesk',monospace"}}>{row.val}</span>
                      </div>
                    ))}
                    <div style={{marginTop:14,padding:"8px 12px",borderRadius:8,background:C.tealTint,border:`1px solid ${C.teal}20`,textAlign:"center"}}>
                      <p style={{color:C.teal,fontSize:10,fontWeight:700,letterSpacing:"0.1em"}}>✓ REPORT READY</p>
                    </div>
                  </div>
                </div>
              </GlassCard>
            </div>
          )}

          {/* DASHBOARD */}
          {dashboardData?.charts?.length>0&&(
            <div className="reveal r4">
              <Panel title="Analytics Dashboard" subtitle="Auto-generated visual insights from your dataset" icon="📊" accent={C.violet}>
                <div style={{display:"flex",flexDirection:"column",gap:16}}>
                  {dashboardData.charts.map((chart,i)=>(
                    <div key={i} style={{border:`1px solid ${C.border}`,borderRadius:14,overflow:"hidden"}}>
                      <iframe src={`${API}/static/reports/${chart}`} title={`chart-${i}`} width="100%" height="540" style={{display:"block",border:"none"}}/>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          )}

          {/* PREDICTION ENGINE */}
          {trainResponse&&(
            <div className="reveal r5">
              <Panel title="Prediction Engine" subtitle={`Enter feature values to predict "${fmt(target)}"`} icon="🤖" accent={C.teal}>
                <div style={{padding:"14px 18px",marginBottom:24,background:`linear-gradient(135deg,${C.tealTint},${C.violetTint}60)`,border:`1px solid ${C.teal}20`,borderRadius:12,display:"flex",gap:10,alignItems:"flex-start"}}>
                  <span style={{fontSize:18,flexShrink:0}}>💡</span>
                  <p style={{color:C.midText,fontSize:13,lineHeight:1.7}}>Fill in the known feature values. The model will predict <strong style={{color:C.hiText}}>{fmt(target)}</strong>.</p>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:14,marginBottom:24}}>
                  {columns.filter(c=>c!==target).map((col,i)=>(
                    <Input key={i} label={fmt(col)} placeholder="value" value={predictionInput[col]||""} onChange={e=>setPredictionInput({...predictionInput,[col]:e.target.value})}/>
                  ))}
                </div>
                <Btn onClick={handlePredict} color={C.teal} size="lg">⚡ Run Prediction</Btn>

                {predictionResult&&(
                  <div className="reveal" style={{marginTop:20}}>
                    <GlassCard accent={predictionResult.error?C.rose:C.teal} hover={false} style={{padding:"24px 26px"}}>
                      <div style={{position:"absolute",top:0,left:0,right:0,height:2,background:predictionResult.error?`linear-gradient(90deg,transparent,${C.rose}80,transparent)`:`linear-gradient(90deg,transparent,${C.primary}50,${C.teal},transparent)`}}/>
                      {predictionResult.error?(
                        <div style={{display:"flex",gap:14,alignItems:"center"}}>
                          <div style={{width:40,height:40,borderRadius:12,background:`${C.rose}12`,border:`1px solid ${C.rose}25`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:20,color:C.rose}}>✕</div>
                          <div><p style={{color:C.rose,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase"}}>Prediction Failed</p><p style={{color:C.midText,fontSize:13,marginTop:4}}>{String(predictionResult.error)}</p></div>
                        </div>
                      ):(
                        <>
                          <div style={{display:"flex",gap:14,alignItems:"center",marginBottom:20}}>
                            <div style={{width:40,height:40,borderRadius:12,background:C.tealTint,border:`1px solid ${C.teal}25`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:20,color:C.teal}}>✓</div>
                            <div><p style={{color:C.teal,fontSize:11,fontWeight:700,letterSpacing:"0.12em",textTransform:"uppercase"}}>Prediction Complete</p><p style={{color:C.midText,fontSize:12,marginTop:2}}>Predicting: <strong style={{color:C.hiText}}>{fmt(target)}</strong></p></div>
                          </div>
                          <div style={{display:"flex",alignItems:"flex-end",justifyContent:"space-between",flexWrap:"wrap",gap:16}}>
                            <div>
                              <p style={{color:C.midText,fontSize:11,marginBottom:8,letterSpacing:"0.1em",textTransform:"uppercase"}}>Predicted Value</p>
                              <p style={{color:C.teal,fontWeight:800,lineHeight:1,fontFamily:"'Space Grotesk',monospace",fontSize:predDisplay.length>10?28:predDisplay.length>6?38:48}}>{predDisplay}</p>
                              {predConfidence!=null&&<p style={{color:`${C.teal}80`,fontSize:12,marginTop:8}}>Confidence: <strong style={{color:C.teal}}>{String(predConfidence)}</strong></p>}
                            </div>
                            <div style={{display:"flex",flexDirection:"column",gap:8,alignItems:"flex-end"}}>
                              <span style={{padding:"6px 14px",borderRadius:20,background:C.violetTint,border:`1px solid ${C.primary}25`,color:C.primary,fontSize:11,fontWeight:700,textTransform:"uppercase"}}>{fmt(predType||"")}</span>
                              <span style={{padding:"6px 14px",borderRadius:20,background:C.tealTint,border:`1px solid ${C.teal}20`,color:C.teal,fontSize:11,fontWeight:600}}>Model: {modelName}</span>
                            </div>
                          </div>
                        </>
                      )}
                    </GlassCard>
                  </div>
                )}
              </Panel>
            </div>
          )}

          <div style={{textAlign:"center",padding:"8px 0",display:"flex",alignItems:"center",justifyContent:"center",gap:16}}>
            <div style={{flex:1,height:1,background:`linear-gradient(90deg,transparent,${C.border})`}}/>
            <p style={{color:C.loText,fontSize:10,letterSpacing:"0.22em",textTransform:"uppercase",fontFamily:"'Space Grotesk',monospace"}}>AutoAnalytica AI Platform · v5.5 · Powered by Machine Intelligence</p>
            <div style={{flex:1,height:1,background:`linear-gradient(90deg,${C.border},transparent)`}}/>
          </div>

        </main>
      </div>

      <InsightDrawer/>
      <Toast text={toast.text} ok={toast.ok} onClose={()=>setToast({text:""})}/>
      <LoadingOverlay show={loading} message={loadMsg}/>
    </>
    </ThemeCtx.Provider>
  );
}