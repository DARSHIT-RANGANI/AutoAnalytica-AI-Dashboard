# =============================================================================
# backend/app/modules/dashboard/service.py
#
# TWO responsibilities in one file:
#
#   PART 1 ── Chart Engine (unchanged original code)
#   ─────────────────────────────────────────────────
#   generate_dashboard_graphs(filename)
#   Accepts any CSV/XLSX → profiles every column → renders 5 pages of
#   matplotlib + seaborn charts → assembles a polished tabbed HTML file.
#
#   PART 2 ── MongoDB Dashboard Stats (NEW — added at the bottom)
#   ──────────────────────────────────────────────────────────────
#   get_dashboard_stats()   → full stats dict for Dashboard.js
#   get_summary_counts()    → lightweight {datasets, models, reports} counts
#
# The router (router.py) calls both parts from this single service file.
# =============================================================================

# ═════════════════════════════════════════════════════════════════════════════
#  PART 1 — Chart Engine (original — completely unchanged)
# ═════════════════════════════════════════════════════════════════════════════

"""
AutoAnalytica — Automatic Dashboard Service
============================================
Drop in ANY CSV/XLSX → engine automatically:

  1. PROFILES   every column (numeric / categorical / datetime / pii / text)
  2. DECIDES    which groupings, correlations, time axes matter
  3. RENDERS    5 pages of matplotlib + seaborn charts (3×3 grids)
  4. ASSEMBLES  one polished, tabbed HTML file — zero user input needed

Usage:
    result = generate_dashboard_graphs("my_file.csv")
    # → {"charts": ["dashboard_abc123.html"], "rows": 400, ...}
"""

from __future__ import annotations

import base64, io, logging, math, re, uuid, warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

UPLOAD_DIR  = Path("uploads")
REPORTS_DIR = Path("app/reports")

_BG    = "#0A0C1A"; _PANEL = "#0F1224"; _CARD  = "#161B30"; _BORDER= "#252A45"
_HI    = "#F0F4FF"; _MID   = "#8892B0"; _LO    = "#3D4466"
PAL    = ["#6C63FF","#00D4FF","#00E5A0","#FFB547","#FF6B8A",
          "#9B59FF","#36CFC9","#F7931E","#A8FF78","#FF7EB3",
          "#5DADE2","#F1948A","#82E0AA","#F8C471","#C39BD3"]
_DIV   = LinearSegmentedColormap.from_list("div", ["#FF6B8A",_PANEL,"#6C63FF"])
_SEQ   = LinearSegmentedColormap.from_list("seq", [_CARD,"#6C63FF","#00D4FF","#00E5A0"])

_RE_ID    = re.compile(r"\b(id|_id|uuid|key|index)\b", re.I)
_RE_NAME  = re.compile(r"\b(name|fname|lname|fullname|patient|customer)\b", re.I)
_RE_EMAIL = re.compile(r"\b(email|e_mail|mail)\b", re.I)
_RE_PHONE = re.compile(r"\b(phone|mobile|tel|fax)\b", re.I)
_RE_DATE  = re.compile(r"\b(date|time|year|month|day|dob|birth|visit|created|updated|timestamp)\b", re.I)
_RE_ADDR  = re.compile(r"\b(address|street|city|zip|postal|state|country)\b", re.I)
_RE_SKIP  = re.compile(r"\b(password|token|secret|hash|url|link|path|image|photo|file)\b", re.I)


class DataProfile:
    def __init__(self, df: pd.DataFrame):
        self.raw = df
        self.df  = df.copy()
        self.n   = len(df)
        self.roles:    Dict[str,str] = {}
        self.col_info: Dict[str,dict]= {}
        self.numeric_cols:     List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols:    List[str] = []
        self.boolean_cols:     List[str] = []
        self.hue_col:  Optional[str] = None
        self.hue_col2: Optional[str] = None
        self.date_col: Optional[str] = None
        self.top_pair: Optional[Tuple[str,str]] = None
        self.missing_pct = round(float(df.isnull().mean().mean()*100), 1)
        self.duplicates  = int(df.duplicated().sum())
        self.health      = max(0, round(100 - self.missing_pct*1.5
                                        - min(self.duplicates/max(self.n,1)*100,20)))
        self._run()

    def _run(self):
        for col in self.df.columns:
            role, parsed = self._classify_col(col)
            self.roles[col] = role
            if role == "datetime" and parsed is not None:
                self.df[col] = parsed
        for col, role in self.roles.items():
            if   role == "numeric":     self.numeric_cols.append(col)
            elif role == "categorical": self.categorical_cols.append(col)
            elif role == "datetime":    self.datetime_cols.append(col)
            elif role == "boolean":     self.boolean_cols.append(col)
        self.hue_col  = self._pick_hue(None)
        self.hue_col2 = self._pick_hue(self.hue_col)
        if self.datetime_cols:
            self.date_col = self.datetime_cols[0]
        if len(self.numeric_cols) >= 2:
            try:
                corr = self.df[self.numeric_cols].corr().abs()
                np.fill_diagonal(corr.values, 0)
                self.top_pair = corr.stack().idxmax()
            except Exception:
                self.top_pair = (self.numeric_cols[0], self.numeric_cols[1])

    def _classify_col(self, col: str) -> Tuple[str, Optional[pd.Series]]:
        raw    = self.df[col]
        name_l = col.lower().strip()
        if _RE_EMAIL.search(name_l): return "email", None
        if _RE_PHONE.search(name_l): return "phone", None
        if _RE_NAME.search(name_l):  return "name",  None
        if _RE_ADDR.search(name_l):  return "address",None
        if _RE_SKIP.search(name_l):  return "skip",  None
        if pd.api.types.is_bool_dtype(raw): return "boolean", None
        if pd.api.types.is_numeric_dtype(raw):
            nuniq = raw.nunique()
            if _RE_ID.search(name_l) and nuniq == self.n: return "id", None
            if nuniq == 2: return "boolean", None
            if nuniq <= 15 and raw.min() >= 0:
                if set(raw.dropna().unique()).issubset(set(range(20))): return "categorical", None
            return "numeric", None
        if pd.api.types.is_datetime64_any_dtype(raw): return "datetime", raw
        series_str = raw.dropna().astype(str).str.strip()
        nuniq      = series_str.nunique()
        n_filled   = max(len(series_str), 1)
        if _RE_DATE.search(name_l):
            parsed = pd.to_datetime(raw, errors="coerce")
            if parsed.notna().sum() >= n_filled * 0.5: return "datetime", parsed
        if nuniq > 5:
            sample = series_str.head(30)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.7:
                full_parsed = pd.to_datetime(raw, errors="coerce")
                if full_parsed.notna().sum() >= n_filled * 0.5: return "datetime", full_parsed
        num_series = pd.to_numeric(series_str, errors="coerce")
        if num_series.notna().mean() > 0.8:
            if _RE_ID.search(name_l) and nuniq == self.n: return "id", None
            if _RE_PHONE.search(name_l): return "phone", None
            self.df[col] = num_series
            if nuniq == 2: return "boolean", None
            return "numeric", None
        if _RE_ID.search(name_l) and nuniq > n_filled * 0.8: return "id", None
        if nuniq > min(60, n_filled * 0.6) and nuniq > 20: return "text", None
        return "categorical", None

    def _pick_hue(self, exclude: Optional[str]) -> Optional[str]:
        _PREFER = re.compile(
            r"\b(condition|disease|status|type|category|class|group|gender|sex|"
            r"diagnosis|outcome|label|result|department|region|segment|tier)\b", re.I)
        candidates = [c for c in self.categorical_cols + self.boolean_cols
                      if c != exclude and 2 <= self.df[c].nunique() <= 12]
        if not candidates:
            candidates = [c for c in self.categorical_cols
                          if c != exclude and self.df[c].nunique() <= 20]
        if not candidates: return None
        def score(c):
            return self.df[c].nunique() + (-5 if _PREFER.search(c) else 0)
        return min(candidates, key=score)

    def nums(self, n=8):   return self.numeric_cols[:n]
    def cats(self, n=4):   return self.categorical_cols[:n]
    def groupby_mean(self, by, cols):
        try:   return self.df.groupby(by)[cols].mean()
        except: return pd.DataFrame()
    def groupby_count(self, by):
        try:   return self.df.groupby(by).size().unstack(by[-1]).fillna(0)
        except: return pd.DataFrame()
    def crosstab(self, r, c):
        try:   return pd.crosstab(self.df[r], self.df[c])
        except: return pd.DataFrame()
    def pivot(self, val, idx, col):
        try:   return self.df.pivot_table(values=val, index=idx, columns=col, aggfunc="mean")
        except: return pd.DataFrame()


# ── Drawing toolkit ───────────────────────────────────────────────────────────

def _theme():
    plt.rcParams.update({
        "figure.facecolor":"#0F1224","axes.facecolor":"#161B30",
        "axes.edgecolor":"#252A45","axes.labelcolor":"#8892B0",
        "axes.titlecolor":"#F0F4FF","axes.titlesize":10,"axes.titlepad":8,
        "axes.labelsize":8,"axes.grid":True,"grid.color":"#252A45",
        "grid.linewidth":0.5,"xtick.color":"#8892B0","ytick.color":"#8892B0",
        "xtick.labelsize":7,"ytick.labelsize":7,"text.color":"#F0F4FF",
        "legend.facecolor":"#161B30","legend.edgecolor":"#252A45",
        "legend.fontsize":7,"legend.title_fontsize":7,
        "figure.dpi":130,"savefig.facecolor":"#0F1224",
        "font.family":["DejaVu Sans"],
    })

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _fig(rows=3, cols=3, title=""):
    _theme()
    fig = plt.figure(figsize=(18, 5.2*rows))
    if title: fig.suptitle(title, fontsize=14, color=_HI, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.40)
    return fig, gs

def _hide(ax): ax.set_visible(False)
def _rotx(ax, deg=30):
    ax.tick_params(axis="x", labelrotation=deg)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right"); lbl.set_fontsize(7)

def _top_cats(s, n=8):
    top = s.value_counts().nlargest(n).index
    return s.where(s.isin(top), "Other")

def _kde(ax, s, color, label=None):
    s = s.dropna()
    if len(s) < 5 or s.std() < 1e-9: return
    x = np.linspace(float(s.quantile(.01)), float(s.quantile(.99)), 300)
    try:
        k = scipy_stats.gaussian_kde(s)
        kw = dict(color=color, lw=1.8, alpha=0.9)
        if label: kw["label"] = label
        ax.plot(x, k(x), **kw)
    except Exception: pass

def _heatmap(ax, df, cmap, title, fmt="{:.2f}", vmin=None, vmax=None):
    if df.empty: _hide(ax); return
    im = ax.imshow(df.values.astype(float), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(c)[:10] for c in df.columns], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(r)[:14] for r in df.index], fontsize=6)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.iloc[i,j]
            if pd.notna(v): ax.text(j,i,fmt.format(float(v)),ha="center",va="center",fontsize=5.8,color=_HI)
    cb = plt.colorbar(im, ax=ax, shrink=0.78); cb.ax.yaxis.set_tick_params(labelsize=6)
    ax.set_title(title)

def _grouped_bar(ax, pivot, title, ylabel="", xlabel=""):
    if pivot.empty: _hide(ax); return
    n_grps=len(pivot); n_bars=len(pivot.columns)
    x=np.arange(n_grps); w=min(0.72/max(n_bars,1),0.25)
    for i,col in enumerate(pivot.columns):
        offset=(i-n_bars/2+0.5)*w
        ax.bar(x+offset,pivot[col].fillna(0),w*0.90,color=PAL[i%len(PAL)],label=str(col)[:14],edgecolor="#0F1224",lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels([str(v)[:12] for v in pivot.index],rotation=30,ha="right",fontsize=7)
    ax.legend(fontsize=6,framealpha=0.6); ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)

def _stacked_bar(ax, pivot, title, pct=False):
    if pivot.empty: _hide(ax); return
    if pct: pivot=pivot.div(pivot.sum(axis=1),axis=0)*100
    bottom=np.zeros(len(pivot))
    for i,col in enumerate(pivot.columns):
        ax.bar(range(len(pivot)),pivot[col].fillna(0),bottom=bottom,color=PAL[i%len(PAL)],label=str(col)[:12],edgecolor="#0F1224",lw=0.3)
        bottom+=pivot[col].fillna(0).values
    ax.set_xticks(range(len(pivot))); ax.set_xticklabels([str(v)[:12] for v in pivot.index],rotation=30,ha="right",fontsize=7)
    ax.legend(fontsize=6,framealpha=0.6); ax.set_title(title); ax.set_ylabel("%"if pct else"Count")

def _violin(ax, df, num_col, cat_col, title):
    plot_df=df[[num_col,cat_col]].copy(); plot_df[cat_col]=_top_cats(plot_df[cat_col])
    cats=sorted(plot_df[cat_col].dropna().unique())
    data=[plot_df.loc[plot_df[cat_col]==c,num_col].dropna() for c in cats]
    data=[d for d in data if len(d)>1]
    if not data: _hide(ax); return
    vp=ax.violinplot(data,showmedians=True,showextrema=True)
    for i,pc in enumerate(vp["bodies"]): pc.set_facecolor(PAL[i%len(PAL)]); pc.set_alpha(0.72)
    for k in ["cmedians","cmins","cmaxes","cbars"]:
        if k in vp: vp[k].set_color(_HI); vp[k].set_linewidth(1)
    ax.set_xticks(range(1,len(data)+1))
    ax.set_xticklabels([str(c)[:10] for c in cats[:len(data)]],rotation=30,ha="right")
    ax.set_title(title); ax.set_ylabel(num_col)

def _boxplot(ax, df, num_col, cat_col, title):
    plot_df=df[[num_col,cat_col]].copy(); plot_df[cat_col]=_top_cats(plot_df[cat_col])
    cats=sorted(plot_df[cat_col].dropna().unique())
    data=[plot_df.loc[plot_df[cat_col]==c,num_col].dropna() for c in cats]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color=_HI,lw=1.5),
                  whiskerprops=dict(color=_MID),capprops=dict(color=_MID),
                  flierprops=dict(marker="o",ms=2.5,mfc=PAL[4],mec="none"))
    for patch,c in zip(bp["boxes"],PAL): patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(range(1,len(cats)+1)); ax.set_xticklabels([str(c)[:10] for c in cats],rotation=30,ha="right")
    ax.set_title(title); ax.set_ylabel(num_col)


# ── Pages ─────────────────────────────────────────────────────────────────────

def _page_overview(p):
    df,hue,hue2=p.df,p.hue_col,p.hue_col2; nums,cats=p.nums(6),p.cats(4); nc,cc=len(nums),len(cats)
    fig,gs=_fig(3,3,"Dataset Overview")
    ax=fig.add_subplot(gs[0,0])
    if nc>=1:
        col=nums[0]
        if hue:
            for i,g in enumerate(df[hue].dropna().unique()[:6]):
                ax.hist(df.loc[df[hue]==g,col].dropna(),bins=22,alpha=0.6,color=PAL[i],label=str(g)[:12],edgecolor="#0F1224",lw=0.3)
            ax.legend(title=hue[:14],fontsize=6)
        else: ax.hist(df[col].dropna(),bins=25,color=PAL[0],alpha=0.8,edgecolor="#0F1224",lw=0.3)
        ax.set_title(f"Distribution — {col}"); ax.set_xlabel(col); ax.set_ylabel("Count")
    else: _hide(ax)
    ax=fig.add_subplot(gs[0,1])
    if nc>=1 and hue: _violin(ax,df,nums[0],hue,f"{nums[0]} by {hue}")
    elif nc>=2:
        ax.scatter(df[nums[0]],df[nums[1]],color=PAL[0],alpha=0.4,s=12)
        ax.set_title(f"{nums[0]} vs {nums[1]}"); ax.set_xlabel(nums[0]); ax.set_ylabel(nums[1])
    else: _hide(ax)
    ax=fig.add_subplot(gs[0,2]); pie_col=hue or(cats[0] if cc else None)
    if pie_col:
        vc=df[pie_col].value_counts().head(8)
        _,_,autos=ax.pie(vc.values,labels=vc.index.astype(str),colors=PAL[:len(vc)],autopct="%1.1f%%",startangle=130,wedgeprops=dict(edgecolor="#0F1224",lw=1.5),textprops=dict(color=_HI,fontsize=7))
        for a in autos: a.set_fontsize(7)
        ax.set_title(f"Distribution — {pie_col}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,0])
    if nc>=1 and hue:
        col=nums[0]; grp=p.groupby_mean(hue,[col]).sort_values(col,ascending=False).head(10)
        bars=ax.bar(range(len(grp)),grp[col].values,color=PAL[:len(grp)],edgecolor="#0F1224",lw=0.4)
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([str(x)[:12] for x in grp.index],rotation=30,ha="right")
        for bar,v in zip(bars,grp[col].values): ax.text(bar.get_x()+bar.get_width()/2,v*1.008,f"{v:.1f}",ha="center",va="bottom",fontsize=6,color=_HI)
        ax.set_title(f"Avg {col} by {hue}"); ax.set_ylabel(f"Mean {col}")
    elif nc>=1:
        col=nums[0]; desc=df[col].describe()
        ax.bar(["min","25%","50%","75%","max"],[desc[k] for k in ["min","25%","50%","75%","max"]],color=PAL[:5])
        ax.set_title(f"{col} — Percentiles")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,1])
    if nc>=1 and hue: _boxplot(ax,df,nums[0],hue,f"{nums[0]} Distribution by {hue}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,2])
    if nc>=1 and hue and hue2:
        col=nums[0]; grp=df.groupby([hue2,hue])[col].mean().unstack(hue).head(10)
        _grouped_bar(ax,grp,f"Avg {col} by {hue2} & {hue}",ylabel=f"Avg {col}",xlabel=hue2)
    elif nc>=2 and hue:
        col=nums[1]; grp=p.groupby_mean(hue,[col]).sort_values(col,ascending=False)
        ax.bar(range(len(grp)),grp[col].values,color=PAL[:len(grp)],edgecolor="#0F1224")
        ax.set_xticks(range(len(grp))); ax.set_xticklabels([str(x)[:12] for x in grp.index],rotation=30,ha="right")
        ax.set_title(f"Avg {col} by {hue}"); ax.set_ylabel(f"Avg {col}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,0])
    if hue and hue2:
        ct=df.groupby([hue,hue2]).size().unstack(hue2).fillna(0).head(10)
        _grouped_bar(ax,ct,f"Count by {hue} & {hue2}",ylabel="Count",xlabel=hue)
    elif hue:
        vc=df[hue].value_counts().head(10)
        ax.bar(range(len(vc)),vc.values,color=PAL[:len(vc)])
        ax.set_xticks(range(len(vc))); ax.set_xticklabels([str(x)[:12] for x in vc.index],rotation=30,ha="right")
        ax.set_title(f"Count — {hue}"); ax.set_ylabel("Count")
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,1])
    if hue and hue2:
        ct=df.groupby([hue,hue2]).size().unstack(hue2).fillna(0).head(10)
        _stacked_bar(ax,ct,f"Stacked Count: {hue} / {hue2}")
    elif nc>=2:
        c1,c2=nums[0],nums[1]; valid=df[[c1,c2]].dropna()
        ax.scatter(valid[c1],valid[c2],alpha=0.4,s=10,color=PAL[0])
        if len(valid)>5:
            m,b,r,*_=scipy_stats.linregress(valid[c1],valid[c2])
            xr=np.linspace(float(valid[c1].min()),float(valid[c1].max()),100)
            ax.plot(xr,m*xr+b,color=PAL[2],lw=1.6,ls="--"); ax.set_title(f"{c1} vs {c2}  (r={r:.2f})")
        ax.set_xlabel(c1); ax.set_ylabel(c2)
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,2])
    if nc>=1 and hue and hue2:
        pv=p.pivot(nums[0],hue,hue2).round(1); _heatmap(ax,pv,_SEQ,f"{nums[0]} Heatmap","{:.1f}"); ax.set_xlabel(hue2); ax.set_ylabel(hue)
    elif nc>=2:
        corr=df[nums[:8]].corr().round(2); _heatmap(ax,corr,_DIV,"Correlation Heatmap","{:.2f}",-1,1)
    else: _hide(ax)
    return _b64(fig)

def _page_numeric(p):
    if len(p.numeric_cols)<1: return None
    df,hue,nums=p.df,p.hue_col,p.nums(8); nc=len(nums); fig,gs=_fig(3,3,"Numeric Feature Deep Dive")
    ax=fig.add_subplot(gs[0,0]); corr=df[nums].corr().round(2); _heatmap(ax,corr,_DIV,"Correlation Matrix","{:.2f}",-1,1)
    ax=fig.add_subplot(gs[0,1])
    for i,col in enumerate(nums[:6]):
        s=df[col].dropna()
        if s.std()<1e-9: continue
        _kde(ax,(s-s.mean())/s.std(),PAL[i],label=col[:12])
    ax.legend(fontsize=6); ax.set_xlabel("Z-Score"); ax.set_ylabel("Density"); ax.set_title("KDE Overlay — All Numerics (Z-Scored)")
    ax=fig.add_subplot(gs[0,2])
    if p.top_pair:
        c1,c2=p.top_pair
        if hue:
            for i,g in enumerate(df[hue].dropna().unique()[:6]):
                mask=df[hue]==g; ax.scatter(df.loc[mask,c1],df.loc[mask,c2],color=PAL[i],alpha=0.5,s=12,label=str(g)[:12])
            ax.legend(title=hue[:12],fontsize=6)
        else: ax.scatter(df[c1],df[c2],color=PAL[0],alpha=0.4,s=12)
        valid=df[[c1,c2]].dropna()
        if len(valid)>5:
            m,b,r,*_=scipy_stats.linregress(valid[c1],valid[c2])
            xr=np.linspace(float(valid[c1].min()),float(valid[c1].max()),100)
            ax.plot(xr,m*xr+b,color=PAL[2],lw=1.8,ls="--"); ax.set_title(f"{c1} vs {c2}  (r = {r:.2f})")
        ax.set_xlabel(c1); ax.set_ylabel(c2)
    for slot in range(3):
        ax=fig.add_subplot(gs[1,slot])
        if slot<nc:
            col=nums[slot]; s=df[col].dropna()
            ax.hist(s,bins=25,color=PAL[slot],alpha=0.75,edgecolor="#0F1224",lw=0.3)
            ax.axvline(float(s.mean()),color=_HI,lw=1.5,ls="-",label="mean")
            ax.axvline(float(s.median()),color=PAL[4],lw=1.5,ls="--",label="median")
            ax.legend(fontsize=6); ax.set_title(f"Histogram — {col}"); ax.set_xlabel(col); ax.set_ylabel("Count")
        else: _hide(ax)
    ax=fig.add_subplot(gs[2,0])
    normed=[(df[c].dropna()-df[c].mean())/(df[c].std()+1e-9) for c in nums[:8]]
    bp=ax.boxplot(normed,patch_artist=True,medianprops=dict(color=_HI,lw=1.5),whiskerprops=dict(color=_MID),capprops=dict(color=_MID),flierprops=dict(marker="o",ms=2.5,mfc=PAL[4],mec="none"))
    for patch,c in zip(bp["boxes"],PAL): patch.set_facecolor(c); patch.set_alpha(0.72)
    ax.set_xticks(range(1,len(nums[:8])+1)); ax.set_xticklabels([c[:8] for c in nums[:8]],rotation=35,ha="right")
    ax.axhline(3,color=PAL[4],lw=0.9,ls="--",alpha=0.7,label="±3σ"); ax.axhline(-3,color=PAL[4],lw=0.9,ls="--",alpha=0.7)
    ax.legend(fontsize=6); ax.set_title("Outlier Detection (Z-Scored Boxplots)")
    ax=fig.add_subplot(gs[2,1])
    desc=df[nums].describe().T[["mean","std","min","max"]].head(8); x=np.arange(len(desc)); w=0.2
    for i,stat in enumerate(["mean","std","min","max"]):
        ax.bar(x+i*w,desc[stat],w*0.9,color=PAL[i],label=stat,edgecolor="#0F1224",lw=0.3)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels([c[:8] for c in desc.index],rotation=35,ha="right")
    ax.legend(fontsize=6); ax.set_title("Descriptive Statistics")
    ax=fig.add_subplot(gs[2,2])
    for i,col in enumerate(nums[:5]):
        s=df[col].dropna().sort_values(); ax.plot(s,np.arange(1,len(s)+1)/len(s),color=PAL[i],lw=1.6,alpha=0.85,label=col[:12])
    ax.axhline(0.5,color=_MID,lw=0.8,ls="--",alpha=0.6); ax.legend(fontsize=6); ax.set_title("ECDF — Cumulative Distribution")
    ax.set_xlabel("Value"); ax.set_ylabel("Cumulative Probability")
    return _b64(fig)

def _page_categorical(p):
    if len(p.categorical_cols)<1: return None
    df,hue,hue2=p.df,p.hue_col,p.hue_col2; nums,cats=p.nums(4),p.cats(4); nc,cc=len(nums),len(cats)
    fig,gs=_fig(3,3,"Categorical Feature Analysis")
    for slot in range(3):
        ax=fig.add_subplot(gs[0,slot])
        if slot<cc:
            col=cats[slot]; vc=df[col].value_counts().head(10)
            bars=ax.barh(range(len(vc)),vc.values,color=PAL[slot%len(PAL)],edgecolor="#0F1224",lw=0.3)
            ax.set_yticks(range(len(vc))); ax.set_yticklabels([str(x)[:18] for x in vc.index],fontsize=7); ax.invert_yaxis()
            for bar,v in zip(bars,vc.values): ax.text(v*1.01,bar.get_y()+bar.get_height()/2,str(v),va="center",fontsize=6,color=_HI)
            ax.set_title(f"Value Counts — {col}"); ax.set_xlabel("Count")
        else: _hide(ax)
    ax=fig.add_subplot(gs[1,0])
    if nc>=1 and hue: _violin(ax,df,nums[0],hue,f"Violin: {nums[0]} by {hue}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,1])
    if hue and hue2:
        ct=p.crosstab(hue,hue2); _heatmap(ax,ct,_SEQ,f"Cross-Tab: {hue} × {hue2}","{:.0f}"); ax.set_xlabel(hue2); ax.set_ylabel(hue)
    elif hue and nc>=1:
        col=nums[0]; grp=p.groupby_mean(hue,[col]).sort_values(col)
        _heatmap(ax,grp.rename(columns={col:f"Avg {col}"}),_SEQ,f"Avg {col} by {hue}","{:.1f}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,2])
    if nc>=2 and hue:
        grp=p.groupby_mean(hue,nums[:6]); normed=(grp-grp.min())/(grp.max()-grp.min()+1e-9)
        _heatmap(ax,normed.round(2),_SEQ,f"Normalised Feature Means by {hue}","{:.2f}")
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,0])
    if nc>=1 and hue:
        col=nums[0]; pdf=df[[col,hue]].copy(); pdf[hue]=_top_cats(pdf[hue]); cats_u=sorted(pdf[hue].dropna().unique())
        for i,cat in enumerate(cats_u):
            vals=pdf.loc[pdf[hue]==cat,col].dropna(); jitter=np.random.uniform(-0.22,0.22,len(vals))
            ax.scatter(np.full(len(vals),i)+jitter,vals,color=PAL[i%len(PAL)],alpha=0.45,s=8,label=str(cat)[:12])
        ax.set_xticks(range(len(cats_u))); ax.set_xticklabels([str(c)[:10] for c in cats_u],rotation=30,ha="right")
        ax.set_title(f"Strip Plot: {col} by {hue}"); ax.set_ylabel(col)
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,1])
    if hue and hue2:
        ct=df.groupby([hue,hue2]).size().unstack(hue2).fillna(0).head(10); _stacked_bar(ax,ct,f"% Stacked: {hue} / {hue2}",pct=True)
    else: _hide(ax)
    ax=fig.add_subplot(gs[2,2])
    if nc>=1 and hue:
        col=nums[0]; grp=df.groupby(hue)[col].agg(["mean","sem","count"]).head(12); grp=grp[grp["count"]>1]; x=np.arange(len(grp))
        ax.errorbar(x,grp["mean"],yerr=grp["sem"]*1.96,fmt="o",color=PAL[0],ecolor=_MID,elinewidth=1.2,capsize=4,ms=6,zorder=3)
        ax.plot(x,grp["mean"],color=PAL[0],lw=1,alpha=0.4,ls="--")
        ax.fill_between(x,grp["mean"]-grp["sem"]*1.96,grp["mean"]+grp["sem"]*1.96,alpha=0.12,color=PAL[0])
        ax.set_xticks(x); ax.set_xticklabels([str(v)[:12] for v in grp.index],rotation=30,ha="right")
        ax.set_title(f"Mean ± 95% CI: {col} by {hue}"); ax.set_ylabel(f"Avg {col}")
    else: _hide(ax)
    return _b64(fig)

def _page_timeseries(p):
    if not p.date_col or len(p.numeric_cols)<1: return None
    df=p.df.copy(); dc=p.date_col; nums=p.nums(4); hue=p.hue_col
    try:
        df[dc]=pd.to_datetime(df[dc],errors="coerce"); df=df.dropna(subset=[dc]).sort_values(dc)
        if len(df)<5: return None
    except Exception: return None
    fig,gs=_fig(2,2,"Time Series Analysis")
    ax=fig.add_subplot(gs[0,0])
    if hue:
        for i,g in enumerate(df[hue].dropna().unique()[:6]):
            sub=df[df[hue]==g].set_index(dc)[nums[0]].resample("ME").mean()
            ax.plot(sub.index,sub.values,color=PAL[i],lw=1.5,alpha=0.85,label=str(g)[:12])
        ax.legend(title=hue[:12],fontsize=6)
    else:
        m=df.set_index(dc)[nums[0]].resample("ME").mean()
        ax.plot(m.index,m.values,color=PAL[0],lw=1.8); ax.fill_between(m.index,m.values,alpha=0.18,color=PAL[0])
    ax.set_title(f"{nums[0]} over Time (Monthly Avg)"); ax.set_ylabel(nums[0]); _rotx(ax,30)
    ax=fig.add_subplot(gs[0,1])
    ct=df.set_index(dc).resample("ME").size()
    ax.bar(ct.index,ct.values,width=20,color=PAL[1],alpha=0.8,edgecolor="#0F1224")
    ax.set_title("Record Count per Month"); ax.set_ylabel("Count"); _rotx(ax,30)
    ax=fig.add_subplot(gs[1,0])
    for i,col in enumerate(nums[:4]):
        s=df.set_index(dc)[col].resample("ME").mean(); n=(s-s.mean())/(s.std()+1e-9)
        ax.plot(n.index,n.values,color=PAL[i],lw=1.4,alpha=0.85,label=col[:12])
    ax.axhline(0,color=_MID,lw=0.7,ls="--",alpha=0.5); ax.legend(fontsize=6); _rotx(ax,30)
    ax.set_title("All Numerics — Normalised Trend")
    ax=fig.add_subplot(gs[1,1])
    if hue:
        try:
            pv=(df.groupby([pd.Grouper(key=dc,freq="QE"),hue]).size().unstack(hue).fillna(0))
            _stacked_bar(ax,pv,f"{hue} Count per Quarter")
            step=max(1,len(pv)//6); ax.set_xticks(range(0,len(pv),step))
            ax.set_xticklabels([str(pv.index[i])[:7] for i in range(0,len(pv),step)],rotation=30,ha="right",fontsize=7)
        except Exception: _hide(ax)
    else: _hide(ax)
    return _b64(fig)

def _page_quality(p):
    df,nums=p.df,p.nums(10); fig,gs=_fig(2,3,"Data Quality Report")
    ax=fig.add_subplot(gs[0,0])
    miss=(df.isnull().mean()*100).round(1).sort_values(ascending=False)
    colors=[PAL[0] if v<5 else PAL[4] if v>20 else PAL[3] for v in miss.values]
    ax.barh(range(len(miss)),miss.values,color=colors,edgecolor="#0F1224",lw=0.3)
    ax.set_yticks(range(len(miss))); ax.set_yticklabels([c[:18] for c in miss.index],fontsize=6); ax.invert_yaxis()
    ax.axvline(5,color=PAL[3],lw=1,ls="--",alpha=0.7,label="5%"); ax.axvline(20,color=PAL[4],lw=1,ls="--",alpha=0.7,label="20%")
    ax.legend(fontsize=6); ax.set_title("Missing Values (%)"); ax.set_xlabel("% Missing")
    ax=fig.add_subplot(gs[0,1])
    step=max(1,len(df)//150); sample=df.isnull().astype(int).iloc[::step,:20]
    im=ax.imshow(sample.T.values,cmap="RdYlGn_r",aspect="auto",interpolation="nearest")
    ax.set_yticks(range(len(sample.columns))); ax.set_yticklabels([c[:14] for c in sample.columns],fontsize=6)
    ax.set_xticks([]); ax.set_xlabel("Rows (sample)"); ax.set_title("Null Pattern Map")
    cb=plt.colorbar(im,ax=ax,shrink=0.7,ticks=[0,1]); cb.ax.set_yticklabels(["Present","Missing"],fontsize=6)
    ax=fig.add_subplot(gs[0,2])
    dup=p.duplicates; total=p.n
    ax.pie([max(0,total-dup),dup],labels=[f"Unique\n{total-dup:,}",f"Duplicate\n{dup:,}"],colors=[PAL[2],PAL[4]],autopct="%1.1f%%",startangle=90,wedgeprops=dict(edgecolor="#0F1224",lw=1.5),textprops=dict(color=_HI,fontsize=8))
    ax.set_title("Duplicate Rows")
    ax=fig.add_subplot(gs[1,0])
    if nums:
        skew=df[nums].skew().sort_values()
        colors=[PAL[4] if abs(v)>1 else PAL[3] if abs(v)>0.5 else PAL[2] for v in skew.values]
        ax.barh(range(len(skew)),skew.values,color=colors,edgecolor="#0F1224")
        ax.set_yticks(range(len(skew))); ax.set_yticklabels([c[:14] for c in skew.index],fontsize=6)
        ax.axvline(0,color=_MID,lw=0.8)
        for v in [1,-1]: ax.axvline(v,color=PAL[4],lw=0.8,ls="--",alpha=0.7)
        ax.set_title("Feature Skewness"); ax.set_xlabel("Skewness")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,1])
    if nums:
        kurt=df[nums].kurt().sort_values()
        colors=[PAL[4] if abs(v)>3 else PAL[3] if abs(v)>1 else PAL[2] for v in kurt.values]
        ax.barh(range(len(kurt)),kurt.values,color=colors,edgecolor="#0F1224")
        ax.set_yticks(range(len(kurt))); ax.set_yticklabels([c[:14] for c in kurt.index],fontsize=6)
        ax.axvline(0,color=_MID,lw=0.8); ax.set_title("Feature Kurtosis"); ax.set_xlabel("Excess Kurtosis")
    else: _hide(ax)
    ax=fig.add_subplot(gs[1,2])
    if nums:
        desc=df[nums].describe().T.head(10); x=np.arange(len(desc))
        ax.barh(x,desc["max"]-desc["min"],left=desc["min"],color=PAL[0],alpha=0.4,label="Range")
        ax.barh(x,desc["75%"]-desc["25%"],left=desc["25%"],color=PAL[2],alpha=0.85,label="IQR")
        ax.scatter(desc["50%"],x,color=_HI,zorder=5,s=20,label="Median")
        ax.set_yticks(x); ax.set_yticklabels([c[:14] for c in desc.index],fontsize=6)
        ax.legend(fontsize=6); ax.set_title("Value Ranges (IQR + Full Range)")
    else: _hide(ax)
    return _b64(fig)


# ── HTML Assembler ────────────────────────────────────────────────────────────

def _html(pages, p, filename):
    hc="#00E5A0" if p.health>=75 else "#FFB547" if p.health>=50 else "#FF6B8A"
    mc="#FF6B8A" if p.missing_pct>5 else "#00E5A0"
    dc="#FF6B8A" if p.duplicates>0 else "#00E5A0"
    kpis=[
        ("📋","Rows",        f"{p.n:,}",                   "#6C63FF"),
        ("📐","Columns",     f"{len(p.df.columns)}",       "#00D4FF"),
        ("🔢","Numeric",     f"{len(p.numeric_cols)}",     "#00E5A0"),
        ("🔤","Categorical", f"{len(p.categorical_cols)}", "#FFB547"),
        ("📅","Datetime",    f"{len(p.datetime_cols)}",    "#9B59FF"),
        ("❓","Missing",     f"{p.missing_pct}%",          mc),
        ("♻️","Duplicates", f"{p.duplicates:,}",           dc),
        ("🏥","Health",      f"{p.health}%",               hc),
    ]
    kpi_html=""
    for icon,label,value,color in kpis:
        kpi_html+=f"""
        <div class="kpi-card" style="--accent:{color};">
          <div class="kpi-glow"></div>
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{color};">{value}</div>
          <div class="kpi-bar"><div class="kpi-bar-fill" style="background:{color};"></div></div>
        </div>"""
    role_map=defaultdict(int)
    for r in p.roles.values(): role_map[r]+=1
    role_color={"numeric":"#00E5A0","categorical":"#FFB547","datetime":"#9B59FF","boolean":"#00D4FF","id":"#8892B0","text":"#FF6B8A","email":"#3D4466","phone":"#3D4466","name":"#3D4466","address":"#3D4466","skip":"#252A45"}
    pills=""
    for role,cnt in sorted(role_map.items()):
        c=role_color.get(role,"#8892B0")
        pills+=f'<span class="pill" style="--pc:{c};"><span class="pill-dot" style="background:{c};box-shadow:0 0 5px {c};"></span>{role.title()} <b>{cnt}</b></span> '
    badges=""
    badge_items=[]
    if p.hue_col:  badge_items.append(("⬡","Primary Group",p.hue_col,"#6C63FF"))
    if p.hue_col2: badge_items.append(("◈","Secondary",p.hue_col2,"#00D4FF"))
    if p.date_col: badge_items.append(("◷","Time Axis",p.date_col,"#9B59FF"))
    if p.top_pair: badge_items.append(("⇌","Top Corr",f"{p.top_pair[0]} ↔ {p.top_pair[1]}","#00E5A0"))
    for sym,lbl,val,clr in badge_items:
        badges+=f'<div class="badge" style="--bc:{clr};"><span class="badge-sym">{sym}</span><span class="badge-lbl">{lbl}</span><span class="badge-val">{val}</span></div>'
    if not badges: badges='<span style="color:#2E3558;font-size:12px;">Single-column dataset</span>'
    tabs,panels="",""
    for i,(title,b64) in enumerate(pages):
        act="active" if i==0 else ""; dsp="block" if i==0 else "none"
        icon=title.split(" ")[0] if title[0] in "📊🔢🔤📅🩺" else "📊"
        clean=title.split(" ",1)[-1] if " " in title else title
        tabs+=(f'<button class="tab-btn {act}" onclick="switchTab({i})">'
               f'<span class="tab-icon">{icon}</span>'
               f'<span class="tab-label">{clean}</span>'
               f'<span class="tab-pip"></span></button>')
        panels+=(f'<div id="panel-{i}" class="chart-panel" style="display:{dsp};">'
                 f'<div class="chart-header"><span class="chart-dot"></span>'
                 f'<span class="chart-title">{title}</span></div>'
                 f'<div class="chart-body"><img src="data:image/png;base64,{b64}" class="chart-img" alt="{title}"></div>'
                 f'</div>')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoAnalytica — {filename}</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
:root{{
  --bg0:#060811;--bg1:#0A0D1C;--bg2:#0F1328;--bg3:#141836;
  --border:#1E2440;--border2:#2A3060;
  --hi:#E8EEFF;--mid:#6B7599;--lo:#2E3558;
  --p1:#6C63FF;--p2:#00D4FF;--p3:#00E5A0;--p4:#FFB547;--p5:#FF6B8A;
  --font:'Syne',sans-serif;--mono:'JetBrains Mono',monospace;
}}
html{{scroll-behavior:smooth;}}
body{{background:var(--bg0);font-family:var(--font);color:var(--mid);min-height:100vh;overflow-x:hidden;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-track{{background:var(--bg0);}}
::-webkit-scrollbar-thumb{{background:var(--border2);border-radius:2px;}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:44px 44px;opacity:0.20;pointer-events:none;z-index:0;}}
body::after{{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 65% 45% at 10% 0%,#6C63FF0E 0%,transparent 60%),radial-gradient(ellipse 45% 35% at 90% 100%,#00D4FF09 0%,transparent 55%);pointer-events:none;z-index:0;}}
.hdr{{position:sticky;top:0;z-index:200;background:rgba(6,8,17,0.92);backdrop-filter:blur(24px);border-bottom:1px solid var(--border);padding:0 28px;height:60px;display:flex;align-items:center;justify-content:space-between;}}
.hdr-line{{position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent 0%,var(--p1)55,var(--p2)55,transparent 100%);opacity:0.55;}}
.logo{{display:flex;align-items:center;gap:13px;}}
.logo-hex{{width:38px;height:38px;background:linear-gradient(135deg,var(--p1),var(--p2));clip-path:polygon(50% 0%,93% 25%,93% 75%,50% 100%,7% 75%,7% 25%);display:flex;align-items:center;justify-content:center;font-size:17px;animation:hue-cycle 14s linear infinite;flex-shrink:0;}}
@keyframes hue-cycle{{to{{filter:hue-rotate(360deg);}}}}
.logo-name{{font-size:15px;font-weight:800;letter-spacing:.13em;text-transform:uppercase;font-family:var(--mono);background:linear-gradient(90deg,var(--p1),var(--p2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.logo-sub{{font-size:9px;letter-spacing:.20em;color:var(--lo);text-transform:uppercase;margin-top:1px;}}
.hdr-file{{font-family:var(--mono);font-size:11px;color:var(--mid);background:var(--bg2);border:1px solid var(--border);padding:5px 13px;border-radius:6px;display:flex;align-items:center;gap:8px;}}
.hdr-file::before{{content:'◦';color:var(--p2);font-size:14px;}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:11px;padding:22px 28px 14px;position:relative;z-index:1;}}
.kpi-card{{background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent,var(--p1));border-radius:12px;padding:14px 15px 12px;position:relative;overflow:hidden;transition:transform .22s,box-shadow .22s,border-color .22s;cursor:default;}}
.kpi-card:hover{{transform:translateY(-4px);box-shadow:0 10px 36px color-mix(in srgb,var(--accent,var(--p1)) 22%,transparent);border-color:var(--accent,var(--p1));}}
.kpi-glow{{position:absolute;top:-22px;right:-22px;width:64px;height:64px;border-radius:50%;background:radial-gradient(circle,var(--accent,var(--p1)) 0%,transparent 70%);opacity:0.18;pointer-events:none;transition:opacity .3s;}}
.kpi-card:hover .kpi-glow{{opacity:0.42;}}
.kpi-icon{{font-size:19px;margin-bottom:7px;display:block;}}
.kpi-label{{font-size:9px;letter-spacing:.16em;text-transform:uppercase;color:var(--mid);font-weight:700;margin-bottom:4px;}}
.kpi-value{{font-size:24px;font-weight:800;font-family:var(--mono);line-height:1.1;margin-bottom:9px;}}
.kpi-bar{{height:2px;background:var(--border);border-radius:1px;overflow:hidden;}}
.kpi-bar-fill{{height:100%;width:68%;border-radius:1px;animation:bar-in .9s ease forwards;transform-origin:left;transform:scaleX(0);}}
@keyframes bar-in{{to{{transform:scaleX(1);}}}}
.schema-bar{{padding:8px 28px 8px;display:flex;flex-wrap:wrap;gap:7px;position:relative;z-index:1;}}
.pill{{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;font-family:var(--mono);background:color-mix(in srgb,var(--pc,#888) 10%,transparent);border:1px solid color-mix(in srgb,var(--pc,#888) 28%,transparent);color:var(--pc,#888);transition:background .2s;}}
.pill:hover{{background:color-mix(in srgb,var(--pc,#888) 18%,transparent);}}
.pill-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0;}}
.detect-row{{padding:6px 28px 18px;display:flex;flex-wrap:wrap;gap:8px;position:relative;z-index:1;}}
.badge{{display:inline-flex;align-items:center;gap:8px;background:var(--bg2);border:1px solid color-mix(in srgb,var(--bc,var(--p1)) 28%,transparent);border-radius:8px;padding:6px 14px;}}
.badge-sym{{font-size:14px;color:var(--bc,var(--p1));font-weight:700;}}
.badge-lbl{{color:var(--mid);font-size:9px;letter-spacing:.13em;text-transform:uppercase;font-weight:600;}}
.badge-val{{color:var(--hi);font-family:var(--mono);font-size:11px;font-weight:500;}}
.divider{{height:1px;background:linear-gradient(90deg,transparent 4%,var(--border) 28%,var(--border) 72%,transparent 96%);margin:0 28px;position:relative;z-index:1;}}
.tabs-wrap{{padding:16px 28px 0;display:flex;gap:5px;flex-wrap:wrap;position:relative;z-index:1;}}
.tab-btn{{position:relative;display:inline-flex;align-items:center;gap:7px;padding:9px 18px;background:var(--bg2);border:1px solid var(--border);border-radius:10px 10px 0 0;color:var(--mid);font-family:var(--font);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;transition:all .18s;outline:none;}}
.tab-btn:hover{{border-color:var(--p1);color:var(--hi);background:color-mix(in srgb,var(--p1) 8%,var(--bg2));}}
.tab-btn.active{{background:color-mix(in srgb,var(--p1) 14%,var(--bg2));border-color:var(--p1);border-bottom-color:var(--bg2);color:var(--hi);box-shadow:0 -4px 18px color-mix(in srgb,var(--p1) 22%,transparent);}}
.tab-icon{{font-size:14px;}}
.tab-pip{{width:5px;height:5px;border-radius:50%;background:var(--p1);opacity:0;transition:opacity .2s;position:absolute;top:7px;right:7px;box-shadow:0 0 7px var(--p1);}}
.tab-btn.active .tab-pip{{opacity:1;}}
.tab-rule{{height:1px;margin:0 28px;background:var(--p1);box-shadow:0 0 14px var(--p1);position:relative;z-index:1;}}
.panels-wrap{{padding:0 28px 36px;position:relative;z-index:1;}}
.chart-panel{{background:var(--bg2);border:1px solid var(--border);border-top:none;border-radius:0 12px 12px 12px;overflow:hidden;animation:fade-up .3s ease;}}
@keyframes fade-up{{from{{opacity:0;transform:translateY(8px);}}to{{opacity:1;transform:translateY(0);}}}}
.chart-header{{display:flex;align-items:center;gap:10px;padding:12px 20px;background:var(--bg3);border-bottom:1px solid var(--border);}}
.chart-dot{{width:8px;height:8px;border-radius:50%;background:var(--p1);box-shadow:0 0 10px var(--p1);flex-shrink:0;}}
.chart-title{{font-size:11px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--hi);}}
.chart-body{{padding:16px;}}
.chart-img{{width:100%;border-radius:8px;border:1px solid var(--border);display:block;}}
.ftr{{text-align:center;padding:16px;color:var(--lo);font-family:var(--mono);font-size:9px;letter-spacing:.22em;text-transform:uppercase;border-top:1px solid var(--border);position:relative;z-index:1;}}
.ftr b{{color:var(--p1);font-weight:400;}}
.kpi-grid,.schema-bar,.detect-row,.tabs-wrap,.panels-wrap{{animation:page-reveal .5s ease both;}}
.kpi-grid{{animation-delay:.04s;}}.schema-bar{{animation-delay:.09s;}}.detect-row{{animation-delay:.13s;}}.tabs-wrap{{animation-delay:.17s;}}.panels-wrap{{animation-delay:.21s;}}
@keyframes page-reveal{{from{{opacity:0;transform:translateY(12px);}}to{{opacity:1;transform:translateY(0);}}}}
</style>
</head>
<body>
<header class="hdr">
  <div class="logo">
    <div class="logo-hex">⚡</div>
    <div><div class="logo-name">AutoAnalytica</div><div class="logo-sub">Zero-Config Dashboard Engine</div></div>
  </div>
  <div class="hdr-file">{filename}</div>
  <div class="hdr-line"></div>
</header>
<div class="kpi-grid">{kpi_html}</div>
<div class="schema-bar">{pills}</div>
<div class="detect-row">{badges}</div>
<div class="divider"></div>
<div class="tabs-wrap">{tabs}</div>
<div class="tab-rule"></div>
<div class="panels-wrap">{panels}</div>
<footer class="ftr">AutoAnalytica &nbsp;<b>·</b>&nbsp; Matplotlib + Seaborn &nbsp;<b>·</b>&nbsp; Zero-Configuration Engine</footer>
<script>
function switchTab(idx){{
  document.querySelectorAll('.chart-panel').forEach((el,i)=>el.style.display=i===idx?'block':'none');
  document.querySelectorAll('.tab-btn').forEach((b,i)=>b.classList.toggle('active',i===idx));
}}
</script>
</body>
</html>"""


# ── Entry point (Part 1) ──────────────────────────────────────────────────────

def _sanitize(obj):
    if isinstance(obj,dict): return {k:_sanitize(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return type(obj)(_sanitize(v) for v in obj)
    if isinstance(obj,float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def generate_dashboard_graphs(filename: str) -> Dict:
    UPLOAD_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    fp=UPLOAD_DIR/filename
    if not fp.exists(): return {"error":f"File not found: {fp}"}
    try:
        if filename.lower().endswith((".xlsx",".xls")): df=pd.read_excel(fp)
        else: df=pd.read_csv(fp,sep=None,engine="python",encoding_errors="replace")
    except Exception as e: return {"error":f"Cannot read file: {e}"}
    if df.empty: return {"error":"File is empty."}
    try: profile=DataProfile(df)
    except Exception as e: return {"error":f"Profiling failed: {e}"}
    pages=[]
    def try_page(label,fn,*args):
        try:
            r=fn(*args)
            if r: pages.append((label,r))
        except Exception as e: log.warning("Page '%s' skipped: %s",label,e)
    try_page("📊 Overview",    _page_overview,    profile)
    try_page("🔢 Numeric",     _page_numeric,     profile)
    try_page("🔤 Categorical", _page_categorical, profile)
    try_page("📅 Time Series", _page_timeseries,  profile)
    try_page("🩺 Data Quality",_page_quality,     profile)
    if not pages: return {"error":"No charts could be generated."}
    html_str=_html(pages,profile,filename)
    name=f"dashboard_{uuid.uuid4().hex[:12]}.html"
    (REPORTS_DIR/name).write_text(html_str,encoding="utf-8")
    return _sanitize({
        "charts":          [name],
        "rows":            profile.n,
        "columns":         len(df.columns),
        "numeric_columns": len(profile.numeric_cols),
        "cat_columns":     len(profile.categorical_cols),
        "missing_cells":   int(df.isnull().sum().sum()),
        "detected_hue":    profile.hue_col,
        "detected_hue2":   profile.hue_col2,
        "detected_date":   profile.date_col,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  PART 2 — MongoDB Dashboard Stats (NEW — appended below chart engine)
#
#  get_dashboard_stats() and get_summary_counts() are async functions called
#  by the /dashboard/stats and /dashboard/counts router endpoints.
#  They are completely independent of the chart engine above.
# ═════════════════════════════════════════════════════════════════════════════

from datetime import datetime as _dt
from typing   import Any as _Any

from app.db.crud import (
    count_datasets,
    count_models,
    count_reports,
    get_best_accuracy,
    get_recent_models,
)


def _fmt_dt(value) -> str:
    """Normalise a datetime object or string to a formatted UTC string."""
    if value is None:
        return ""
    if isinstance(value, _dt):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


async def get_dashboard_stats() -> Dict[str, _Any]:
    """
    Return the full stats payload consumed by the frontend Dashboard.js page.

    Shape:
    ──────
    {
        "datasets_uploaded":  3,
        "models_trained":     7,
        "reports_generated":  2,
        "best_accuracy":      94.32,       # percentage
        "recent_trainings": [
            {
                "model_id":     "...",
                "model_name":   "RandomForest",
                "dataset":      "abc123.csv",
                "target":       "price",
                "problem_type": "regression",
                "accuracy":     91.5,
                "trained_at":   "2026-03-14 10:22:00"
            },
            ...
        ]
    }
    """
    total_datasets = await count_datasets()
    total_models   = await count_models()
    total_reports  = await count_reports()
    best_acc       = await get_best_accuracy()
    recent_raw     = await get_recent_models(limit=5)

    recent_trainings: List[Dict[str, _Any]] = []
    for m in recent_raw:
        acc = m.get("accuracy")
        recent_trainings.append({
            "model_id":     m.get("_id", ""),
            "model_name":   m.get("model_name", "Unknown"),
            "dataset":      m.get("dataset_filename", ""),
            "target":       m.get("target_column", ""),
            "problem_type": m.get("problem_type", ""),
            "accuracy":     round(float(acc) * 100, 2) if acc is not None else None,
            "trained_at":   _fmt_dt(m.get("created_at")),
        })

    return {
        "datasets_uploaded": total_datasets,
        "models_trained":    total_models,
        "reports_generated": total_reports,
        "best_accuracy":     best_acc,
        "recent_trainings":  recent_trainings,
    }


async def get_summary_counts() -> Dict[str, int]:
    """
    Return the three entity counts.
    Cheaper than get_dashboard_stats() — used for header widgets or polling.
    """
    return {
        "datasets": await count_datasets(),
        "models":   await count_models(),
        "reports":  await count_reports(),
    }