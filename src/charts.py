from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

COLORS = {"id": "#167D8D", "vn": "#E07A3F", "text": "#18212A", "muted": "#596773", "grid": "#D9E0E4", "light": "#E8EEF1", "highlight": "#C43E3E"}


def _font(size: int, bold: bool = False):
    for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FT = _font(38, True); FS = _font(19); FL = _font(18); FB = _font(18, True); FZ = _font(15)


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _fmt(value: float) -> str:
    if abs(value) >= 1_000_000: return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000: return f"{value/1_000:.1f}K"
    if abs(value) >= 100: return f"{value:,.0f}"
    return f"{value:.2f}"


def _base(title: str, subtitle: str, height: int = 900):
    im = Image.new("RGB", (1600, height), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 38), title, fill=_rgb(COLORS["text"]), font=FT)
    y = 94
    for line in textwrap.wrap(subtitle, 130):
        d.text((60, y), line, fill=_rgb(COLORS["muted"]), font=FS); y += 25
    return im, d


def _footer(d: ImageDraw.ImageDraw, note: str, y: int = 855):
    d.text((60, y), note, fill=_rgb(COLORS["muted"]), font=FZ)


def _save(im: Image.Image, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    im.save(path, "PNG", optimize=True)
    return path


def grouped_bars(labels: list[str], series: dict[str, list[float]], title: str, subtitle: str, out: Path, name: str, percent: bool = False, note: str = "") -> Path:
    im, d = _base(title, subtitle)
    left, right, top, bottom = 130, 1530, 185, 735
    vmax = max(v for values in series.values() for v in values) or 1
    if percent: vmax = max(1, vmax)
    for i in range(6):
        value = vmax * i / 5; y = bottom - (bottom-top)*i/5
        d.line((left, y, right, y), fill=_rgb(COLORS["grid"]))
        lab = f"{100*value:.0f}%" if percent else _fmt(value)
        d.text((left-12, y), lab, anchor="rm", fill=_rgb(COLORS["muted"]), font=FZ)
    gw = (right-left)/len(labels); bw = min(62, gw*.7/max(1,len(series)))
    for sidx, (sname, values) in enumerate(series.items()):
        color = _rgb(COLORS["id"] if sidx == 0 else COLORS["vn"])
        for i, value in enumerate(values):
            cx = left+gw*(i+.5); x = cx-len(series)*bw/2+sidx*bw
            h = (bottom-top)*value/vmax
            d.rounded_rectangle((x,bottom-h,x+bw-5,bottom),radius=5,fill=color)
            lab = f"{100*value:.1f}%" if percent else _fmt(value)
            d.text((x+(bw-5)/2,bottom-h-7),lab,anchor="ms",fill=_rgb(COLORS["text"]),font=FZ)
    for i,label in enumerate(labels):
        d.text((left+gw*(i+.5),bottom+22),str(label)[:24],anchor="ma",fill=_rgb(COLORS["text"]),font=FL)
    x=1070
    for sidx,sname in enumerate(series):
        d.rectangle((x,145,x+18,163),fill=_rgb(COLORS["id"] if sidx==0 else COLORS["vn"]))
        d.text((x+25,154),sname,anchor="lm",fill=_rgb(COLORS["text"]),font=FZ); x+=190
    _footer(d,note)
    return _save(im,out,name)


def panel_bars(panels: list[tuple[str,list[str],list[float],str]], title: str, subtitle: str, out: Path, name: str, note: str = "") -> Path:
    im,d=_base(title,subtitle,940)
    for pidx,(ptitle,labels,values,color) in enumerate(panels):
        x0=40+pidx*790; left,right,top,bottom=x0+320,x0+750,205,810
        d.text((x0+25,160),ptitle,fill=_rgb(COLORS["text"]),font=_font(25,True))
        step=(bottom-top)/max(1,len(labels)); vmax=max(values) or 1
        for i,(label,value) in enumerate(zip(labels,values)):
            y=top+i*step+step*.18; h=min(38,step*.58)
            d.text((left-12,y+h/2),str(label)[:36],anchor="rm",fill=_rgb(COLORS["text"]),font=FZ)
            w=(right-left)*value/vmax
            d.rounded_rectangle((left,y,left+w,y+h),radius=5,fill=_rgb(color))
            d.text((min(right,left+w+8),y+h/2),_fmt(value),anchor="lm",fill=_rgb(COLORS["text"]),font=FZ)
    _footer(d,note,900)
    return _save(im,out,name)


def scatter_panels(frame: pd.DataFrame, x: str, y: str, highlight: str | None, title: str, subtitle: str, out: Path, name: str, x_label: str, y_label: str, note: str = "") -> Path:
    im,d=_base(title,subtitle)
    for pidx,country in enumerate(["id","vn"]):
        g=frame.loc[frame.country_code==country].dropna(subset=[x,y])
        xv=g[x].to_numpy(float); yv=g[y].to_numpy(float)
        x0=50+pidx*790; left,right,top,bottom=x0+110,x0+750,195,735
        xmin,xmax=np.nanmin(xv),np.nanmax(xv); ymin,ymax=np.nanmin(yv),np.nanmax(yv)
        if xmax==xmin:xmax+=1
        if ymax==ymin:ymax+=1
        d.text(((left+right)/2,155),"Indonesia" if country=="id" else "Vietnam",anchor="mm",fill=_rgb(COLORS["text"]),font=_font(24,True))
        for i in range(6):
            xx=left+(right-left)*i/5; yy=bottom-(bottom-top)*i/5
            d.line((xx,top,xx,bottom),fill=_rgb(COLORS["grid"])); d.line((left,yy,right,yy),fill=_rgb(COLORS["grid"]))
            d.text((xx,bottom+12),_fmt(xmin+(xmax-xmin)*i/5),anchor="ma",fill=_rgb(COLORS["muted"]),font=FZ)
            d.text((left-10,yy),_fmt(ymin+(ymax-ymin)*i/5),anchor="rm",fill=_rgb(COLORS["muted"]),font=FZ)
        overlay=Image.new("RGBA",im.size,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        flags=g[highlight].astype(bool).to_numpy() if highlight else np.zeros(len(g),bool)
        for xx,yy,flag in zip(xv,yv,flags):
            px=left+(right-left)*(xx-xmin)/(xmax-xmin); py=bottom-(bottom-top)*(yy-ymin)/(ymax-ymin)
            color=(*_rgb(COLORS["highlight"] if flag else COLORS[country]),180 if flag else 75)
            r=6 if flag else 4; od.ellipse((px-r,py-r,px+r,py+r),fill=color)
        im.paste(overlay,(0,0),overlay); d=ImageDraw.Draw(im)
        d.text(((left+right)/2,bottom+58),x_label,anchor="mm",fill=_rgb(COLORS["text"]),font=FL)
        d.text((left-75,(top+bottom)/2),y_label,anchor="mm",fill=_rgb(COLORS["text"]),font=FL)
    _footer(d,note)
    return _save(im,out,name)


def feature_importance_chart(importance: pd.DataFrame, out: Path) -> Path:
    agg=importance.groupby("feature_label",as_index=False).importance.mean().nlargest(15,"importance").sort_values("importance")
    im,d=_base("Global feature importance for the best actionable models","Mean normalized importance across the separately trained Indonesia and Vietnam models")
    left,right,top,bottom=510,1510,180,780; step=(bottom-top)/len(agg); vmax=agg.importance.max()
    for i,row in enumerate(agg.itertuples()):
        y=top+i*step+step*.18; h=step*.6
        d.text((left-15,y+h/2),str(row.feature_label)[:48],anchor="rm",fill=_rgb(COLORS["text"]),font=FL)
        w=(right-left)*row.importance/vmax
        d.rounded_rectangle((left,y,left+w,y+h),radius=5,fill=_rgb(COLORS["id"]))
        d.text((left+w+9,y+h/2),f"{row.importance:.3f}",anchor="lm",fill=_rgb(COLORS["text"]),font=FZ)
    _footer(d,"Importance is model-specific and explanatory; it is not causal effect size.")
    return _save(im,out,"06_global_feature_importance.png")


def create_required_charts(scored: pd.DataFrame, metrics: pd.DataFrame, importance: pd.DataFrame, sensitivity: pd.DataFrame, out: Path) -> list[Path]:
    paths: list[Path]=[]
    bins=np.linspace(0,100,11)
    hist={c:np.histogram(scored.loc[scored.country_code==c,"opportunity_score"],bins=bins)[0].tolist() for c in ["id","vn"]}
    paths.append(grouped_bars([f"{int(bins[i])}–{int(bins[i+1])}" for i in range(10)],{"Indonesia":hist["id"],"Vietnam":hist["vn"]},"Opportunity scores separate prioritization tiers","Count of latest listings by transparent 0–100 score band",out,"01_opportunity_score_distribution.png",note="Scores use peer-normalized signals within country and platform category."))
    labels=sorted(scored.recommendation_label.unique())
    counts=scored.groupby(["country_code","recommendation_label"]).size().unstack(fill_value=0)
    paths.append(grouped_bars(labels,{"Indonesia":[counts.loc["id"].get(x,0) for x in labels],"Vietnam":[counts.loc["vn"].get(x,0) for x in labels]},"Recommendation mix differs by market context","Decision-support labels assigned from score components and peer position",out,"02_recommendation_label_distribution.png",note="Labels guide review; they do not guarantee outcomes."))
    panels=[]
    for c in ["id","vn"]:
        top=scored.loc[scored.country_code==c].nlargest(10,"opportunity_score")
        panels.append(("Indonesia" if c=="id" else "Vietnam",[str(x)[:34] for x in top.product_name],top.opportunity_score.tolist(),COLORS[c]))
    paths.append(panel_bars(panels,"Top 10 product opportunities by market","Balanced opportunity score; prices are never compared across currencies",out,"03_top_10_opportunities.png",note="Full product identifiers and explanations are in product_recommendations.csv."))
    paths.append(scatter_panels(scored,"engagement_strength","sold_pct_peer","conversion_gap_candidate","Conversion opportunities combine high engagement with a sold-value gap","Red points meet the transparent conversion-gap rule",out,"04_engagement_vs_sold_conversion_gap.png","Engagement strength","Sold-value peer percentile",note="Both axes are peer-normalized within country and platform category."))
    paths.append(scatter_panels(scored,"discount_pct_peer","sold_pct_peer","deep_discount_flag","Deep discounts do not guarantee strong peer-relative response","Displayed discount percentile versus historical sold-value percentile",out,"05_discount_vs_peer_response.png","Discount peer percentile","Sold-value peer percentile",note="Red points have displayed discounts of at least 50%; association is not causal."))
    paths.append(feature_importance_chart(importance,out))
    group=metrics.loc[metrics.validation_scheme=="group_5_fold"].copy()
    labels=["DummyMedian","Ridge","GradientBoostedStumpsFallback"]
    summary=group.groupby("model").rmse_log.mean()
    paths.append(grouped_bars(["RMSE log"],{"Dummy median":[summary.get(labels[0],0)],"Best non-trivial":[min(summary.get(labels[1],99),summary.get(labels[2],99))]},"Non-trivial models are tested against a median baseline","Mean shop-grouped cross-validation RMSE across countries and feature experiments",out,"07_model_baseline_comparison.png",note="Lower is better. Descriptive features are leakage-prone for future prediction and are reported separately."))
    sens=sensitivity.pivot(index="configuration",columns="country_code",values="top20_jaccard_vs_balanced")
    configs=["balanced","growth_opportunity","hero_protection"]
    paths.append(grouped_bars(configs,{"Indonesia":[sens.loc[x,"id"] for x in configs],"Vietnam":[sens.loc[x,"vn"] for x in configs]},"Top-ranked products remain reasonably stable across score weights","Jaccard overlap of each configuration's top 20 with the balanced top 20",out,"08_score_weight_sensitivity.png",percent=True,note="Balanced is 100% by definition; higher alternative overlap means greater ranking robustness."))
    return paths


def contact_sheet(paths: list[Path], out: Path) -> Path:
    thumbs=[]
    for path in paths:
        im=Image.open(path).convert("RGB"); im.thumbnail((620,360)); thumbs.append(im.copy())
    sheet=Image.new("RGB",(1280,math.ceil(len(thumbs)/2)*380),_rgb(COLORS["light"]))
    for i,im in enumerate(thumbs): sheet.paste(im,((i%2)*640+10,(i//2)*380+10))
    path=out/"proposal_chart_contact_sheet.png"; sheet.save(path,"PNG",optimize=True); return path
