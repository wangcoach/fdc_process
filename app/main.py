"""
FDC 分析演示 API：元数据下拉 + Plotly 图表 JSON（前端 plotly.js 渲染）。
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="FDC Rawtrace API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 演示用元数据（可改为 DB / Parquet / API）----------

FABS = [
    {"id": "fab_a", "label": "Fab A (12\")"},
    {"id": "fab_b", "label": "Fab B (8\")"},
]

CHAMBERS_BY_FAB: dict[str, list[dict[str, str]]] = {
    "fab_a": [
        {"id": "CVD01", "label": "CVD-01"},
        {"id": "ETCH02", "label": "ETCH-02"},
    ],
    "fab_b": [
        {"id": "PVD01", "label": "PVD-01"},
    ],
}

STEPS_BY_CHAMBER: dict[str, list[dict[str, str]]] = {
    "CVD01": [{"id": "S10", "label": "Step 10 Depo"}, {"id": "S20", "label": "Step 20 Clean"}],
    "ETCH02": [{"id": "E01", "label": "Etch Main"}],
    "PVD01": [{"id": "P05", "label": "PVD Coat"}],
}

PARAMS_BY_STEP: dict[str, list[dict[str, str]]] = {
    "S10": [
        {"id": "pressure", "label": "Chamber Pressure (Torr)"},
        {"id": "rf_fwd", "label": "RF Forward Power (W)"},
        {"id": "temp_ped", "label": "Pedestal Temperature (°C)"},
        {"id": "rf_refl", "label": "RF Reflected Power (W)"},
        {"id": "gas_ar", "label": "Ar Flow (sccm)"},
        {"id": "gas_n2", "label": "N2 Flow (sccm)"},
        {"id": "dc_bias", "label": "DC Bias (V)"},
        {"id": "esc_volt", "label": "ESC Voltage (V)"},
        {"id": "throttle", "label": "Throttle Valve (%)"},
        {"id": "chamber_lk", "label": "Chamber Leak Rate (mTorr/min)"},
    ],
    "S20": [{"id": "o2_flow", "label": "O2 Flow (sccm)"}],
    "E01": [{"id": "bias", "label": "Bias Voltage (V)"}],
    "P05": [{"id": "thickness", "label": "Film Thickness (nm)"}],
}


def _param_label(step_id: str, param_id: str) -> str:
    for p in PARAMS_BY_STEP.get(step_id, []):
        if p["id"] == param_id:
            return p["label"]
    return param_id


def _demo_trace_y(seed: str, n: int = 400) -> list[float]:
    """可复现的演示曲线；生产环境替换为真实采样数据。"""
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    phase = (h % 10000) / 10000.0 * 2 * math.pi
    out: list[float] = []
    for i in range(n):
        t = i / 25.0
        base = 10.0 + 2.2 * math.sin(t + phase) + 0.35 * math.sin(3.2 * t + phase * 0.7)
        out.append(base + 0.08 * math.sin(17 * t))  # 轻微高频抖动
    return out


# 与设备 Step（下拉）无关：原始 FDC 中 process_step 列的取值（演示）
_PROCESS_STEP_POOL = ("DEP_PRIME", "MAIN_DEP", "STAB", "PURGE", "COOL_DOWN")


def _process_step_intervals(n: int, seed: str) -> tuple[list[int], list[str]]:
    """按 sample index 将数据划分为若干区间，每段对应 process_step 列的一个取值。"""
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    k = 3 + (h % 3)  # 3..5 段
    k = max(2, min(k, max(2, n // 50)))
    starts: list[int] = [0]
    acc = 0
    for i in range(k):
        chunk = n // k + (1 if i < (n % k) else 0)
        if i == k - 1:
            break
        acc += chunk
        starts.append(acc)
    names = [_PROCESS_STEP_POOL[(h + i) % len(_PROCESS_STEP_POOL)] for i in range(k)]
    return starts, names


def _process_step_column(n: int, starts: list[int], names: list[str]) -> list[str]:
    """每个 sample 对应的 process_step 列值（与真实数据列一致）。"""
    out: list[str] = []
    seg = 0
    for i in range(n):
        while seg + 1 < len(starts) and i >= starts[seg + 1]:
            seg += 1
        out.append(names[seg])
    return out


# 前端缩略图会过滤掉该 name 的 trace；大图保留（用 Scatter 竖线比 layout.shapes 更稳定）
PROCESS_STEP_VLINE_TRACE_NAME = "__process_step_vline__"


class ChartRequest(BaseModel):
    wafer_ids: list[str] = Field(..., min_length=1, description="一个或多个 Wafer ID")
    fab_id: str
    chamber_id: str
    step_id: str
    param_ids: list[str] = Field(..., min_length=1, description="一个或多个 Parameter")
    process_step: str | None = Field(default=None, description="保留字段；分段由数据中的 process_step 列决定")
    time_start: datetime | None = None
    time_end: datetime | None = None


DESIGN_FONT = "-apple-system, system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
DESIGN_TEXT = "#181d26"
DESIGN_BORDER = "#e0e2e6"
DESIGN_PLOT_BG = "#f8fafc"
DESIGN_PAPER = "#ffffff"


def build_single_param_figure(body: ChartRequest, pid: str) -> go.Figure:
    """单参数单图：前端每张卡片独立渲染，各自右上角 Plotly 工具栏。"""
    n = 400
    x = list(range(n))
    colors = ("#1b61c9", "#e11d48", "#059669", "#d97706", "#7c3aed", "#0d9488", "#db2777", "#4f46e5")
    # 演示：与原始 FDC 一致，每条 trace 对应一列 process_step，按 sample 将曲线分成多段
    interval_seed = f"{body.step_id}|{pid}"
    seg_starts, seg_names = _process_step_intervals(n, interval_seed)
    _process_step_col = _process_step_column(n, seg_starts, seg_names)
    assert len(_process_step_col) == n

    fig = go.Figure()
    all_y_vals: list[float] = []
    for j, wid in enumerate(body.wafer_ids):
        seed = f"{wid}|{pid}|{body.step_id}"
        y = _demo_trace_y(seed, n)
        all_y_vals.extend(y)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=wid,
                legendgroup=wid,
                showlegend=True,
                line=dict(width=1.4, color=colors[j % len(colors)]),
                # 不在坐标轴 / hover 中展示 process_step；分段由竖线标出
                hovertemplate="wafer=%{fullData.name}<br>sample=%{x}<br>y=%{y:.4f}<extra></extra>",
            )
        )

    ymin = min(all_y_vals)
    ymax = max(all_y_vals)
    yspan = ymax - ymin
    pad = max(yspan * 0.06, 1e-6)
    y_lo = ymin - pad
    y_hi = ymax + pad
    # 每个新区间起点（第二段起）画竖线，与 plotly.js layout.shapes 无关，避免被 react/resize 吞掉
    for x0 in seg_starts[1:]:
        fig.add_trace(
            go.Scatter(
                x=[x0, x0],
                y=[y_lo, y_hi],
                mode="lines",
                name=PROCESS_STEP_VLINE_TRACE_NAME,
                showlegend=False,
                hoverinfo="skip",
                line=dict(color="#64748b", width=2, dash="dot"),
            )
        )

    fig.update_layout(
        title=None,
        paper_bgcolor=DESIGN_PAPER,
        plot_bgcolor=DESIGN_PLOT_BG,
        font=dict(family=DESIGN_FONT, color=DESIGN_TEXT, size=11),
        # 右侧留空给 modebar（缩放/平移/下载等）
        margin=dict(l=52, r=52, t=36, b=48),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=DESIGN_TEXT),
        ),
        hovermode="closest",
        height=260,
        autosize=False,
    )
    fig.update_xaxes(
        title=dict(text="Sample index", font=dict(size=11)),
        gridcolor=DESIGN_BORDER,
        zerolinecolor="#cbd5e1",
        showgrid=True,
        linecolor=DESIGN_BORDER,
    )
    fig.update_yaxes(
        title=dict(text=_param_label(body.step_id, pid), font=dict(size=11)),
        gridcolor=DESIGN_BORDER,
        zerolinecolor="#cbd5e1",
        showgrid=True,
        linecolor=DESIGN_BORDER,
    )
    return fig


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta/fabs")
def list_fabs() -> list[dict[str, str]]:
    return FABS


@app.get("/api/meta/chambers")
def list_chambers(fab_id: str = Query(...)) -> list[dict[str, str]]:
    return CHAMBERS_BY_FAB.get(fab_id, [])


@app.get("/api/meta/steps")
def list_steps(chamber_id: str = Query(...)) -> list[dict[str, str]]:
    return STEPS_BY_CHAMBER.get(chamber_id, [])


@app.get("/api/meta/params")
def list_params(step_id: str = Query(...)) -> list[dict[str, str]]:
    return PARAMS_BY_STEP.get(step_id, [])


@app.post("/api/chart")
def chart_figure(body: ChartRequest) -> JSONResponse:
    """每个 param 独立 figure，前端一卡一图，各自 modebar。"""
    figures: list[dict] = []
    for pid in body.param_ids:
        fig = build_single_param_figure(body, pid)
        d = json.loads(fig.to_json())
        figures.append(
            {
                "param_id": pid,
                "label": _param_label(body.step_id, pid),
                "data": d["data"],
                "layout": d["layout"],
            }
        )
    return JSONResponse(content={"figures": figures})


@app.get("/")
def index() -> FileResponse:
    """不要用 StaticFiles 挂到 '/'，否则在部分环境下会抢在 /api 之前匹配，导致下拉元数据请求拿不到 JSON。"""
    return FileResponse(STATIC_DIR / "index.html")
