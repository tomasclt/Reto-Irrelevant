import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

GOLD = "#D6B36A"
CREAM = "#F4F1E8"
MUTED = "#9B9EA7"
GRID = "rgba(255,255,255,.08)"


def _layout(fig, title: str):
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CREAM),
        margin=dict(l=10, r=10, t=55, b=10),
        hoverlabel=dict(bgcolor="#15171C", font_color=CREAM),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def sales_line(monthly_df: pd.DataFrame):
    fig = px.line(monthly_df, x="periodo", y="ventas", markers=True)
    fig.update_traces(line_color=GOLD, marker_color=GOLD, line_width=3)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig, "Evolución de ventas")


def orders_bar(monthly_df: pd.DataFrame):
    fig = px.bar(monthly_df, x="periodo", y="pedidos")
    fig.update_traces(marker_color=GOLD)
    return _layout(fig, "Pedidos por mes")


def horizontal_bar(df: pd.DataFrame, label: str, title: str):
    view = df.head(10).sort_values("ventas")
    fig = px.bar(view, x="ventas", y=label, orientation="h")
    fig.update_traces(marker_color=GOLD)
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig, title)


def concentration_donut(df: pd.DataFrame, label: str):
    view = df.head(5).copy()
    others = max(0, 1 - view["participacion"].sum())
    labels = view[label].tolist() + (["Otros"] if others > 0.0001 else [])
    values = view["participacion"].tolist() + ([others] if others > 0.0001 else [])
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.62, textinfo="percent+label"))
    fig.update_traces(marker=dict(line=dict(color="#15171C", width=2)))
    return _layout(fig, "Concentración de ventas")
