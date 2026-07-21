from __future__ import annotations

import pandas as pd

MONTHS_ES = {1:"enero",2:"febrero",3:"marzo",4:"abril",5:"mayo",6:"junio",7:"julio",8:"agosto",9:"septiembre",10:"octubre",11:"noviembre",12:"diciembre"}


def apply_filters(df: pd.DataFrame, start, end, clients: list[str], products: list[str]) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out[out["fecha"].dt.date >= start]
    if end is not None:
        out = out[out["fecha"].dt.date <= end]
    if clients:
        out = out[out["cliente"].isin(clients)]
    if products:
        out = out[out["producto"].isin(products)]
    return out


def kpis(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {"ventas": 0.0, "pedidos": 0, "clientes": 0, "productos": 0, "ticket": 0.0}
    return {
        "ventas": float(df["valor"].sum()),
        "pedidos": int(len(df)),
        "clientes": int(df["cliente"].nunique()),
        "productos": int(df["producto"].nunique()),
        "ticket": float(df["valor"].mean()),
    }


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["periodo", "ventas", "pedidos", "ticket"])
    work = df.assign(periodo=df["fecha"].dt.to_period("M").dt.to_timestamp())
    return (
        work.groupby("periodo", as_index=False)
        .agg(ventas=("valor", "sum"), pedidos=("valor", "size"), ticket=("valor", "mean"))
        .sort_values("periodo")
    )


def by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[dimension, "ventas", "pedidos", "ticket", "participacion"])
    result = (
        df.groupby(dimension, as_index=False)
        .agg(ventas=("valor", "sum"), pedidos=("valor", "size"), ticket=("valor", "mean"))
        .sort_values("ventas", ascending=False)
    )
    total = result["ventas"].sum()
    result["participacion"] = result["ventas"] / total if total else 0
    return result


def executive_insights(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No hay datos para analizar con los filtros actuales."]
    insights: list[str] = []
    m = monthly(df)
    if len(m) >= 2:
        prev, curr = m.iloc[-2], m.iloc[-1]
        sales_delta = ((curr.ventas / prev.ventas) - 1) * 100 if prev.ventas else 0
        orders_delta = ((curr.pedidos / prev.pedidos) - 1) * 100 if prev.pedidos else 0
        ticket_delta = ((curr.ticket / prev.ticket) - 1) * 100 if prev.ticket else 0
        insights.append(
            f"En {MONTHS_ES[curr.periodo.month]} de {curr.periodo.year}, las ventas variaron {sales_delta:+.1f}% frente al mes anterior; "
            f"los pedidos {orders_delta:+.1f}% y el ticket promedio {ticket_delta:+.1f}%."
        )
    products = by_dimension(df, "producto")
    clients = by_dimension(df, "cliente")
    if not products.empty:
        top = products.iloc[0]
        insights.append(f"{top.producto} concentra {top.participacion:.1%} de la facturación del periodo seleccionado.")
    if not clients.empty:
        top3 = clients.head(3)["participacion"].sum()
        insights.append(f"Los tres principales clientes representan {top3:.1%} de las ventas.")
    best_day = df.groupby(df["fecha"].dt.date)["valor"].sum().idxmax()
    best_value = df.groupby(df["fecha"].dt.date)["valor"].sum().max()
    insights.append(f"El día de mayor facturación fue {best_day:%d/%m/%Y}, con ${best_value:,.0f}.")
    insights.append("El archivo no contiene costos, cantidades, descuentos ni márgenes; las causas comerciales deben tratarse como hipótesis.")
    return insights


def ai_context(df: pd.DataFrame) -> str:
    metrics = kpis(df)
    m = monthly(df).tail(12).copy()
    clients = by_dimension(df, "cliente").head(10)
    products = by_dimension(df, "producto").head(10)
    parts = [
        "METRICAS VERIFICADAS",
        f"Periodo: {df.fecha.min():%Y-%m-%d} a {df.fecha.max():%Y-%m-%d}" if not df.empty else "Periodo: sin datos",
        f"Ventas: {metrics['ventas']:.2f}; pedidos: {metrics['pedidos']}; clientes: {metrics['clientes']}; ticket: {metrics['ticket']:.2f}",
        "SERIE MENSUAL:\n" + m.to_csv(index=False),
        "TOP CLIENTES:\n" + clients.to_csv(index=False),
        "TOP PRODUCTOS:\n" + products.to_csv(index=False),
        "LIMITACIONES: no hay costos, cantidades, descuentos, margen, vendedor ni región.",
    ]
    return "\n".join(parts)
