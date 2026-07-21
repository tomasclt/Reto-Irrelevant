from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


MONTHS_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}

MONTH_NAME_TO_NUMBER = {
    name: number for number, name in MONTHS_ES.items()
}

AI_COLUMNS = ["fecha", "cliente", "producto", "valor"]


# -----------------------------------------------------------------------------
# Filtros y métricas generales
# -----------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    start,
    end,
    clients: list[str],
    products: list[str],
) -> pd.DataFrame:
    """Aplica los filtros seleccionados en el dashboard."""
    out = df.copy()

    if out.empty:
        return out

    if start is not None:
        out = out[out["fecha"].dt.date >= start]

    if end is not None:
        out = out[out["fecha"].dt.date <= end]

    if clients:
        out = out[out["cliente"].isin(clients)]

    if products:
        out = out[out["producto"].isin(products)]

    return out.copy()


def kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """
    Calcula indicadores generales.

    La clave 'pedidos' se conserva por compatibilidad con la interfaz
    actual, pero representa el número de filas o registros de venta
    cuando no existe un identificador único de pedido.
    """
    if df.empty:
        return {
            "ventas": 0.0,
            "pedidos": 0,
            "clientes": 0,
            "productos": 0,
            "ticket": 0.0,
        }

    return {
        "ventas": float(df["valor"].sum()),
        "pedidos": int(len(df)),
        "clientes": int(df["cliente"].nunique()),
        "productos": int(df["producto"].nunique()),
        "ticket": float(df["valor"].mean()),
    }


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las ventas por mes."""
    if df.empty:
        return pd.DataFrame(
            columns=["periodo", "ventas", "pedidos", "ticket"]
        )

    work = df.assign(
        periodo=df["fecha"].dt.to_period("M").dt.to_timestamp()
    )

    return (
        work.groupby("periodo", as_index=False)
        .agg(
            ventas=("valor", "sum"),
            pedidos=("valor", "size"),
            ticket=("valor", "mean"),
        )
        .sort_values("periodo")
        .reset_index(drop=True)
    )


def by_dimension(
    df: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """Agrupa ventas por cliente o producto."""
    if dimension not in {"cliente", "producto"}:
        raise ValueError(
            "La dimensión debe ser 'cliente' o 'producto'."
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                dimension,
                "ventas",
                "pedidos",
                "ticket",
                "participacion",
            ]
        )

    result = (
        df.groupby(dimension, as_index=False)
        .agg(
            ventas=("valor", "sum"),
            pedidos=("valor", "size"),
            ticket=("valor", "mean"),
        )
        .sort_values("ventas", ascending=False)
        .reset_index(drop=True)
    )

    total = float(result["ventas"].sum())
    result["participacion"] = (
        result["ventas"] / total if total else 0.0
    )

    return result


def executive_insights(df: pd.DataFrame) -> list[str]:
    """
    Genera hallazgos ejecutivos calculados exclusivamente
    con Pandas.
    """
    if df.empty:
        return [
            "No hay datos para analizar con los filtros actuales."
        ]

    insights: list[str] = []
    month_table = monthly(df)

    if len(month_table) >= 2:
        previous = month_table.iloc[-2]
        current = month_table.iloc[-1]

        sales_delta = (
            (
                (current["ventas"] / previous["ventas"]) - 1
            )
            * 100
            if previous["ventas"]
            else 0.0
        )

        records_delta = (
            (
                (current["pedidos"] / previous["pedidos"]) - 1
            )
            * 100
            if previous["pedidos"]
            else 0.0
        )

        ticket_delta = (
            (
                (current["ticket"] / previous["ticket"]) - 1
            )
            * 100
            if previous["ticket"]
            else 0.0
        )

        insights.append(
            f"En {MONTHS_ES[current['periodo'].month]} de "
            f"{current['periodo'].year}, las ventas variaron "
            f"{sales_delta:+.1f}% frente al mes anterior; "
            f"los registros {records_delta:+.1f}% y el ticket "
            f"promedio {ticket_delta:+.1f}%."
        )

    products = by_dimension(df, "producto")
    clients = by_dimension(df, "cliente")

    if not products.empty:
        top_product = products.iloc[0]

        insights.append(
            f"{top_product['producto']} concentra "
            f"{top_product['participacion']:.1%} de la "
            "facturación del periodo seleccionado."
        )

    if not clients.empty:
        top_three_share = float(
            clients.head(3)["participacion"].sum()
        )

        insights.append(
            "Los tres principales clientes representan "
            f"{top_three_share:.1%} de las ventas."
        )

    daily_sales = df.groupby(
        df["fecha"].dt.date
    )["valor"].sum()

    best_day = daily_sales.idxmax()
    best_value = float(daily_sales.max())

    insights.append(
        f"El día de mayor facturación fue "
        f"{best_day:%d/%m/%Y}, con ${best_value:,.0f}."
    )

    insights.append(
        "El archivo no contiene necesariamente costos, "
        "cantidades, descuentos ni márgenes; las causas "
        "comerciales deben tratarse como hipótesis."
    )

    return insights


# -----------------------------------------------------------------------------
# Detección de entidades para el contexto dinámico de IA
# -----------------------------------------------------------------------------


def normalize_text(value: object) -> str:
    """
    Normaliza texto: minúsculas, sin tildes y sin espacios
    repetidos.
    """
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(r"\s+", " ", text)


def find_mentions(
    question: str,
    values: list[str],
) -> list[str]:
    """
    Encuentra clientes o productos mencionados literalmente
    en la consulta.
    """
    normalized_question = normalize_text(question)
    matches: list[str] = []

    ordered_values = sorted(
        values,
        key=lambda item: len(str(item)),
        reverse=True,
    )

    for value in ordered_values:
        normalized_value = normalize_text(value)

        if (
            normalized_value
            and normalized_value in normalized_question
        ):
            matches.append(value)

    return matches


def detect_months(question: str) -> list[int]:
    """Detecta meses escritos en español dentro de la pregunta."""
    normalized_question = normalize_text(question)
    detected: list[int] = []

    for (
        month_name,
        month_number,
    ) in MONTH_NAME_TO_NUMBER.items():
        if re.search(
            rf"\b{re.escape(month_name)}\b",
            normalized_question,
        ):
            detected.append(month_number)

    return detected


def detect_years(question: str) -> list[int]:
    """Detecta años escritos con cuatro cifras."""
    return sorted(
        {
            int(year)
            for year in re.findall(
                r"\b(?:19|20)\d{2}\b",
                question,
            )
        }
    )


def history_to_text(
    history: list[dict[str, Any]] | None,
    max_messages: int = 6,
) -> str:
    """
    Convierte los últimos mensajes del usuario en texto.

    Sirve como apoyo para preguntas de seguimiento como:
    '¿y cuál fue su última compra?'.
    """
    if not history:
        return ""

    user_messages = [
        str(message.get("content", "")).strip()
        for message in history[-max_messages:]
        if message.get("role") == "user"
        and str(message.get("content", "")).strip()
    ]

    return " ".join(user_messages)


def retrieve_relevant_rows(
    question: str,
    df: pd.DataFrame,
    history: list[dict[str, Any]] | None = None,
    max_rows: int = 150,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Selecciona las filas más relacionadas con la pregunta.

    Utiliza la pregunta actual y, cuando sea necesario,
    el historial reciente de la conversación.
    """
    empty_entities: dict[str, Any] = {
        "clientes": [],
        "productos": [],
        "meses": [],
        "anios": [],
        "filtro_especifico": False,
        "contexto_parcial": False,
        "registros_encontrados": 0,
        "registros_enviados": 0,
    }

    if df.empty:
        return df.copy(), empty_entities

    current_text = question
    expanded_text = (
        f"{history_to_text(history)} {question}".strip()
    )

    client_values = (
        df["cliente"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    product_values = (
        df["producto"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Primero busca entidades en la pregunta actual.
    clients = find_mentions(
        current_text,
        client_values,
    )

    products = find_mentions(
        current_text,
        product_values,
    )

    months = detect_months(current_text)
    years = detect_years(current_text)

    # Si no encuentra entidades, revisa el historial.
    if not clients:
        clients = find_mentions(
            expanded_text,
            client_values,
        )

    if not products:
        products = find_mentions(
            expanded_text,
            product_values,
        )

    if not months:
        months = detect_months(expanded_text)

    if not years:
        years = detect_years(expanded_text)

    selected = df.copy()

    if clients:
        selected = selected[
            selected["cliente"].isin(clients)
        ]

    if products:
        selected = selected[
            selected["producto"].isin(products)
        ]

    if months:
        selected = selected[
            selected["fecha"].dt.month.isin(months)
        ]

    if years:
        selected = selected[
            selected["fecha"].dt.year.isin(years)
        ]

    specific_filter_applied = bool(
        clients or products or months or years
    )

    selected = selected.sort_values(
        "fecha",
        ascending=False,
    )

    original_count = int(len(selected))
    context_is_partial = original_count > max_rows

    if context_is_partial:
        selected = selected.head(max_rows)

    entities: dict[str, Any] = {
        "clientes": clients,
        "productos": products,
        "meses": months,
        "anios": years,
        "filtro_especifico": specific_filter_applied,
        "contexto_parcial": context_is_partial,
        "registros_encontrados": original_count,
        "registros_enviados": int(len(selected)),
    }

    return selected.copy(), entities


# -----------------------------------------------------------------------------
# Construcción del contexto verificado para el modelo
# -----------------------------------------------------------------------------


def build_specific_metrics(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Calcula métricas verificadas sobre las filas recuperadas
    para responder la consulta.
    """
    metrics = kpis(df)

    if df.empty:
        return {
            **metrics,
            "fecha_inicial": None,
            "fecha_final": None,
            "ultima_compra": None,
            "venta_maxima": 0.0,
            "venta_minima": 0.0,
        }

    return {
        **metrics,
        "fecha_inicial": (
            df["fecha"].min().strftime("%Y-%m-%d")
        ),
        "fecha_final": (
            df["fecha"].max().strftime("%Y-%m-%d")
        ),
        "ultima_compra": (
            df["fecha"].max().strftime("%Y-%m-%d")
        ),
        "venta_maxima": float(df["valor"].max()),
        "venta_minima": float(df["valor"].min()),
    }


def prepare_records_for_ai(
    df: pd.DataFrame,
) -> str:
    """
    Convierte las filas relevantes en CSV estructurado
    para enviarlas al modelo.
    """
    if df.empty:
        return (
            "No se encontraron registros relacionados "
            "con la consulta."
        )

    available_columns = [
        column
        for column in AI_COLUMNS
        if column in df.columns
    ]

    records = df[available_columns].copy()

    if "fecha" in records.columns:
        records["fecha"] = (
            records["fecha"].dt.strftime("%Y-%m-%d")
        )

        records = records.sort_values(
            "fecha",
            ascending=False,
        )

    return records.to_csv(index=False)


def dashboard_reference() -> str:
    """
    Describe las secciones reales de la aplicación para
    que el modelo indique dónde verificar la información.
    """
    return """
SECCIONES DISPONIBLES PARA VERIFICACIÓN:

- Resumen Ejecutivo:
  indicadores generales, evolución mensual e insights.

- Análisis de Clientes:
  ventas, registros, ticket y participación por cliente.

- Análisis de Productos:
  ventas, registros, ticket y participación por producto.

- Asistente IA:
  consultas conversacionales basadas en los datos filtrados.

- Informe y Descargas:
  consolidado y archivos exportables.

REGLA DE TRAZABILIDAD:

- Solo menciona una sección cuando la cifra pueda verificarse allí.
- Si el dato proviene de filas específicas y no de un gráfico,
  indica que puede comprobarse en el consolidado disponible
  en Informe y Descargas.
""".strip()


def ai_context(
    df: pd.DataFrame,
    question: str,
    history: list[dict[str, Any]] | None = None,
    max_rows: int = 150,
) -> str:
    """
    Construye el contexto dinámico que recibe el modelo.

    Incluye:

    - métricas generales verificadas por Python;
    - métricas específicas de los registros relevantes;
    - serie mensual;
    - rankings de clientes y productos;
    - filas relacionadas con la pregunta;
    - historial conversacional;
    - trazabilidad hacia el dashboard;
    - limitaciones de los datos.
    """
    if df.empty:
        return """
FUENTE DE VERDAD

No hay registros disponibles con los filtros actuales.

INSTRUCCIÓN

No es posible realizar cálculos ni emitir conclusiones
numéricas. Indica al usuario que debe modificar los filtros
o cargar datos válidos.
""".strip()

    general_metrics = kpis(df)

    monthly_table = (
        monthly(df)
        .tail(18)
        .copy()
    )

    clients_table = (
        by_dimension(df, "cliente")
        .head(20)
        .copy()
    )

    products_table = (
        by_dimension(df, "producto")
        .head(20)
        .copy()
    )

    relevant_rows, entities = retrieve_relevant_rows(
        question=question,
        df=df,
        history=history,
        max_rows=max_rows,
    )

    specific_metrics = build_specific_metrics(
        relevant_rows
    )

    if not monthly_table.empty:
        monthly_table["periodo"] = (
            monthly_table["periodo"].dt.strftime(
                "%Y-%m"
            )
        )

    context_parts = [
        "FUENTE DE VERDAD",

        (
            "Toda la información siguiente fue obtenida "
            "del conjunto de datos cargado y calculada por "
            "Python. Las métricas verificadas tienen prioridad "
            "sobre cualquier inferencia del modelo."
        ),

        "\nPREGUNTA ACTUAL",

        question,

        "\nFILTROS GENERALES ACTIVOS",

        (
            f"Periodo disponible: "
            f"{df['fecha'].min():%Y-%m-%d} a "
            f"{df['fecha'].max():%Y-%m-%d}"
        ),

        f"Registros disponibles: {len(df)}",

        "\nMÉTRICAS GENERALES VERIFICADAS POR PYTHON",

        (
            f"Ventas totales: "
            f"{general_metrics['ventas']:.2f}"
        ),

        (
            f"Registros de venta: "
            f"{general_metrics['pedidos']}"
        ),

        (
            f"Clientes únicos: "
            f"{general_metrics['clientes']}"
        ),

        (
            f"Productos únicos: "
            f"{general_metrics['productos']}"
        ),

        (
            f"Ticket promedio por registro: "
            f"{general_metrics['ticket']:.2f}"
        ),

        "\nENTIDADES DETECTADAS EN LA CONSULTA",

        (
            f"Clientes: "
            f"{entities['clientes'] or 'ninguno'}"
        ),

        (
            f"Productos: "
            f"{entities['productos'] or 'ninguno'}"
        ),

        (
            f"Meses: "
            f"{entities['meses'] or 'ninguno'}"
        ),

        (
            f"Años: "
            f"{entities['anios'] or 'ninguno'}"
        ),

        (
            "Se aplicó un filtro específico para recuperar "
            "registros."
            if entities["filtro_especifico"]
            else
            "No se detectó una entidad específica; se incluyó "
            "una muestra de los registros más recientes."
        ),

        "\nMÉTRICAS DE LOS REGISTROS RELEVANTES",

        (
            f"Ventas: "
            f"{specific_metrics['ventas']:.2f}"
        ),

        (
            f"Registros de venta: "
            f"{specific_metrics['pedidos']}"
        ),

        (
            f"Clientes únicos: "
            f"{specific_metrics['clientes']}"
        ),

        (
            f"Productos únicos: "
            f"{specific_metrics['productos']}"
        ),

        (
            f"Ticket promedio por registro: "
            f"{specific_metrics['ticket']:.2f}"
        ),

        (
            f"Fecha inicial: "
            f"{specific_metrics['fecha_inicial']}"
        ),

        (
            f"Fecha final: "
            f"{specific_metrics['fecha_final']}"
        ),

        (
            f"Última fecha registrada: "
            f"{specific_metrics['ultima_compra']}"
        ),

        (
            f"Valor máximo por registro: "
            f"{specific_metrics['venta_maxima']:.2f}"
        ),

        (
            f"Valor mínimo por registro: "
            f"{specific_metrics['venta_minima']:.2f}"
        ),

        "\nSERIE MENSUAL VERIFICADA",

        monthly_table.to_csv(index=False),

        "\nPRINCIPALES CLIENTES",

        clients_table.to_csv(index=False),

        "\nPRINCIPALES PRODUCTOS",

        products_table.to_csv(index=False),

        "\nREGISTROS RELEVANTES PARA LA PREGUNTA",

        prepare_records_for_ai(relevant_rows),

        "\nCOBERTURA DEL CONTEXTO",

        (
            f"Registros encontrados: "
            f"{entities['registros_encontrados']}"
        ),

        (
            f"Registros enviados al modelo: "
            f"{entities['registros_enviados']}"
        ),

        (
            "El listado de registros es parcial porque "
            f"superó el límite de {max_rows} filas. No "
            "presentes la muestra como si fuera la totalidad."
            if entities["contexto_parcial"]
            else
            "Se enviaron todos los registros encontrados."
        ),

        "\nTRAZABILIDAD EN EL DASHBOARD",

        dashboard_reference(),

        "\nLIMITACIONES DEL ARCHIVO",

        (
            "El conjunto no contiene necesariamente costos, "
            "cantidades, márgenes, descuentos, vendedores, "
            "regiones, canales ni un identificador único "
            "de pedido."
        ),

        (
            "La métrica 'pedidos' de la interfaz corresponde "
            "al número de filas o registros de venta mientras "
            "no exista un identificador único de pedido. En las "
            "respuestas utiliza preferentemente la expresión "
            "'registros de venta'."
        ),
    ]

    return "\n".join(
        str(part)
        for part in context_parts
    )
