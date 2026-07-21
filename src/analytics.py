from __future__ import annotations

import re
import unicodedata

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
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def normalize_text(value: object) -> str:
    """
    Normaliza texto para comparar preguntas con clientes y productos.
    Elimina mayúsculas, tildes y espacios repetidos.
    """
    text = str(value).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", text)


def find_mentions(
    question: str,
    values: list[str],
) -> list[str]:
    """
    Identifica valores de una columna que fueron mencionados
    directamente en la pregunta.
    """
    normalized_question = normalize_text(question)
    matches: list[str] = []

    for value in values:
        normalized_value = normalize_text(value)

        if normalized_value and normalized_value in normalized_question:
            matches.append(value)

    return matches


def detect_months(question: str) -> list[int]:
    """
    Detecta nombres de meses presentes en la pregunta.
    """
    normalized_question = normalize_text(question)
    detected: list[int] = []

    for month_name, month_number in MONTH_NAME_TO_NUMBER.items():
        if month_name in normalized_question:
            detected.append(month_number)

    return detected


def detect_years(question: str) -> list[int]:
    """
    Detecta años escritos con cuatro cifras.
    """
    return [
        int(year)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", question)
    ]


def retrieve_relevant_rows(
    question: str,
    df: pd.DataFrame,
    max_rows: int = 150,
) -> tuple[pd.DataFrame, dict]:
    """
    Selecciona los registros más relacionados con la pregunta.

    Devuelve:
    - DataFrame con registros relevantes.
    - Diccionario con entidades detectadas.
    """
    if df.empty:
        return df.copy(), {
            "clientes": [],
            "productos": [],
            "meses": [],
            "anios": [],
            "contexto_parcial": False,
        }

    clients = find_mentions(
        question,
        df["cliente"].dropna().astype(str).unique().tolist(),
    )

    products = find_mentions(
        question,
        df["producto"].dropna().astype(str).unique().tolist(),
    )

    months = detect_months(question)
    years = detect_years(question)

    selected = df.copy()

    if clients:
        selected = selected[selected["cliente"].isin(clients)]

    if products:
        selected = selected[selected["producto"].isin(products)]

    if months:
        selected = selected[selected["fecha"].dt.month.isin(months)]

    if years:
        selected = selected[selected["fecha"].dt.year.isin(years)]

    # Si no se detectó ninguna entidad concreta, se usa una muestra reciente.
    specific_filter_applied = bool(
        clients or products or months or years
    )

    if not specific_filter_applied:
        selected = selected.sort_values(
            "fecha",
            ascending=False,
        )

    original_count = len(selected)
    context_is_partial = original_count > max_rows

    if context_is_partial:
        selected = selected.head(max_rows)

    entities = {
        "clientes": clients,
        "productos": products,
        "meses": months,
        "anios": years,
        "contexto_parcial": context_is_partial,
        "registros_encontrados": original_count,
        "registros_enviados": len(selected),
    }

    return selected.copy(), entities


def build_specific_metrics(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Calcula métricas verificadas para el conjunto seleccionado.
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
        "fecha_inicial": df["fecha"].min().strftime("%Y-%m-%d"),
        "fecha_final": df["fecha"].max().strftime("%Y-%m-%d"),
        "ultima_compra": df["fecha"].max().strftime("%Y-%m-%d"),
        "venta_maxima": float(df["valor"].max()),
        "venta_minima": float(df["valor"].min()),
    }


def prepare_records_for_ai(
    df: pd.DataFrame,
) -> str:
    """
    Convierte registros a CSV controlado para enviarlos al modelo.
    """
    if df.empty:
        return "No se encontraron registros relacionados."

    columns = [
        column
        for column in ["fecha", "cliente", "producto", "valor"]
        if column in df.columns
    ]

    records = df[columns].copy()

    if "fecha" in records.columns:
        records["fecha"] = records["fecha"].dt.strftime("%Y-%m-%d")

    records = records.sort_values(
        "fecha",
        ascending=False,
    )

    return records.to_csv(index=False)


def dashboard_reference() -> str:
    """
    Describe únicamente las secciones que realmente existen
    en la aplicación.
    """
    return """
SECCIONES DISPONIBLES PARA VERIFICACIÓN:
- Resumen Ejecutivo: indicadores generales, evolución mensual e insights.
- Análisis de Clientes: ventas, registros, ticket y participación por cliente.
- Análisis de Productos: ventas, registros, ticket y participación por producto.
- Asistente IA: consultas conversacionales basadas en los datos filtrados.
- Informe y Descargas: consolidado y archivos exportables.

REGLA:
Solo indica una sección cuando la cifra o análisis pueda verificarse allí.
Si el resultado proviene de registros específicos no visibles directamente
en un gráfico, indica que puede comprobarse en el consolidado exportable.
""".strip()


def ai_context(
    df: pd.DataFrame,
    question: str,
) -> str:
    """
    Construye un contexto dinámico según la pregunta del usuario.
    """
    if df.empty:
        return """
ESTADO DEL CONJUNTO:
No hay registros disponibles con los filtros actuales.

No es posible realizar cálculos ni emitir conclusiones.
""".strip()

    general_metrics = kpis(df)
    monthly_table = monthly(df).tail(18).copy()
    clients_table = by_dimension(df, "cliente").head(20)
    products_table = by_dimension(df, "producto").head(20)

    relevant_rows, entities = retrieve_relevant_rows(
        question=question,
        df=df,
        max_rows=150,
    )

    specific_metrics = build_specific_metrics(relevant_rows)

    context_parts = [
        "FUENTE DE VERDAD",
        (
            "Toda la información siguiente fue obtenida del conjunto "
            "de datos cargado y calculada por Python."
        ),

        "\nPREGUNTA ACTUAL",
        question,

        "\nFILTROS GENERALES ACTIVOS",
        f"Periodo disponible: {df['fecha'].min():%Y-%m-%d} "
        f"a {df['fecha'].max():%Y-%m-%d}",
        f"Registros disponibles: {len(df)}",

        "\nMÉTRICAS GENERALES VERIFICADAS POR PYTHON",
        f"Ventas totales: {general_metrics['ventas']:.2f}",
        f"Registros de venta: {general_metrics['pedidos']}",
        f"Clientes únicos: {general_metrics['clientes']}",
        f"Productos únicos: {general_metrics['productos']}",
        f"Ticket promedio por registro: {general_metrics['ticket']:.2f}",

        "\nENTIDADES DETECTADAS EN LA PREGUNTA",
        f"Clientes: {entities['clientes'] or 'ninguno'}",
        f"Productos: {entities['productos'] or 'ninguno'}",
        f"Meses: {entities['meses'] or 'ninguno'}",
        f"Años: {entities['anios'] or 'ninguno'}",

        "\nMÉTRICAS DE LOS REGISTROS RELEVANTES",
        f"Ventas: {specific_metrics['ventas']:.2f}",
        f"Registros: {specific_metrics['pedidos']}",
        f"Clientes: {specific_metrics['clientes']}",
        f"Productos: {specific_metrics['productos']}",
        f"Ticket promedio: {specific_metrics['ticket']:.2f}",
        f"Fecha inicial: {specific_metrics['fecha_inicial']}",
        f"Fecha final: {specific_metrics['fecha_final']}",
        f"Valor máximo por registro: {specific_metrics['venta_maxima']:.2f}",
        f"Valor mínimo por registro: {specific_metrics['venta_minima']:.2f}",

        "\nSERIE MENSUAL VERIFICADA",
        monthly_table.to_csv(index=False),

        "\nPRINCIPALES CLIENTES",
        clients_table.to_csv(index=False),

        "\nPRINCIPALES PRODUCTOS",
        products_table.to_csv(index=False),

        "\nREGISTROS RELEVANTES PARA LA PREGUNTA",
        prepare_records_for_ai(relevant_rows),

        "\nCOBERTURA DEL CONTEXTO",
        f"Registros encontrados: {entities['registros_encontrados']}",
        f"Registros enviados al modelo: {entities['registros_enviados']}",
        (
            "El listado de registros es parcial porque superó el límite."
            if entities["contexto_parcial"]
            else "Se enviaron todos los registros encontrados."
        ),

        "\nTRAZABILIDAD EN EL DASHBOARD",
        dashboard_reference(),

        "\nLIMITACIONES DEL ARCHIVO",
        (
            "El conjunto no contiene necesariamente costos, cantidades, "
            "márgenes, descuentos, vendedores, regiones, canales ni un "
            "identificador único de pedido."
        ),
        (
            "La métrica denominada pedidos corresponde actualmente al "
            "número de filas o registros de venta, salvo que el archivo "
            "incluya posteriormente un identificador único de pedido."
        ),
    ]

    return "\n".join(str(part) for part in context_parts)
