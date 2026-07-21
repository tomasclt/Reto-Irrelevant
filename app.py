from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from src.ai import ask_nvidia
from src.analytics import (
    ai_context,
    apply_filters,
    by_dimension,
    executive_insights,
    kpis,
    monthly,
)
from src.charts import (
    concentration_donut,
    horizontal_bar,
    orders_bar,
    sales_line,
)
from src.config import (
    APP_NAME,
    CSS_PATH,
    DEMO_DATA_PATH,
    MAX_CHAT_MESSAGES,
    MAX_FILES,
    MAX_QUESTION_LENGTH,
)
from src.data import (
    consolidate,
    load_demo,
    validate_and_normalize,
)
from src.exports import (
    csv_bytes,
    excel_bytes,
    pdf_bytes,
)
from src.security import sanitize_question


st.set_page_config(
    page_title=APP_NAME,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    """
    Convierte un valor numérico a formato monetario
    sin decimales.
    """
    return f"${value:,.0f}".replace(",", ".")


def init_state() -> None:
    """
    Inicializa los datos y el historial de conversación.
    """
    if "dataset" not in st.session_state:
        demo = load_demo(DEMO_DATA_PATH)

        st.session_state.dataset = demo.dataframe
        st.session_state.file_results = [demo]
        st.session_state.data_mode = (
            "Datos de demostración"
        )

    if "chat" not in st.session_state:
        st.session_state.chat = []


def get_secret() -> str | None:
    """
    Obtiene la API key desde Streamlit Secrets
    o desde una variable de entorno.
    """
    try:
        value = st.secrets.get(
            "NVIDIA_API_KEY"
        )

        if value:
            return str(value).strip()

    except Exception:
        pass

    value = os.getenv(
        "NVIDIA_API_KEY"
    )

    return (
        value.strip()
        if value
        else None
    )


init_state()


# -------------------------------------------------------------------------
# Barra lateral
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="eyebrow">'
        "Sales intelligence"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "## ◈ Reto_Irrelevant"
    )

    st.caption(
        "Datos claros. Decisiones mejores."
    )

    page = st.radio(
        "Navegación",
        [
            "Inicio",
            "Carga de datos",
            "Dashboard",
            "Clientes",
            "Productos",
            "Asistente IA",
            "Informe y descargas",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    df_all = st.session_state.dataset

    st.caption(
        f"Fuente activa: "
        f"{st.session_state.data_mode}"
    )

    if not df_all.empty:
        st.caption(
            f"{len(df_all):,} registros · "
            f"{df_all.fecha.min():%d/%m/%Y} a "
            f"{df_all.fecha.max():%d/%m/%Y}"
        )
    else:
        st.caption(
            "Sin registros"
        )


# -------------------------------------------------------------------------
# Filtros globales
# -------------------------------------------------------------------------

filtered = df_all

if (
    page != "Carga de datos"
    and not df_all.empty
):
    with st.expander(
        "Filtros globales",
        expanded=False,
    ):
        c1, c2, c3, c4 = st.columns(
            [1, 1, 1.3, 1.3]
        )

        min_date = (
            df_all.fecha.min().date()
        )

        max_date = (
            df_all.fecha.max().date()
        )

        start = c1.date_input(
            "Desde",
            min_date,
            min_value=min_date,
            max_value=max_date,
        )

        end = c2.date_input(
            "Hasta",
            max_date,
            min_value=min_date,
            max_value=max_date,
        )

        clients = c3.multiselect(
            "Clientes",
            sorted(
                df_all.cliente
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
        )

        products = c4.multiselect(
            "Productos",
            sorted(
                df_all.producto
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
        )

        if start > end:
            st.error(
                "La fecha inicial no puede "
                "ser posterior a la fecha final."
            )
            st.stop()

        filtered = apply_filters(
            df=df_all,
            start=start,
            end=end,
            clients=clients,
            products=products,
        )


# -------------------------------------------------------------------------
# Inicio
# -------------------------------------------------------------------------

if page == "Inicio":
    st.markdown(
        '<div class="eyebrow">'
        "Centro de inteligencia comercial"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">'
        "Entiende tus ventas.<br>"
        "<span>Decide con evidencia.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtle">'
        "Un tablero gerencial y asistente "
        "conversacional que transforma archivos "
        "CSV en señales claras de negocio."
        "</p>",
        unsafe_allow_html=True,
    )

    st.write("")

    metrics = kpis(filtered)

    columns = st.columns(4)

    labels = [
        "Ventas",
        "Registros de venta",
        "Clientes activos",
        "Ticket promedio",
    ]

    values = [
        money(metrics["ventas"]),
        f'{metrics["pedidos"]:,}',
        f'{metrics["clientes"]:,}',
        money(metrics["ticket"]),
    ]

    for (
        column,
        label,
        value,
    ) in zip(
        columns,
        labels,
        values,
    ):
        column.markdown(
            '<div class="card">'
            f'<div class="metric-label">'
            f"{label}"
            "</div>"
            f'<div class="metric-value">'
            f"{value}"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns(
        [1.35, 0.65]
    )

    with left:
        st.markdown(
            "### Resumen ejecutivo"
        )

        for item in executive_insights(
            filtered
        ):
            st.markdown(
                f'<div class="insight">'
                f"{item}"
                "</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            "### Estado del negocio"
        )

        month_table = monthly(
            filtered
        )

        status = (
            "Información inicial"
        )

        description = (
            "Carga más periodos para "
            "detectar tendencias."
        )

        if len(month_table) >= 2:
            sales_down = (
                month_table.iloc[-1].ventas
                < month_table.iloc[-2].ventas
            )

            ticket_down = (
                month_table.iloc[-1].ticket
                < month_table.iloc[-2].ticket
            )

            status = (
                "Atención"
                if sales_down
                and ticket_down
                else "Estable"
            )

            description = (
                "Las ventas y el ticket "
                "requieren revisión."
                if status == "Atención"
                else
                "No se observan dos señales "
                "negativas simultáneas."
            )

        st.markdown(
            '<div class="card">'
            '<div class="metric-label">'
            "Diagnóstico"
            "</div>"
            '<div class="metric-value">'
            f"{status}"
            "</div>"
            '<p class="subtle">'
            f"{description}"
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.info(
            "El estado es una señal "
            "orientativa basada en reglas, "
            "no una predicción financiera."
        )


# -------------------------------------------------------------------------
# Carga de datos
# -------------------------------------------------------------------------

elif page == "Carga de datos":
    st.markdown(
        "# Carga y consolidación"
    )

    st.write(
        "Carga varios CSV de diferentes "
        "periodos. La aplicación valida, "
        "normaliza y consolida los registros "
        "dentro de esta sesión."
    )

    uploaded = st.file_uploader(
        "Selecciona hasta 12 archivos CSV",
        type=["csv"],
        accept_multiple_files=True,
        help=(
            "Máximo 10 MB por archivo. "
            "Columnas requeridas: fecha, "
            "cliente, producto y valor."
        ),
    )

    remove_duplicates = st.checkbox(
        "Excluir duplicados exactos "
        "al consolidar",
        value=False,
        help=(
            "Se considera duplicado exacto "
            "la coincidencia de fecha, cliente, "
            "producto y valor."
        ),
    )

    if uploaded:
        if len(uploaded) > MAX_FILES:
            st.error(
                f"Seleccionaste {len(uploaded)} "
                f"archivos. El máximo permitido "
                f"es {MAX_FILES}."
            )

        else:
            results = [
                validate_and_normalize(
                    file.name,
                    file.getvalue(),
                )
                for file in uploaded
            ]

            summary = pd.DataFrame(
                [
                    {
                        "archivo": result.name,
                        "filas_originales": (
                            result.raw_rows
                        ),
                        "filas_validas": (
                            result.valid_rows
                        ),
                        "estado": (
                            "Aceptado"
                            if result.dataframe
                            is not None
                            else "Rechazado"
                        ),
                        "observaciones": (
                            " | ".join(
                                result.errors
                                + result.warnings
                            )
                        ),
                    }
                    for result in results
                ]
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            valid_results = [
                result
                for result in results
                if result.dataframe
                is not None
            ]

            if valid_results:
                try:
                    candidate = consolidate(
                        results,
                        remove_duplicates=(
                            remove_duplicates
                        ),
                    )

                    st.success(
                        "Consolidado listo: "
                        f"{len(candidate):,} "
                        "registros válidos."
                    )

                    st.dataframe(
                        candidate.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "Usar este consolidado",
                        type="primary",
                    ):
                        st.session_state.dataset = (
                            candidate
                        )

                        st.session_state.file_results = (
                            results
                        )

                        st.session_state.data_mode = (
                            f"{len(valid_results)} "
                            "archivo(s) válido(s)"
                        )

                        st.session_state.chat = []

                        st.success(
                            "Datos activados."
                        )

                        st.rerun()

                except ValueError as exc:
                    st.error(
                        str(exc)
                    )

    c1, c2 = st.columns(2)

    if c1.button(
        "Restaurar datos de demostración"
    ):
        demo = load_demo(
            DEMO_DATA_PATH
        )

        st.session_state.dataset = (
            demo.dataframe
        )

        st.session_state.file_results = [
            demo
        ]

        st.session_state.data_mode = (
            "Datos de demostración"
        )

        st.session_state.chat = []

        st.rerun()

    with c2:
        st.download_button(
            "Descargar plantilla CSV",
            data=(
                "fecha,cliente,producto,valor\n"
                "2026-01-01,Cliente ejemplo,"
                "Producto ejemplo,100000\n"
            ).encode("utf-8-sig"),
            file_name=(
                "plantilla_pedidos.csv"
            ),
            mime="text/csv",
        )


# -------------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------------

elif page == "Dashboard":
    st.markdown(
        "# Dashboard gerencial"
    )

    if filtered.empty:
        st.warning(
            "No hay datos con los "
            "filtros seleccionados."
        )
        st.stop()

    metrics = kpis(
        filtered
    )

    columns = st.columns(4)

    columns[0].metric(
        "Ventas",
        money(metrics["ventas"]),
    )

    columns[1].metric(
        "Registros de venta",
        f'{metrics["pedidos"]:,}',
    )

    columns[2].metric(
        "Clientes",
        f'{metrics["clientes"]:,}',
    )

    columns[3].metric(
        "Ticket promedio",
        money(metrics["ticket"]),
    )

    month_table = monthly(
        filtered
    )

    c1, c2 = st.columns(2)

    c1.plotly_chart(
        sales_line(
            month_table
        ),
        use_container_width=True,
    )

    c2.plotly_chart(
        orders_bar(
            month_table
        ),
        use_container_width=True,
    )

    clients_table = by_dimension(
        filtered,
        "cliente",
    )

    products_table = by_dimension(
        filtered,
        "producto",
    )

    c3, c4 = st.columns(2)

    c3.plotly_chart(
        horizontal_bar(
            clients_table,
            "cliente",
            "Principales clientes",
        ),
        use_container_width=True,
    )

    c4.plotly_chart(
        horizontal_bar(
            products_table,
            "producto",
            "Principales productos",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        concentration_donut(
            products_table,
            "producto",
        ),
        use_container_width=True,
    )


# -------------------------------------------------------------------------
# Clientes
# -------------------------------------------------------------------------

elif page == "Clientes":
    st.markdown(
        "# Inteligencia de clientes"
    )

    table = by_dimension(
        filtered,
        "cliente",
    )

    if table.empty:
        st.warning(
            "No hay datos disponibles."
        )
        st.stop()

    st.dataframe(
        table.style.format(
            {
                "ventas": "${:,.0f}",
                "ticket": "${:,.0f}",
                "participacion": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox(
        "Explorar cliente",
        table.cliente.tolist(),
    )

    client_df = filtered[
        filtered.cliente == selected
    ]

    client_metrics = kpis(
        client_df
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ventas",
        money(
            client_metrics["ventas"]
        ),
    )

    c2.metric(
        "Registros de venta",
        client_metrics["pedidos"],
    )

    c3.metric(
        "Ticket promedio",
        money(
            client_metrics["ticket"]
        ),
    )

    st.plotly_chart(
        sales_line(
            monthly(client_df)
        ),
        use_container_width=True,
    )

    st.dataframe(
        by_dimension(
            client_df,
            "producto",
        ).style.format(
            {
                "ventas": "${:,.0f}",
                "ticket": "${:,.0f}",
                "participacion": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------------
# Productos
# -------------------------------------------------------------------------

elif page == "Productos":
    st.markdown(
        "# Inteligencia de productos"
    )

    table = by_dimension(
        filtered,
        "producto",
    )

    if table.empty:
        st.warning(
            "No hay datos disponibles."
        )
        st.stop()

    st.dataframe(
        table.style.format(
            {
                "ventas": "${:,.0f}",
                "ticket": "${:,.0f}",
                "participacion": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox(
        "Explorar producto",
        table.producto.tolist(),
    )

    product_df = filtered[
        filtered.producto == selected
    ]

    product_metrics = kpis(
        product_df
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Ventas",
        money(
            product_metrics["ventas"]
        ),
    )

    c2.metric(
        "Registros de venta",
        product_metrics["pedidos"],
    )

    c3.metric(
        "Clientes compradores",
        product_metrics["clientes"],
    )

    st.plotly_chart(
        sales_line(
            monthly(product_df)
        ),
        use_container_width=True,
    )

    st.dataframe(
        by_dimension(
            product_df,
            "cliente",
        ).style.format(
            {
                "ventas": "${:,.0f}",
                "ticket": "${:,.0f}",
                "participacion": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------------
# Asistente IA
# -------------------------------------------------------------------------

elif page == "Asistente IA":
    st.markdown(
        "# Asistente de inteligencia comercial"
    )

    st.caption(
        "Python calcula y verifica las "
        "métricas; NVIDIA interpreta los "
        "resultados y los registros relevantes."
    )

    api_key = get_secret()

    if not api_key:
        st.warning(
            "La API key de NVIDIA no está "
            "configurada. El dashboard funciona, "
            "pero el chat está deshabilitado."
        )

        st.code(
            'NVIDIA_API_KEY = "nvapi-TU_CLAVE_REAL"',
            language="toml",
        )

    prompts = [
        "¿Qué cambió en el último mes?",
        "¿Cuál es el producto más importante?",
        "¿Existe concentración de clientes?",
        "Dame un resumen ejecutivo del periodo.",
    ]

    prompt_columns = st.columns(2)

    chosen = None

    for index, prompt in enumerate(
        prompts
    ):
        clicked = prompt_columns[
            index % 2
        ].button(
            prompt,
            key=f"prompt_{index}",
        )

        if clicked:
            chosen = prompt

    for message in st.session_state.chat:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    typed = st.chat_input(
        "Pregunta sobre ventas, clientes, "
        "productos o periodos",
        max_chars=MAX_QUESTION_LENGTH,
        disabled=not bool(api_key),
    )

    question = chosen or typed

    if question and api_key:
        try:
            question = sanitize_question(
                question,
                MAX_QUESTION_LENGTH,
            )

        except Exception as exc:
            st.error(
                "La pregunta no pudo validarse.\n\n"
                f"Tipo: {type(exc).__name__}\n\n"
                f"Detalle: {str(exc)[:500]}"
            )
            st.stop()

        conversation_history = (
            st.session_state.chat.copy()
        )

        st.session_state.chat.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "Analizando datos..."
            ):
                try:
                    context = ai_context(
                        df=filtered,
                        question=question,
                        history=(
                            conversation_history
                        ),
                        max_rows=100,
                    )

                    answer = ask_nvidia(
                        api_key=api_key,
                        question=question,
                        context=context,
                        history=(
                            conversation_history
                        ),
                    )

                except AuthenticationError as exc:
                    answer = (
                        "NVIDIA rechazó la clave API. "
                        "Revisa el secreto "
                        "`NVIDIA_API_KEY` en Streamlit.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                except RateLimitError as exc:
                    answer = (
                        "NVIDIA rechazó temporalmente "
                        "la solicitud por límite de uso "
                        "o cuota.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                except APITimeoutError as exc:
                    answer = (
                        "La consulta superó el tiempo "
                        "máximo de espera. Prueba una "
                        "pregunta más específica.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                except APIConnectionError as exc:
                    answer = (
                        "No fue posible establecer "
                        "conexión con NVIDIA.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                except APIStatusError as exc:
                    status_code = getattr(
                        exc,
                        "status_code",
                        "desconocido",
                    )

                    request_id = getattr(
                        exc,
                        "request_id",
                        None,
                    )

                    answer = (
                        "NVIDIA respondió con un error "
                        f"HTTP {status_code}.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                    if request_id:
                        answer += (
                            "\n\nID de solicitud: "
                            f"{request_id}"
                        )

                except ValueError as exc:
                    answer = (
                        "No fue posible procesar "
                        "la consulta.\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                except Exception as exc:
                    answer = (
                        "Se produjo un error interno "
                        "al preparar o procesar la "
                        "consulta.\n\n"
                        f"Tipo: "
                        f"{type(exc).__name__}\n\n"
                        f"Detalle técnico: "
                        f"{str(exc)[:700]}"
                    )

                st.markdown(
                    answer
                )

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.chat = (
            st.session_state.chat[
                -MAX_CHAT_MESSAGES:
            ]
        )

        st.rerun()

    if st.button(
        "Limpiar conversación"
    ):
        st.session_state.chat = []
        st.rerun()


# -------------------------------------------------------------------------
# Informe y descargas
# -------------------------------------------------------------------------

elif page == "Informe y descargas":
    st.markdown(
        "# Informe ejecutivo "
        "y exportaciones"
    )

    if filtered.empty:
        st.warning(
            "No hay datos para exportar."
        )
        st.stop()

    for item in executive_insights(
        filtered
    ):
        st.markdown(
            f'<div class="insight">'
            f"{item}"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")

    c1, c2, c3 = st.columns(3)

    c1.download_button(
        "Descargar CSV",
        data=csv_bytes(
            filtered
        ),
        file_name=(
            "ventas_filtradas.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    c2.download_button(
        "Descargar Excel",
        data=excel_bytes(
            filtered
        ),
        file_name=(
            "reporte_ventas.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    c3.download_button(
        "Descargar PDF",
        data=pdf_bytes(
            filtered
        ),
        file_name=(
            "informe_ejecutivo.pdf"
        ),
        mime="application/pdf",
        use_container_width=True,
    )

    st.markdown(
        "### Vista de datos exportados"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )
