from __future__ import annotations

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.analytics import by_dimension, executive_insights, kpis, monthly


def csv_bytes(df: pd.DataFrame) -> bytes:
    export = df.copy()
    if "fecha" in export:
        export["fecha"] = export["fecha"].dt.strftime("%Y-%m-%d")
    return export.to_csv(index=False).encode("utf-8-sig")


def excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        metrics = pd.DataFrame([kpis(df)])
        metrics.to_excel(writer, sheet_name="Resumen", index=False, startrow=2)
        monthly(df).to_excel(writer, sheet_name="Ventas mensuales", index=False)
        by_dimension(df, "cliente").to_excel(writer, sheet_name="Clientes", index=False)
        by_dimension(df, "producto").to_excel(writer, sheet_name="Productos", index=False)
        df.to_excel(writer, sheet_name="Datos consolidados", index=False)
        workbook = writer.book
        title = workbook.add_format({"bold": True, "font_size": 18, "font_color": "#8A6A24"})
        header = workbook.add_format({"bold": True, "bg_color": "#D6B36A", "font_color": "#111111", "border": 0})
        money = workbook.add_format({"num_format": "$#,##0"})
        percent = workbook.add_format({"num_format": "0.0%"})
        writer.sheets["Resumen"].write("A1", "Reto_Irrelevant - Resumen ejecutivo", title)
        for sheet_name, ws in writer.sheets.items():
            ws.freeze_panes(1 if sheet_name != "Resumen" else 3, 0)
            ws.set_default_row(19)
            ws.set_column(0, 0, 24)
            ws.set_column(1, 8, 18)
            # Style detected header row
            header_row = 2 if sheet_name == "Resumen" else 0
            if sheet_name == "Resumen":
                for col, value in enumerate(metrics.columns): ws.write(header_row, col, value, header)
            else:
                source = {"Ventas mensuales": monthly(df), "Clientes": by_dimension(df, "cliente"), "Productos": by_dimension(df, "producto"), "Datos consolidados": df}[sheet_name]
                for col, value in enumerate(source.columns): ws.write(header_row, col, value, header)
            ws.set_column("B:B", 18, money)
            if sheet_name in {"Clientes", "Productos"}: ws.set_column("E:E", 15, percent)
    return output.getvalue()


def pdf_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleGold", parent=styles["Title"], textColor=colors.HexColor("#8A6A24"), fontSize=20, leading=24, alignment=TA_LEFT)
    heading = ParagraphStyle("HeadingGold", parent=styles["Heading2"], textColor=colors.HexColor("#8A6A24"), spaceBefore=10)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    story = [Paragraph("Reto_Irrelevant", title), Paragraph("Informe ejecutivo de ventas", styles["Heading3"]), Spacer(1, 8)]
    if df.empty:
        story.append(Paragraph("No hay datos disponibles para el informe.", body))
    else:
        story.append(Paragraph(f"Periodo analizado: {df.fecha.min():%d/%m/%Y} - {df.fecha.max():%d/%m/%Y}", body))
        metric = kpis(df)
        table_data = [
            ["Ventas", "Pedidos", "Clientes", "Ticket promedio"],
            [f"${metric['ventas']:,.0f}", f"{metric['pedidos']:,}", f"{metric['clientes']:,}", f"${metric['ticket']:,.0f}"],
        ]
        table = Table(table_data, colWidths=[42*mm]*4)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D6B36A")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111111")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#BBBBBB")),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 7),
        ]))
        story += [Spacer(1, 10), table, Paragraph("Hallazgos principales", heading)]
        for insight in executive_insights(df):
            story.append(Paragraph(f"• {insight}", body))
            story.append(Spacer(1, 3))
        story.append(Paragraph("Principales clientes", heading))
        top_clients = by_dimension(df, "cliente").head(8)
        cdata = [["Cliente", "Ventas", "Pedidos", "Participación"]] + [
            [r.cliente, f"${r.ventas:,.0f}", str(int(r.pedidos)), f"{r.participacion:.1%}"] for r in top_clients.itertuples()
        ]
        ctable = Table(cdata, colWidths=[72*mm, 38*mm, 24*mm, 32*mm], repeatRows=1)
        ctable.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D6B36A")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#BBBBBB")),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (1,1), (-1,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(ctable)
        story.append(Paragraph("Alcance y limitaciones", heading))
        story.append(Paragraph("El informe analiza facturación y frecuencia de pedidos. No permite calcular margen, unidades, descuentos ni causas comerciales porque esas variables no están presentes en los archivos.", body))
    doc.build(story)
    return output.getvalue()
