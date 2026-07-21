from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd

from src.config import COLUMN_ALIASES, MAX_ROWS, REQUIRED_COLUMNS
from src.security import sanitize_text


@dataclass
class FileResult:
    name: str
    dataframe: pd.DataFrame | None
    errors: list[str]
    warnings: list[str]
    raw_rows: int = 0
    valid_rows: int = 0


def _normalized_label(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text


def infer_column_mapping(columns: list[str]) -> dict[str, str]:
    normalized = {_normalized_label(col): col for col in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = {_normalized_label(alias) for alias in aliases} | {_normalized_label(canonical)}
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding, sep=None, engine="python")
        except Exception as exc:  # try safe fallbacks
            last_error = exc
    raise ValueError(f"No fue posible leer el CSV: {last_error}")


def _parse_currency(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,\.\-]", "", regex=True)
        .str.strip()
    )

    def convert(value: str) -> float | None:
        if not value or value in {"-", ".", ","}:
            return None
        # Colombian/European: 1.234.567,89; US: 1,234,567.89
        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")
        elif "," in value:
            parts = value.split(",")
            value = value.replace(",", ".") if len(parts[-1]) in (1, 2) else value.replace(",", "")
        elif value.count(".") > 1:
            value = value.replace(".", "")
        try:
            return float(value)
        except ValueError:
            return None

    return cleaned.map(convert)


def validate_and_normalize(name: str, content: bytes) -> FileResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        raw = _read_csv_bytes(content)
    except Exception as exc:
        return FileResult(name, None, [str(exc)], [])

    if raw.empty:
        return FileResult(name, None, ["El archivo está vacío."], [], raw_rows=0)
    if len(raw) > MAX_ROWS:
        return FileResult(name, None, [f"Supera el máximo de {MAX_ROWS:,} filas."], [], raw_rows=len(raw))

    mapping = infer_column_mapping([str(c) for c in raw.columns])
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        return FileResult(
            name,
            None,
            ["Faltan columnas reconocibles: " + ", ".join(missing)],
            ["Columnas detectadas: " + ", ".join(map(str, raw.columns))],
            raw_rows=len(raw),
        )

    df = raw[[mapping[c] for c in REQUIRED_COLUMNS]].copy()
    df.columns = list(REQUIRED_COLUMNS)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=False)
    df["valor"] = _parse_currency(df["valor"])
    for col in ("cliente", "producto"):
        df[col] = df[col].map(lambda x: sanitize_text(x, 180))
        df.loc[df[col].eq(""), col] = pd.NA

    invalid_mask = (
        df["fecha"].isna()
        | df["valor"].isna()
        | (df["valor"] < 0)
        | df["cliente"].isna()
        | df["producto"].isna()
    )
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        warnings.append(f"Se excluyeron {invalid_count} filas inválidas o incompletas.")
    df = df.loc[~invalid_mask].copy()
    if df.empty:
        return FileResult(name, None, ["No quedaron filas válidas después de la validación."], warnings, len(raw), 0)

    df["valor"] = df["valor"].astype(float)
    df["archivo_origen"] = sanitize_text(name, 150)
    df["huella"] = df.apply(
        lambda row: hashlib.sha256(
            f"{row.fecha.date()}|{row.cliente.casefold()}|{row.producto.casefold()}|{row.valor:.2f}".encode()
        ).hexdigest()[:16],
        axis=1,
    )
    exact_dupes = int(df.duplicated(subset=["fecha", "cliente", "producto", "valor"]).sum())
    if exact_dupes:
        warnings.append(f"Se detectaron {exact_dupes} duplicados exactos dentro del archivo; se conservaron para revisión.")
    return FileResult(name, df, errors, warnings, len(raw), len(df))


def consolidate(results: list[FileResult], remove_duplicates: bool = False) -> pd.DataFrame:
    frames = [r.dataframe for r in results if r.dataframe is not None]
    if not frames:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "archivo_origen", "huella"])
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > MAX_ROWS:
        raise ValueError(f"El consolidado supera el máximo de {MAX_ROWS:,} filas.")
    if remove_duplicates:
        combined = combined.drop_duplicates(subset=["fecha", "cliente", "producto", "valor"], keep="first")
    return combined.sort_values("fecha").reset_index(drop=True)


def load_demo(path) -> FileResult:
    return validate_and_normalize(path.name, path.read_bytes())
