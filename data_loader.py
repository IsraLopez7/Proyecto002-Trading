"""
data_loader.py
Módulo para cargar y preparar datos de BTCUSDT
"""

import pandas as pd
import numpy as np

# =========================
# Helpers
# =========================

def _normalize_columns(cols: pd.Index) -> pd.Index:
    return (
        cols.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("__", "_", regex=False)
    )

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """
    Parsea una serie a datetime (UTC) soportando:
    - cadenas (ISO/mixed)
    - epoch en segundos o milisegundos
    """
    dt = pd.to_datetime(s, errors='coerce', utc=True)
    ok_ratio = dt.notna().mean()

    if ok_ratio >= 0.5:
        return dt

    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notna().mean() < 0.5:
        return dt

    median_val = s_num.dropna().median()
    unit = 'ms' if median_val and median_val > 1e11 else 's'
    dt2 = pd.to_datetime(s_num, unit=unit, errors='coerce', utc=True)
    return dt2 if dt2.notna().mean() > ok_ratio else dt

def _find_header_row(filepath: str, max_scan: int = 300) -> int:
    """
    Devuelve el índice (0-based) de la línea que parece cabecera real (contiene OHLC/fecha).
    Si no la encuentra, retorna 0 (fallback).
    """
    def is_header_line(line: str) -> bool:
        low = line.strip().lower()
        if not low:
            return False
        for sep in [",", ";", "\t", "|"]:
            parts = [p.strip().lower() for p in line.split(sep)]
            if len(parts) >= 5:
                score = sum(any(k in p for k in ["date","time","timestamp","open","high","low","close"]) for p in parts)
                if score >= 4:
                    return True
        return False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_scan:
                break
            if is_header_line(line):
                return i
    return 0

# =========================
# API principal
# =========================

def load_data(filepath='Binance_BTCUSDT_1h.csv', datetime_col: str | None = None) -> pd.DataFrame:
    """
    Carga y preprocesa BTCUSDT (formato CryptoDataDownload u otros):

    - Detecta la línea de cabecera real (salta URLs/comentarios previos).
    - Autodetecta separador (sep=None, engine='python').
    - Normaliza nombres de columnas.
    - Crea/parsea df['date'] (UTC) desde `datetime_col` o detectando la mejor candidata.
    - Convierte OHLC/volúmenes a numérico.
    - Ordena por fecha ascendente y limpia NaN críticos.
    """
    # 1) detectar cabecera real
    header_idx = _find_header_row(filepath)

    # 2) leer CSV de forma robusta (engine='python' no admite low_memory)
    try:
        df = pd.read_csv(
            filepath,
            skiprows=header_idx,   # salta líneas previas a la cabecera real
            header=0,              # la primera línea tras el salto es la cabecera
            sep=None,              # autodetecta separador
            engine='python',
            comment='#'
        )
    except Exception:
        # Fallback: asume coma y usa engine por defecto (C)
        df = pd.read_csv(
            filepath,
            skiprows=header_idx,
            header=0,
            sep=',',
            comment='#'
        )

    # 3) normalizar nombres
    df.columns = _normalize_columns(df.columns)

    # 4) fecha/tiempo → df['date']
    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise KeyError(f"No existe la columna temporal '{datetime_col}'. Columnas: {list(df.columns)}")
        cand = df[datetime_col]
    else:
        preferred = ['date', 'datetime', 'open_time', 'close_time', 'time', 'timestamp']
        candidates = [c for c in preferred if c in df.columns]
        if not candidates:
            candidates = [c for c in df.columns if any(k in c for k in ['date', 'time', 'timestamp'])]
        if not candidates:
            raise ValueError(
                "No se halló columna de fecha/tiempo tras detectar cabecera.\n"
                f"Columnas: {list(df.columns)}"
            )
        cand = df[candidates[0]]

    df['date'] = _parse_datetime_series(cand)
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # 5) mapear alias comunes de volumen
    if 'quote_asset_volume' in df.columns and 'volume_usdt' not in df.columns:
        df = df.rename(columns={'quote_asset_volume': 'volume_usdt'})
    if 'volume' in df.columns and 'volume_btc' not in df.columns:
        df = df.rename(columns={'volume': 'volume_btc'})

    # 6) asegurar OHLC
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas OHLC requeridas para el proyecto.\n"
            f"Faltantes: {missing}\n"
            f"Disponibles: {list(df.columns)}"
        )

    # 7) convertir numéricos clave
    for col in ['open', 'high', 'low', 'close', 'volume_btc', 'volume_usdt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 8) limpiar filas con NaN críticos
    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)
    return df

def split_data(df: pd.DataFrame, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):
    """
    Divide el dataset en train, test y validation manteniendo orden temporal.
    """
    assert np.isclose(train_ratio + test_ratio + val_ratio, 1.0), "Los ratios deben sumar 1.0"

    n = len(df)
    train_size = int(n * train_ratio)
    test_size = int(n * test_ratio)

    train_df = df.iloc[:train_size].copy()
    test_df  = df.iloc[train_size:train_size + test_size].copy()
    val_df   = df.iloc[train_size + test_size:].copy()

    return train_df, test_df, val_df

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega retornos simple y logarítmico.
    """
    df = df.copy()
    df['return'] = df['close'].pct_change().fillna(0.0)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def get_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de rango, cercanía a high/low, y cambio de volumen.
    """
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close']
    df['close_to_high'] = (df['high'] - df['close']) / df['close']
    df['close_to_low'] = (df['close'] - df['low']) / df['close']
    if 'volume_btc' in df.columns:
        df['volume_change'] = df['volume_btc'].pct_change()
    elif 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change()
    else:
        df['volume_change'] = np.nan
    return df

# =========================
# Test rápido
# =========================
if __name__ == "__main__":
    df = load_data()
    df = add_returns(df)
    df = get_price_features(df)
    train_df, test_df, val_df = split_data(df)
    print(f"✓ Registros: {len(df)} | Periodo: {df['date'].min()} → {df['date'].max()}")
    print(f"✓ Train/Test/Val: {len(train_df)}/{len(test_df)}/{len(val_df)}")
    print(f"✓ Columnas: {list(df.columns)}")