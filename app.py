import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import io
from pathlib import Path

# Page config
st.set_page_config(page_title="Reactive Power Purchase", layout="wide")

st.title("Reactive Power Purchase Decision Tool")
st.markdown("Upload Norwegian case CSV and PTF price CSV to compute reactive power purchase decisions.")

# Initialize session state for persistent backlog
if 'backlog_prev_total' not in st.session_state:
    st.session_state.backlog_prev_total = 0.0
if 'accumulate_backlog' not in st.session_state:
    st.session_state.accumulate_backlog = True

# =========================================================
# A) File Loading Functions
# =========================================================

def get_data_paths():
    """
    Get paths to data directories using pathlib (OS-independent).
    Returns: (base_dir, data_dir, scenario_dir)
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    scenario_dir = data_dir / "senaryo"
    return base_dir, data_dir, scenario_dir


def list_scenario_files():
    """
    List all CSV files in data/scenarios/ directory.
    Returns: list of (filename, full_path) tuples, sorted alphabetically
    """
    _, _, scenario_dir = get_data_paths()
    scenarios = []
    
    if scenario_dir.exists() and scenario_dir.is_dir():
        for csv_file in sorted(scenario_dir.glob("*.csv")):
            scenarios.append((csv_file.stem, csv_file))
    
    return scenarios


def list_ptf_files():
    """
    List all CSV files in data/ directory (excluding scenarios subfolder).
    Returns: list of (filename, full_path) tuples, sorted alphabetically
    """
    _, data_dir, scenario_dir = get_data_paths()
    ptf_files = []
    
    if data_dir.exists() and data_dir.is_dir():
        for csv_file in sorted(data_dir.glob("*.csv")):
            # Exclude files in scenarios subfolder
            if csv_file.parent != scenario_dir:
                ptf_files.append((csv_file.name, csv_file))
    
    return ptf_files


def get_default_ptf_path():
    """
    Get the default PTF file path.
    Returns: Path object or None if file doesn't exist
    """
    _, data_dir, _ = get_data_paths()
    default_file = data_dir / "Piyasa_Takas_Fiyati-20112025-20112025.csv"
    if default_file.exists():
        return default_file
    return None

def load_case(csv_file):
    """
    Load Norwegian_case CSV with columns: hour, P_total, Q_total, P_PV_max
    Accepts either a file path (string/Path) or file-like object.
    Returns: DataFrame with exactly 48 hours, with hour labels 0..47
    If only 24h, duplicate to 48h for display (but label clearly).
    """
    # Handle both file paths and file-like objects with encoding fallback
    if isinstance(csv_file, (str, Path)):
        # Try different encodings
        encodings = ['utf-8', 'cp1254', 'latin1']
        df = None
        last_error = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
        if df is None:
            # Last resort: try with error handling
            df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
    else:
        # File-like object - try to read with encoding fallback
        csv_file.seek(0)
        content = csv_file.read()
        if isinstance(content, bytes):
            encodings = ['utf-8', 'cp1254', 'latin1']
            for encoding in encodings:
                try:
                    content = content.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                content = content.decode("utf-8", errors='ignore')
            df = pd.read_csv(io.StringIO(content))
        else:
            csv_file.seek(0)
            df = pd.read_csv(csv_file)
    df = df.sort_values("hour").reset_index(drop=True)
    
    required_cols = ['hour', 'P_total', 'Q_total', 'P_PV_max']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    n_rows = len(df)
    
    # If only 24h, duplicate to 48h for display
    if n_rows == 24:
        df_dup = df.copy()
        df_dup['hour'] = df_dup['hour'] + 24
        df = pd.concat([df, df_dup], ignore_index=True)
        st.info("⚠️ Case file has 24 hours. Duplicated to 48 hours for display.")
    
    # Always take the latest 48 hours and reindex hours to 0..47
    if len(df) > 48:
        df = df.tail(48).reset_index(drop=True)
        st.info(f"⚠️ Case file has {n_rows} hours. Showing latest 48 hours only.")
    
    # Ensure hour labels are 0..47 for consistency
    df['hour'] = np.arange(48)
    
    return df


def _read_file_with_encoding_fallback(file_path):
    """
    Helper function to read a file with encoding fallback.
    Tries utf-8, then cp1254 (Turkish Windows), then latin1.
    Returns: content as string
    """
    encodings = ['utf-8', 'cp1254', 'latin1']
    last_error = None
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    
    # If all encodings fail, raise the last error
    raise ValueError(f"Could not decode file with any encoding. Last error: {last_error}")


def load_prices(csv_file):
    """
    Load price CSV - accepts EPİAŞ format or simple (hour, price_TL_per_MWh)
    Accepts either a file path (string/Path) or file-like object.
    Returns: numpy array of prices in TL/MWh, length 24 or 48 (or more)
    If only 24 prices, tile to 48. If 48 are given, use as-is.
    """
    # Handle both file paths and file-like objects
    is_file_path = isinstance(csv_file, (str, Path))
    content = None
    
    if is_file_path:
        # File path - read with encoding fallback
        try:
            content = _read_file_with_encoding_fallback(csv_file)
            lines = content.splitlines()
        except Exception as e:
            # Fallback to utf-8 directly
            with open(csv_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
    else:
        # File-like object
        csv_file.seek(0)
        content = csv_file.read()
        if isinstance(content, bytes):
            # Try to decode with fallback encodings
            encodings = ['utf-8', 'cp1254', 'latin1']
            for encoding in encodings:
                try:
                    content = content.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                content = content.decode("utf-8", errors='ignore')  # Last resort
        lines = content.splitlines()
    
    reader = csv.reader(lines)
    header = next(reader, None)
    
    prices = []
    
    # Try EPİAŞ format first (Tarih, Saat, PTF (TL/MWh), ...)
    if header and len(header) >= 3 and "PTF" in header[2]:
        # EPİAŞ format: parse price from row like ["20.11.2025", "00:00", "2.517", "01", ...]
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                # Price is in column 2 and 3: "2.517" + "," + "01" = "2.517,01"
                price_str = row[2].strip() + "," + row[3].strip()
                # Normalize: remove thousand separator, replace comma with dot
                price_str_norm = price_str.replace(".", "").replace(",", ".")
                price = float(price_str_norm)
                prices.append(price)
            except (ValueError, IndexError):
                continue
    else:
        # Simple format: try to find hour and price columns
        if is_file_path:
            # File path - read directly
            df = pd.read_csv(csv_file)
        else:
            # File-like object - reset and read
            csv_file.seek(0)
            if content:
                df = pd.read_csv(io.StringIO(content))
            else:
                df = pd.read_csv(csv_file)
        
        # Look for price column
        price_col = None
        for col in df.columns:
            if 'price' in col.lower() or 'ptf' in col.lower() or 'tl' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            # Assume second column is price
            if len(df.columns) >= 2:
                price_col = df.columns[1]
            else:
                raise ValueError("Could not find price column in CSV")
        
        prices = df[price_col].values.tolist()
    
    prices = np.array(prices, dtype=float)
    
    # If only 24 prices, tile to 48
    if len(prices) == 24:
        prices = np.tile(prices, 2)
        st.info("⚠️ Price file has 24 hours. Tiled to 48 hours.")
    
    # Always take the latest 48 prices
    if len(prices) > 48:
        prices = prices[-48:]
        st.info(f"⚠️ Price file has {len(prices)} hours. Using latest 48 hours only.")
    
    return prices


# =========================================================
# A.1) Multi-Day File Loading Functions
# =========================================================

def load_case_multiday(csv_file, horizon_days):
    """
    Load Norwegian_case CSV with columns: hour, P_total, Q_total, P_PV_max
    For multi-day horizon: returns DataFrame with 24*horizon_days hours.
    Accepts either a file path (string/Path) or file-like object.
    Returns: DataFrame with hour index 0..(24*horizon_days-1)
    If only 24h, tiles to fill horizon. If more hours, takes latest horizon_days*24.
    """
    # Handle both file paths and file-like objects with encoding fallback
    if isinstance(csv_file, (str, Path)):
        # Try different encodings
        encodings = ['utf-8', 'cp1254', 'latin1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if df is None:
            df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
    else:
        # File-like object
        csv_file.seek(0)
        content = csv_file.read()
        if isinstance(content, bytes):
            encodings = ['utf-8', 'cp1254', 'latin1']
            for encoding in encodings:
                try:
                    content = content.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                content = content.decode("utf-8", errors='ignore')
            df = pd.read_csv(io.StringIO(content))
        else:
            csv_file.seek(0)
            df = pd.read_csv(csv_file)
    
    df = df.sort_values("hour").reset_index(drop=True)
    
    required_cols = ['hour', 'P_total', 'Q_total', 'P_PV_max']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    n_rows = len(df)
    target_hours = 24 * horizon_days
    
    # If only 24h, tile to fill horizon
    if n_rows == 24:
        tiles_needed = horizon_days
        df_list = [df.copy()]
        for i in range(1, tiles_needed):
            df_tile = df.copy()
            df_tile['hour'] = df_tile['hour'] + (i * 24)
            df_list.append(df_tile)
        df = pd.concat(df_list, ignore_index=True)
        st.info(f"⚠️ Case file has 24 hours. Tiled {horizon_days} times to fill {horizon_days}-day horizon.")
    
    # If more than needed, take latest target_hours
    if len(df) > target_hours:
        df = df.tail(target_hours).reset_index(drop=True)
        st.info(f"⚠️ Case file has {n_rows} hours. Showing latest {target_hours} hours ({horizon_days} days).")
    
    # Ensure hour labels are 0..(target_hours-1) for consistency
    df['hour'] = np.arange(target_hours)
    
    return df


def load_prices_multiday(csv_file, horizon_days):
    """
    Load price CSV for multi-day horizon - accepts EPİAŞ format or simple format.
    Accepts either a file path (string/Path) or file-like object.
    Returns: numpy array of prices in TL/MWh, length 24*horizon_days
    If only 24 prices, tiles to fill horizon. If more, takes latest horizon_days*24.
    """
    # Handle both file paths and file-like objects
    is_file_path = isinstance(csv_file, (str, Path))
    content = None
    
    if is_file_path:
        # File path - read with encoding fallback
        try:
            content = _read_file_with_encoding_fallback(csv_file)
            lines = content.splitlines()
        except Exception:
            with open(csv_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
    else:
        # File-like object
        csv_file.seek(0)
        content = csv_file.read()
        if isinstance(content, bytes):
            encodings = ['utf-8', 'cp1254', 'latin1']
            for encoding in encodings:
                try:
                    content = content.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                content = content.decode("utf-8", errors='ignore')
        lines = content.splitlines()
    
    reader = csv.reader(lines)
    header = next(reader, None)
    
    prices = []
    
    # Try EPİAŞ format first (Tarih, Saat, PTF (TL/MWh), ...)
    if header and len(header) >= 3 and "PTF" in header[2]:
        # EPİAŞ format: parse price from row like ["20.11.2025", "00:00", "2.517", "01", ...]
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                price_str = row[2].strip() + "," + row[3].strip()
                price_str_norm = price_str.replace(".", "").replace(",", ".")
                price = float(price_str_norm)
                prices.append(price)
            except (ValueError, IndexError):
                continue
    else:
        # Simple format: try to find hour and price columns
        if is_file_path:
            # Try encodings
            encodings = ['utf-8', 'cp1254', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if df is None:
                df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
        else:
            csv_file.seek(0)
            if content:
                df = pd.read_csv(io.StringIO(content))
            else:
                csv_file.seek(0)
                df = pd.read_csv(csv_file)
        
        # Look for price column
        price_col = None
        for col in df.columns:
            if 'price' in col.lower() or 'ptf' in col.lower() or 'tl' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            if len(df.columns) >= 2:
                price_col = df.columns[1]
            else:
                raise ValueError("Could not find price column in CSV")
        
        prices = df[price_col].values.tolist()
    
    prices = np.array(prices, dtype=float)
    target_length = 24 * horizon_days
    
    # If only 24 prices, tile to fill horizon
    if len(prices) == 24:
        prices = np.tile(prices, horizon_days)
        st.info(f"⚠️ Price file has 24 hours. Tiled {horizon_days} times to fill {horizon_days}-day horizon.")
    
    # If more than needed, take latest target_length
    if len(prices) > target_length:
        prices = prices[-target_length:]
        st.info(f"⚠️ Price file has {len(prices)} hours. Using latest {target_length} hours ({horizon_days} days).")
    
    return prices


# =========================================================
# B) Core Computation Functions
# =========================================================

def compute_q_limit(P, pf_limit=0.98):
    """
    Compute Q_limit from PF constraint.
    Q_limit(t) = P(t) * tan(arccos(pf_limit))
    """
    tan_phi = np.tan(np.arccos(pf_limit))
    return P * tan_phi


def compute_q_avail(S_rated, P_PV_max):
    """
    Compute Q_avail (market availability / purchase limit) from S and PV.
    if P_PV_max(t) > 0:
        Q_avail(t) = sqrt(max(S_rated^2 - P_PV_max(t)^2, 0))
    else:
        Q_avail(t) = S_rated / 2
    """
    # Validate S_rated
    S_rated = float(S_rated)
    if S_rated <= 0:
        raise ValueError(f"S_rated must be > 0, got {S_rated}")
    
    # Ensure P_PV_max is numeric array
    P_PV_max = np.array(P_PV_max, dtype=float)
    
    Q_avail = np.zeros_like(P_PV_max, dtype=float)
    for i, pv in enumerate(P_PV_max):
        pv_val = float(pv)
        if pv_val > 0:
            Q_avail[i] = np.sqrt(max(S_rated**2 - pv_val**2, 0))
    else:
            Q_avail[i] = S_rated / 2.0
    return Q_avail


def compute_q_need(Q_D, Q_limit):
    """
    Compute Q_need (reactive need beyond PF limit).
    Q_need(t) = max(Q_D(t) - Q_limit(t), 0)
    """
    return np.maximum(Q_D - Q_limit, 0)


# =========================================================
# C) Backlog Allocation Strategies
# =========================================================

def allocate_backlog_uniform(backlog_total, n_hours=24):
    """Strategy 1: Uniform distribution (raw allocation, not capacity-aware)"""
    return np.full(n_hours, backlog_total / n_hours)


def allocate_backlog_price_weighted(backlog_total, prices):
    """Strategy 2: Price-weighted distribution (raw allocation, not capacity-aware)"""
    inv_prices = 1.0 / (prices + 1e-10)  # avoid division by zero
    weights = inv_prices / np.sum(inv_prices)
    return backlog_total * weights


def apply_capacity_aware_reallocation(A_alloc_raw, backlog_total, freecap, prices=None, 
                                     strategy="uniform", effective_reactive_price=None, 
                                     penalty_rate=None):
    """
    Generic capacity-aware reallocation procedure.
    
    Ensures:
    1) 0 <= A_alloc(t) <= freecap(t) for all t
    2) sum(A_alloc(t)) <= backlog_total
    
    Args:
        A_alloc_raw: initial allocation proposal (24,)
        backlog_total: total backlog to allocate
        freecap: free capacity at each hour (24,)
        prices: prices for price-weighted strategies (24,)
        strategy: "uniform", "price_weighted", or "economic_feasible"
        effective_reactive_price: for economic strategy (24,)
        penalty_rate: for economic strategy (scalar)
    
    Returns:
        A_alloc: final capacity-aware allocation (24,)
        leftover: amount that could not be allocated (scalar)
    """
    n_hours = len(A_alloc_raw)
    
    # Step 1: Clip initial allocation to free capacity
    A_alloc = np.clip(A_alloc_raw, 0, freecap)
    
    # Step 2: Compute leftover
    leftover = backlog_total - np.sum(A_alloc)
    
    # Step 3: Reallocate leftover to hours with remaining capacity
    if leftover > 1e-9:
        # Compute remaining capacity at each hour
        remcap = freecap - A_alloc
        
        # Find hours with remaining capacity
        available_hours = np.where(remcap > 1e-9)[0]
        
        if len(available_hours) > 0:
            if strategy == "uniform":
                # Distribute leftover evenly across hours with remcap > 0
                per_hour = leftover / len(available_hours)
                for h in available_hours:
                    alloc_amount = min(per_hour, remcap[h])
                    A_alloc[h] += alloc_amount
                    leftover -= alloc_amount
                    if leftover <= 1e-9:
                        break
                
                # If still leftover, distribute proportionally to remaining capacity
                if leftover > 1e-9:
                    remaining_remcap = freecap - A_alloc
                    remaining_available = np.where(remaining_remcap > 1e-9)[0]
                    if len(remaining_available) > 0:
                        remcap_sum = np.sum(remaining_remcap[remaining_available])
                        if remcap_sum > 1e-9:
                            for h in remaining_available:
                                alloc_amount = min(leftover * (remaining_remcap[h] / remcap_sum), 
                                                 remaining_remcap[h])
                                A_alloc[h] += alloc_amount
                                leftover -= alloc_amount
                                if leftover <= 1e-9:
                                    break
            
            elif strategy == "price_weighted":
                # Distribute leftover to cheaper hours first
                # Sort available hours by price (cheapest first)
                sorted_available = available_hours[np.argsort(prices[available_hours])]
                
                for h in sorted_available:
                    if leftover <= 1e-9:
                        break
                    alloc_amount = min(leftover, remcap[h])
                    A_alloc[h] += alloc_amount
                    leftover -= alloc_amount
                    remcap[h] -= alloc_amount
                
                # If still leftover, distribute proportionally by inverse price
                if leftover > 1e-9:
                    remaining_remcap = freecap - A_alloc
                    remaining_available = np.where(remaining_remcap > 1e-9)[0]
                    if len(remaining_available) > 0:
                        inv_prices_rem = 1.0 / (prices[remaining_available] + 1e-10)
                        weights_rem = inv_prices_rem / np.sum(inv_prices_rem)
                        
                        for idx, h in enumerate(remaining_available):
                            if leftover <= 1e-9:
                                break
                            alloc_amount = min(leftover * weights_rem[idx], remaining_remcap[h])
                            A_alloc[h] += alloc_amount
                            leftover -= alloc_amount
            
            elif strategy == "economic_feasible":
                # Distribute leftover ONLY to hours where effective_price < penalty_rate
                # Sort by increasing effective_reactive_price (cheapest first)
                eligible_mask = (effective_reactive_price < penalty_rate) & (remcap > 1e-9)
                eligible_hours = np.where(eligible_mask)[0]
                
                if len(eligible_hours) > 0:
                    sorted_eligible = eligible_hours[np.argsort(effective_reactive_price[eligible_hours])]
                    
                    for h in sorted_eligible:
                        if leftover <= 1e-9:
                            break
                        alloc_amount = min(leftover, remcap[h])
                        A_alloc[h] += alloc_amount
                        leftover -= alloc_amount
                        remcap[h] -= alloc_amount
        # If no available hours or strategy doesn't match, leftover remains
    
    return A_alloc, leftover


def allocate_backlog_price_feasible(backlog_total, Q_need, Q_avail, prices):
    """
    Strategy 3: Price-feasible reallocation (legacy, non-economic version)
    
    Start with Strategy 2 allocation A0(t).
    If desired(t) = Q_need(t) + A0(t) > Q_avail(t), reallocate excess
    to other hours where Q_avail(t) - desired(t) > 0, preferring cheaper hours.
    
    IMPORTANT: Total allocation must always equal backlog_total.
    
    Returns:
        A: final allocation (24,) such that sum(A) = backlog_total
        reallocated_amount: amount reallocated to each hour (24,)
    """
    n_hours = len(Q_need)
    
    # Step 1: Initial allocation using Strategy 2
    A0 = allocate_backlog_price_weighted(backlog_total, prices)
    
    # Step 2: Compute desired load
    desired = Q_need + A0
    
    # Step 3: Compute overload and free capacity
    overload = np.maximum(desired - Q_avail, 0)
    free_capacity = np.maximum(Q_avail - desired, 0)
    
    # Step 4: Reallocate overload to free capacity hours
    A = A0.copy()
    reallocated_amount = np.zeros(n_hours, dtype=float)
    
    total_overload = np.sum(overload)
    if total_overload > 1e-6:
        # Find deficit and surplus hours
        deficit_hours = np.where(overload > 1e-6)[0]
        surplus_hours = np.where(free_capacity > 1e-6)[0]
        
        if len(surplus_hours) > 0:
            # Step 4a: Cap deficit hours at what's feasible
            # This reduces A in deficit hours and creates "saved" amount
            saved_amount = 0.0
            for h_deficit in deficit_hours:
                max_feasible = max(0, Q_avail[h_deficit] - Q_need[h_deficit])
                reduction = A[h_deficit] - max_feasible
                if reduction > 0:
                    saved_amount += reduction
                    A[h_deficit] = max_feasible
            
            # Step 4b: Redistribute saved amount to surplus hours (preferring cheaper hours)
            if saved_amount > 1e-6:
                # Compute weights for surplus hours (cheaper = higher weight)
                inv_prices_surplus = 1.0 / (prices[surplus_hours] + 1e-10)
                weights_surplus = inv_prices_surplus / np.sum(inv_prices_surplus)
            
                # Redistribute saved amount to surplus hours
                remaining_saved = saved_amount
                for idx, h_surplus in enumerate(surplus_hours):
                    if remaining_saved <= 1e-6:
                        break
                
                    # How much can we allocate to this hour?
                    # Consider both free capacity and weight
                    max_realloc = min(free_capacity[h_surplus], remaining_saved * weights_surplus[idx])
                    reallocated_amount[h_surplus] = max_realloc
                    A[h_surplus] += max_realloc
                    remaining_saved -= max_realloc
                # If there's still remaining saved amount (due to capacity limits),
                # redistribute it proportionally to all surplus hours that still have capacity
                if remaining_saved > 1e-6:
                    # Find hours that still have free capacity
                    current_desired = Q_need + A
                    remaining_free_capacity = np.maximum(Q_avail - current_desired, 0)
                    still_available_hours = np.where(remaining_free_capacity > 1e-6)[0]
                    
                    if len(still_available_hours) > 0:
                        inv_prices_available = 1.0 / (prices[still_available_hours] + 1e-10)
                        weights_available = inv_prices_available / np.sum(inv_prices_available)
                        
                        for idx, h_avail in enumerate(still_available_hours):
                            if remaining_saved <= 1e-6:
                                break
                            
                            max_realloc = min(remaining_free_capacity[h_avail], 
                                            remaining_saved * weights_available[idx])
                            reallocated_amount[h_avail] += max_realloc
                            A[h_avail] += max_realloc
                            remaining_saved -= max_realloc
        else:
            # No surplus hours available - just cap at Q_avail
            for h_deficit in deficit_hours:
                A[h_deficit] = max(0, Q_avail[h_deficit] - Q_need[h_deficit])
    
    # Final check: ensure total equals backlog_total (within numerical precision)
    # If there's a small discrepancy due to rounding, adjust proportionally
    current_total = np.sum(A)
    if abs(current_total - backlog_total) > 1e-6:
        # Adjust proportionally to maintain relative distribution
        if current_total > 1e-6:
            A = A * (backlog_total / current_total)
    else:
            # If somehow everything is zero, fall back to uniform
            A = np.full(n_hours, backlog_total / n_hours)
    
    return A, reallocated_amount


def allocate_backlog_economic_feasible(backlog_total, Q_need_day2, Q_avail_day2, prices_day2, 
                                        alpha, penalty_rate):
    """
    Strategy 3 (Economic + Feasible): Allocate backlog to cheap hours with free capacity
    AND only when price < penalty_rate.
    
    Greedy allocation algorithm:
    1. Compute free capacity for backlog: freecap(t) = max(Q_avail(t) - Q_need_day2(t), 0)
    2. Define economic eligibility: eligible(t) = (effective_reactive_price(t) < penalty_rate)
    3. Allocate greedily to eligible hours sorted by increasing effective_reactive_price
    
    Returns:
        A_alloc: final allocation (24,) - may be less than backlog_total if not economical
        economic_info: dict with economic details for verification
    """
    n_hours = len(Q_need_day2)
    A_alloc = np.zeros(n_hours, dtype=float)
    
    # Compute effective reactive price
    effective_reactive_price = alpha * prices_day2
    
    # Compute free capacity for backlog
    freecap = np.maximum(Q_avail_day2 - Q_need_day2, 0)
    
    # Economic eligibility
    eligible = effective_reactive_price < penalty_rate
    
    # Sort eligible hours by increasing effective_reactive_price
    eligible_hours = np.where(eligible)[0]
    if len(eligible_hours) > 0:
        sorted_indices = np.argsort(effective_reactive_price[eligible_hours])
        sorted_eligible = eligible_hours[sorted_indices]
        
        # Greedy allocation
        remaining = float(backlog_total)
        for t in sorted_eligible:
            if remaining <= 1e-6:
                break
            x = min(remaining, freecap[t])
            A_alloc[t] = x
            remaining -= x
    
    # Economic info for verification
    economic_info = {
        'effective_reactive_price': effective_reactive_price,
        'penalty_rate': penalty_rate,
        'eligible': eligible,
        'freecap': freecap,
        'allocated_total': np.sum(A_alloc),
        'remaining_unallocated': backlog_total - np.sum(A_alloc)
    }
    
    return A_alloc, economic_info


# =========================================================
# D) Two-Day Simulation
# =========================================================

def simulate_two_days(df_case, prices, S_rated, pf_limit, strategy, backlog_prev_total, 
                       alpha=0.10, penalty_rate=200.0):
    """
    Simulate two days (48 hours) with backlog logic.
    
    Core logic:
    - The system requests Q_need from the market.
    - The market delivers Q_inv.
    - Any unmet part becomes Q_deficit and is carried forward.
    
    Args:
        alpha: reactive price multiplier (for economic strategy)
        penalty_rate: penalty rate in TL/MVArh (for economic strategy)
    
    Returns:
        results_df: DataFrame with core variables only
        Day1_backlog_total: scalar (sum of Q_deficit from Day 1)
        Day2_backlog_total: scalar (sum of Q_deficit from Day 2)
        reallocation_info: dict with reallocation details (for Strategy 3, internal use only)
        economic_info: dict with economic details (for Strategy 3 economic, internal use only)
    """
    # Extract data (already 48 hours)
    hours = df_case['hour'].values
    P = df_case['P_total'].values
    Q_D = df_case['Q_total'].values
    P_PV_max = df_case['P_PV_max'].values
    
    # Ensure prices are 48 hours
    if len(prices) < 48:
        prices = np.tile(prices, 2)[:48]
    elif len(prices) > 48:
        prices = prices[-48:]
    
    # Split into Day 1 and Day 2 (hours 0..23 and 24..47)
    day1_mask = (hours >= 0) & (hours < 24)
    day2_mask = (hours >= 24) & (hours < 48)
    
    P_day1 = P[day1_mask]
    Q_D_day1 = Q_D[day1_mask]
    P_PV_max_day1 = P_PV_max[day1_mask]
    prices_day1 = prices[day1_mask]
    
    P_day2 = P[day2_mask]
    Q_D_day2 = Q_D[day2_mask]
    P_PV_max_day2 = P_PV_max[day2_mask]
    prices_day2 = prices[day2_mask]
    
    # Initialize arrays for core variables
    Q_limit = np.zeros(48, dtype=float)
    Q_need = np.zeros(48, dtype=float)
    Q_avail = np.zeros(48, dtype=float)
    Q_inv = np.zeros(48, dtype=float)
    Q_deficit = np.zeros(48, dtype=float)
    
    # Additional columns for Day 2 visualization
    Q_need_day2 = np.zeros(48, dtype=float)  # Day2 own need (from PF rule)
    A_alloc = np.zeros(48, dtype=float)  # Day1 deficit allocation into Day2
    Q_need_total = np.zeros(48, dtype=float)  # Total request (Q_need_day2 + A_alloc)
    
    # Internal helper variables (not shown to user)
    backlog_allocation_A = np.zeros(48, dtype=float)  # Helper: allocation of Day1 backlog to Day2
    reallocated_amount = np.zeros(48, dtype=float)  # Helper: for Strategy 3 only
    
    # =========================================================
    # DAY 1 (hours 0..23)
    # =========================================================
    Q_limit_day1 = compute_q_limit(P_day1, pf_limit)
    Q_need_base_day1 = compute_q_need(Q_D_day1, Q_limit_day1)  # Base need from physical demand
    Q_avail_day1 = compute_q_avail(S_rated, P_PV_max_day1)
    
    # Day 1: The system requests Q_need (no backlog from previous day)
    Q_need_day1 = Q_need_base_day1.copy()
    
    # The market delivers Q_inv (limited by Q_avail)
    Q_inv_day1 = np.minimum(Q_need_day1, Q_avail_day1)
    
    # Any unmet part becomes Q_deficit
    Q_deficit_day1 = Q_need_day1 - Q_inv_day1
    
    # Store Day 1 results
    day1_indices = np.where(day1_mask)[0]
    Q_limit[day1_indices] = Q_limit_day1
    Q_need[day1_indices] = Q_need_day1
    Q_avail[day1_indices] = Q_avail_day1
    Q_inv[day1_indices] = Q_inv_day1
    Q_deficit[day1_indices] = Q_deficit_day1
    
    # For Day 1: Q_need_day2 = Q_need (for consistency)
    Q_need_day2[day1_indices] = Q_need_day1
    # For Day 1: A_alloc = 0 (no backlog allocation)
    A_alloc[day1_indices] = 0.0
    # For Day 1: Q_need_total = Q_need
    Q_need_total[day1_indices] = Q_need_day1
    
    # Day1 backlog total (sum of deficits)
    Day1_backlog_total = np.sum(Q_deficit_day1)
    
    # =========================================================
    # DAY 2 (hours 24..47)
    # =========================================================
    Q_limit_day2 = compute_q_limit(P_day2, pf_limit)
    Q_need_base_day2 = compute_q_need(Q_D_day2, Q_limit_day2)  # Base need from physical demand (WITHOUT backlog)
    Q_avail_day2 = compute_q_avail(S_rated, P_PV_max_day2)
    
    # Step 1: Define FREE CAPACITY for backlog at each hour (Day 2)
    # freecap(t) = max(Q_avail(t) - Q_need_base(t), 0)
    # This is the maximum backlog amount that can be cleared at hour t
    # while still allowing the hour's own Q_need to be served as much as possible
    freecap = np.maximum(Q_avail_day2 - Q_need_base_day2, 0)
    
    # Step 2: Generate raw allocation proposal using chosen strategy
    economic_info = None
    reallocated_amount_day2 = np.zeros(24, dtype=float)
    A_alloc_raw = None
    
    if strategy == "uniform":
        A_alloc_raw = allocate_backlog_uniform(Day1_backlog_total, n_hours=24)
    elif strategy == "price_weighted":
        A_alloc_raw = allocate_backlog_price_weighted(Day1_backlog_total, prices_day2)
    elif strategy == "price_feasible":
        # For price_feasible, use price_weighted as initial proposal
        A_alloc_raw = allocate_backlog_price_weighted(Day1_backlog_total, prices_day2)
    elif strategy == "economic_feasible":
        # For economic_feasible, start with zero and let the economic strategy handle it
        A_alloc_raw = np.zeros(24, dtype=float)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Step 3: Apply capacity-aware reallocation
    if strategy == "economic_feasible":
        # Economic strategy has its own logic, but we still apply capacity awareness
        A_alloc_day2, economic_info = allocate_backlog_economic_feasible(
            Day1_backlog_total, Q_need_base_day2, Q_avail_day2, prices_day2,
            alpha, penalty_rate
        )
        # Economic strategy already respects freecap internally
        leftover_unallocated = Day1_backlog_total - np.sum(A_alloc_day2)
    else:
        # Apply capacity-aware reallocation for uniform, price_weighted, price_feasible
        A_alloc_day2, leftover_unallocated = apply_capacity_aware_reallocation(
            A_alloc_raw, Day1_backlog_total, freecap, prices_day2, strategy,
            None, None  # effective_reactive_price and penalty_rate not needed for non-economic strategies
        )
        
        # For price_feasible, compute reallocated_amount for diagnostics
        if strategy == "price_feasible":
            # Reallocated amount is the difference from initial price-weighted allocation
            A_initial = allocate_backlog_price_weighted(Day1_backlog_total, prices_day2)
            reallocated_amount_day2 = A_alloc_day2 - A_initial

    # Step 4: Market check after applying A_alloc
    # Day 2: Q_need_day2 is the Day2 own need (from PF rule, WITHOUT backlog)
    Q_need_day2_values = Q_need_base_day2.copy()
    
    # Day 2: Total request to market = Day2 own need + Day1 deficit allocation
    Q_need_total_day2 = Q_need_day2_values + A_alloc_day2
    
    # The market delivers Q_inv (limited by Q_avail)
    Q_inv_day2 = np.minimum(Q_need_total_day2, Q_avail_day2)
    
    # Any unmet part becomes Q_deficit
    Q_deficit_day2 = Q_need_total_day2 - Q_inv_day2
    
    # Step 5: Verification assertions
    # Assert that A_alloc respects freecap
    violation_mask = A_alloc_day2 > freecap + 1e-9
    if np.any(violation_mask):
        first_violation_hour = np.where(violation_mask)[0][0]
        st.warning(f"⚠️ Capacity violation: A_alloc[{first_violation_hour}] = {A_alloc_day2[first_violation_hour]:.6f} > freecap[{first_violation_hour}] = {freecap[first_violation_hour]:.6f}")
    
    # Assert that Q_inv respects Q_avail
    violation_mask = Q_inv_day2 > Q_avail_day2 + 1e-9
    if np.any(violation_mask):
        first_violation_hour = np.where(violation_mask)[0][0]
        st.warning(f"⚠️ Market capacity violation: Q_inv[{first_violation_hour}] = {Q_inv_day2[first_violation_hour]:.6f} > Q_avail[{first_violation_hour}] = {Q_avail_day2[first_violation_hour]:.6f}")
    
    # Store Day 2 results
    day2_indices = np.where(day2_mask)[0]
    Q_limit[day2_indices] = Q_limit_day2
    Q_need[day2_indices] = Q_need_total_day2  # Total request (for consistency with core variable)
    Q_avail[day2_indices] = Q_avail_day2
    Q_inv[day2_indices] = Q_inv_day2
    Q_deficit[day2_indices] = Q_deficit_day2
    Q_need_day2[day2_indices] = Q_need_day2_values  # Day2 own need
    A_alloc[day2_indices] = A_alloc_day2  # Day1 deficit allocation
    Q_need_total[day2_indices] = Q_need_total_day2  # Total request
    backlog_allocation_A[day2_indices] = A_alloc_day2  # Helper, not shown to user
    reallocated_amount[day2_indices] = reallocated_amount_day2  # Helper, not shown to user
    
    # Day2 backlog total (sum of deficits)
    Day2_backlog_total = np.sum(Q_deficit_day2)
    
    # Build results DataFrame with CORE VARIABLES + Day 2 visualization columns
    results_df = pd.DataFrame({
        'hour': hours,
        'day_index': (hours // 24).astype(int),
        'hour_in_day': (hours % 24).astype(int),
        'P': P,
        'Q_D': Q_D,
        'P_PV_max': P_PV_max,
        'price': prices,
        'Q_limit': Q_limit,
        'Q_need': Q_need,  # Total request (Q_need_total for Day2, Q_need for Day1)
        'Q_avail': Q_avail,
        'Q_inv': Q_inv,
        'Q_deficit': Q_deficit,
        'Q_need_day2': Q_need_day2,  # Day2 own need (from PF rule)
        'A_alloc': A_alloc,  # Day1 deficit allocation into Day2
        'Q_need_total': Q_need_total,  # Total request (Q_need_day2 + A_alloc)
    })
    
    # Reallocation info for Strategy 3 (internal use only, not shown in main table)
    reallocation_info = None
    if strategy == "price_feasible":
        A_initial = allocate_backlog_price_weighted(Day1_backlog_total, prices_day2)
        reallocation_info = {
            'day2_hours': np.arange(24),
            'A_initial': A_initial,
            'A_final': A_alloc_day2,
            'reallocated_amount': reallocated_amount_day2,
            'Q_need_base_day2': Q_need_base_day2,
            'Q_need_day2': Q_need_day2_values,
            'Q_avail_day2': Q_avail_day2,
            'prices_day2': prices_day2
        }
    
    # Diagnostics info for Day 2 (for verification table)
    diagnostics_info = {
        'hour_in_day': np.arange(24),
        'price': prices_day2,
        'Q_avail': Q_avail_day2,
        'base_Q_need': Q_need_base_day2,
        'freecap': freecap,
        'A_alloc_raw': A_alloc_raw if A_alloc_raw is not None else np.zeros(24),
        'A_alloc': A_alloc_day2,
        'Q_need_final': Q_need_total_day2,
        'Q_inv': Q_inv_day2,
        'Q_deficit': Q_deficit_day2,
        'leftover_unallocated': leftover_unallocated if strategy != "economic_feasible" else (Day1_backlog_total - np.sum(A_alloc_day2))
    }
    
    return results_df, Day1_backlog_total, Day2_backlog_total, reallocation_info, economic_info, diagnostics_info


# =========================================================
# D.1) Multi-Day Simulation
# =========================================================

def simulate_multi_days(df_case_N, prices_N, S_rated, pf_limit, strategy, backlog_prev_total,
                        alpha=0.10, penalty_rate=200.0):
    """
    Simulate multi-day horizon (7/14/30 days) with backlog logic.
    Reuses the SAME per-day logic from simulate_two_days.
    
    Core logic (per day d):
    - Day 1: compute Q_limit, Q_need_base, Q_avail; deliver Q_inv = min(Q_need, Q_avail); 
      Q_deficit = Q_need - Q_inv; backlog_total = sum(Q_deficit)
    - For day d>=2:
      - compute day's base need (from PF rule): Q_need_base_d = max(Q_total - Q_limit, 0)
      - compute freecap_d = max(Q_avail_d - Q_need_base_d, 0)
      - allocate previous day's backlog_total into today using SAME strategy functions
      - total request: Q_need_total_d = Q_need_base_d + A_alloc_d
      - deliver: Q_inv_d = min(Q_need_total_d, Q_avail_d)
      - deficit: Q_deficit_d = Q_need_total_d - Q_inv_d
      - backlog_total_for_next_day = sum(Q_deficit_d)
    
    Args:
        df_case_N: DataFrame with 24*horizon_days rows, columns: hour, P_total, Q_total, P_PV_max
        prices_N: array with 24*horizon_days prices
        S_rated: rated apparent power (MVA)
        pf_limit: power factor limit
        strategy: backlog allocation strategy ("uniform", "price_weighted", "price_feasible", "economic_feasible")
        backlog_prev_total: backlog from previous simulation (scalar)
        alpha: reactive price multiplier (for economic strategy)
        penalty_rate: penalty rate in TL/MVArh (for economic strategy)
    
    Returns:
        results_df_N: DataFrame with all hours, columns: hour, day_index, hour_in_day, P, Q_D, P_PV_max, price,
                     Q_limit, Q_need, Q_avail, Q_inv, Q_deficit, Q_need_base, A_alloc, Q_need_total
        daily_summary_df: DataFrame with one row per day: day, backlog_in, backlog_allocated, backlog_cleared,
                         backlog_out, new_need_sum, delivered_sum
        cohort_df: DataFrame tracking deficit by origin day at end of day 7 (if horizon>=7):
                  origin_day, remaining_amount
    """
    horizon_days = len(df_case_N) // 24
    hours = df_case_N['hour'].values
    P = df_case_N['P_total'].values
    Q_D = df_case_N['Q_total'].values
    P_PV_max = df_case_N['P_PV_max'].values
    
    # Ensure prices are correct length
    if len(prices_N) < len(df_case_N):
        # Tile if needed
        tiles_needed = (len(df_case_N) + len(prices_N) - 1) // len(prices_N)
        prices_N = np.tile(prices_N, tiles_needed)[:len(df_case_N)]
    elif len(prices_N) > len(df_case_N):
        prices_N = prices_N[:len(df_case_N)]
    
    # Initialize result arrays (one per hour)
    Q_limit = np.zeros(len(df_case_N), dtype=float)
    Q_need_base = np.zeros(len(df_case_N), dtype=float)  # Base need from PF rule
    Q_avail = np.zeros(len(df_case_N), dtype=float)
    Q_inv = np.zeros(len(df_case_N), dtype=float)
    Q_deficit = np.zeros(len(df_case_N), dtype=float)
    A_alloc = np.zeros(len(df_case_N), dtype=float)  # Backlog allocation
    Q_need_total = np.zeros(len(df_case_N), dtype=float)  # Total request (base + allocation)
    
    # Daily summary tracking
    daily_summary = []
    
    # Cohort tracking: track deficit by origin day (FIFO clearing)
    # cohort[origin_day] = remaining amount from that day
    cohort = {}  # Will track by origin day index (1-indexed)
    # We'll track how much backlog was cleared each day to update cohorts
    
    # Track backlog cleared per day for cohort updates
    backlog_cleared_per_day = {}  # day -> amount cleared that day
    
    # Current backlog (carries over day by day)
    current_backlog_total = float(backlog_prev_total)
    if current_backlog_total > 1e-6:
        # Initialize cohort for previous backlog (assign to "day 0")
        cohort[0] = current_backlog_total
    
    # =========================================================
    # DAY-BY-DAY SIMULATION
    # =========================================================
    for day_idx in range(horizon_days):
        day_num = day_idx + 1  # 1-indexed day number
        
        # Get hours for this day
        day_start_hour = day_idx * 24
        day_end_hour = (day_idx + 1) * 24
        day_mask = (hours >= day_start_hour) & (hours < day_end_hour)
        day_indices = np.where(day_mask)[0]
        
        # Extract day data
        P_day = P[day_mask]
        Q_D_day = Q_D[day_mask]
        P_PV_max_day = P_PV_max[day_mask]
        prices_day = prices_N[day_mask]
        
        # Compute base quantities for this day
        Q_limit_day = compute_q_limit(P_day, pf_limit)
        Q_need_base_day = compute_q_need(Q_D_day, Q_limit_day)
        Q_avail_day = compute_q_avail(S_rated, P_PV_max_day)
        
        # Store base quantities
        Q_limit[day_indices] = Q_limit_day
        Q_need_base[day_indices] = Q_need_base_day
        Q_avail[day_indices] = Q_avail_day
        
        # Track backlog allocation for this day
        backlog_allocated_day = 0.0
        backlog_cleared_day = 0.0
        
        if day_idx == 0:
            # Day 1: no backlog allocation, just base need
            Q_need_total_day = Q_need_base_day.copy()
            A_alloc_day = np.zeros(24, dtype=float)
        else:
            # Day 2+: allocate backlog from previous day
            # Step 1: Compute free capacity for backlog
            freecap_day = np.maximum(Q_avail_day - Q_need_base_day, 0)
            
            # Step 2: Generate raw allocation proposal using chosen strategy
            A_alloc_raw = None
            if strategy == "uniform":
                A_alloc_raw = allocate_backlog_uniform(current_backlog_total, n_hours=24)
            elif strategy == "price_weighted":
                A_alloc_raw = allocate_backlog_price_weighted(current_backlog_total, prices_day)
            elif strategy == "price_feasible":
                A_alloc_raw = allocate_backlog_price_weighted(current_backlog_total, prices_day)
            elif strategy == "economic_feasible":
                A_alloc_raw = np.zeros(24, dtype=float)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Step 3: Apply capacity-aware reallocation
            if strategy == "economic_feasible":
                A_alloc_day, economic_info_day = allocate_backlog_economic_feasible(
                    current_backlog_total, Q_need_base_day, Q_avail_day, prices_day,
                    alpha, penalty_rate
                )
            else:
                # Apply capacity-aware reallocation for uniform, price_weighted, price_feasible
                A_alloc_day, leftover_unallocated = apply_capacity_aware_reallocation(
                    A_alloc_raw, current_backlog_total, freecap_day, prices_day, strategy,
                    None, None
                )
            
            # Total allocation amount
            backlog_allocated_day = np.sum(A_alloc_day)
            
            # Total request = base need + backlog allocation
            Q_need_total_day = Q_need_base_day + A_alloc_day
        
        # Market delivers Q_inv (limited by Q_avail)
        Q_inv_day = np.minimum(Q_need_total_day, Q_avail_day)
        
        # Compute backlog cleared (actually realized backlog allocation)
        # This is the portion of A_alloc that was actually delivered
        backlog_cleared_day = np.sum(np.maximum(Q_inv_day - Q_need_base_day, 0))
        
        # Any unmet part becomes Q_deficit
        Q_deficit_day = Q_need_total_day - Q_inv_day
        
        # Store results for this day
        Q_inv[day_indices] = Q_inv_day
        Q_deficit[day_indices] = Q_deficit_day
        A_alloc[day_indices] = A_alloc_day if day_idx > 0 else np.zeros(24)
        Q_need_total[day_indices] = Q_need_total_day
        
        # Update backlog for next day
        # For day 1: backlog_in is the previous backlog (backlog_prev_total)
        # For day 2+: backlog_in is the current_backlog_total (which is backlog_out from previous day)
        if day_idx == 0:
            backlog_in = float(backlog_prev_total)
        else:
            backlog_in = current_backlog_total
        
        backlog_out = np.sum(Q_deficit_day)
        current_backlog_total = backlog_out
        
        # Update cohorts using FIFO rule: when backlog is cleared, consume oldest cohorts first
        if day_idx > 0 and backlog_cleared_day > 1e-6:
            remaining_to_clear = backlog_cleared_day
            # Sort cohort keys (origin days) to process oldest first (including day 0 for previous backlog)
            sorted_origin_days = sorted(cohort.keys())
            for origin_day in sorted_origin_days:
                if remaining_to_clear <= 1e-9:
                    break
                if cohort[origin_day] > 1e-9:
                    amount_to_clear_from_origin = min(remaining_to_clear, cohort[origin_day])
                    cohort[origin_day] -= amount_to_clear_from_origin
                    remaining_to_clear -= amount_to_clear_from_origin
                    if cohort[origin_day] < 1e-9:
                        cohort[origin_day] = 0.0
        
        # Add new deficit from this day to cohort (after clearing previous backlog)
        if backlog_out > 1e-6:
            if day_num not in cohort:
                cohort[day_num] = 0.0
            cohort[day_num] += backlog_out
        else:
            # Ensure cohort entry exists even if zero (for consistency)
            if day_num not in cohort:
                cohort[day_num] = 0.0
        
        # Store backlog cleared for this day
        backlog_cleared_per_day[day_num] = backlog_cleared_day
        
        # Daily summary
        # backlog_in is already correctly set above based on day_idx
        daily_summary.append({
            'day': day_num,
            'backlog_in': backlog_in,
            'backlog_allocated': backlog_allocated_day,
            'backlog_cleared': backlog_cleared_day,
            'backlog_out': backlog_out,
            'new_need_sum': np.sum(Q_need_base_day),
            'delivered_sum': np.sum(Q_inv_day)
        })
    
    # Build results DataFrame
    # Note: Q_need represents total request (base + allocation), same as Q_need_total
    # We include both for consistency with 48h path, but they are identical
    results_df_N = pd.DataFrame({
        'hour': hours,
        'day_index': (hours // 24).astype(int),
        'hour_in_day': (hours % 24).astype(int),
        'P': P,
        'Q_D': Q_D,
        'P_PV_max': P_PV_max,
        'price': prices_N,
        'Q_limit': Q_limit,
        'Q_need': Q_need_total,  # Total request (base + allocation)
        'Q_avail': Q_avail,
        'Q_inv': Q_inv,
        'Q_deficit': Q_deficit,
        'Q_need_base': Q_need_base,  # Base need from PF rule
        'A_alloc': A_alloc,  # Backlog allocation
        'Q_need_total': Q_need_total,  # Same as Q_need (for consistency with 48h)
    })
    
    # Build daily summary DataFrame
    daily_summary_df = pd.DataFrame(daily_summary)
    
    # Build cohort DataFrame (at end of day 7, if horizon >= 7)
    cohort_df = None
    if horizon_days >= 7:
        # Filter cohorts to origin days 1..7
        cohort_list = []
        for origin_day in range(1, 8):  # Days 1-7
            remaining = cohort.get(origin_day, 0.0)
            if remaining > 1e-9:
                cohort_list.append({
                    'origin_day': origin_day,
                    'remaining_amount': remaining
                })
        if cohort_list:
            cohort_df = pd.DataFrame(cohort_list)
        else:
            # Empty cohort (all cleared)
            cohort_df = pd.DataFrame(columns=['origin_day', 'remaining_amount'])
    
    return results_df_N, daily_summary_df, cohort_df


# =========================================================
# E) Plotting Functions
# =========================================================

def plot_48hour_timeline(results_df):
    """48-hour stacked bar visualization showing REALIZED allocation only (Q_inv base + backlog_cleared + Q_deficit)"""
    
    fig = go.Figure()
    
    # Compute realized backlog cleared for each hour
    # Formula: base_Q_need = Q_need(t) - A_alloc(t)  (Q_need before backlog)
    #          backlog_cleared = max(Q_inv(t) - base_Q_need, 0)
    #          backlog_cleared = min(backlog_cleared, A_alloc(t))
    
    # Get base_Q_need (Q_need before backlog allocation)
    # For Day 1: base_Q_need = Q_need (since A_alloc = 0)
    # For Day 2: base_Q_need = Q_need_day2 (already stored in results_df)
    base_Q_need = results_df['Q_need'].copy()
    day2_mask = results_df['day_index'] == 1
    base_Q_need[day2_mask] = results_df.loc[day2_mask, 'Q_need_day2']
    base_Q_need = base_Q_need.values
    
    # Compute backlog_cleared (actually realized backlog)
    Q_inv = results_df['Q_inv'].values
    A_alloc = results_df['A_alloc'].values
    backlog_cleared = np.maximum(Q_inv - base_Q_need, 0)
    backlog_cleared = np.minimum(backlog_cleared, A_alloc)
    
    # Compute Q_inv_base (part that serves base_Q_need)
    Q_inv_base = Q_inv - backlog_cleared
    
    # Stacked bars (bottom to top):
    # Bottom: Q_inv_base (delivered to serve base need)
    fig.add_trace(go.Bar(
        x=results_df['hour'],
        y=Q_inv_base,
        name='Q_inv (delivered)',
        marker_color='green',
        opacity=0.7
    ))
    
    # Middle: backlog_cleared (actually realized backlog allocation)
    fig.add_trace(go.Bar(
        x=results_df['hour'],
        y=backlog_cleared,
        name='Backlog cleared (allocated & delivered)',
        marker_color='yellow',
        opacity=0.7,
        base=Q_inv_base  # Stack on top of Q_inv_base
    ))
    
    # Top: Q_deficit (unmet reactive power, shown with pattern)
    # For Day 1: deficit carried into Day 2
    # For Day 2: deficit carried into Day 3
    # Use pattern to distinguish from filled bars
    fig.add_trace(go.Bar(
        x=results_df['hour'],
        y=results_df['Q_deficit'],
        name='Remaining deficit (carried forward)',
        marker=dict(
            color='red',
            pattern=dict(shape='/', fillmode='overlay', solidity=0.5)
        ),
        opacity=0.6,
        base=Q_inv_base + backlog_cleared  # Stack on top of Q_inv_base + backlog_cleared
    ))
    
    # Q_need as thin line overlay (what was requested, including A_alloc for Day 2)
    Q_need_total = results_df['Q_need'].copy()  # Already includes A_alloc for Day 2
    fig.add_trace(go.Scatter(
        x=results_df['hour'],
        y=Q_need_total,
        name='Q_need (requested)',
        mode='lines',
        line=dict(color='darkgreen', width=1.5)
    ))
    
    # A_alloc as dotted line for Day 2 only (PLANNED backlog allocation, not necessarily realized)
    day2_mask = results_df['day_index'] == 1
    day2_hours = results_df[day2_mask]['hour'].values
    day2_A_alloc = results_df[day2_mask]['A_alloc'].values
    fig.add_trace(go.Scatter(
        x=day2_hours,
        y=day2_A_alloc,
        name='A_alloc (planned, Day2)',
        mode='lines+markers',
        line=dict(color='orange', width=2, dash='dot'),
        marker=dict(size=4),
        opacity=0.6
    ))
    
    # Q_avail line (capacity limit) across all 48 hours on same y-axis
    fig.add_trace(go.Scatter(
        x=results_df['hour'],
        y=results_df['Q_avail'],
        name='Q_avail',
        mode='lines',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Q_limit line (EPDK / PF limit) across all 48 hours on same y-axis
    fig.add_trace(go.Scatter(
        x=results_df['hour'],
        y=results_df['Q_limit'],
        name='Q_limit',
        mode='lines',
        line=dict(color='purple', width=2, dash='dot')
    ))
    
    # Add vertical line to separate Day 1 and Day 2
    fig.add_vline(
        x=23.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="Day 1 → Day 2",
        annotation_position="top"
    )
    
    # Verify visual consistency: Q_inv (including backlog_cleared) never exceeds Q_avail
    total_delivered = Q_inv_base + backlog_cleared
    max_exceed = np.max(total_delivered - results_df['Q_avail'].values)
    if max_exceed > 1e-6:
        st.warning(f"⚠️ Visualization consistency check: Some hours show delivered > Q_avail (max exceed: {max_exceed:.6f} MVAr)")
    
    fig.update_layout(
        barmode='stack',
        xaxis_title='Hour (0..47)',
        yaxis_title='Q (MVAr)',
        title='48-hour reactive power operation with deficit carry-over',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_strategy_allocation(results_df, reallocation_info, strategy):
    """Plot 3: Strategy allocation visualization - shows how Day1 backlog affects Q_need in Day2"""
    day2_df = results_df[results_df['day_index'] == 1].copy()
    
    fig = go.Figure()
    
    # Show Q_need in Day 2 (which includes backlog allocation)
    fig.add_trace(go.Bar(
        x=day2_df['hour_in_day'],
        y=day2_df['Q_need'],
        name='Q_need (includes backlog allocation)',
        marker_color='blue',
        opacity=0.7
    ))
    
    # For Strategy 3, show reallocation details if available (internal helper visualization)
    if strategy == "price_feasible" and reallocation_info:
        # This is a technical detail, shown separately in reallocation table
        pass
    
    fig.update_layout(
        barmode='group',
        xaxis_title='Hour in Day 2',
        yaxis_title='Q_need (MVAr)',
        title=f'Strategy {strategy}: Q_need in Day 2 (includes Day1 backlog allocation)',
        height=400,
        showlegend=True
    )
    
    return fig


def plot_multi_day_timeline(results_df_N, selected_day=None, show_full_horizon=True):
    """
    Multi-day stacked bar visualization showing REALIZED allocation.
    Similar to plot_48hour_timeline but for multi-day horizons.
    
    Args:
        results_df_N: DataFrame with all hours for the horizon
        selected_day: If provided, filter to this day only (1-indexed). If None and show_full_horizon=False, use day 1.
        show_full_horizon: If True, plot all days. If False and selected_day is provided, plot only that day.
    
    Returns:
        fig: Plotly figure
    """
    # Filter to selected day if specified
    if selected_day is not None and not show_full_horizon:
        plot_df = results_df_N[results_df_N['day_index'] == (selected_day - 1)].copy()
        title_suffix = f" (Day {selected_day} only)"
    else:
        plot_df = results_df_N.copy()
        if selected_day is not None:
            title_suffix = f" (Full horizon, Day {selected_day} highlighted)"
        else:
            title_suffix = " (Full horizon)"
    
    fig = go.Figure()
    
    # Get base_Q_need (Q_need before backlog allocation)
    # For Day 1: base_Q_need = Q_need (since A_alloc = 0)
    # For Day 2+: base_Q_need = Q_need_base
    base_Q_need = plot_df['Q_need_base'].values.copy()
    day1_mask = plot_df['day_index'] == 0
    base_Q_need[day1_mask] = plot_df.loc[day1_mask, 'Q_need'].values
    
    # Compute backlog_cleared (actually realized backlog)
    Q_inv = plot_df['Q_inv'].values
    A_alloc = plot_df['A_alloc'].values
    backlog_cleared = np.maximum(Q_inv - base_Q_need, 0)
    backlog_cleared = np.minimum(backlog_cleared, A_alloc)
    
    # Compute Q_inv_base (part that serves base_Q_need)
    Q_inv_base = Q_inv - backlog_cleared
    
    # Stacked bars (bottom to top):
    # Bottom: Q_inv_base (delivered to serve base need)
    fig.add_trace(go.Bar(
        x=plot_df['hour'],
        y=Q_inv_base,
        name='Q_inv (delivered)',
        marker_color='green',
        opacity=0.7
    ))
    
    # Middle: backlog_cleared (actually realized backlog allocation)
    fig.add_trace(go.Bar(
        x=plot_df['hour'],
        y=backlog_cleared,
        name='Backlog cleared (allocated & delivered)',
        marker_color='yellow',
        opacity=0.7,
        base=Q_inv_base
    ))
    
    # Top: Q_deficit (unmet reactive power, shown with pattern)
    fig.add_trace(go.Bar(
        x=plot_df['hour'],
        y=plot_df['Q_deficit'],
        name='Remaining deficit (carried forward)',
        marker=dict(
            color='red',
            pattern=dict(shape='/', fillmode='overlay', solidity=0.5)
        ),
        opacity=0.6,
        base=Q_inv_base + backlog_cleared
    ))
    
    # Q_need_total as thin line overlay
    Q_need_total = plot_df['Q_need_total'].values
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=Q_need_total,
        name='Q_need (requested)',
        mode='lines',
        line=dict(color='darkgreen', width=1.5)
    ))
    
    # A_alloc as dotted line for Day 2+ only
    day2_plus_mask = plot_df['day_index'] > 0
    if day2_plus_mask.any():
        day2_plus_hours = plot_df.loc[day2_plus_mask, 'hour'].values
        day2_plus_A_alloc = plot_df.loc[day2_plus_mask, 'A_alloc'].values
        fig.add_trace(go.Scatter(
            x=day2_plus_hours,
            y=day2_plus_A_alloc,
            name='A_alloc (planned, Day2+)',
            mode='lines+markers',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=4),
            opacity=0.6
        ))
    
    # Q_avail line (capacity limit)
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_df['Q_avail'],
        name='Q_avail',
        mode='lines',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Q_limit line (EPDK / PF limit)
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_df['Q_limit'],
        name='Q_limit',
        mode='lines',
        line=dict(color='purple', width=2, dash='dot')
    ))
    
    # Add vertical lines to separate days
    horizon_days = plot_df['day_index'].max() + 1
    for day_idx in range(1, horizon_days):
        x_pos = day_idx * 24 - 0.5
        fig.add_vline(
            x=x_pos,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Day {day_idx} → Day {day_idx+1}",
            annotation_position="top"
        )
    
    # Highlight selected day if provided
    if selected_day is not None and show_full_horizon:
        day_start_hour = (selected_day - 1) * 24
        day_end_hour = selected_day * 24
        fig.add_vrect(
            x0=day_start_hour - 0.5,
            x1=day_end_hour - 0.5,
            fillcolor="yellow",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text=f"Day {selected_day}",
            annotation_position="top left"
        )
    
    fig.update_layout(
        barmode='stack',
        xaxis_title=f'Hour (0..{plot_df["hour"].max()})',
        yaxis_title='Q (MVAr)',
        title=f'Multi-day reactive power operation with deficit carry-over{title_suffix}',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_input_timeseries(df_input, prices, selected_day=None, show_full_horizon=True, title_prefix="Input"):
    """
    Plot input time series (P_total, Q_total, P_PV_max, prices).
    
    Args:
        df_input: DataFrame with columns: hour, P_total, Q_total, P_PV_max
        prices: array of prices
        selected_day: If provided and show_full_horizon=False, filter to this day
        show_full_horizon: If True, plot all days
        title_prefix: Prefix for plot title
    
    Returns:
        fig: Plotly figure
    """
    # Filter to selected day if specified
    if selected_day is not None and not show_full_horizon:
        day_mask = (df_input['hour'] >= (selected_day - 1) * 24) & (df_input['hour'] < selected_day * 24)
        plot_df = df_input[day_mask].copy()
        plot_prices = prices[day_mask.values] if len(prices) == len(df_input) else prices
        title_suffix = f" (Day {selected_day} only)"
    else:
        plot_df = df_input.copy()
        plot_prices = prices if len(prices) == len(df_input) else np.tile(prices, (len(df_input) + len(prices) - 1) // len(prices))[:len(df_input)]
        if selected_day is not None:
            title_suffix = f" (Full horizon, Day {selected_day} highlighted)"
        else:
            title_suffix = " (Full horizon)"
    
    fig = go.Figure()
    
    # P_total
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_df['P_total'],
        name='P_total',
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Q_total
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_df['Q_total'],
        name='Q_total',
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    # P_PV_max
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_df['P_PV_max'],
        name='P_PV_max',
        mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    # Prices (on secondary axis - but we'll use same axis with different scale in subtitle)
    # Actually, let's keep it on same axis but use a different color/style
    fig.add_trace(go.Scatter(
        x=plot_df['hour'],
        y=plot_prices,
        name='price (TL/MWh)',
        mode='lines',
        line=dict(color='orange', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Highlight selected day if provided
    if selected_day is not None and show_full_horizon:
        day_start_hour = (selected_day - 1) * 24
        day_end_hour = selected_day * 24
        fig.add_vrect(
            x0=day_start_hour - 0.5,
            x1=day_end_hour - 0.5,
            fillcolor="yellow",
            opacity=0.1,
            layer="below",
            line_width=0
        )
    
    # Add vertical lines to separate days
    horizon_days = (plot_df['hour'].max() + 1) // 24
    for day_idx in range(1, horizon_days):
        x_pos = day_idx * 24 - 0.5
        fig.add_vline(
            x=x_pos,
            line_dash="dot",
            line_color="gray",
            opacity=0.5
        )
    
    fig.update_layout(
        xaxis_title=f'Hour (0..{plot_df["hour"].max()})',
        yaxis_title='P/Q/P_PV_max (MW/MVAr)',
        yaxis2=dict(
            title='Price (TL/MWh)',
            overlaying='y',
            side='right'
        ),
        title=f'{title_prefix} Time Series{title_suffix}',
        height=500,
        showlegend=True
    )
    
    return fig


# =========================================================
# F) Main UI
# =========================================================

# Sidebar
st.sidebar.header("Demo Data")

# Checkbox to toggle between demo files and manual upload
use_demo_files = st.sidebar.checkbox(
    "Use demo files (recommended)",
    value=True,
    help="If checked, select files from dropdowns. If unchecked, use file upload widgets below."
)

norwegian_file = None
ptf_file = None

if use_demo_files:
    # Demo file selection via dropdowns
    scenarios = list_scenario_files()
    ptf_files = list_ptf_files()
    default_ptf = get_default_ptf_path()
    
    if not scenarios:
        st.sidebar.warning("⚠️ No scenario files found in data/scenarios/")
    else:
        # Scenario dropdown
        scenario_options = {f.replace("_", " "): path for f, path in scenarios}
        selected_scenario_label = st.sidebar.selectbox(
            "Select Scenario",
            options=list(scenario_options.keys()),
            help="Select a scenario CSV file from data/scenarios/"
        )
        selected_scenario_path = scenario_options[selected_scenario_label]
        norwegian_file = str(selected_scenario_path)
    
    if not ptf_files:
        st.sidebar.warning("⚠️ No PTF files found in data/")
    else:
        # PTF dropdown with default selection
        ptf_options = {name: path for name, path in ptf_files}
        ptf_option_names = list(ptf_options.keys())
        
        # Find default index
        default_index = 0
        if default_ptf and default_ptf.exists():
            default_ptf_name = default_ptf.name
            if default_ptf_name in ptf_option_names:
                default_index = ptf_option_names.index(default_ptf_name)
        
        selected_ptf_name = st.sidebar.selectbox(
            "Select PTF Price File",
            options=ptf_option_names,
            index=default_index,
            help="Select a PTF price CSV file from data/"
        )
        ptf_file = str(ptf_options[selected_ptf_name])
    
    # Show selected files info
    if norwegian_file and ptf_file:
        st.sidebar.info(f"""
        **Selected Files:**
        - Scenario: `{Path(norwegian_file).stem}`
        - PTF: `{Path(ptf_file).name}`
        """)
else:
    # Manual file upload (original behavior)
    st.sidebar.header("Input Files")
    norwegian_file = st.sidebar.file_uploader(
        "Upload Norwegian_case.csv",
        type=['csv'],
        key='norwegian'
    )
    
    ptf_file = st.sidebar.file_uploader(
        "Upload PTF Price CSV",
        type=['csv'],
        key='ptf'
    )

st.sidebar.header("Parameters")

S_rated = st.sidebar.number_input(
    "S_rated (MVA)",
    min_value=0.0,
    value=10.0,
    step=0.5
)

pf_limit = st.sidebar.slider(
    "PF_limit",
    min_value=0.90,
    max_value=0.999,
    value=0.98,
    step=0.001
)

strategy = st.sidebar.selectbox(
    "Backlog Allocation Strategy",
    ["uniform", "price_weighted", "price_feasible", "economic_feasible"],
    format_func=lambda x: {
        "uniform": "Strategy 1: Uniform",
        "price_weighted": "Strategy 2: Price-weighted",
        "price_feasible": "Strategy 3: Price-feasible reallocation",
        "economic_feasible": "Strategy 3: Economic + Feasible (buy vs penalty)"
    }[x]
)

# Economic parameters (for Strategy 3: Economic + Feasible)
st.sidebar.header("Economic Parameters")
alpha = st.sidebar.slider(
    "Alpha (reactive price multiplier)",
    min_value=0.01,
    max_value=1.0,
    value=0.10,
    step=0.01,
    help="effective_reactive_price = alpha * PTF_price"
)

penalty_rate_TL_per_MVArh = st.sidebar.number_input(
    "Penalty rate (TL/MVArh)",
    min_value=0.0,
    value=200.0,
    step=10.0,
    help="If effective_reactive_price >= penalty_rate, buying is not economical"
)

# Horizon selection
st.sidebar.header("Simulation Horizon")
horizon_option = st.sidebar.selectbox(
    "Horizon",
    ["48h (current)", "7 days", "14 days", "30 days"],
    index=0,
    help="Select simulation horizon. 48h preserves existing behavior."
)

# Map horizon option to days
horizon_map = {
    "48h (current)": 2,
    "7 days": 7,
    "14 days": 14,
    "30 days": 30
}
horizon_days = horizon_map[horizon_option]

# Day selector for multi-day views
day_to_view = None
if horizon_days > 2:
    day_to_view = st.sidebar.selectbox(
        "Day Selector",
        options=list(range(1, horizon_days + 1)),
        index=0,
        help="Select which day to inspect in detailed views"
    )

accumulate_backlog = st.sidebar.checkbox(
    "Accumulate backlog",
    value=st.session_state.accumulate_backlog,
    help="If checked, new backlog accumulates. If unchecked, backlog is replaced."
)

if st.sidebar.button("Reset Backlog", help="Reset persistent backlog to 0"):
    st.session_state.backlog_prev_total = 0.0
    st.success("Backlog reset to 0")

# Main area
if norwegian_file is None or ptf_file is None:
    st.info("Please upload both CSV files to begin.")
else:
    try:
        # =========================================================
        # BRANCH: 48h (existing) OR multi-day path
        # =========================================================
        if horizon_days == 2:
            # =========================================================
            # EXISTING 48H PATH (UNCHANGED)
            # =========================================================
            # Load files
            df_case = load_case(norwegian_file)
            prices = load_prices(ptf_file)
            
            # Run simulation
            results_df, Day1_backlog_total, Day2_backlog_total, reallocation_info, economic_info, diagnostics_info = simulate_two_days(
                df_case, prices, S_rated, pf_limit, strategy, st.session_state.backlog_prev_total,
                alpha=alpha, penalty_rate=penalty_rate_TL_per_MVArh
            )
            
            # Update persistent backlog
            if accumulate_backlog:
                st.session_state.backlog_prev_total += Day2_backlog_total
            else:
                st.session_state.backlog_prev_total = Day2_backlog_total
            
            # Display metrics
            st.info("""
            **Market Interaction Logic:**
            - **Day 1:** The system requests Q_need from the market. The market delivers Q_inv (limited by Q_avail). 
              Any unmet part becomes Q_deficit and is carried forward.
            - **Day 2:** The system requests Q_need_total = Q_need_day2 (Day2 own need) + A_alloc (Day1 deficit allocation).
              The market delivers Q_inv (limited by Q_avail). Any unmet part becomes Q_deficit (carried to Day3/backlog).
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Day 1 Q_deficit Total", f"{Day1_backlog_total:.3f} MVArh", 
                         help="Sum of Q_deficit from Day 1 (becomes backlog for Day 2)")
            with col2:
                st.metric("Day 2 Q_deficit Total", f"{Day2_backlog_total:.3f} MVArh",
                         help="Sum of Q_deficit from Day 2")
            with col3:
                st.metric("Persistent Q_deficit (next run)", f"{st.session_state.backlog_prev_total:.3f} MVArh",
                         help="Q_deficit carried to next simulation run")
            
            # Display main results table
            st.header("Main Results Table (48 hours)")
            display_df = results_df.copy()
            display_df = display_df.round(4)
            st.dataframe(display_df, use_container_width=True)
            
            # Debug expander for Q_avail verification
            with st.expander("🔍 Debug: Q_avail Calculation Verification"):
                st.caption("Verify that Q_avail = S_rated/2 when P_PV_max = 0 (night hours)")
                debug_df = results_df[['hour', 'day_index', 'hour_in_day', 'P_PV_max', 'Q_avail']].copy()
                debug_df['S_rated'] = S_rated
                debug_df['Expected_Q_avail_night'] = np.where(debug_df['P_PV_max'] == 0, S_rated / 2.0, 
                                                              np.sqrt(np.maximum(S_rated**2 - debug_df['P_PV_max']**2, 0)))
                debug_df = debug_df.round(4)
                st.dataframe(debug_df, use_container_width=True)
                
                # Sanity check
                night_mask = debug_df['P_PV_max'] == 0
                if night_mask.any():
                    night_q_avail = debug_df.loc[night_mask, 'Q_avail'].values
                    expected_night = S_rated / 2.0
                    if np.any(np.abs(night_q_avail - expected_night) > 1e-3):
                        st.warning(f"⚠️ Some night hours have Q_avail ≠ S_rated/2. Expected: {expected_night:.3f}")
                    else:
                        st.success(f"✅ Night hours (P_PV_max=0) correctly show Q_avail = {expected_night:.3f} MVAr")
            
            # Plots
            st.header("Visualizations")
            
            # 48-hour timeline visualization
            st.subheader("48-hour Reactive Power Operation Timeline")
            fig_48h = plot_48hour_timeline(results_df)
            st.plotly_chart(fig_48h, use_container_width=True)
            
            # Day 2 Diagnostics Table
            st.subheader("Day 2 Diagnostics Table")
            st.caption("Verification table showing capacity-aware backlog allocation. Includes freecap, A_alloc_raw, and final allocation.")
            
            # Build diagnostics table from diagnostics_info
            diagnostics_df = pd.DataFrame({
                'hour_in_day': diagnostics_info['hour_in_day'],
                'price': diagnostics_info['price'],
                'Q_avail': diagnostics_info['Q_avail'],
                'base_Q_need': diagnostics_info['base_Q_need'],
                'freecap': diagnostics_info['freecap'],
                'A_alloc_raw': diagnostics_info['A_alloc_raw'],
                'A_alloc': diagnostics_info['A_alloc'],
                'Q_need_final': diagnostics_info['Q_need_final'],
                'Q_inv': diagnostics_info['Q_inv'],
                'Q_deficit': diagnostics_info['Q_deficit']
            })
            
            # Add economic columns if using economic strategy
            if strategy == "economic_feasible" and economic_info:
                diagnostics_df['effective_reactive_price'] = economic_info['effective_reactive_price']
                diagnostics_df['penalty_rate'] = economic_info['penalty_rate']
                diagnostics_df['eligible'] = economic_info['eligible'].astype(int)  # Convert bool to int for display
                
                # Reorder columns
                diagnostics_df = diagnostics_df[['hour_in_day', 'price', 'effective_reactive_price', 'penalty_rate', 
                                                 'eligible', 'Q_avail', 'base_Q_need', 'freecap', 
                                                 'A_alloc_raw', 'A_alloc', 'Q_need_final', 'Q_inv', 'Q_deficit']]
            
            diagnostics_df = diagnostics_df.round(4)
            st.dataframe(diagnostics_df, use_container_width=True)
            
            # Show leftover unallocated if any
            leftover = diagnostics_info['leftover_unallocated']
            if leftover > 1e-6:
                st.info(f"ℹ️ **Leftover unallocated backlog:** {leftover:.4f} MVArh (could not be allocated due to capacity constraints)")
            
            # Show economic summary if using economic strategy
            if strategy == "economic_feasible" and economic_info:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Allocated", f"{economic_info['allocated_total']:.3f} MVArh",
                             help="Amount of Day1 backlog allocated to Day2")
                with col2:
                    st.metric("Remaining Unallocated", f"{economic_info['remaining_unallocated']:.3f} MVArh",
                             help="Intentionally left as deficit (buying not economical or no capacity)")
            
            # Strategy 3 reallocation table (technical details, internal use)
            if strategy == "price_feasible" and reallocation_info:
                st.subheader("Strategy 3: Technical Reallocation Details (Internal)")
                st.caption("This table shows technical details of how backlog allocation was adjusted to respect Q_avail limits.")
                realloc_df = pd.DataFrame({
                    'hour_in_day': reallocation_info['day2_hours'],
                    'price': reallocation_info['prices_day2'],
                    'Q_need_base': reallocation_info['Q_need_base_day2'],
                    'Q_need_final': reallocation_info['Q_need_day2'],
                    'Q_avail': reallocation_info['Q_avail_day2'],
                    'A_initial': reallocation_info['A_initial'],
                    'A_final': reallocation_info['A_final'],
                    'reallocated_amount': reallocation_info['reallocated_amount']
                })
                realloc_df = realloc_df.round(4)
                st.dataframe(realloc_df, use_container_width=True)
            
            # Download button
            st.header("Download Results")
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="reactive_power_results.csv",
                mime="text/csv"
            )
        
        else:
            # =========================================================
            # MULTI-DAY PATH (7/14/30 days)
            # =========================================================
            # Load files for multi-day
            df_case_N = load_case_multiday(norwegian_file, horizon_days)
            prices_N = load_prices_multiday(ptf_file, horizon_days)
            
            # Run multi-day simulation
            results_df_N, daily_summary_df, cohort_df = simulate_multi_days(
                df_case_N, prices_N, S_rated, pf_limit, strategy, st.session_state.backlog_prev_total,
                alpha=alpha, penalty_rate=penalty_rate_TL_per_MVArh
            )
            
            # Update persistent backlog (use last day's backlog_out)
            last_day_backlog = daily_summary_df.iloc[-1]['backlog_out'] if len(daily_summary_df) > 0 else 0.0
            if accumulate_backlog:
                st.session_state.backlog_prev_total += last_day_backlog
            else:
                st.session_state.backlog_prev_total = last_day_backlog
            
            # Display metrics
            st.info(f"""
            **Multi-Day Simulation ({horizon_days} days):**
            - Each day computes base Q_need from PF rule, allocates previous day's backlog, delivers Q_inv, and tracks deficit.
            - Backlog is allocated using the selected strategy (uniform/price_weighted/price_feasible/economic_feasible).
            - Deficit is carried forward day-by-day using the same logic as the 48h simulation.
            """)
            
            # Display daily summary table
            st.header("Daily Summary Table")
            st.caption("Summary of backlog flow and delivery for each day")
            display_summary = daily_summary_df.copy()
            display_summary = display_summary.round(4)
            st.dataframe(display_summary, use_container_width=True)
            
            # Day 8 deficit KPI (when horizon >= 7)
            if horizon_days >= 7:
                st.header("Day 8 Deficit Metrics")
                day7_backlog = daily_summary_df.iloc[6]['backlog_out'] if len(daily_summary_df) > 6 else 0.0
                st.metric(
                    "Deficit carried into Day 8",
                    f"{day7_backlog:.3f} MVArh",
                    help="Sum of Q_deficit from Day 7 (backlog at end of Day 7)"
                )
                
                # Cohort breakdown table
                if cohort_df is not None and len(cohort_df) > 0:
                    st.subheader("Deficit Cohort Breakdown (at end of Day 7)")
                    st.caption("How much deficit from each origin day (Day 1-7) is still not covered at Day 8. Uses FIFO clearing rule.")
                    display_cohort = cohort_df.copy()
                    display_cohort = display_cohort.round(4)
                    # Sort by origin_day
                    display_cohort = display_cohort.sort_values('origin_day')
                    st.dataframe(display_cohort, use_container_width=True)
                    
                    # Show Day 1 remaining if present
                    day1_remaining = cohort_df[cohort_df['origin_day'] == 1]['remaining_amount'].values
                    if len(day1_remaining) > 0 and day1_remaining[0] > 1e-6:
                        st.metric(
                            "Uncovered deficit from Day 1 still remaining at Day 8",
                            f"{day1_remaining[0]:.3f} MVArh",
                            help="Portion of Day 1 deficit that was not cleared by the end of Day 7"
                        )
                else:
                    st.success("✅ All deficit from Days 1-7 was cleared by the end of Day 7!")
            
            # Input visualization section
            st.header("Input Visualization")
            
            # Toggle for showing selected day only
            show_day_only = st.checkbox(
                f"Show only Day {day_to_view}" if day_to_view else "Show selected day only",
                value=False,
                key="show_day_only_input"
            )
            
            # Input time series plot
            st.subheader("Scenario Inputs and PTF Prices")
            fig_input = plot_input_timeseries(
                df_case_N[['hour', 'P_total', 'Q_total', 'P_PV_max']],
                prices_N,
                selected_day=day_to_view,
                show_full_horizon=not show_day_only,
                title_prefix="Input"
            )
            st.plotly_chart(fig_input, use_container_width=True)
            
            # Input tables
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Scenario Inputs Table")
                input_df = df_case_N[['hour', 'P_total', 'Q_total', 'P_PV_max']].copy()
                if show_day_only and day_to_view:
                    day_mask = (input_df['hour'] >= (day_to_view - 1) * 24) & (input_df['hour'] < day_to_view * 24)
                    input_df = input_df[day_mask]
                input_df = input_df.round(4)
                st.dataframe(input_df, use_container_width=True, height=400)
            
            with col2:
                st.subheader("PTF Prices Table")
                prices_df = pd.DataFrame({
                    'hour': np.arange(len(prices_N)),
                    'price': prices_N
                })
                if show_day_only and day_to_view:
                    day_mask = (prices_df['hour'] >= (day_to_view - 1) * 24) & (prices_df['hour'] < day_to_view * 24)
                    prices_df = prices_df[day_mask]
                prices_df = prices_df.round(4)
                st.dataframe(prices_df, use_container_width=True, height=400)
            
            # Output visualization section
            st.header("Output Visualization")
            
            # Toggle for showing selected day only
            show_day_only_output = st.checkbox(
                f"Show only Day {day_to_view}" if day_to_view else "Show selected day only",
                value=False,
                key="show_day_only_output"
            )
            
            # Multi-day timeline plot
            st.subheader("Multi-Day Reactive Power Operation Timeline")
            fig_multiday = plot_multi_day_timeline(
                results_df_N,
                selected_day=day_to_view,
                show_full_horizon=not show_day_only_output
            )
            st.plotly_chart(fig_multiday, use_container_width=True)
            
            # Results table (optional: filter by selected day)
            st.subheader("Detailed Results Table")
            display_results = results_df_N.copy()
            if show_day_only_output and day_to_view:
                day_mask = (display_results['day_index'] == (day_to_view - 1))
                display_results = display_results[day_mask]
            # Remove internal columns if any
            display_results = display_results.round(4)
            st.dataframe(display_results, use_container_width=True, height=400)
            
            # Download button
            st.header("Download Results")
            csv = results_df_N.to_csv(index=False)
            st.download_button(
                label="Download Multi-Day Results as CSV",
                data=csv,
                file_name=f"reactive_power_results_{horizon_days}days.csv",
                mime="text/csv"
            )
            
            # Download daily summary
            csv_summary = daily_summary_df.to_csv(index=False)
            st.download_button(
                label="Download Daily Summary as CSV",
                data=csv_summary,
                file_name=f"daily_summary_{horizon_days}days.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)
