"""
Parking Compliance Advisor - Streamlit Web Application

A historical, probabilistic, and compliance-focused tool for analyzing parking
enforcement patterns and 311 complaint hotspots. This app does NOT track real-time
enforcement vehicle locations and does NOT suggest illegal parking.

Architecture:
    - Data Layer: CSV loading with validation and caching
    - Processing Layer: Grid-based risk scoring with temporal decay
    - Presentation Layer: Streamlit UI with pydeck visualizations

Author: Parking Compliance Advisor Team
Version: 1.1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta

# Configure logging for debugging (logs to console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Centralized constants with documentation
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """
    Immutable configuration for the Parking Compliance Advisor.
    
    All magic numbers are centralized here with explanations.
    Frozen dataclass ensures config cannot be accidentally mutated.
    """
    # Grid resolution: ~200m at Bay Area latitude
    # Calculation: 1° lat ≈ 111km, so 0.0018° ≈ 200m
    GRID_RESOLUTION_DEG: float = 0.0018
    
    # Geographic bounds for input validation (Bay Area)
    LAT_MIN: float = 36.5   # South of Monterey
    LAT_MAX: float = 38.5   # North of SF
    LON_MIN: float = -123.0  # Pacific coast
    LON_MAX: float = -121.0  # East Bay hills
    
    # Default map center (midpoint between SF and SJ)
    DEFAULT_LAT: float = 37.5
    DEFAULT_LON: float = -122.1
    
    # Risk score weights (must sum to 1.0)
    SF_WEIGHT: float = 0.7
    SJ_WEIGHT: float = 0.3
    
    # Temporal decay settings
    DEFAULT_HALFLIFE_DAYS: int = 7
    MIN_HALFLIFE_DAYS: int = 1
    MAX_HALFLIFE_DAYS: int = 30
    
    # Search parameters
    DEFAULT_SEARCH_RADIUS_CELLS: int = 5
    MAX_NEARBY_RESULTS: int = 5
    
    # Visualization
    HEATMAP_RADIUS_PIXELS: int = 30
    HEATMAP_OPACITY: float = 0.6
    GRID_POINT_RADIUS_METERS: int = 100
    
    def validate(self) -> None:
        """Validate configuration invariants."""
        assert abs(self.SF_WEIGHT + self.SJ_WEIGHT - 1.0) < 0.001, \
            "Weights must sum to 1.0"
        assert self.LAT_MIN < self.LAT_MAX, "Invalid latitude bounds"
        assert self.LON_MIN < self.LON_MAX, "Invalid longitude bounds"


# Global config instance
CONFIG = AppConfig()
CONFIG.validate()  # Fail fast if config is invalid

# =============================================================================
# INPUT VALIDATION
# =============================================================================

class ValidationError(Exception):
    """Custom exception for input validation failures."""
    pass


def validate_coordinates(lat: float, lon: float, context: str = "input") -> Tuple[float, float]:
    """
    Validate that coordinates are within acceptable bounds.
    
    Args:
        lat: Latitude value to validate
        lon: Longitude value to validate
        context: Description of where these coordinates came from (for error messages)
    
    Returns:
        Tuple of (lat, lon) if valid
        
    Raises:
        ValidationError: If coordinates are outside Bay Area bounds
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationError(f"Coordinates must be numeric, got lat={type(lat)}, lon={type(lon)}")
    
    if np.isnan(lat) or np.isnan(lon):
        raise ValidationError(f"Coordinates cannot be NaN in {context}")
    
    if not (CONFIG.LAT_MIN <= lat <= CONFIG.LAT_MAX):
        raise ValidationError(
            f"Latitude {lat} in {context} outside valid range "
            f"[{CONFIG.LAT_MIN}, {CONFIG.LAT_MAX}]"
        )
    
    if not (CONFIG.LON_MIN <= lon <= CONFIG.LON_MAX):
        raise ValidationError(
            f"Longitude {lon} in {context} outside valid range "
            f"[{CONFIG.LON_MIN}, {CONFIG.LON_MAX}]"
        )
    
    return float(lat), float(lon)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str], source: str) -> pd.DataFrame:
    """
    Validate DataFrame has required columns and clean data.
    
    Args:
        df: DataFrame to validate
        required_cols: List of column names that must exist
        source: Name of data source for error messages
        
    Returns:
        Cleaned DataFrame with invalid rows removed
        
    Raises:
        ValidationError: If required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(f"{source} missing required columns: {missing}")
    
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with missing values from {source}")
    
    return df


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sf_tickets(filepath: str = "sf_tickets_last30.csv") -> pd.DataFrame:
    """
    Load San Francisco parking citations data with validation.
    
    Expected columns: timestamp, latitude, longitude
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with parsed datetime and extracted time features.
        Returns empty DataFrame on error (with warning displayed).
    """
    required_cols = ['timestamp', 'latitude', 'longitude']
    
    try:
        logger.info(f"Loading SF tickets from {filepath}")
        df = pd.read_csv(filepath)
        df = validate_dataframe(df, required_cols, "SF tickets")
        
        # Parse timestamp with explicit format handling
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Validate coordinates are within bounds
        valid_mask = (
            (df['latitude'] >= CONFIG.LAT_MIN) & 
            (df['latitude'] <= CONFIG.LAT_MAX) &
            (df['longitude'] >= CONFIG.LON_MIN) & 
            (df['longitude'] <= CONFIG.LON_MAX)
        )
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Filtered {invalid_count} SF tickets with out-of-bounds coordinates")
        df = df[valid_mask].copy()
        
        # Extract time features for filtering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_name'] = df['timestamp'].dt.day_name()
        
        logger.info(f"Loaded {len(df)} valid SF tickets")
        return df
        
    except FileNotFoundError:
        st.warning(f"File not found: {filepath}. Using sample data for demonstration.")
        logger.warning(f"File not found: {filepath}, generating sample data")
        return generate_sample_sf_data()
    except ValidationError as e:
        st.error(f"Data validation error: {e}")
        logger.error(f"Validation error in SF tickets: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading SF tickets: {e}")
        logger.exception(f"Unexpected error loading SF tickets")
        return pd.DataFrame()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sj_complaints(filepath: str = "sj_illegal_parking_last30.csv") -> pd.DataFrame:
    """
    Load San Jose 311 illegal parking complaints data with validation.
    
    Expected columns: timestamp, latitude, longitude
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with parsed datetime and extracted time features.
        Returns empty DataFrame on error (with warning displayed).
    """
    required_cols = ['timestamp', 'latitude', 'longitude']
    
    try:
        logger.info(f"Loading SJ complaints from {filepath}")
        df = pd.read_csv(filepath)
        df = validate_dataframe(df, required_cols, "SJ complaints")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Validate coordinates
        valid_mask = (
            (df['latitude'] >= CONFIG.LAT_MIN) & 
            (df['latitude'] <= CONFIG.LAT_MAX) &
            (df['longitude'] >= CONFIG.LON_MIN) & 
            (df['longitude'] <= CONFIG.LON_MAX)
        )
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Filtered {invalid_count} SJ complaints with out-of-bounds coordinates")
        df = df[valid_mask].copy()
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        
        logger.info(f"Loaded {len(df)} valid SJ complaints")
        return df
        
    except FileNotFoundError:
        st.warning(f"File not found: {filepath}. Using sample data for demonstration.")
        logger.warning(f"File not found: {filepath}, generating sample data")
        return generate_sample_sj_data()
    except ValidationError as e:
        st.error(f"Data validation error: {e}")
        logger.error(f"Validation error in SJ complaints: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading SJ complaints: {e}")
        logger.exception(f"Unexpected error loading SJ complaints")
        return pd.DataFrame()


def generate_sample_sf_data() -> pd.DataFrame:
    """
    Generate sample SF ticket data for demonstration purposes.
    
    Creates realistic-looking data clustered around downtown SF.
    Uses fixed seed for reproducibility.
    
    Returns:
        DataFrame with ~500 sample ticket records
    """
    np.random.seed(42)  # Reproducibility
    n_samples = 500
    
    # San Francisco downtown center
    lat_center, lon_center = 37.7749, -122.4194
    
    # Generate timestamps over the last 30 days with realistic distribution
    # More tickets during business hours
    base_time = datetime.now() - timedelta(days=30)
    timestamps = []
    for _ in range(n_samples):
        day_offset = np.random.randint(0, 30)
        # Weighted hour selection: more during 8am-6pm
        hour = np.random.choice(
            range(24),
            p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05,
               0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07,
               0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
        )
        timestamps.append(base_time + timedelta(days=day_offset, hours=hour))
    
    # Generate locations with clustering (downtown hotspots)
    lats = lat_center + np.random.normal(0, 0.015, n_samples)
    lons = lon_center + np.random.normal(0, 0.015, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons
    })
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    
    logger.info(f"Generated {len(df)} sample SF tickets")
    return df


def generate_sample_sj_data() -> pd.DataFrame:
    """
    Generate sample SJ complaint data for demonstration purposes.
    
    Creates realistic-looking data clustered around downtown San Jose.
    Uses different seed than SF for variety.
    
    Returns:
        DataFrame with ~300 sample complaint records
    """
    np.random.seed(123)  # Different seed for variety
    n_samples = 300
    
    # San Jose downtown center
    lat_center, lon_center = 37.3382, -121.8863
    
    # Generate timestamps - complaints more spread throughout day
    base_time = datetime.now() - timedelta(days=30)
    timestamps = []
    for _ in range(n_samples):
        day_offset = np.random.randint(0, 30)
        hour = np.random.choice(
            range(24),
            p=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04,
               0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
               0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02]
        )
        timestamps.append(base_time + timedelta(days=day_offset, hours=hour))
    
    # Generate locations
    lats = lat_center + np.random.normal(0, 0.012, n_samples)
    lons = lon_center + np.random.normal(0, 0.012, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'latitude': lats,
        'longitude': lons
    })
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    
    logger.info(f"Generated {len(df)} sample SJ complaints")
    return df


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def filter_by_time(
    df: pd.DataFrame, 
    hours: List[int], 
    days: List[int]
) -> pd.DataFrame:
    """
    Filter DataFrame by selected hours and days of week.
    
    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns
        hours: List of hours (0-23) to include
        days: List of day indices (0=Monday, 6=Sunday) to include
    
    Returns:
        Filtered DataFrame (copy to avoid modifying original)
        
    Note:
        Returns empty DataFrame if input is empty or no matches found
    """
    if df.empty:
        return df.copy()
    
    if not hours or not days:
        logger.warning("Empty filter criteria provided, returning empty DataFrame")
        return df.iloc[0:0].copy()  # Return empty with same columns
    
    mask = df['hour'].isin(hours) & df['day_of_week'].isin(days)
    filtered = df[mask].copy()
    
    logger.debug(f"Filtered {len(df)} rows to {len(filtered)} rows")
    return filtered


# =============================================================================
# RISK SCORING FUNCTIONS - Core business logic
# =============================================================================

def lat_lon_to_grid_cell(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert latitude/longitude to grid cell center coordinates.
    
    Uses floor division to snap to grid, then adds half resolution
    to get cell center.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
    
    Returns:
        Tuple of (grid_lat, grid_lon) representing cell center
        
    Example:
        >>> lat_lon_to_grid_cell(37.7749, -122.4194)
        (37.7748, -122.4194)  # Snapped to nearest grid cell
    """
    resolution = CONFIG.GRID_RESOLUTION_DEG
    grid_lat = round(lat / resolution) * resolution
    grid_lon = round(lon / resolution) * resolution
    return (grid_lat, grid_lon)


def compute_recency_weights(
    timestamps: pd.Series, 
    halflife_days: float,
    reference_time: Optional[datetime] = None
) -> np.ndarray:
    """
    Compute exponential decay weights based on recency.
    
    More recent events get higher weights following:
        weight = 0.5^(days_ago / halflife)
    
    Args:
        timestamps: Series of datetime objects
        halflife_days: Number of days for weight to decay to 0.5
        reference_time: Time to measure from (default: now)
    
    Returns:
        Array of weights in range (0, 1]
        
    Raises:
        ValueError: If halflife_days <= 0
    """
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be positive, got {halflife_days}")
    
    if reference_time is None:
        reference_time = pd.Timestamp.now()
    else:
        reference_time = pd.Timestamp(reference_time)
    
    days_ago = (reference_time - timestamps).dt.total_seconds() / (24 * 3600)
    
    # Handle negative days (future timestamps) by clamping to 0
    days_ago = np.maximum(days_ago.values, 0)
    
    # Exponential decay: weight = 0.5^(days_ago / halflife)
    # Equivalent to: weight = exp(-ln(2) * days_ago / halflife)
    decay_rate = np.log(2) / halflife_days
    weights = np.exp(-decay_rate * days_ago)
    
    return weights


def compute_grid_scores(
    df: pd.DataFrame, 
    halflife_days: float
) -> pd.DataFrame:
    """
    Compute risk scores for each grid cell from event data.
    
    Algorithm:
    1. Assign each event to a grid cell
    2. Apply recency-based weights
    3. Sum weighted counts per cell
    4. Normalize to [0, 1] range
    
    Args:
        df: DataFrame with 'latitude', 'longitude', 'timestamp' columns
        halflife_days: Halflife for exponential decay weighting
    
    Returns:
        DataFrame with columns: 
        - grid_lat, grid_lon: Cell center coordinates
        - raw_score: Sum of recency-weighted events
        - event_count: Raw count of events
        - normalized_score: Score scaled to [0, 1]
    """
    if df.empty:
        logger.debug("Empty DataFrame provided to compute_grid_scores")
        return pd.DataFrame(columns=[
            'grid_lat', 'grid_lon', 'raw_score', 'event_count', 'normalized_score'
        ])
    
    # Vectorized grid cell assignment (more efficient than apply)
    resolution = CONFIG.GRID_RESOLUTION_DEG
    grid_lats = np.round(df['latitude'].values / resolution) * resolution
    grid_lons = np.round(df['longitude'].values / resolution) * resolution
    
    # Compute recency weights
    weights = compute_recency_weights(df['timestamp'], halflife_days)
    
    # Create working DataFrame
    work_df = pd.DataFrame({
        'grid_lat': grid_lats,
        'grid_lon': grid_lons,
        'weight': weights
    })
    
    # Aggregate scores per grid cell
    grid_scores = work_df.groupby(['grid_lat', 'grid_lon']).agg(
        raw_score=('weight', 'sum'),
        event_count=('weight', 'count')
    ).reset_index()
    
    # Normalize scores to [0, 1] with safe division
    max_score = grid_scores['raw_score'].max()
    if max_score > 0:
        grid_scores['normalized_score'] = grid_scores['raw_score'] / max_score
    else:
        grid_scores['normalized_score'] = 0.0
    
    logger.debug(f"Computed scores for {len(grid_scores)} grid cells")
    return grid_scores


def compute_combined_risk_grid(
    sf_scores: pd.DataFrame, 
    sj_scores: pd.DataFrame,
    sf_weight: float = CONFIG.SF_WEIGHT,
    sj_weight: float = CONFIG.SJ_WEIGHT
) -> pd.DataFrame:
    """
    Combine SF and SJ scores into a unified risk grid.
    
    Uses outer join to include all cells from both datasets.
    Missing values filled with 0 (no data = no risk).
    
    Args:
        sf_scores: Grid scores from SF tickets
        sj_scores: Grid scores from SJ complaints
        sf_weight: Weight for SF scores (default from config)
        sj_weight: Weight for SJ scores (default from config)
    
    Returns:
        Combined risk grid DataFrame with per-source and combined scores
        
    Raises:
        ValueError: If weights are negative or don't sum to ~1.0
    """
    # Validate weights
    if sf_weight < 0 or sj_weight < 0:
        raise ValueError("Weights must be non-negative")
    if abs(sf_weight + sj_weight - 1.0) > 0.01:
        logger.warning(f"Weights sum to {sf_weight + sj_weight}, not 1.0")
    
    # Handle empty inputs
    if sf_scores.empty and sj_scores.empty:
        return pd.DataFrame(columns=[
            'grid_lat', 'grid_lon', 'sf_score', 'sj_score',
            'combined_score', 'combined_score_normalized'
        ])
    
    # Rename columns for clarity before merging
    sf_renamed = sf_scores.rename(columns={
        'normalized_score': 'sf_score',
        'raw_score': 'sf_raw_score',
        'event_count': 'sf_count'
    })
    
    sj_renamed = sj_scores.rename(columns={
        'normalized_score': 'sj_score',
        'raw_score': 'sj_raw_score',
        'event_count': 'sj_count'
    })
    
    # Merge on grid cell (outer join to include all cells)
    combined = pd.merge(
        sf_renamed, sj_renamed,
        on=['grid_lat', 'grid_lon'],
        how='outer'
    ).fillna(0)
    
    # Compute weighted combined score
    combined['combined_score'] = (
        sf_weight * combined['sf_score'] + 
        sj_weight * combined['sj_score']
    )
    
    # Normalize combined score to [0, 1]
    max_combined = combined['combined_score'].max()
    if max_combined > 0:
        combined['combined_score_normalized'] = combined['combined_score'] / max_combined
    else:
        combined['combined_score_normalized'] = 0.0
    
    logger.debug(f"Combined risk grid has {len(combined)} cells")
    return combined


def get_cell_risk(
    risk_grid: pd.DataFrame, 
    lat: float, 
    lon: float
) -> Dict[str, Any]:
    """
    Get risk information for a specific location.
    
    Looks up the grid cell containing the given coordinates.
    
    Args:
        risk_grid: Combined risk grid DataFrame
        lat: Target latitude
        lon: Target longitude
    
    Returns:
        Dictionary with:
        - grid_lat, grid_lon: Cell center
        - combined_score: Normalized combined risk (0-1)
        - sf_score, sj_score: Individual source scores
        - has_data: Whether any historical data exists for this cell
    """
    try:
        lat, lon = validate_coordinates(lat, lon, "destination lookup")
    except ValidationError as e:
        logger.warning(f"Invalid coordinates in get_cell_risk: {e}")
        return {
            'grid_lat': lat,
            'grid_lon': lon,
            'combined_score': 0.0,
            'sf_score': 0.0,
            'sj_score': 0.0,
            'has_data': False,
            'error': str(e)
        }
    
    grid_lat, grid_lon = lat_lon_to_grid_cell(lat, lon)
    
    # Find matching cell
    cell_data = risk_grid[
        (np.abs(risk_grid['grid_lat'] - grid_lat) < 1e-7) & 
        (np.abs(risk_grid['grid_lon'] - grid_lon) < 1e-7)
    ]
    
    if cell_data.empty:
        return {
            'grid_lat': grid_lat,
            'grid_lon': grid_lon,
            'combined_score': 0.0,
            'sf_score': 0.0,
            'sj_score': 0.0,
            'has_data': False
        }
    
    row = cell_data.iloc[0]
    return {
        'grid_lat': grid_lat,
        'grid_lon': grid_lon,
        'combined_score': float(row.get('combined_score_normalized', 0.0)),
        'sf_score': float(row.get('sf_score', 0.0)),
        'sj_score': float(row.get('sj_score', 0.0)),
        'has_data': True
    }


def find_nearby_lower_risk_cells(
    risk_grid: pd.DataFrame, 
    center_lat: float, 
    center_lon: float,
    search_radius_cells: int = CONFIG.DEFAULT_SEARCH_RADIUS_CELLS,
    max_results: int = CONFIG.MAX_NEARBY_RESULTS
) -> pd.DataFrame:
    """
    Find nearby grid cells with lower risk scores.
    
    Searches in a square grid around the target location and returns
    cells with lower risk, sorted by distance then risk.
    
    Args:
        risk_grid: Combined risk grid DataFrame
        center_lat: Target latitude
        center_lon: Target longitude
        search_radius_cells: Number of grid cells to search in each direction
        max_results: Maximum number of results to return
    
    Returns:
        DataFrame of lower-risk nearby cells, sorted by distance
    """
    if risk_grid.empty:
        return pd.DataFrame()
    
    center_grid_lat, center_grid_lon = lat_lon_to_grid_cell(center_lat, center_lon)
    center_risk = get_cell_risk(risk_grid, center_lat, center_lon)
    
    # Define search bounds
    search_distance = search_radius_cells * CONFIG.GRID_RESOLUTION_DEG
    lat_min = center_grid_lat - search_distance
    lat_max = center_grid_lat + search_distance
    lon_min = center_grid_lon - search_distance
    lon_max = center_grid_lon + search_distance
    
    # Filter to nearby cells
    nearby = risk_grid[
        (risk_grid['grid_lat'] >= lat_min) & 
        (risk_grid['grid_lat'] <= lat_max) &
        (risk_grid['grid_lon'] >= lon_min) & 
        (risk_grid['grid_lon'] <= lon_max)
    ].copy()
    
    if nearby.empty:
        return pd.DataFrame()
    
    # Filter to cells with strictly lower risk
    nearby = nearby[
        nearby['combined_score_normalized'] < center_risk['combined_score']
    ]
    
    if nearby.empty:
        return pd.DataFrame()
    
    # Calculate distance from center (in grid cells, for ranking)
    nearby['distance'] = np.sqrt(
        ((nearby['grid_lat'] - center_grid_lat) / CONFIG.GRID_RESOLUTION_DEG) ** 2 +
        ((nearby['grid_lon'] - center_grid_lon) / CONFIG.GRID_RESOLUTION_DEG) ** 2
    )
    
    # Sort by distance first, then by score
    nearby = nearby.sort_values(['distance', 'combined_score_normalized'])
    
    return nearby.head(max_results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_heatmap_layer(
    df: pd.DataFrame, 
    color: List[int], 
    layer_id: str
) -> pdk.Layer:
    """
    Create a pydeck HeatmapLayer from point data.
    
    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        color: RGB color for the layer (not used directly, heatmap uses gradient)
        layer_id: Unique identifier for the layer
    
    Returns:
        pydeck HeatmapLayer configured for visualization
    """
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=['longitude', 'latitude'],
        aggregation='"MEAN"',
        get_weight=1,
        radiusPixels=CONFIG.HEATMAP_RADIUS_PIXELS,
        opacity=CONFIG.HEATMAP_OPACITY,
        id=layer_id
    )


def create_scatterplot_layer(
    df: pd.DataFrame, 
    color: List[int], 
    radius: int = 50,
    layer_id: str = "scatter"
) -> pdk.Layer:
    """
    Create a pydeck ScatterplotLayer from point data.
    
    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        color: RGBA color for the points
        radius: Point radius in meters
        layer_id: Unique identifier for the layer
    
    Returns:
        pydeck ScatterplotLayer configured for visualization
    """
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=['longitude', 'latitude'],
        get_color=color,
        get_radius=radius,
        pickable=True,
        id=layer_id
    )


def score_to_color(score: float) -> List[int]:
    """
    Convert a risk score (0-1) to an RGBA color.
    
    Color gradient: Green (0) -> Yellow (0.5) -> Red (1.0)
    
    Args:
        score: Risk score between 0 and 1
        
    Returns:
        RGBA color as list of 4 integers [R, G, B, A]
    """
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    
    if score < 0.5:
        # Green to Yellow: increase red
        r = int(255 * (score * 2))
        g = 255
    else:
        # Yellow to Red: decrease green
        r = 255
        g = int(255 * (1 - (score - 0.5) * 2))
    
    return [r, g, 0, 150]  # Semi-transparent


def create_grid_layer(risk_grid: pd.DataFrame) -> pdk.Layer:
    """
    Create a pydeck layer showing risk grid cells with color-coded risk.
    
    Args:
        risk_grid: Combined risk grid DataFrame with 'combined_score_normalized'
    
    Returns:
        pydeck ScatterplotLayer with risk-colored cells
    """
    if risk_grid.empty:
        return pdk.Layer("ScatterplotLayer", data=[], id="risk_grid")
    
    # Create copy to avoid modifying original
    display_grid = risk_grid.copy()
    
    # Apply color mapping
    display_grid['color'] = display_grid['combined_score_normalized'].apply(score_to_color)
    
    return pdk.Layer(
        "ScatterplotLayer",
        data=display_grid,
        get_position=['grid_lon', 'grid_lat'],
        get_color='color',
        get_radius=CONFIG.GRID_POINT_RADIUS_METERS,
        pickable=True,
        id="risk_grid"
    )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to CSV bytes for download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        UTF-8 encoded CSV bytes
    """
    return df.to_csv(index=False).encode('utf-8')


# =============================================================================
# UI COMPONENTS
# =============================================================================

def inject_custom_css() -> None:
    """Inject custom CSS for a professional, non-generic look."""
    st.markdown("""
    <style>
    /* Global background and font - using system fonts */
    .stApp {
        background: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-top: 1rem;
    }
    
    /* Headers */
    h1 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        letter-spacing: -0.5px;
    }
    
    h2, h3, h4 {
        font-family: Georgia, 'Times New Roman', serif !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    /* Regular text */
    p, span, label, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #4a5568 !important;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1a202c !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #4a5568 !important;
    }
    
    /* Metrics cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #f7fafc, #edf2f7);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetric"] label {
        color: #2b6cb0 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-family: 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4299e1 0%, #2b6cb0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 500;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(66, 153, 225, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #63b3ed 0%, #4299e1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        box-shadow: 0 2px 8px rgba(72, 187, 120, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
        box-shadow: 0 4px 12px rgba(72, 187, 120, 0.4);
    }
    
    /* Input fields */
    .stNumberInput input,
    .stTextInput input {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #1a202c !important;
        font-family: 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background: transparent;
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(66, 153, 225, 0.15) !important;
        border: 1px solid rgba(66, 153, 225, 0.3) !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
    }
    
    /* Checkboxes */
    .stCheckbox span {
        color: #4a5568 !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: #f7fafc;
        border-radius: 10px;
        border-left: 4px solid #4299e1;
    }
    
    /* Tables */
    table {
        background: #ffffff;
        border-radius: 8px;
    }
    
    th {
        background: #edf2f7 !important;
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    td {
        color: #4a5568 !important;
        border-color: #e2e8f0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f7fafc;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_disclaimer() -> None:
    """Render the ethical disclaimer banner at the top of the app."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #f59e0b;
        padding: 20px 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">
        <h4 style="color: #b45309; margin: 0 0 12px 0; font-size: 1.1rem;">⚠️ Important Disclaimer</h4>
        <p style="color: #78350f; margin: 0 0 15px 0; font-size: 0.95rem; line-height: 1.7;">
            This tool provides <strong style="color: #b45309;">historical, probabilistic analysis</strong> of parking enforcement 
            patterns and complaint hotspots. It helps drivers find <strong style="color: #b45309;">legal parking</strong> 
            in compliance with local regulations.
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;">
            <span style="color: #dc2626;">✗ Does NOT track real-time enforcement</span>
            <span style="color: #16a34a;">✓ Historical data only</span>
            <span style="color: #dc2626;">✗ Does NOT suggest illegal parking</span>
            <span style="color: #16a34a;">✓ Always check posted signs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_filters() -> Dict[str, Any]:
    """
    Render sidebar filter controls and return selected values.
    
    Returns:
        Dictionary with filter settings:
        - selected_hours: List of selected hours (0-23)
        - selected_days: List of selected day indices (0=Mon, 6=Sun)
        - halflife_days: Recency decay halflife
        - sf_weight: Weight for SF ticket data
        - sj_weight: Weight for SJ complaint data
        - show_sf_heatmap: Whether to show SF heatmap
        - show_sj_heatmap: Whether to show SJ heatmap
        - show_risk_grid: Whether to show risk grid overlay
    """
    st.header("⚙️ Settings")
    
    # Data source info
    st.subheader("📁 Data Sources")
    st.info("""
    - **SF Tickets**: sf_tickets_last30.csv
    - **SJ Complaints**: sj_illegal_parking_last30.csv
    """)
    
    # Time filters
    st.subheader("🕐 Time Filters")
    
    # Hour selection with range slider
    hour_range = st.slider(
        "Hour of Day (0-23)",
        min_value=0,
        max_value=23,
        value=(0, 23),
        help="Filter data by hour of day. Useful for analyzing patterns during specific times."
    )
    selected_hours = list(range(hour_range[0], hour_range[1] + 1))
    
    # Day of week selection
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_day_names = st.multiselect(
        "Days of Week",
        options=day_names,
        default=day_names,
        help="Filter data by day of week. Weekdays may have different patterns than weekends."
    )
    selected_days = [day_names.index(d) for d in selected_day_names]
    
    # Risk scoring settings
    st.subheader("📊 Risk Scoring")
    
    halflife_days = st.slider(
        "Recency Halflife (days)",
        min_value=CONFIG.MIN_HALFLIFE_DAYS,
        max_value=CONFIG.MAX_HALFLIFE_DAYS,
        value=CONFIG.DEFAULT_HALFLIFE_DAYS,
        help="Days for event weight to decay by half. Lower values emphasize recent events more."
    )
    
    sf_weight = st.slider(
        "SF Ticket Weight",
        min_value=0.0,
        max_value=1.0,
        value=CONFIG.SF_WEIGHT,
        step=0.1,
        help="Weight for SF ticket data in combined score. Higher = more influence."
    )
    sj_weight = 1.0 - sf_weight
    st.caption(f"SJ Complaint Weight: {sj_weight:.1f}")
    
    # Layer visibility
    st.subheader("🗺️ Map Layers")
    show_sf_heatmap = st.checkbox("SF Enforcement Heatmap", value=True)
    show_sj_heatmap = st.checkbox("SJ Complaint Heatmap", value=True)
    show_risk_grid = st.checkbox("Risk Grid Overlay", value=False)
    
    return {
        'selected_hours': selected_hours,
        'selected_days': selected_days,
        'halflife_days': halflife_days,
        'sf_weight': sf_weight,
        'sj_weight': sj_weight,
        'show_sf_heatmap': show_sf_heatmap,
        'show_sj_heatmap': show_sj_heatmap,
        'show_risk_grid': show_risk_grid
    }


def render_metrics(
    sf_filtered: pd.DataFrame,
    sf_total: int,
    sj_filtered: pd.DataFrame,
    sj_total: int,
    risk_grid: pd.DataFrame
) -> None:
    """Render the summary metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "SF Tickets (Filtered)", 
            len(sf_filtered), 
            delta=f"of {sf_total} total"
        )
    
    with col2:
        st.metric(
            "SJ Complaints (Filtered)", 
            len(sj_filtered),
            delta=f"of {sj_total} total"
        )
    
    with col3:
        st.metric("Risk Grid Cells", len(risk_grid))
    
    with col4:
        if not risk_grid.empty:
            avg_risk = risk_grid['combined_score_normalized'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        else:
            st.metric("Avg Risk Score", "N/A")


def render_destination_lookup(risk_grid: pd.DataFrame) -> None:
    """Render the destination risk lookup section."""
    st.subheader("📍 Destination Risk Lookup")
    
    st.markdown("""
    Enter your destination coordinates to check the historical risk score and 
    find nearby lower-risk areas for **legal parking**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dest_lat = st.number_input(
            "Latitude",
            value=37.7749,
            min_value=CONFIG.LAT_MIN,
            max_value=CONFIG.LAT_MAX,
            format="%.6f",
            help=f"Enter destination latitude ({CONFIG.LAT_MIN} to {CONFIG.LAT_MAX})"
        )
    
    with col2:
        dest_lon = st.number_input(
            "Longitude",
            value=-122.4194,
            min_value=CONFIG.LON_MIN,
            max_value=CONFIG.LON_MAX,
            format="%.6f",
            help=f"Enter destination longitude ({CONFIG.LON_MIN} to {CONFIG.LON_MAX})"
        )
    
    if st.button("🔍 Check Risk", type="primary"):
        # Get risk for destination
        cell_risk = get_cell_risk(risk_grid, dest_lat, dest_lon)
        
        # Check for validation errors
        if 'error' in cell_risk:
            st.error(f"Invalid coordinates: {cell_risk['error']}")
            return
        
        # Display results
        st.markdown("---")
        
        if cell_risk['has_data']:
            # Determine risk level indicator
            if cell_risk['combined_score'] < 0.33:
                risk_color = "🟢"
                risk_level = "Low"
            elif cell_risk['combined_score'] < 0.66:
                risk_color = "🟡"
                risk_level = "Medium"
            else:
                risk_color = "🔴"
                risk_level = "High"
            
            st.markdown(f"""
            ### Results for ({dest_lat:.6f}, {dest_lon:.6f})
            
            | Metric | Value |
            |--------|-------|
            | **Grid Cell** | ({cell_risk['grid_lat']:.6f}, {cell_risk['grid_lon']:.6f}) |
            | **Combined Risk Score** | {risk_color} {cell_risk['combined_score']:.3f} ({risk_level}) |
            | **SF Ticket Score** | {cell_risk['sf_score']:.3f} |
            | **SJ Complaint Score** | {cell_risk['sj_score']:.3f} |
            """)
        else:
            st.info(
                "No historical data available for this location. "
                "Risk score: 0.0 (no recorded events in this grid cell)"
            )
        
        # Find nearby lower-risk cells
        st.markdown("### 🅿️ Nearby Lower-Risk Areas (Legal Parking Suggestions)")
        
        nearby_cells = find_nearby_lower_risk_cells(risk_grid, dest_lat, dest_lon)
        
        if not nearby_cells.empty:
            st.markdown("""
            *The following areas have historically lower enforcement activity 
            and fewer complaints. Always verify parking is legal at your chosen location.*
            """)
            
            display_cols = ['grid_lat', 'grid_lon', 'combined_score_normalized', 
                           'sf_score', 'sj_score', 'distance']
            nearby_display = nearby_cells[display_cols].copy()
            nearby_display.columns = ['Latitude', 'Longitude', 'Risk Score', 
                                      'SF Score', 'SJ Score', 'Distance (cells)']
            
            st.dataframe(
                nearby_display.style.format({
                    'Latitude': '{:.6f}',
                    'Longitude': '{:.6f}',
                    'Risk Score': '{:.3f}',
                    'SF Score': '{:.3f}',
                    'SJ Score': '{:.3f}',
                    'Distance (cells)': '{:.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info(
                "No lower-risk areas found nearby. This may already be a low-risk area, "
                "or there's insufficient data in the surrounding grid cells."
            )


def render_export_section(risk_grid: pd.DataFrame, sf_filtered: pd.DataFrame) -> None:
    """Render the data export section."""
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not risk_grid.empty:
            st.download_button(
                label="⬇️ Download Risk Grid CSV",
                data=convert_df_to_csv(risk_grid),
                file_name="parking_risk_grid.csv",
                mime="text/csv",
                help="Download the combined risk grid with all scores"
            )
        else:
            st.button("⬇️ Download Risk Grid CSV", disabled=True)
            st.caption("No risk grid data available")
    
    with col2:
        if not sf_filtered.empty:
            st.download_button(
                label="⬇️ Download Filtered SF Tickets CSV",
                data=convert_df_to_csv(sf_filtered),
                file_name="sf_tickets_filtered.csv",
                mime="text/csv",
                help="Download SF ticket data with current filters applied"
            )
        else:
            st.button("⬇️ Download Filtered SF Tickets CSV", disabled=True)
            st.caption("No SF ticket data available")


def render_footer() -> None:
    """Render the app footer."""
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center;
        padding: 25px;
        background: #f7fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
    ">
        <p style="color: #2b6cb0; font-size: 1rem; font-weight: 600; margin-bottom: 8px;">
            Parking Compliance Advisor
        </p>
        <p style="color: #718096; font-size: 0.85rem; margin-bottom: 5px;">
            Historical parking pattern analysis for the Bay Area
        </p>
        <p style="color: #a0aec0; font-size: 0.75rem; margin: 0;">
            Data is for informational purposes only • Always park legally • Check posted signs
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """
    Main Streamlit application entry point.
    
    Orchestrates the app flow:
    1. Page configuration
    2. Load and filter data
    3. Compute risk scores
    4. Render UI components
    """
    
    # Page configuration - must be first Streamlit command
    st.set_page_config(
        page_title="Parking Compliance Advisor",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom styling
    inject_custom_css()
    
    # Title and disclaimer
    st.title("🚗 Parking Compliance Advisor")
    render_disclaimer()
    
    # =========================================================================
    # SIDEBAR - Controls
    # =========================================================================
    
    with st.sidebar:
        filters = render_sidebar_filters()
    
    # =========================================================================
    # LOAD AND PROCESS DATA
    # =========================================================================
    
    # Load data (cached)
    sf_data = load_sf_tickets()
    sj_data = load_sj_complaints()
    
    # Apply time filters
    sf_filtered = filter_by_time(
        sf_data, 
        filters['selected_hours'], 
        filters['selected_days']
    )
    sj_filtered = filter_by_time(
        sj_data, 
        filters['selected_hours'], 
        filters['selected_days']
    )
    
    # Compute risk scores
    sf_scores = compute_grid_scores(sf_filtered, filters['halflife_days'])
    sj_scores = compute_grid_scores(sj_filtered, filters['halflife_days'])
    risk_grid = compute_combined_risk_grid(
        sf_scores, sj_scores, 
        filters['sf_weight'], filters['sj_weight']
    )
    
    # =========================================================================
    # DATA SUMMARY
    # =========================================================================
    
    render_metrics(sf_filtered, len(sf_data), sj_filtered, len(sj_data), risk_grid)
    
    # =========================================================================
    # MAP VISUALIZATION
    # =========================================================================
    
    st.subheader("🗺️ Interactive Map")
    
    # Build layers list based on user selections
    layers = []
    
    if filters['show_sf_heatmap'] and not sf_filtered.empty:
        layers.append(create_heatmap_layer(
            sf_filtered, 
            [255, 0, 0],  # Red
            "sf_heatmap"
        ))
    
    if filters['show_sj_heatmap'] and not sj_filtered.empty:
        layers.append(create_heatmap_layer(
            sj_filtered,
            [0, 0, 255],  # Blue
            "sj_heatmap"
        ))
    
    if filters['show_risk_grid'] and not risk_grid.empty:
        layers.append(create_grid_layer(risk_grid))
    
    # Calculate map center from data
    all_lats: List[float] = []
    all_lons: List[float] = []
    
    if not sf_filtered.empty:
        all_lats.extend(sf_filtered['latitude'].tolist())
        all_lons.extend(sf_filtered['longitude'].tolist())
    
    if not sj_filtered.empty:
        all_lats.extend(sj_filtered['latitude'].tolist())
        all_lons.extend(sj_filtered['longitude'].tolist())
    
    if all_lats and all_lons:
        center_lat = float(np.mean(all_lats))
        center_lon = float(np.mean(all_lons))
    else:
        center_lat = CONFIG.DEFAULT_LAT
        center_lon = CONFIG.DEFAULT_LON
    
    # Create map view
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=10,
        pitch=0
    )
    
    # Build deck with layers
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={
            "text": "Risk Score: {combined_score_normalized:.3f}\n"
                   "SF Score: {sf_score:.3f}\n"
                   "SJ Score: {sj_score:.3f}"
        }
    )
    
    st.pydeck_chart(deck)
    
    # Map legend
    st.markdown("""
    <div style="
        background: #f7fafc;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
    ">
        <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
            <strong style="color: #2d3748;">Legend:</strong>&nbsp;&nbsp;
            <span style="color: #dc2626;">●</span> SF Tickets&nbsp;&nbsp;
            <span style="color: #2563eb;">●</span> SJ Complaints&nbsp;&nbsp;
            <span style="color: #16a34a;">●</span>→<span style="color: #ca8a04;">●</span>→<span style="color: #dc2626;">●</span> Risk (Low→High)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # DESTINATION LOOKUP & EXPORTS
    # =========================================================================
    
    render_destination_lookup(risk_grid)
    render_export_section(risk_grid, sf_filtered)
    render_footer()


if __name__ == "__main__":
    main()
