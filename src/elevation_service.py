"""
Elevation Service for EV Trip Modeling.
Uses NASA/ESA SRTM 30m Digital Elevation Model (N42W084.hgt) for Ann Arbor, MI,
with spatial sampling (~50m), local caching (data/processed/elevation_cache.parquet),
and deadband filtering (0.5m) to avoid altimetric noise.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANN_ARBOR_MEAN_ELEVATION_M = 260.0
DEM_PATH = Path("data/raw/dem/N42W084.hgt")


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)


class ElevationService:
    """
    Manages elevation data retrieval using the 30m NASA/ESA SRTM DEM grid
    covering Ann Arbor, MI ([42N, 43N] x [-84W, -83W]), caching points
    and computing robust trip elevation metrics.
    """
    def __init__(
        self,
        cache_path: str = "data/processed/elevation_cache.parquet",
        dem_path: str = "data/raw/dem/N42W084.hgt"
    ):
        self.cache_path = Path(cache_path)
        self.dem_path = Path(dem_path)
        self.cache: Dict[Tuple[float, float], float] = {}
        self.dem_grid: Optional[np.ndarray] = None
        
        self._load_dem()
        self._load_cache()

    def _load_dem(self):
        """Loads 3601 x 3601 16-bit big-endian signed integer SRTM DEM grid."""
        if self.dem_path.exists():
            try:
                with open(self.dem_path, "rb") as f:
                    self.dem_grid = np.fromfile(f, dtype=">i2").reshape((3601, 3601))
                logger.info(f"Loaded SRTM 30m DEM tile from {self.dem_path} (shape: {self.dem_grid.shape})")
            except Exception as e:
                logger.error(f"Error loading DEM file {self.dem_path}: {e}")
                self.dem_grid = None
        else:
            logger.warning(f"DEM file not found at {self.dem_path}. Will rely on cache or fallback.")

    def _load_cache(self):
        """Loads elevation cache from parquet if it exists."""
        if self.cache_path.exists():
            try:
                df = pd.read_parquet(self.cache_path)
                for _, row in df.iterrows():
                    self.cache[(round(row['lat_round'], 4), round(row['lon_round'], 4))] = float(row['elevation_m'])
                logger.info(f"Loaded {len(self.cache)} cached elevation coordinates from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Could not load elevation cache from {self.cache_path}: {e}")
        else:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def save_cache(self):
        """Saves memory cache to parquet file."""
        if not self.cache:
            return
        records = [
            {'lat_round': lat, 'lon_round': lon, 'elevation_m': elev}
            for (lat, lon), elev in self.cache.items()
            if not np.isnan(elev)
        ]
        df = pd.DataFrame(records)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        logger.info(f"Saved {len(records)} elevation points to {self.cache_path}")

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Looks up elevation (in meters) for a single coordinate.
        Queries memory cache first, then SRTM DEM grid.
        Returns np.nan if outside DEM or invalid (avoiding constant 260.0 cliffs).
        """
        pt_key = (round(lat, 4), round(lon, 4))
        if pt_key in self.cache:
            return self.cache[pt_key]

        if self.dem_grid is not None:
            # Tile N42W084 covers latitude [42.0, 43.0] and longitude [-84.0, -83.0]
            if 42.0 <= lat <= 43.0 and -84.0 <= lon <= -83.0:
                row = int(np.clip(np.round((43.0 - lat) * 3600.0), 0, 3600))
                col = int(np.clip(np.round((lon - (-84.0)) * 3600.0), 0, 3600))
                elev = float(self.dem_grid[row, col])
                if elev > -100:  # Valid SRTM reading (voids in SRTM are -32768)
                    self.cache[pt_key] = elev
                    return elev

        return np.nan

    def fetch_elevations_for_coords(self, coords: List[Tuple[float, float]]) -> None:
        """
        Populates elevation cache for all provided coordinates.
        """
        for lat, lon in coords:
            self.get_elevation(lat, lon)
        self.save_cache()

    def sample_trip_coordinates(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        min_step_m: float = 50.0
    ) -> List[Tuple[float, float]]:
        """
        Spatially samples trip coordinates so points are spaced by at least min_step_m,
        always including the start and end of the trip.
        """
        if len(lats) == 0:
            return []
        if len(lats) == 1:
            return [(round(float(lats[0]), 4), round(float(lons[0]), 4))]

        sampled = [(round(float(lats[0]), 4), round(float(lons[0]), 4))]
        last_lat = float(lats[0])
        last_lon = float(lons[0])

        for i in range(1, len(lats) - 1):
            cur_lat = float(lats[i])
            cur_lon = float(lons[i])
            dist = haversine_distance_m(last_lat, last_lon, cur_lat, cur_lon)
            if dist >= min_step_m:
                sampled.append((round(cur_lat, 4), round(cur_lon, 4)))
                last_lat = cur_lat
                last_lon = cur_lon

        # Always include the destination point
        end_pt = (round(float(lats[-1]), 4), round(float(lons[-1]), 4))
        if end_pt != sampled[-1]:
            sampled.append(end_pt)

        return sampled

    def calculate_trip_elevation_metrics(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        distance_km: float,
        deadband_m: float = 0.5,
        min_step_m: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculates:
        - elevation_gain_m: sum of positive elevation deltas >= deadband_m
        - elevation_loss_m: sum of negative elevation deltas <= -deadband_m
        - net_elevation_m: final_elevation - initial_elevation
        - initial_elevation_m: elevation at start of trip
        - final_elevation_m: elevation at end of trip
        - climbing_grade_percent: (elevation_gain_m / (distance_km * 1000)) * 100
        - net_grade_percent: (net_elevation_m / (distance_km * 1000)) * 100
        """
        sampled_coords = self.sample_trip_coordinates(lats, lons, min_step_m=min_step_m)
        
        if not sampled_coords:
            return {
                'elevation_gain_m': 0.0,
                'elevation_loss_m': 0.0,
                'net_elevation_m': 0.0,
                'initial_elevation_m': ANN_ARBOR_MEAN_ELEVATION_M,
                'final_elevation_m': ANN_ARBOR_MEAN_ELEVATION_M,
                'climbing_grade_percent': 0.0,
                'net_grade_percent': 0.0
            }

        raw_elevations = [self.get_elevation(lat, lon) for lat, lon in sampled_coords]
        
        # Robust linear interpolation along trajectory for any isolated missing points
        s_elev = pd.Series(raw_elevations, dtype=float)
        if s_elev.isna().any():
            s_elev = s_elev.interpolate(method='linear').bfill().ffill()
            if s_elev.isna().all():
                s_elev = s_elev.fillna(ANN_ARBOR_MEAN_ELEVATION_M)

        elevations = s_elev.tolist()

        initial_elev = elevations[0]
        final_elev = elevations[-1]
        net_elev = final_elev - initial_elev

        gain_m = 0.0
        loss_m = 0.0

        for i in range(len(elevations) - 1):
            dh = elevations[i+1] - elevations[i]
            if dh >= deadband_m:
                gain_m += dh
            elif dh <= -deadband_m:
                loss_m += abs(dh)

        dist_m = max(distance_km * 1000.0, 1.0)
        climbing_grade = (gain_m / dist_m) * 100.0
        net_grade = (net_elev / dist_m) * 100.0

        return {
            'elevation_gain_m': float(round(gain_m, 2)),
            'elevation_loss_m': float(round(loss_m, 2)),
            'net_elevation_m': float(round(net_elev, 2)),
            'initial_elevation_m': float(round(initial_elev, 2)),
            'final_elevation_m': float(round(final_elev, 2)),
            'climbing_grade_percent': float(round(climbing_grade, 4)),
            'net_grade_percent': float(round(net_grade, 4))
        }
