"""
Trip Feature Extraction Pipeline for EV Energy Consumption Modeling.
Extracts complete trip-level features (1 row per trip, no fragmentation)
from 1Hz telemetry, incorporating:
- Distance & kinematics (speed, accelerations, idling)
- Battery state of charge (initial, final, delta SOC)
- Regeneration and Gross/Net energy (kWh)
- Climate & HVAC auxiliary power consumption (Heater + AC, OAT)
- Altimetric features (Open-Meteo elevation, gain, loss, grade)
- Driver profile distributions and dominant cluster
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.elevation_service import ElevationService, haversine_distance_m

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_1HZ_PATH = Path("data/processed/driving_features/ev_trips_1hz.parquet")
DATA_CLUSTERS_PATH = Path("data/processed/driving_features/driving_behavior_clusters.parquet")
OUTPUT_DIR = Path("data/processed/trips")
RESULTS_DIR = Path("outputs/results/trips")

PROFILE_NAMES = {
    0: "Econômico / Suave",
    1: "Moderado / Regular",
    2: "Agressivo / Dinâmico"
}


def extract_trip_features(
    df_1hz: pd.DataFrame,
    df_clusters: pd.DataFrame,
    elevation_service: ElevationService
) -> pd.DataFrame:
    """
    Processes all complete trips from 1Hz telemetry and computes 1 consolidated tuple per trip.
    """
    logger.info("Extracting coordinates to pre-warm elevation cache...")
    all_sampled_coords = []
    trip_groups = list(df_1hz.groupby(["VehId", "Trip"]))
    logger.info(f"Total trips to process: {len(trip_groups)}")

    # 1. Pre-warm elevation cache for all sampled points across all trips
    for (veh_id, trip_id), group in trip_groups:
        lats = group["Latitude[deg]"].values
        lons = group["Longitude[deg]"].values
        sampled = elevation_service.sample_trip_coordinates(lats, lons, min_step_m=50.0)
        all_sampled_coords.extend(sampled)

    logger.info(f"Total sampled points across all trips: {len(all_sampled_coords)}")
    elevation_service.fetch_elevations_for_coords(all_sampled_coords)
    logger.info("Elevation cache pre-warming complete.")

    # 2. Extract trip-level features
    rows = []
    
    # Pre-index clusters by (VehId, Trip) for fast lookup
    clusters_by_trip = dict(tuple(df_clusters.groupby(["VehId", "Trip"])))

    for (veh_id, trip_id), grp in trip_groups:
        dur_s = float(grp["dt"].sum()) if "dt" in grp else float(len(grp))
        sample_count = len(grp)
        day_num = float(grp["DayNum"].iloc[0]) if "DayNum" in grp else 0.0

        # Kinematics
        speeds_kmh = grp["Vehicle Speed[km/h]"].values
        dt_arr = grp["dt"].values if "dt" in grp else np.ones(sample_count)
        
        # Distance from wheel speed
        dist_km = float(np.sum(speeds_kmh * dt_arr) / 3600.0)
        
        # GPS distance
        lats = grp["Latitude[deg]"].values
        lons = grp["Longitude[deg]"].values
        gps_dist_m = 0.0
        for i in range(len(lats) - 1):
            gps_dist_m += haversine_distance_m(lats[i], lons[i], lats[i+1], lons[i+1])
        dist_gps_km = float(gps_dist_m / 1000.0)

        mean_speed = float(np.mean(speeds_kmh))
        max_speed = float(np.max(speeds_kmh))
        std_speed = float(np.std(speeds_kmh))
        
        # Idling
        idle_samples = np.sum(speeds_kmh < 1.0)
        idle_ratio = float(idle_samples / sample_count) if sample_count > 0 else 0.0

        # Accelerations
        accels = grp["accel_ms2"].values if "accel_ms2" in grp else np.zeros(sample_count)
        pos_accels = accels[accels > 0.1]
        neg_accels = accels[accels < -0.1]

        mean_pos_accel = float(np.mean(pos_accels)) if len(pos_accels) > 0 else 0.0
        max_pos_accel = float(np.max(pos_accels)) if len(pos_accels) > 0 else 0.0
        mean_neg_accel = float(np.mean(neg_accels)) if len(neg_accels) > 0 else 0.0
        min_neg_accel = float(np.min(neg_accels)) if len(neg_accels) > 0 else 0.0

        hard_braking_events = int(np.sum(accels <= -2.5))
        rapid_accel_events = int(np.sum(accels >= 2.0))

        # Battery & Electrical
        soc_series = grp["HV Battery SOC[%]"].values
        init_soc = float(soc_series[0])
        final_soc = float(soc_series[-1])
        delta_soc = float(init_soc - final_soc)  # Positive = battery depleted

        volt_series = grp["HV Battery Voltage[V]"].values
        init_volt = float(volt_series[0])
        final_volt = float(volt_series[-1])
        mean_volt = float(np.mean(volt_series))

        # Power & Energy (kWh)
        # In ev_trips_1hz.parquet, power_kw = -(V_bat * I_bat) / 1000
        # P > 0: discharge/traction. P < 0: regen/charge.
        power_arr = grp["power_kw"].values
        
        pos_power = np.maximum(power_arr, 0.0)
        neg_power = np.maximum(-power_arr, 0.0)

        energy_gross_kwh = float(np.sum(pos_power * dt_arr) / 3600.0)
        energy_regen_kwh = float(np.sum(neg_power * dt_arr) / 3600.0)
        energy_net_kwh = float(energy_gross_kwh - energy_regen_kwh)
        
        regen_efficiency = float(energy_regen_kwh / energy_gross_kwh) if energy_gross_kwh > 1e-4 else 0.0
        net_consumption_100km = float((energy_net_kwh / dist_km) * 100.0) if dist_km > 0.05 else 0.0

        # Climate & HVAC
        heater_w = grp["Heater Power[Watts]"].values if "Heater Power[Watts]" in grp else np.zeros(sample_count)
        ac_w = grp["Air Conditioning Power[Watts]"].values if "Air Conditioning Power[Watts]" in grp else np.zeros(sample_count)
        total_hvac_w = heater_w + ac_w
        aux_hvac_energy_kwh = float(np.sum(total_hvac_w * dt_arr) / 3600000.0)
        
        hvac_share_percent = float((aux_hvac_energy_kwh / energy_gross_kwh) * 100.0) if energy_gross_kwh > 1e-4 else 0.0

        oat_series = grp["OAT[DegC]"].values if "OAT[DegC]" in grp else np.zeros(sample_count)
        mean_oat_c = float(np.mean(oat_series))
        min_oat_c = float(np.min(oat_series))
        max_oat_c = float(np.max(oat_series))

        # Altimetry from ElevationService
        elev_metrics = elevation_service.calculate_trip_elevation_metrics(
            lats, lons, distance_km=dist_km, deadband_m=0.5, min_step_m=50.0
        )

        # Driving Profile from driving_behavior_clusters
        trip_key = (veh_id, trip_id)
        if trip_key in clusters_by_trip:
            trip_w = clusters_by_trip[trip_key]
            n_w = len(trip_w)
            eco_ratio = float((trip_w["cluster_id"] == 0).sum() / n_w)
            mod_ratio = float((trip_w["cluster_id"] == 1).sum() / n_w)
            agg_ratio = float((trip_w["cluster_id"] == 2).sum() / n_w)
            dom_id = int(trip_w["cluster_id"].mode()[0])
            dom_label = PROFILE_NAMES.get(dom_id, "Moderado / Regular")
        else:
            eco_ratio, mod_ratio, agg_ratio = 0.0, 1.0, 0.0
            dom_id = 1
            dom_label = PROFILE_NAMES[1]

        # Valid trip filter criteria
        is_valid = bool((dur_s >= 60.0) and (dist_km >= 0.5))

        trip_dict = {
            "trip_id": f"{veh_id}_{trip_id}",
            "VehId": int(veh_id),
            "Trip": int(trip_id),
            "DayNum": day_num,
            "duration_s": round(dur_s, 2),
            "sample_count": sample_count,
            "is_valid_trip": is_valid,

            # Distance & Kinematics
            "distance_km": round(dist_km, 4),
            "distance_gps_km": round(dist_gps_km, 4),
            "mean_speed_kmh": round(mean_speed, 2),
            "max_speed_kmh": round(max_speed, 2),
            "std_speed_kmh": round(std_speed, 2),
            "idle_ratio": round(idle_ratio, 4),
            "mean_pos_accel_ms2": round(mean_pos_accel, 4),
            "max_pos_accel_ms2": round(max_pos_accel, 4),
            "mean_neg_accel_ms2": round(mean_neg_accel, 4),
            "min_neg_accel_ms2": round(min_neg_accel, 4),
            "hard_braking_events": hard_braking_events,
            "rapid_accel_events": rapid_accel_events,

            # Battery State of Charge & Voltage
            "initial_soc_percent": round(init_soc, 2),
            "final_soc_percent": round(final_soc, 2),
            "delta_soc_percent": round(delta_soc, 2),
            "initial_voltage_v": round(init_volt, 2),
            "final_voltage_v": round(final_volt, 2),
            "mean_voltage_v": round(mean_volt, 2),

            # Energy Consumption & Regeneration (Targets)
            "energy_gross_kwh": round(energy_gross_kwh, 4),
            "energy_regen_kwh": round(energy_regen_kwh, 4),
            "energy_net_kwh": round(energy_net_kwh, 4),
            "regen_efficiency_ratio": round(regen_efficiency, 4),
            "net_consumption_kwh_100km": round(net_consumption_100km, 2),

            # Climate & HVAC
            "aux_hvac_energy_kwh": round(aux_hvac_energy_kwh, 4),
            "hvac_share_percent": round(hvac_share_percent, 2),
            "mean_oat_c": round(mean_oat_c, 2),
            "min_oat_c": round(min_oat_c, 2),
            "max_oat_c": round(max_oat_c, 2),

            # Altimetry & Elevation
            "elevation_gain_m": elev_metrics["elevation_gain_m"],
            "elevation_loss_m": elev_metrics["elevation_loss_m"],
            "net_elevation_m": elev_metrics["net_elevation_m"],
            "initial_elevation_m": elev_metrics["initial_elevation_m"],
            "final_elevation_m": elev_metrics["final_elevation_m"],
            "climbing_grade_percent": elev_metrics["climbing_grade_percent"],
            "net_grade_percent": elev_metrics["net_grade_percent"],

            # Driving Behavior Profile
            "driver_profile_dominant": dom_label,
            "driver_profile_dominant_id": dom_id,
            "eco_window_ratio": round(eco_ratio, 4),
            "moderate_window_ratio": round(mod_ratio, 4),
            "aggressive_window_ratio": round(agg_ratio, 4)
        }
        rows.append(trip_dict)

    df_trips = pd.DataFrame(rows)
    return df_trips


def main():
    logger.info("=== Starting EV Complete Trip Feature Extraction ===")
    
    if not DATA_1HZ_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_1HZ_PATH}")
    if not DATA_CLUSTERS_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_CLUSTERS_PATH}")

    df_1hz = pd.read_parquet(DATA_1HZ_PATH)
    df_clusters = pd.read_parquet(DATA_CLUSTERS_PATH)
    logger.info(f"Loaded 1Hz data: {len(df_1hz):,} rows | Clusters data: {len(df_clusters):,} rows")

    elevation_service = ElevationService(cache_path="data/processed/elevation_cache.parquet")

    df_trips = extract_trip_features(df_1hz, df_clusters, elevation_service)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "ev_trips_complete_dataset.parquet"
    csv_path = OUTPUT_DIR / "ev_trips_complete_dataset.csv"
    
    df_trips.to_parquet(parquet_path, index=False)
    df_trips.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved complete trip dataset to {parquet_path} and {csv_path}")

    # Summary statistics for valid trips
    df_valid = df_trips[df_trips["is_valid_trip"]].copy()
    logger.info(f"Total trips: {len(df_trips)} | Valid modeling trips: {len(df_valid)}")

    summary_stats = {
        "total_trips": int(len(df_trips)),
        "valid_trips": int(len(df_valid)),
        "invalid_trips": int(len(df_trips) - len(df_valid)),
        "trips_by_vehicle": df_trips["VehId"].value_counts().to_dict(),
        "valid_trips_by_vehicle": df_valid["VehId"].value_counts().to_dict(),
        "driver_profile_distribution": df_valid["driver_profile_dominant"].value_counts().to_dict(),
        "metrics_summary": {
            "distance_km": {
                "mean": float(round(df_valid["distance_km"].mean(), 2)),
                "min": float(round(df_valid["distance_km"].min(), 2)),
                "max": float(round(df_valid["distance_km"].max(), 2)),
                "sum": float(round(df_valid["distance_km"].sum(), 2))
            },
            "duration_min": {
                "mean": float(round(df_valid["duration_s"].mean() / 60.0, 2)),
                "min": float(round(df_valid["duration_s"].min() / 60.0, 2)),
                "max": float(round(df_valid["duration_s"].max() / 60.0, 2))
            },
            "energy_net_kwh": {
                "mean": float(round(df_valid["energy_net_kwh"].mean(), 3)),
                "min": float(round(df_valid["energy_net_kwh"].min(), 3)),
                "max": float(round(df_valid["energy_net_kwh"].max(), 3)),
                "sum": float(round(df_valid["energy_net_kwh"].sum(), 3))
            },
            "energy_gross_kwh": {
                "mean": float(round(df_valid["energy_gross_kwh"].mean(), 3)),
                "sum": float(round(df_valid["energy_gross_kwh"].sum(), 3))
            },
            "energy_regen_kwh": {
                "mean": float(round(df_valid["energy_regen_kwh"].mean(), 3)),
                "sum": float(round(df_valid["energy_regen_kwh"].sum(), 3))
            },
            "regen_efficiency_ratio": {
                "mean": float(round(df_valid["regen_efficiency_ratio"].mean(), 4))
            },
            "aux_hvac_energy_kwh": {
                "mean": float(round(df_valid["aux_hvac_energy_kwh"].mean(), 3)),
                "sum": float(round(df_valid["aux_hvac_energy_kwh"].sum(), 3))
            },
            "elevation_gain_m": {
                "mean": float(round(df_valid["elevation_gain_m"].mean(), 2)),
                "max": float(round(df_valid["elevation_gain_m"].max(), 2))
            },
            "mean_oat_c": {
                "mean": float(round(df_valid["mean_oat_c"].mean(), 2)),
                "min": float(round(df_valid["mean_oat_c"].min(), 2)),
                "max": float(round(df_valid["mean_oat_c"].max(), 2))
            }
        }
    }

    summary_file = RESULTS_DIR / "ev_trips_summary_statistics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved summary statistics to {summary_file}")
    print(json.dumps(summary_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
