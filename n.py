import sqlite3
import pandas as pd

# Connect to SQLite DB
conn = sqlite3.connect("argo.db")

def run_query(query, desc):
    print(f"\n🔹 {desc}")
    df = pd.read_sql_query(query, conn)
    print(df.head(20))  # show first 20 rows

# 1. Number of floats in dataset
run_query("""
SELECT COUNT(DISTINCT float_id) AS num_floats
FROM argo_profiles;
""", "Number of unique floats")

# 2. Profiles per float
run_query("""
SELECT float_id, COUNT(DISTINCT profile_number) AS num_profiles
FROM argo_profiles
GROUP BY float_id
ORDER BY num_profiles DESC;
""", "Profiles per float")

# 3. Depth range per float
run_query("""
SELECT float_id,
       MIN(depth) AS min_depth,
       MAX(depth) AS max_depth
FROM argo_profiles
GROUP BY float_id;
""", "Depth range per float")

# 4. Date range (min/max time)
run_query("""
SELECT MIN(time) AS min_time,
       MAX(time) AS max_time
FROM argo_profiles;
""", "Date range of dataset")

# 5. Average surface temperature (depth < 10m) per float
run_query("""
SELECT float_id,
       ROUND(AVG(temperature), 3) AS avg_surface_temp
FROM argo_profiles
WHERE depth < 10
GROUP BY float_id;
""", "Average surface temperature (depth < 10m)")

# 6. Temperature/salinity at 1000m depth (approx) per float
run_query("""
SELECT float_id,
       ROUND(AVG(temperature), 3) AS temp_1000m,
       ROUND(AVG(salinity), 3) AS sal_1000m
FROM argo_profiles
WHERE depth BETWEEN 950 AND 1050
GROUP BY float_id;
""", "Average T/S near 1000m depth per float")

# 7. Spatial bounding box
run_query("""
SELECT MIN(lat) AS lat_min,
       MAX(lat) AS lat_max,
       MIN(lon) AS lon_min,
       MAX(lon) AS lon_max
FROM argo_profiles;
""", "Geographical bounding box")

conn.close()
