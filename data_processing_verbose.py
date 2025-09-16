# --- data_processing_verbose.py ---
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import logging
import traceback # Used for printing detailed error messages

# Ensure all necessary modules can be imported
try:
    from database_manager import DatabaseManager
    from config import DATA_PROCESSING_CONFIG, USE_SQLITE
    from rag_system import VectorStoreManager
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules. Please check your project structure. Details: {e}")
    exit()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArgoDataProcessor:
    def __init__(self, data_dir=None):
        print("-> Initializing ArgoDataProcessor...")
        self.data_dir = data_dir or DATA_PROCESSING_CONFIG["data_dir"]
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
        print("-> Processor initialized successfully.")
    
    def _decode_if_bytes(self, value):
        """Safely decodes a value if it's a byte string."""
        try:
            return value.decode('utf-8').strip()
        except (UnicodeDecodeError, AttributeError):
            return str(value).strip()

    def _convert_time(self, val):
        try:
            if np.issubdtype(type(val), np.number) and not np.isnan(val):
                return pd.to_datetime("1950-01-01") + pd.to_timedelta(float(val), unit="D")
        except Exception:
            pass
        return pd.NaT

    def _extract_metadata(self, ds, float_id_str):
        try:
            params = [var for var in ds.data_vars if '_ADJUSTED' in var]
            first_prof = ds.isel(N_PROF=0)
            
            metadata = {
                "float_id": float_id_str,
                "wmo_id": self._decode_if_bytes(first_prof['WMO_INST_TYPE'].values),
                "project_name": self._decode_if_bytes(first_prof['PROJECT_NAME'].values),
                "institution": self._decode_if_bytes(ds.attrs.get('institution', 'Unknown')),
                "date_launched": str(ds.get('LAUNCH_DATE', 'N/A')),
                "parameters": params
            }
            return metadata
        except Exception as e:
            print(f"--> ERROR: Could not extract metadata for float {float_id_str}. Details: {e}")
            return None

    def _generate_metadata_summary(self, metadata):
        if not metadata: return None
        summary = (
            f"ARGO float ID {metadata['float_id']} from the {metadata['project_name']} project, "
            f"managed by {metadata['institution']}. It measures parameters including: "
            f"{', '.join(metadata['parameters'])}."
        )
        return summary

    def process_netcdf_file(self, file_path):
        try:
            with xr.open_dataset(file_path, decode_times=False) as ds:
                
                # --- NEW LOGIC INSPIRED BY YOUR SCRIPT ---
                # Extract all data into NumPy arrays first for robust access
                platform_numbers = ds['PLATFORM_NUMBER'].values
                juld_array = ds['JULD'].values
                lat_array = ds['LATITUDE'].values
                lon_array = ds['LONGITUDE'].values
                cycle_array = ds['CYCLE_NUMBER'].values
                
                # Always prefer adjusted values
                pres_array = ds['PRES_ADJUSTED'].values
                temp_array = ds['TEMP_ADJUSTED'].values
                sal_array = ds['PSAL_ADJUSTED'].values

                num_profiles = ds.sizes['N_PROF']
                num_levels = ds.sizes['N_LEVELS']
                print(f"--> Found {num_profiles} profiles and {num_levels} levels in this file.")

                records = []
                # Loop through profiles
                for i in range(num_profiles):
                    profile_time = self._convert_time(juld_array[i])
                    
                    # Critical check: if time is invalid for a profile, we must skip it
                    if pd.isna(profile_time):
                        print(f"--> Skipping profile index {i} due to invalid timestamp.")
                        continue
                    
                    # Loop through depth levels
                    for j in range(num_levels):
                        pres_val = pres_array[i, j]
                        temp_val = temp_array[i, j]
                        sal_val = sal_array[i, j]

                        # Only add record if the core data is valid
                        if not np.isnan(pres_val) and not np.isnan(temp_val) and not np.isnan(sal_val):
                            records.append({
                                'float_id': self._decode_if_bytes(platform_numbers[i]),
                                'profile_number': int(cycle_array[i]),
                                'time': profile_time,
                                'lat': float(lat_array[i]),
                                'lon': float(lon_array[i]),
                                'depth': float(pres_val),
                                'temperature': float(temp_val),
                                'salinity': float(sal_val),
                                # BGC data is not present, will correctly be None
                                'doxy': None,
                                'chla': None
                            })
                
                if not records:
                    print("--> No valid measurement records found after processing all profiles.")
                    return [] # Return empty list
                
                # Group records by float_id to handle files with multiple floats
                df = pd.DataFrame(records)
                results = []
                for float_id, group_df in df.groupby('float_id'):
                    metadata = self._extract_metadata(ds, float_id)
                    if metadata:
                        results.append((group_df, metadata))
                
                print(f"--> Compiled {len(df)} valid records across {len(results)} floats from this file.")
                return results

        except Exception as e:
            print(f"--> FATAL ERROR processing file {os.path.basename(file_path)}. Details: {e}")
            traceback.print_exc()
            return []

    def process_directory(self, max_files=None):
        print("\n-> Starting directory processing...")
        try:
            self.db_manager.initialize_database()
            print("-> Database connection successful and tables are ready.")
        except Exception as e:
            print(f"-> FATAL ERROR: Could not initialize database. Is PostgreSQL running and configured correctly?")
            print(f"-> Details: {e}")
            traceback.print_exc()
            return

        nc_files_path = os.path.join(self.data_dir, "*.nc")
        print(f"-> Searching for NetCDF files in: {os.path.abspath(nc_files_path)}")
        
        nc_files = sorted(glob.glob(nc_files_path))
        print(f"-> Found {len(nc_files)} NetCDF files.")
        
        if not nc_files:
            print("-> WARNING: No files found. Please ensure .nc files are in the 'data' directory.")
            return

        if max_files: 
            nc_files = nc_files[:max_files]
            print(f"-> Will process a maximum of {len(nc_files)} files.")
        
        processed_count = 0
        for file_path in nc_files:
            print(f"\n-> Processing file: {os.path.basename(file_path)}")
            results = self.process_netcdf_file(file_path)
            
            if not results:
                print("--> SKIPPED: No valid data could be extracted from this file.")
                continue

            file_processed_successfully = True
            for df, metadata in results:
                if df is not None and not df.empty and metadata:
                    if not USE_SQLITE:
                        df['geom'] = df.apply(lambda row: f'SRID=4326;POINT({row.lon} {row.lat})', axis=1)

                    success = self.db_manager.insert_argo_data(df, metadata)
                    
                    if success:
                        print(f"--> SUCCESS: Inserted {len(df)} records for float {metadata['float_id']} into SQL database.")
                        summary_text = self._generate_metadata_summary(metadata)
                        self.vector_store.add_document(
                            doc_id=metadata['float_id'],
                            document=summary_text,
                            metadata={"float_id": metadata['float_id']}
                        )
                        print(f"--> SUCCESS: Added/updated metadata for float {metadata['float_id']} in vector store.")
                    else:
                        print(f"--> FAILED to insert data into SQL database for float {metadata['float_id']}.")
                        file_processed_successfully = False
            
            if file_processed_successfully:
                processed_count += 1
        
        print(f"\n-> Processing complete. Successfully processed data from {processed_count}/{len(nc_files)} files.")

if __name__ == "__main__":
    print("=============================================")
    print("   Running ARGO Data Processor in Verbose Mode   ")
    print("=============================================")
    try:
        processor = ArgoDataProcessor()
        processor.process_directory(max_files=DATA_PROCESSING_CONFIG.get('max_files'))
    except Exception as e:
        print(f"\n-> A critical error occurred during script execution.")
        print(f"-> Details: {e}")
        traceback.print_exc()