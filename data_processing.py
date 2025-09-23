# --- data_processing.py ---
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import logging

from database_manager import DatabaseManager
from config import DATA_PROCESSING_CONFIG, USE_SQLITE
from rag_system import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArgoDataProcessor:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_PROCESSING_CONFIG["data_dir"]
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStoreManager()
    
    # ... (No changes to _convert_time, _extract_metadata, _generate_metadata_summary, process_netcdf_file) ...
    def _convert_time(self, val):
        try:
            if np.issubdtype(type(val), np.number) and not np.isnan(val):
                return pd.to_datetime("1950-01-01") + pd.to_timedelta(float(val), unit="D")
        except Exception:
            return pd.NaT
        return pd.NaT

    def _extract_metadata(self, ds):
        try:
            float_id = ds['PLATFORM_NUMBER'].values[0].decode('utf-8').strip()
            params = [var for var in ds.data_vars if '_ADJUSTED' in var]
            
            metadata = {
                "float_id": float_id,
                "wmo_id": ds['WMO_INST_TYPE'].values[0].decode('utf-8').strip(),
                "project_name": ds['PROJECT_NAME'].values[0].decode('utf-8').strip(),
                "institution": ds.attrs.get('institution', 'Unknown'),
                "date_launched": str(ds.get('LAUNCH_DATE', 'N/A')),
                "parameters": params
            }
            return metadata
        except Exception as e:
            logger.error(f"Could not extract metadata: {e}")
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
                file_metadata = self._extract_metadata(ds)
                if not file_metadata:
                    return None, None

                records = []
                num_profiles = ds.dims['N_PROF']

                for i in range(num_profiles):
                    profile_data = ds.isel(N_PROF=i)
                    
                    profile_time = self._convert_time(profile_data['JULD'].values)
                    profile_lat = profile_data['LATITUDE'].values
                    profile_lon = profile_data['LONGITUDE'].values
                    cycle_num = int(profile_data['CYCLE_NUMBER'].values)

                    depths = profile_data['PRES_ADJUSTED'].values
                    temps = profile_data['TEMP_ADJUSTED'].values
                    sals = profile_data['PSAL_ADJUSTED'].values
                    
                    for j in range(len(depths)):
                        if not np.isnan(depths[j]) and not np.isnan(temps[j]) and not np.isnan(sals[j]):
                            records.append({
                                'float_id': file_metadata['float_id'],
                                'profile_number': cycle_num,
                                'time': profile_time,
                                'lat': float(profile_lat),
                                'lon': float(profile_lon),
                                'depth': float(depths[j]),
                                'temperature': float(temps[j]),
                                'salinity': float(sals[j]),
                                'doxy': float(profile_data.get('DOXY_ADJUSTED', [np.nan])[j]) if 'DOXY_ADJUSTED' in profile_data else None,
                                'chla': float(profile_data.get('CHLA_ADJUSTED', [np.nan])[j]) if 'CHLA_ADJUSTED' in profile_data else None
                            })
                
                if not records:
                    return None, None
                    
                return pd.DataFrame(records), file_metadata

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, None

    # --- MODIFIED: process_directory now calls the reset methods first ---
    def process_directory(self, max_files=None):
        # Step 1: Reset both databases to ensure a clean slate.
        logger.info("Preparing to process new data. Clearing old data first...")
        self.db_manager.reset_database()
        self.vector_store.reset_vector_store()
        
        # Step 2: Proceed with finding and processing files.
        nc_files = sorted(glob.glob(os.path.join(self.data_dir, "*.nc")))
        if max_files: nc_files = nc_files[:max_files]
        
        logger.info(f"Found {len(nc_files)} NetCDF files to process.")
        processed_count = 0
        
        for file_path in nc_files:
            df, metadata = self.process_netcdf_file(file_path)
            
            if df is not None and not df.empty and metadata:
                logger.info(f"Extracted {len(df)} valid measurements from {os.path.basename(file_path)}")
                if not USE_SQLITE:
                    df['geom'] = df.apply(lambda row: f'SRID=4326;POINT({row.lon} {row.lat})', axis=1)

                success = self.db_manager.insert_argo_data(df, metadata)
                
                if success:
                    summary_text = self._generate_metadata_summary(metadata)
                    self.vector_store.add_document(
                        doc_id=metadata['float_id'],
                        document=summary_text,
                        metadata={"float_id": metadata['float_id']}
                    )
                    logger.info(f"Added metadata for float {metadata['float_id']} to vector store.")
                    processed_count += 1
        
        logger.info(f"Processing complete. Successfully processed {processed_count}/{len(nc_files)} files.")