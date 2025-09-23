# --- database_manager.py ---
import sqlalchemy as sa
from sqlalchemy import create_engine, text, JSON
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import pandas as pd
import logging
from config import get_db_url, USE_SQLITE
from config import DATA_PROCESSING_CONFIG

from geoalchemy2 import Geometry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

class ArgoProfile(Base):
    __tablename__ = 'argo_profiles'
    
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    float_id = sa.Column(sa.String, index=True, nullable=False)
    profile_number = sa.Column(sa.Integer)
    time = sa.Column(sa.DateTime, index=True)
    lat = sa.Column(sa.Float)
    lon = sa.Column(sa.Float)
    depth = sa.Column(sa.Float)
    temperature = sa.Column(sa.Float, nullable=True)
    salinity = sa.Column(sa.Float, nullable=True)
    doxy = sa.Column(sa.Float, nullable=True)
    chla = sa.Column(sa.Float, nullable=True)
    ph = sa.Column(sa.Float, nullable=True)
    bbp = sa.Column(sa.Float, nullable=True)
    
    geom = sa.Column(Geometry(geometry_type='POINT', srid=4326), index=True, nullable=True)

class FloatMetadata(Base):
    __tablename__ = 'float_metadata'
    
    float_id = sa.Column(sa.String, primary_key=True, index=True)
    wmo_id = sa.Column(sa.String, nullable=True)
    project_name = sa.Column(sa.String, nullable=True)
    institution = sa.Column(sa.String, nullable=True)
    date_launched = sa.Column(sa.String, nullable=True)
    parameters = sa.Column(JSON, nullable=True)

class DatabaseManager:
    def __init__(self, db_url=None):
        self.db_url = db_url or get_db_url()
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.is_postgres = not USE_SQLITE and self.db_url.startswith('postgresql')

    def initialize_database(self):
        try:
            if self.is_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                    conn.commit()
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    # --- NEW: Method to drop and recreate all tables ---
    def reset_database(self):
        """
        Drops all tables defined in the Base metadata and recreates them.
        This effectively clears the entire database.
        """
        try:
            logger.warning("--- RESETTING SQL DATABASE ---")
            Base.metadata.drop_all(self.engine)
            logger.info("All tables dropped successfully.")
            self.initialize_database() # Recreate tables and ensure PostGIS is enabled
            logger.info("--- SQL DATABASE RESET COMPLETE ---")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

    def insert_argo_data(self, df: pd.DataFrame, metadata_dict: dict):
        try:
            df.to_sql(
                name=ArgoProfile.__tablename__,
                con=self.engine,
                if_exists='append',
                index=False,
                chunksize=DATA_PROCESSING_CONFIG.get("chunk_size", 5000)
            )
            
            with self.Session() as session:
                metadata_obj = FloatMetadata(**metadata_dict)
                session.merge(metadata_obj)
                session.commit()
            
            logger.info(f"Inserted {len(df)} rows for float {metadata_dict.get('float_id')}")
            return True
        except Exception as e:
            logger.error(f"Error during bulk insert: {e}")
            return False

    def execute_query(self, query: str):
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection)
            return {
                "success": True,
                "data": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "rowcount": len(df)
            }
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {"success": False, "error": str(e), "data": []}

    def spatial_query(self, lat: float, lon: float, radius_km: int = 10):
        if not self.is_postgres:
            logger.warning("Spatial queries are only efficiently supported on PostgreSQL with PostGIS.")
            query = f"""
            SELECT *, (lat - {lat})*(lat - {lat}) + (lon - {lon})*(lon - {lon}) as dist_sq
            FROM argo_profiles ORDER BY dist_sq LIMIT 10;
            """
            return self.execute_query(query)

        query = f"""
        SELECT *
        FROM argo_profiles
        WHERE ST_DWithin(
            geom,
            ST_MakePoint({lon}, {lat})::geography,
            {radius_km * 1000} -- Radius in meters
        )
        ORDER BY ST_Distance(geom, ST_MakePoint({lon}, {lat})::geography)
        LIMIT 20;
        """
        return self.execute_query(query)