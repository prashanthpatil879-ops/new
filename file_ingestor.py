from datetime import datetime
import fnmatch
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
from rda.io.unity_catalog import UnityCatalog
from rda.utils.logger import get_logger

class FileIngestor:

    def __init__(self, env_config, spark, dbutils):
        self._logger = get_logger(self.__class__.__name__)
        self._spark = spark
        self._dbutils = dbutils
        self._env_config = env_config
        self._uc = UnityCatalog(env_config, spark)
    
    def list_and_filter_files(self, landing: str, file_pattern: str) -> list:
        self._logger.info(f"Looking for file_pattern='{file_pattern}' in '{landing}'")
        all_files = self._dbutils.fs.ls(landing)
        files = [file for file in all_files if fnmatch.fnmatch(file.name, file_pattern)]
        return files

    def read(self, file_info: str, dataset_cfg) -> DataFrame:
        self._logger.info(f"Reading data from '{file_info.path}'")
        df = (
            self._spark.read
            .format(dataset_cfg.file_format)
            .options(**dataset_cfg.options)
            .load(file_info.path)
        )

        if dataset_cfg.get('column_renames', None):
            old_col_names, new_col_names = [v for k, v in dataset_cfg.column_renames.items()]
            
            for i in range(len(old_col_names)):
                if old_col_names[i] in df.columns:
                    df = df.withColumnRenamed(old_col_names[i], new_col_names[i])
        
        df = df.toDF(*[c.upper() for c in df.columns])
        return df
    
    def read_excel(self, file_info: str, dataset_cfg): 
        self._logger.info(f"Reading data from '{file_info.path}'")
        pdf = pd.read_excel(file_info.path, **dataset_cfg.options, storage_options=self._get_storage_options())

        if dataset_cfg.get('column_renames', None):
            old_col_names, new_col_names = [v for k, v in dataset_cfg.column_renames.items()]
            pdf.rename(columns=dict(zip(old_col_names, new_col_names)), inplace=True)
        
        if dataset_cfg.get('column_drops', None):
            pdf.drop(dataset_cfg.column_drops, axis=1, inplace=True)

        if dataset_cfg.get('column_adds', None):
            for col_name, col_value in dataset_cfg.column_adds.items():
                if 'NaN' == col_value:
                    pdf[col_name] = np.nan
                else:
                    pdf[col_name] = col_value
        
        column_type_overrides = dataset_cfg.get("column_type_overrides", {})
        pdf = self.apply_type_overrides(pdf, column_type_overrides)
        
        pdf.columns = [str(c).upper() for c in pdf.columns]
        return pdf
    
    def add_audit_columns(self, df: DataFrame, file_name: str) -> DataFrame:
        self._logger.info("Adding audit columns")
        df = df.withColumn("SOURCE", F.lit(file_name)) \
        .withColumn('CREATED_BY', F.lit('ADB')) \
        .withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))

        df = df.toDF(*[c.upper() for c in df.columns])
        return df
    
    def write(self, df: DataFrame, dataset_cfg) -> None:
        self._logger.info(f"Writting data into '{dataset_cfg.raw_table}' table")
        self._uc.write(df, dataset_cfg.raw_table, dataset_cfg.raw_ext_table_loc, partition_col=dataset_cfg.get('partition_col'))

    def archive(self, file_info, archive_base: str) -> None:
        date_folder = datetime.now().strftime("%Y%m%d")
        archive_base = archive_base.replace("YYYYMMDD", date_folder)
        self._dbutils.fs.mkdirs(archive_base)
        archive_path = f"{archive_base}/{date_folder}/{file_info.name}"
        self._dbutils.fs.mv(file_info.path, archive_path)
        self._logger.info(f"Archived '{file_info.name}' to '{archive_path}' successfully")
        

    def _get_storage_options(self):
        return {
            "account_name": f"{self._env_config.get('adls.storage_account')}",
            "tenant_id": f"{self._env_config.get('secrets.tenant_id')}",
            "client_id": f"{self._env_config.get('secrets.client_id')}",
            "client_secret": f"{self._env_config.get('secrets.client_secret')}",
        }

    def apply_type_overrides(self, pdf, column_type_overrides):        
        pdf.columns = [str(c).lower() for c in pdf.columns]

        for col, expected_type in column_type_overrides.items():
            col = str(col).lower()
            
            if col not in pdf.columns:
                continue

            expected_type = expected_type.lower()

            if expected_type == "date":
                if pd.api.types.is_numeric_dtype(pdf[col]):
                    pdf[col] = pd.to_datetime(
                        pdf[col],
                        unit="D",
                        origin="1899-12-30",
                        errors="coerce"
                    ).dt.date
                else:
                    pdf[col] = pd.to_datetime(
                        pdf[col],
                        errors="coerce"
                    ).dt.date

            elif expected_type == "timestamp":
                if pd.api.types.is_numeric_dtype(pdf[col]):
                    pdf[col] = pd.to_datetime(
                        pdf[col],
                        unit="D",
                        origin="1899-12-30",
                        errors="coerce"
                    )
                else:
                    pdf[col] = pd.to_datetime(
                        pdf[col],
                        errors="coerce"
                    )

            elif expected_type == "string":
                pdf[col] = pdf[col].astype("string")

            elif expected_type == "integer":
                pdf[col] = pd.to_numeric(
                    pdf[col],
                    errors="coerce"
                ).astype("Int64")

            elif expected_type == "decimal":
                pdf[col] = pd.to_numeric(
                    pdf[col],
                    errors="coerce"
                )

            elif expected_type == "boolean":
                pdf[col] = (
                    pdf[col]
                    .replace({
                        "Y": True,
                        "N": False,
                        "Yes": True,
                        "No": False,
                        "TRUE": True,
                        "FALSE": False,
                        1: True,
                        0: False
                    })
                    .astype("boolean")
                )

            else:
                raise ValueError(
                    f"Unsupported column type override '{expected_type}' "
                    f"for column '{col}'"
                )

        return pdf



