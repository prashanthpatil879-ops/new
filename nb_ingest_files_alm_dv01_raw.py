# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC dbutils.library.restartPython()
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook to load EAR ALM dv01 all asset derivitives data to RAW from Files

# COMMAND ----------

# MAGIC %run ../../common/nb_load_rda_package

# COMMAND ----------

dataset_key = "dv01_all_asset_derivative"

# COMMAND ----------

dataset_cfg = env_config.get(f"ingest.file.{dataset_key}")

# COMMAND ----------

from rda.ingestion.file_ingestor import FileIngestor
from rda.utils import date_utils as du
ingestor = FileIngestor(env_config, spark, dbutils)

# COMMAND ----------

files = ingestor.list_and_filter_files(dataset_cfg.landing, dataset_cfg.file_pattern)

if not files:
    dbutils.notebook.exit(f"No new files found in {dataset_cfg.landing} matching pattern '{dataset_cfg.file_pattern}'")
else:
    print(f"Found {str(files)} to process...") 

# COMMAND ----------

import pandas as pd
pdf = pd.DataFrame()

for file in files:
    for sheet in dataset_cfg.sheets.keys():
        pdf_temp = ingestor.read_excel(file, dataset_cfg.sheets.get(sheet))
        if pdf.empty:
            pdf = pdf_temp
        else:
            pdf = pd.concat([pdf, pdf_temp], ignore_index=True)

    df = spark.createDataFrame(pdf)
    df = ingestor.add_audit_columns(df, file.name)
    # ingestor.write(df, dataset_cfg)
    # ingestor.archive(file, dataset_cfg.archive)


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from gfdo_dev_catalog.gft_rda_raw.irr_ear_reporting_dv01_all_asset_derivative_raw where risk_factor_name='3m'