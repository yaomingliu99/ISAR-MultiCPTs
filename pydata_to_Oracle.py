# Run this on Windows Environment

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
import numpy as np
import re

import oracledb
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2, CLOB, DATE

pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

cptlin_folder = Path("/mnt/d/Research/SAR With Python/Nutbolt_2027/Jul_2026/Multiple CPTs")

# =============================================================================
# Supporting functions for uploading CPT linear risk to Oracle
# =============================================================================

def get_oracle_engine(
    dsn: str = "STATS_CQITEST",
    wallet_dir: str = "/opt/oracle/network-admin",
    client_dir: str = "/opt/oracle/instantclient_23_26",
):
    """
    Create an Oracle SQLAlchemy engine using the WSL client and wallet.
    """
    config_dir = os.path.expanduser(wallet_dir)
    os.environ["TNS_ADMIN"] = config_dir

    oracledb.init_oracle_client(
        lib_dir=client_dir,
        config_dir=config_dir,
    )

    return create_engine(
        "oracle+oracledb://",
        connect_args={"dsn": dsn},
        pool_pre_ping=True,
    )


def clean_oracle_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with Oracle-safe uppercase column names.
    """
    out = df.copy()

    out.columns = (
        out.columns
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return out


def build_oracle_dtype_map(
    df: pd.DataFrame,
    float_precision: int = 18,
    float_scale: int = 6,
    int_precision: int = 18,
):
    """
    Build SQLAlchemy Oracle dtype map for pandas.to_sql().
    Avoids Oracle FLOAT issues by mapping floats to NUMBER.
    """
    dtype_map = {}

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_integer_dtype(s):
            dtype_map[col] = NUMBER(int_precision, 0)

        elif pd.api.types.is_float_dtype(s):
            dtype_map[col] = NUMBER(float_precision, float_scale)

        elif pd.api.types.is_datetime64_any_dtype(s):
            dtype_map[col] = DATE()

        else:
            max_len = s.dropna().astype(str).str.len().max()
            max_len = int(max_len) if pd.notna(max_len) else 1

            if max_len <= 4000:
                dtype_map[col] = VARCHAR2(max(max_len, 1))
            else:
                dtype_map[col] = CLOB()

    return dtype_map


def drop_oracle_table(engine, schema_name: str, table_name: str):
    """
    Drop Oracle table if it exists.
    Ignores ORA-00942: table or view does not exist.
    """
    schema_name = schema_name.upper()
    table_name = table_name.upper()

    drop_sql = f"""
    BEGIN
       EXECUTE IMMEDIATE 'DROP TABLE {schema_name}.{table_name} PURGE';
    EXCEPTION
       WHEN OTHERS THEN
          IF SQLCODE != -942 THEN
             RAISE;
          END IF;
    END;
    """

    with engine.begin() as conn:
        conn.execute(text(drop_sql))

def read_table_file(file_path, read_kwargs: dict | None = None) -> pd.DataFrame:
    """
    Read CSV or Parquet into a pandas DataFrame based on file extension.
    """
    file_path = Path(file_path)

    if read_kwargs is None:
        read_kwargs = {}

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, **read_kwargs)

    elif suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path, **read_kwargs)

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Expected .csv, .parquet, or .pq"
        )

def upload_csv_to_oracle_replace(
    file_path,
    table_name: str,
    schema_name: str = "STATS",
    dsn: str = "STATS_CQITEST",
    wallet_dir: str = "/opt/oracle/network-admin",
    chunksize: int = 50_000,
    float_precision: int = 18,
    float_scale: int = 6,
    int_precision: int = 18,
    read_kwargs: dict | None = None,
    primary_key_cols: tuple[str, ...] | None = ("DIVISION", "ASM1"),
    use_native_batch: bool = False,
):
    """
    Fully replace an Oracle table using data from a CSV or Parquet file.

    Supported inputs:
      - .csv
      - .parquet
      - .pq

    This avoids pandas if_exists='replace' because Oracle reflection can fail.
    Instead:
      1. DROP TABLE schema.table PURGE
      2. Insert with pandas.to_sql(), or python-oracledb array DML when
         use_native_batch=True
      3. ALTER TABLE schema.table ADD CONSTRAINT primary key
    """
    schema_name = schema_name.upper()
    table_name = table_name.upper()
    file_path = Path(file_path)

    engine = get_oracle_engine(dsn=dsn, wallet_dir=wallet_dir)

    with engine.connect() as conn:
        sysdate = conn.execute(text("SELECT sysdate FROM dual")).scalar()
        current_user = conn.execute(text("SELECT USER FROM dual")).scalar()
        print(f"Connected to Oracle as {current_user}; sysdate = {sysdate}")

    print(f"Reading file: {file_path}")
    df = read_table_file(file_path, read_kwargs=read_kwargs)

    print(f"Original shape: {df.shape}")

    df_upload = clean_oracle_column_names(df)

    print("Oracle columns:")
    print(df_upload.columns.tolist())

    if primary_key_cols is not None:
        primary_key_cols = tuple(col.upper() for col in primary_key_cols)

        missing_pk_cols = [
            col for col in primary_key_cols
            if col not in df_upload.columns
        ]

        if missing_pk_cols:
            raise ValueError(
                f"Primary key columns are missing after Oracle column cleaning: "
                f"{missing_pk_cols}. Available columns: {df_upload.columns.tolist()}"
            )

        duplicate_pk_count = int(
            df_upload.duplicated(list(primary_key_cols)).sum()
        )

        if duplicate_pk_count > 0:
            raise ValueError(
                f"Cannot create primary key on {primary_key_cols}. "
                f"Found {duplicate_pk_count:,} duplicate key rows."
            )

        null_pk_counts = df_upload[list(primary_key_cols)].isna().sum()
        null_pk_counts = null_pk_counts[null_pk_counts > 0]

        if not null_pk_counts.empty:
            raise ValueError(
                f"Cannot create primary key on {primary_key_cols}. "
                f"Primary key columns contain nulls: {null_pk_counts.to_dict()}"
            )

    dtype_map = build_oracle_dtype_map(
        df_upload,
        float_precision=float_precision,
        float_scale=float_scale,
        int_precision=int_precision,
    )

    print(f"Dropping table if exists: {schema_name}.{table_name}")
    drop_oracle_table(engine, schema_name=schema_name, table_name=table_name)

    print(f"Uploading to Oracle: {schema_name}.{table_name}")
    if use_native_batch:
        # Let SQLAlchemy create the empty table with the requested Oracle types,
        # then bypass pandas' per-row dictionaries during the large insert.
        df_upload.head(0).to_sql(
            name=table_name,
            con=engine,
            schema=schema_name,
            if_exists="append",
            index=False,
            dtype=dtype_map,
        )

        column_sql = ", ".join(f'"{col}"' for col in df_upload.columns)
        bind_sql = ", ".join(f":{i}" for i in range(1, len(df_upload.columns) + 1))
        insert_sql = (
            f'INSERT INTO "{schema_name}"."{table_name}" '
            f"({column_sql}) VALUES ({bind_sql})"
        )

        total_rows = len(df_upload)
        raw_conn = engine.raw_connection()
        try:
            cursor = raw_conn.cursor()
            try:
                for start in range(0, total_rows, chunksize):
                    end = min(start + chunksize, total_rows)
                    cursor.executemany(insert_sql, df_upload.iloc[start:end])
                    raw_conn.commit()
                    print(
                        f"Uploaded {end:,} of {total_rows:,} rows "
                        f"({end / total_rows:.1%})"
                    )
            finally:
                cursor.close()
        finally:
            raw_conn.close()
    else:
        df_upload.to_sql(
            name=table_name,
            con=engine,
            schema=schema_name,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            dtype=dtype_map,
        )

    if primary_key_cols is not None:
        pk_name = f"PK_{table_name}"

        # Oracle constraint names max out at 30 bytes in older versions.
        # This keeps the name safe.
        if len(pk_name) > 30:
            pk_name = pk_name[:30]

        pk_cols_sql = ", ".join(primary_key_cols)

        alter_sql = f"""
            ALTER TABLE {schema_name}.{table_name}
            ADD CONSTRAINT {pk_name}
            PRIMARY KEY ({pk_cols_sql})
        """

        print(
            f"Adding primary key constraint {pk_name} "
            f"on ({pk_cols_sql})"
        )

        with engine.begin() as conn:
            conn.execute(text(alter_sql))

    with engine.connect() as conn:
        n = conn.execute(
            text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
        ).scalar()

    print(f"Done. {schema_name}.{table_name}: {n:,} rows")

    return {
        "engine": engine,
        "schema_name": schema_name,
        "table_name": table_name,
        "row_count": n,
        "columns": df_upload.columns.tolist(),
        "dtype_map": dtype_map,
        "primary_key_cols": primary_key_cols,
    }
# =============================================================================

# =============================================================================
# #upload Essential CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Essential_cptlin.parquet",
    table_name="ESSENTIAL_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
    use_native_batch=True,
)

# =============================================================================
# #upload Target Appendectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Appendectomy_cptlin.parquet",
    table_name="TAR_APP_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Colectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Colectomy_cptlin.parquet",
    table_name="TAR_COL_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Cystectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Cystectomy_cptlin.parquet",
    table_name="TAR_CYST_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Esophagectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Esophagectomy_cptlin.parquet",
    table_name="TAR_ESO_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Gyne-Recon CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_GynRecon_cptlin.parquet",
    table_name="TAR_GYNE_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Hepatectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Hepatectomy_cptlin.parquet",
    table_name="TAR_HEP_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Hip CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Hip_cptlin.parquet",
    table_name="TAR_HIP_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Hysterectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Hysterectomy_cptlin.parquet",
    table_name="TAR_HYST_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Nephrectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Nephrectomy_cptlin.parquet",
    table_name="TAR_NEPH_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Pancreatectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Pancreatectomy_cptlin.parquet",
    table_name="TAR_PAN_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Proctectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Proctectomy_cptlin.parquet",
    table_name="TAR_PRO_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Prostatectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Prostatectomy_cptlin.parquet",
    table_name="TAR_PRST_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Thyroidectomy CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Thyroidectomy_cptlin.parquet",
    table_name="TAR_THY_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC AAA CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_AAA_cptlin.parquet",
    table_name="TAR_VAS_AAA_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC AIE CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_AIE_cptlin.parquet",
    table_name="TAR_VAS_AIE_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC AIO CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_AIO_cptlin.parquet",
    table_name="TAR_VAS_AIO_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC CAS CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_CAS_cptlin.parquet",
    table_name="TAR_VAS_CAS_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC CEA CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_CEA_cptlin.parquet",
    table_name="TAR_VAS_CEA_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC EVAR CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_EVAR_cptlin.parquet",
    table_name="TAR_VAS_EVAR_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC LEE CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_LEE_cptlin.parquet",
    table_name="TAR_VAS_LEE_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VASC LEO CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_LEO_cptlin.parquet",
    table_name="TAR_VAS_LEO_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target VHR CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Vascular_VHR_cptlin.parquet",
    table_name="TAR_VHR_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# #upload Target Geriatrics CPT linear risk to Oracle
# =============================================================================
result = upload_csv_to_oracle_replace(
    file_path=cptlin_folder / "Target_Geri_cptlin.parquet",
    table_name="TAR_GERI_CPTLIN",
    schema_name="STATS",
    dsn="STATS_CQITEST",
)

# =============================================================================
# # Test wallet connection and Oracle engine
# =============================================================================
# Config Wallet to WSL
config_dir = os.path.expanduser("/opt/oracle/network-admin")
client_dir = "/opt/oracle/instantclient_23_26"

os.environ["TNS_ADMIN"] = config_dir

oracledb.init_oracle_client(
    lib_dir=client_dir,
    config_dir=config_dir,
)

engine = create_engine(
    "oracle+oracledb://",
    connect_args={"dsn": "STATS_CQITEST"},
    pool_pre_ping=True,
)

# Test the Oracle connection
with engine.connect() as conn:
    result = conn.execute(text("SELECT sysdate FROM dual"))
    print(result.fetchone())

# =============================================================================
# # Check if Primary Key was set correctly  
# =============================================================================
# engine = get_oracle_engine(dsn="STATS_CQITEST")

# with engine.connect() as conn:
#     pk_check = conn.execute(
#         text("""
#             SELECT
#                 ac.constraint_name,
#                 ac.constraint_type,
#                 acc.column_name,
#                 acc.position,
#                 ac.status
#             FROM all_constraints ac
#             JOIN all_cons_columns acc
#                 ON ac.owner = acc.owner
#                AND ac.constraint_name = acc.constraint_name
#                AND ac.table_name = acc.table_name
#             WHERE ac.owner = :schema_name
#               AND ac.table_name = :table_name
#               AND ac.constraint_type = 'P'
#             ORDER BY acc.position
#         """),
#         {
#             "schema_name": "STATS",
#             "table_name": "TAR_APP_CPTLIN",
#         },
#     ).fetchall()

# print(pk_check)
