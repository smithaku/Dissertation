@ECHO OFF
SETLOCAL

REM --- Configuration ---
SET TPC_USER=tpch_user
SET TPC_PASS=tpch_user
REM --- This path should now be correct and point to the SQL script in the same folder ---
SET SCRIPT_PATH=%~dp0run_and_log_query.sql

REM --- List all your PDBs ---
SET PDB_LIST=PDB_CONTROL PDB_PERF_TUNE PDB_ANOMALY_DET PDB_AI_TRAIN PDB_LOAD_TEST PDB_REALTIME PDB_QUERY_OPT PDB_RESOURCE_MGMT PDB_LOG_ANALYTICS

ECHO Starting TPC-H Benchmark Execution...

REM --- Loop through each PDB ---
FOR %%p IN (%PDB_LIST%) DO (
    ECHO.
    ECHO ##################################################
    ECHO ## Processing PDB: %%p
    ECHO ##################################################
    
    REM --- Loop through each query from 1 to 22 ---
    FOR /L %%q IN (1,1,22) DO (
        ECHO   - Running Query %%q on %%p...
        sqlplus -S %TPC_USER%/%TPC_PASS%@%%p @%SCRIPT_PATH% %%p %%q
    )
)

ECHO.
ECHO Benchmark execution completed for all PDBs.
ENDLOCAL