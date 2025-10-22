-- run_and_log_query.sql 
-- Suppress terminal output for setup commands
SET TERMOUT OFF
SET VERIFY OFF
SET FEEDBACK OFF

-- Define substitution variables from the batch script arguments
DEFINE PDB_NAME = '&1'
DEFINE QUERY_NO = '&2'
DEFINE QUERY_FILE_PATH =D:\TPC-HV3.0.1\dbgen\queries\&QUERY_NO..sql

-- Capture the start time into a substitution variable
COLUMN S_TIME NEW_VALUE START_TIME_VAL NOPRINT
SELECT TO_CHAR(SYSTIMESTAMP, 'YYYY-MM-DD HH24:MI:SS.FF') AS S_TIME FROM DUAL;

-- Now, turn OFF substitution variable processing before running the TPC-H query
SET DEFINE OFF

-- Execute the TPC-H query using the pre-constructed file path
@ &QUERY_FILE_PATH

-- Turn substitution variable processing back ON for the final logging steps
SET DEFINE ON

-- Capture the end time into a substitution variable
COLUMN E_TIME NEW_VALUE END_TIME_VAL NOPRINT
SELECT TO_CHAR(SYSTIMESTAMP, 'YYYY-MM-DD HH24:MI:SS.FF') AS E_TIME FROM DUAL;

-- Perform the INSERT using the captured substitution variables
DECLARE
    v_start_time TIMESTAMP := TO_TIMESTAMP('&START_TIME_VAL', 'YYYY-MM-DD HH24:MI:SS.FF');
    v_end_time   TIMESTAMP := TO_TIMESTAMP('&END_TIME_VAL', 'YYYY-MM-DD HH24:MI:SS.FF');
BEGIN
    INSERT INTO tpch_query_log (pdb_name, query_no, start_time, end_time, execution_seconds, execution_date)
    VALUES (
        '&PDB_NAME',
        &QUERY_NO,
        v_start_time,
        v_end_time,
        (EXTRACT(DAY FROM (v_end_time - v_start_time)) * 86400) +
        (EXTRACT(HOUR FROM (v_end_time - v_start_time)) * 3600) +
        (EXTRACT(MINUTE FROM (v_end_time - v_start_time)) * 60) +
        (EXTRACT(SECOND FROM (v_end_time - v_start_time))),
        SYSDATE
    );
    COMMIT;
END;
/

-- Restore settings and exit
SET TERMOUT ON
SET VERIFY ON
SET FEEDBACK ON
EXIT;
