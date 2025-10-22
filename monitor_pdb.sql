-- =================================================================
-- Automated Monitoring Script for Dissertation RQ3
-- This script saves its output to a uniquely named log file.
-- =================================================================

-- Set up formatting for the report
SET LINESIZE 200
SET FEEDBACK OFF
COLUMN STATISTIC FORMAT A50
COLUMN VALUE FORMAT 999,999,999,999

-- Generate a unique filename using a timestamp
COLUMN dt_col NEW_VALUE log_timestamp
SELECT TO_CHAR(SYSDATE, 'YYYYMMDD_HH24MISS') AS dt_col FROM DUAL;
DEFINE log_file = 'monitoring_results_&log_timestamp..log'

-- Start spooling (redirecting) all output to the new file
SPOOL &_log_file

-- --- Report Header ---
PROMPT =================================================================
PROMPT Monitoring Report for PDB: &&_CONNECT_IDENTIFIER
PROMPT Report generated at: &_log_timestamp
PROMPT =================================================================

-- === Section 1: CPU Usage ===
PROMPT
PROMPT ## CPU Usage Statistics ##
SELECT name AS statistic, value
FROM V$SYSSTAT
WHERE name IN ('CPU used by this session', 'CPU used when call started');

-- === Section 2: Memory Usage ===
PROMPT
PROMPT ## Memory (PGA) Usage Statistics (MB) ##
SELECT name AS statistic, ROUND(value / 1024 / 1024, 2) AS value_mb
FROM V$PGASTAT
WHERE name IN ('aggregate PGA target parameter', 'total PGA allocated', 'total PGA in use');

-- === Section 3: Transactions and Throughput ===
PROMPT
PROMPT ## Transaction Statistics ##
SELECT name AS statistic, value
FROM V$SYSSTAT
WHERE name = 'user commits';

-- === Section 4: Disk I/O Statistics ===
PROMPT
PROMPT ## Disk I/O Statistics ##
SELECT name AS statistic, value
FROM V$SYSSTAT
WHERE name IN ('physical reads', 'physical writes', 'physical read total bytes', 'physical write total bytes');

-- === Section 5: Network Statistics ===
PROMPT
PROMPT ## Network Statistics ##
SELECT name AS statistic, value
FROM V$SYSSTAT
WHERE name IN ('SQL*Net roundtrips to/from client', 'SQL*Net roundtrip time');

-- === Section 6: Session Statistics ===
PROMPT
PROMPT ## Concurrent Session Count ##
SELECT 'Active User Sessions' AS statistic, COUNT(*) AS value
FROM V$SESSION
WHERE type = 'USER' AND status = 'ACTIVE';

PROMPT
PROMPT --- End of Report ---

-- Stop spooling output
SPOOL OFF

PROMPT Report saved to &_log_file

-- Exit SQL*Plus
EXIT;