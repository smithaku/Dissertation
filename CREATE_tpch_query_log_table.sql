CREATE TABLE tpch_query_log (
    pdb_name          VARCHAR2(30),
    query_no          NUMBER,
    start_time        TIMESTAMP,
    end_time          TIMESTAMP,
    execution_seconds NUMBER,
    execution_date    DATE
);

