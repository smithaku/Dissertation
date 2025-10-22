@echo OFF
ECHO Loading data into PDB_PERF_TUNE...
#sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_region_load.bad direct=true parallel=true rows=100000
#sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-H V3.0.1\dbgen\nation.ctl log=D:\TPC-H V3.0.1\dbgen\logs\PDB_PERF_TUNE_nation_load.log bad=D:\TPC-H V3.0.1\dbgen\logs\PDB_PERF_TUNE_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_PERF_TUNE control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_PERF_TUNE_lineitem_load.bad direct=true parallel=true rows=100000
ECHO Finished PDB_PERF_TUNE.

ECHO Loading data into PDB_ANOMALY_DET...

sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_ANOMALY_DET control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_ANOMALY_DET_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_ANOMALY_DET.

ECHO Loading data into PDB_AI_TRAIN...

sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_AI_TRAIN control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_AI_TRAIN_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_AI_TRAIN.

ECHO Loading data into PDB_LOAD_TEST...

sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOAD_TEST control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOAD_TEST_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_LOAD_TEST.

ECHO Loading data into PDB_REALTIME...

sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_REALTIME control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_REALTIME_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_REALTIME.

ECHO Loading data into PDB_QUERY_OPT...

sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_QUERY_OPT control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_QUERY_OPT_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_QUERY_OPT.

ECHO Loading data into PDB_RESOURCE_MGMT...

sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_RESOURCE_MGMT control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_RESOURCE_MGMT_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_RESOURCE_MGMT.

ECHO Loading data into PDB_LOG_ANALYTICS...

sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_LOG_ANALYTICS control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_LOG_ANALYTICS_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_LOG_ANALYTICS.

ECHO Loading data into PDB_CONTROL...

sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\region.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_region_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_region_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\nation.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_nation_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_nation_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\part.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_part_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_part_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\supplier.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_supplier_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_supplier_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\partsupp.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_partsupp_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_partsupp_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\customer.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_customer_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_customer_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\orders.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_orders_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_orders_load.bad direct=true parallel=true rows=100000
sqlldr userid=tpch_user/tpch_user@PDB_CONTROL control=D:\TPC-HV3.0.1\dbgen\lineitem.ctl log=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_lineitem_load.log bad=D:\TPC-HV3.0.1\dbgen\logs\PDB_CONTROL_lineitem_load.bad direct=true parallel=true rows=100000

ECHO Finished PDB_CONTROL.
ECHO ALL PDB LOADS ATTEMPTED.
















































































































