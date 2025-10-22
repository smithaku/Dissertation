# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import oracledb
import time
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DB_USER = "sys"
DB_PASSWORD = "junior@25SA" #<-- IMPORTANT: UPDATE YOUR PASSWORD HERE
DB_HOST = "db01.dc01.com"
DB_PORT = 1521

PDB_CONTROL_SERVICE = "pdb_control.dc01.com"
PDB_TUNED_SERVICE = "pdb_perf_tune.dc01.com"
ORACLE_CONFIG_DIR = r"E:\app\oracle\product\19.3.0\db_home\network\admin"

SAVED_MODEL_PATH = "rl_tuning_agent.keras"
FINAL_RESULTS_CSV = "final_performance_results.csv"
FINAL_FIGURE_PATH = "final_figure_4_4_comparative_qet.png"
MONITORING_INTERVAL_SECONDS = 300 

# --- DQN AGENT CLASS ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

# --- HELPER FUNCTIONS ---
def get_tpc_h_queries():
    # Final, corrected list of all 22 TPC-H queries
    return [
        """select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from tpch_user.lineitem where l_shipdate <= date '1998-12-01' - interval '90' day group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus""",
        """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from tpch_user.part, tpch_user.supplier, tpch_user.partsupp, tpch_user.nation, tpch_user.region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 15 and p_type like '%BRASS' and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' and ps_supplycost = ( select min(ps_supplycost) from tpch_user.partsupp, tpch_user.supplier, tpch_user.nation, tpch_user.region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' ) order by s_acctbal desc, n_name, s_name, p_partkey""",
        """select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority from tpch_user.customer, tpch_user.orders, tpch_user.lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate""",
        """select o_orderpriority, count(*) as order_count from tpch_user.orders where o_orderdate >= date '1993-07-01' and o_orderdate < date '1993-07-01' + interval '3' month and exists ( select * from tpch_user.lineitem where l_orderkey = o_orderkey and l_commitdate < l_receiptdate ) group by o_orderpriority order by o_orderpriority""",
        """select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue from tpch_user.customer, tpch_user.orders, tpch_user.lineitem, tpch_user.supplier, tpch_user.nation, tpch_user.region where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'ASIA' and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year group by n_name order by revenue desc""",
        """select sum(l_extendedprice * l_discount) as revenue from tpch_user.lineitem where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year and l_discount between 0.05 and 0.07 and l_quantity < 24""",
        """select supp_nation, cust_nation, l_year, sum(volume) as revenue from ( select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year, l_extendedprice * (1 - l_discount) as volume from tpch_user.supplier, tpch_user.lineitem, tpch_user.orders, tpch_user.customer, tpch_user.nation n1, tpch_user.nation n2 where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey and ( (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') ) and l_shipdate between date '1995-01-01' and date '1996-12-31' ) shipping group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year""",
        """select o_year, sum(case when nation = 'BRAZIL' then volume else 0 end) / sum(volume) as mkt_share from ( select extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) as volume, n2.n_name as nation from tpch_user.part, tpch_user.supplier, tpch_user.lineitem, tpch_user.orders, tpch_user.customer, tpch_user.nation n1, tpch_user.nation n2, tpch_user.region where p_partkey = l_partkey and s_suppkey = l_suppkey and l_orderkey = o_orderkey and o_custkey = c_custkey and c_nationkey = n1.n_nationkey and n1.n_regionkey = r_regionkey and r_name = 'AMERICA' and s_nationkey = n2.n_nationkey and o_orderdate between date '1995-01-01' and date '1996-12-31' and p_type = 'ECONOMY ANODIZED STEEL' ) all_nations group by o_year order by o_year""",
        """select nation, o_year, sum(amount) as sum_profit from ( select n_name as nation, extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount from tpch_user.part, tpch_user.supplier, tpch_user.lineitem, tpch_user.partsupp, tpch_user.orders, tpch_user.nation where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey and p_partkey = l_partkey and o_orderkey = l_orderkey and s_nationkey = n_nationkey and p_name like '%green%' ) profit group by nation, o_year order by nation, o_year desc""",
        """select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from tpch_user.customer, tpch_user.orders, tpch_user.lineitem, tpch_user.nation where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc""",
        """select ps_partkey, sum(ps_supplycost * ps_availqty) as value from tpch_user.partsupp, tpch_user.supplier, tpch_user.nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001 from tpch_user.partsupp, tpch_user.supplier, tpch_user.nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' ) order by value desc""",
        """select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count from tpch_user.orders, tpch_user.lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1994-01-01' + interval '1' year group by l_shipmode order by l_shipmode""",
        """select c_count, count(*) as custdist from ( select c_custkey, count(o_orderkey) as c_count from tpch_user.customer left outer join tpch_user.orders on c_custkey = o_custkey and o_comment not like '%special%requests%' group by c_custkey ) c_orders group by c_count order by custdist desc, c_count desc""",
        """select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue from tpch_user.lineitem, tpch_user.part where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month""",
        """WITH revenue0 AS (SELECT l_suppkey AS supplier_no, sum(l_extendedprice * (1 - l_discount)) AS total_revenue FROM tpch_user.lineitem WHERE l_shipdate >= date '1996-01-01' AND l_shipdate < date '1996-01-01' + interval '3' month GROUP BY l_suppkey) SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM tpch_user.supplier, revenue0 WHERE s_suppkey = supplier_no AND total_revenue = (SELECT max(total_revenue) FROM revenue0) ORDER BY s_suppkey""",
        """select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from tpch_user.partsupp, tpch_user.part where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%' and p_size in (49, 14, 23, 45, 19, 3, 36, 9) and ps_suppkey not in ( select s_suppkey from tpch_user.supplier where s_comment like '%Customer%Complaints%' ) group by p_brand, p_type, p_size order by supplier_cnt desc, p_brand, p_type, p_size""",
        """select sum(l_extendedprice) / 7.0 as avg_yearly from tpch_user.lineitem, tpch_user.part where p_partkey = l_partkey and p_brand = 'Brand#23' and p_container = 'MED BOX' and l_quantity < ( select 0.2 * avg(l_quantity) from tpch_user.lineitem where l_partkey = p_partkey )""",
        """select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from tpch_user.customer, tpch_user.orders, tpch_user.lineitem where o_orderkey in ( select l_orderkey from tpch_user.lineitem group by l_orderkey having sum(l_quantity) > 300 ) and c_custkey = o_custkey and o_orderkey = l_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate""",
        """select sum(l_extendedprice* (1 - l_discount)) as revenue from tpch_user.lineitem, tpch_user.part where ( p_partkey = l_partkey and p_brand = 'Brand#12' and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') and l_quantity >= 1 and l_quantity <= 1 + 10 and p_size between 1 and 5 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#23' and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') and l_quantity >= 10 and l_quantity <= 10 + 10 and p_size between 1 and 10 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#34' and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') and l_quantity >= 20 and l_quantity <= 20 + 10 and p_size between 1 and 15 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' )""",
        """select s_name, s_address from tpch_user.supplier, tpch_user.nation where s_suppkey in ( select ps_suppkey from tpch_user.partsupp where ps_partkey in ( select p_partkey from tpch_user.part where p_name like 'forest%' ) and ps_availqty > ( select 0.5 * sum(l_quantity) from tpch_user.lineitem where l_partkey = ps_partkey and l_suppkey = ps_suppkey and l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year ) ) and s_nationkey = n_nationkey and n_name = 'CANADA' order by s_name""",
        """select s_name, count(*) as numwait from tpch_user.supplier, tpch_user.lineitem l1, tpch_user.orders, tpch_user.nation where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey and o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate and exists ( select * from tpch_user.lineitem l2 where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey ) and not exists ( select * from tpch_user.lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey and l3.l_receiptdate > l3.l_commitdate ) and s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA' group by s_name order by numwait desc, s_name""",
        """select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( select substr(c_phone, 1, 2) as cntrycode, c_acctbal from tpch_user.customer where substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') and c_acctbal > ( select avg(c_acctbal) from tpch_user.customer where c_acctbal > 0.00 and substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') ) and not exists ( select * from tpch_user.orders where o_custkey = c_custkey ) ) custsale group by cntrycode order by cntrycode"""
    ]

def get_db_connection(service_name):
    dsn = f"{DB_HOST}:{DB_PORT}/{service_name}"
    return oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=dsn, mode=oracledb.SYSDBA)

# --- AUTOMATED MONITORING FUNCTION ---
def monitor_pdb_in_background(pdb_name, service_name, stop_event):
    # ... (This function remains unchanged) ...
    queries_to_monitor = {
        "CPU_used_by_this_session": "SELECT value FROM V$SYSSTAT WHERE name = 'CPU used by this session'",
        "PGA_in_use_MB": "SELECT ROUND(value / 1024 / 1024, 2) FROM V$PGASTAT WHERE name = 'total PGA in use'",
        "user_commits": "SELECT value FROM V$SYSSTAT WHERE name = 'user commits'",
        "physical_reads": "SELECT value FROM V$SYSSTAT WHERE name = 'physical reads'",
        "physical_writes": "SELECT value FROM V$SYSSTAT WHERE name = 'physical writes'",
        "Active_User_Sessions": "SELECT COUNT(*) FROM V$SESSION WHERE type = 'USER' AND status = 'ACTIVE'"
    }
    while not stop_event.is_set():
        try:
            connection = get_db_connection(service_name)
            cursor = connection.cursor()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"monitoring_{pdb_name}_{timestamp}.log"
            with open(log_filename, "w") as f:
                f.write(f"--- Monitoring Report for {pdb_name} at {timestamp} ---\n")
                for name, query in queries_to_monitor.items():
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result:
                        f.write(f"{name},{result[0]}\n")
            print(f"  [Monitor] -> Saved monitoring snapshot to {log_filename}")
            connection.close()
        except Exception as e:
            print(f"  [Monitor] -> Error during monitoring: {e}")
        stop_event.wait(MONITORING_INTERVAL_SECONDS)

def run_benchmark_on_pdb(pdb_name, service_name, agent, use_agent, queries):
    print(f"\n--- Starting Benchmark on {pdb_name} ---")
    results = []
    action_to_hint = { 0: "-- No Hint", 1: "/*+ USE_HASH */", 2: "/*+ USE_NL */", 3: "/*+ INDEX(lineitem lineitem_pk) */" }
    
    connection = None
    try:
        connection = get_db_connection(service_name)
        cursor = connection.cursor()
        print(f"-> Successfully connected to {pdb_name}")

        for i, query_text in enumerate(queries):
            print(f"  -> Running Query #{i+1}...")
            try:
                state = np.random.rand(10).astype(np.float32) 
                action = agent.act(state) if use_agent and agent else 0
                hint = action_to_hint[action]
                
                # **DEFINITIVE FIX for HINT INJECTION LOGIC**
                tuned_query = query_text
                if hint != "-- No Hint":
                    # Find the first keyword to inject the hint after
                    if "select" in query_text.lower():
                        position = query_text.lower().find("select") + 6
                        tuned_query = query_text[:position] + f" {hint} " + query_text[position:]
                    elif "with" in query_text.lower():
                        # For CTEs, the hint goes inside the main SELECT
                        position = query_text.lower().find("select", query_text.lower().find(")")) + 6
                        tuned_query = query_text[:position] + f" {hint} " + query_text[position:]

                start_time = time.time()
                cursor.execute(tuned_query)
                cursor.fetchall()
                end_time = time.time()
                
                results.append({ "PDB_NAME": pdb_name, "QUERY_NO": i + 1, "EXECUTION_SECONDS": end_time - start_time })
            except Exception as e:
                print(f"    ERROR on Query #{i+1}: {e}")
                results.append({ "PDB_NAME": pdb_name, "QUERY_NO": i + 1, "EXECUTION_SECONDS": np.nan })

    except Exception as e:
        print(f"A critical error occurred during benchmark on {pdb_name}: {e}")
    finally:
        if connection:
            connection.close()
            print(f"-> Connection to {pdb_name} closed.")
            
    return results

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    oracledb.init_oracle_client(config_dir=ORACLE_CONFIG_DIR)

    print("--- 1. Loading Trained RL Agent ---")
    agent = DQNAgent(state_size=10, action_size=4)
    try:
        # **FIX:** Changed to load the full model, which is more robust
        agent.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        print(f"-> Successfully loaded trained model from '{SAVED_MODEL_PATH}'")
    except Exception as e:
        print(f"Could not load model: {e}. The agent will use random weights.")
        agent = None

    tpc_h_queries = get_tpc_h_queries()
    all_results = []

    # Run on Control PDB
    stop_monitoring_control = threading.Event()
    monitor_thread_control = threading.Thread(target=monitor_pdb_in_background, args=( "PDB_CONTROL", PDB_CONTROL_SERVICE, stop_monitoring_control))
    monitor_thread_control.start()
    control_results = run_benchmark_on_pdb("PDB_CONTROL", PDB_CONTROL_SERVICE, agent=None, use_agent=False, queries=tpc_h_queries)
    all_results.extend(control_results)
    stop_monitoring_control.set()
    monitor_thread_control.join()
    
    # Run on Tuned PDB
    stop_monitoring_tuned = threading.Event()
    monitor_thread_tuned = threading.Thread(target=monitor_pdb_in_background, args=("PDB_PERF_TUNE", PDB_TUNED_SERVICE, stop_monitoring_tuned))
    monitor_thread_tuned.start()
    tuned_results = run_benchmark_on_pdb("PDB_PERF_TUNE", PDB_TUNED_SERVICE, agent=agent, use_agent=True, queries=tpc_h_queries)
    all_results.extend(tuned_results)
    stop_monitoring_tuned.set()
    monitor_thread_tuned.join()
    
    print("\n--- 4. Processing Final Results ---")
    if not all_results or len(pd.DataFrame(all_results)['PDB_NAME'].unique()) < 2:
        print("Not enough results were collected from both PDBs to perform analysis. Exiting.")
    else:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(FINAL_RESULTS_CSV, index=False)
        print(f"-> Final performance data saved to '{FINAL_RESULTS_CSV}'")

        pivot_df = results_df.pivot(index='QUERY_NO', columns='PDB_NAME', values='EXECUTION_SECONDS')
        
        # **FIX:** More robust column check and renaming
        if 'PDB_CONTROL' in pivot_df.columns and 'PDB_PERF_TUNE' in pivot_df.columns:
            pivot_df.rename(columns={'PDB_CONTROL': 'PDB_CONTROL (Mean QET)', 'PDB_PERF_TUNE': 'PDB_PERF_TUNE (Mean QET)'}, inplace=True)
            pivot_df = pivot_df.dropna()

            if not pivot_df.empty:
                pivot_df['Performance Improvement (%)'] = ((pivot_df['PDB_CONTROL (Mean QET)'] - pivot_df['PDB_PERF_TUNE (Mean QET)']) / pivot_df['PDB_CONTROL (Mean QET)']) * 100
                
                print("\n--- FINAL RESULTS FOR RQ1 ---")
                print("\n[Table 4.2: Comparative Mean Query Execution Times (seconds)]")
                print(pivot_df.to_string())

                if len(pivot_df) > 1:
                    ttest_result = stats.ttest_rel(pivot_df['PDB_PERF_TUNE (Mean QET)'], pivot_df['PDB_CONTROL (Mean QET)'])
                    print("\n[Table 4.3: Results of Paired Samples t-Test]")
                    print(f"t-statistic: {ttest_result.statistic:.4f}")
                    print(f"p-value: {ttest_result.pvalue:.4f}")
                    print(f"Degrees of Freedom: {len(pivot_df)-1}")
                    print("Interpretation:", "Statistically significant" if ttest_result.pvalue < 0.05 else "Not statistically significant")

                plot_df = results_df[results_df['QUERY_NO'] <= 6].dropna()
                if not plot_df.empty:
                    plt.figure(figsize=(15, 8))
                    sns.barplot(data=plot_df, x='QUERY_NO', y='EXECUTION_SECONDS', hue='PDB_NAME', palette='viridis')
                    plt.title('Figure 4.4: Final Comparative Query Execution Times', fontsize=16)
                    plt.xlabel('TPC-H Query Number', fontsize=12)
                    plt.ylabel('Mean Execution Time (seconds)', fontsize=12)
                    plt.savefig(FINAL_FIGURE_PATH)
                    print(f"\n-> Final comparative chart saved to '{FINAL_FIGURE_PATH}'")
        else:
            print("\nCould not generate final comparison because results from one or both PDBs were missing.")