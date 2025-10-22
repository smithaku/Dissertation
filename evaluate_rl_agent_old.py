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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# ACTION REQUIRED: Replace "YourPassword" with your actual SYS password.
DB_USER = "sys"
DB_PASSWORD = "junior25SA"
DB_HOST = "db01.dc01.com"
DB_PORT = 1521

PDB_CONTROL_SERVICE = "pdb_control.dc01.com"
PDB_TUNED_SERVICE = "pdb_perf_tune.dc01.com"
ORACLE_CONFIG_DIR = r"E:\app\oracle\product\19.3.0\db_home\network\admin"

SAVED_MODEL_PATH = "rl_tuning_agent.keras"
FINAL_RESULTS_CSV = "final_performance_results.csv"
FINAL_FIGURE_PATH = "final_figure_4_4_comparative_qet.png"

# --- DQN AGENT CLASS (Required to reconstruct the model) ---
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
        # In deployment, we only exploit, no exploration (no epsilon)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

# --- HELPER FUNCTIONS ---
def get_tpc_h_queries():
    # Final list of all 22 TPC-H queries
    return [
        """select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order from lineitem where l_shipdate <= date '1998-12-01' - interval '90' day group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus""",
        """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part, supplier, partsupp, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = 15 and p_type like '%BRASS' and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' and ps_supplycost = ( select min(ps_supplycost) from partsupp, supplier, nation, region where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'EUROPE' ) order by s_acctbal desc, n_name, s_name, p_partkey""",
        """select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment = 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-15' and l_shipdate > date '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate""",
        """select o_orderpriority, count(*) as order_count from orders where o_orderdate >= date '1993-07-01' and o_orderdate < date '1993-07-01' + interval '3' month and exists ( select * from lineitem where l_orderkey = o_orderkey and l_commitdate < l_receiptdate ) group by o_orderpriority order by o_orderpriority""",
        """select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue from customer, orders, lineitem, supplier, nation, region where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = 'ASIA' and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year group by n_name order by revenue desc""",
        """select sum(l_extendedprice * l_discount) as revenue from lineitem where l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year and l_discount between .06 - 0.01 and .06 + 0.01 and l_quantity < 24""",
        """select supp_nation, cust_nation, l_year, sum(volume) as revenue from ( select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year, l_extendedprice * (1 - l_discount) as volume from supplier, lineitem, orders, customer, nation n1, nation n2 where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey and ( (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') ) and l_shipdate between date '1995-01-01' and date '1996-12-31' ) shipping group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year""",
        """select o_year, sum(case when nation = 'BRAZIL' then volume else 0 end) / sum(volume) as mkt_share from ( select extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) as volume, n2.n_name as nation from part, supplier, lineitem, orders, customer, nation n1, nation n2, region where p_partkey = l_partkey and s_suppkey = l_suppkey and l_orderkey = o_orderkey and o_custkey = c_custkey and c_nationkey = n1.n_nationkey and n1.n_regionkey = r_regionkey and r_name = 'AMERICA' and s_nationkey = n2.n_nationkey and o_orderdate between date '1995-01-01' and date '1996-12-31' and p_type = 'ECONOMY ANODIZED STEEL' ) all_nations group by o_year order by o_year""",
        """select nation, o_year, sum(amount) as sum_profit from ( select n_name as nation, extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount from part, supplier, lineitem, partsupp, orders, nation where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey and p_partkey = l_partkey and o_orderkey = l_orderkey and s_nationkey = n_nationkey and p_name like '%green%' ) profit group by nation, o_year order by nation, o_year desc""",
        """select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from customer, orders, lineitem, nation where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc""",
        """select ps_partkey, sum(ps_supplycost * ps_availqty) as value from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001 from partsupp, supplier, nation where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY' ) order by value desc""",
        """select l_shipmode, sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH' then 1 else 0 end) as high_line_count, sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH' then 1 else 0 end) as low_line_count from orders, lineitem where o_orderkey = l_orderkey and l_shipmode in ('MAIL', 'SHIP') and l_commitdate < l_receiptdate and l_shipdate < l_commitdate and l_receiptdate >= date '1994-01-01' and l_receiptdate < date '1994-01-01' + interval '1' year group by l_shipmode order by l_shipmode""",
        """select c_count, count(*) as custdist from ( select c_custkey, count(o_orderkey) as c_count from customer left outer join orders on c_custkey = o_custkey and o_comment not like '%special%requests%' group by c_custkey ) c_orders group by c_count order by custdist desc, c_count desc""",
        """select 100.00 * sum(case when p_type like 'PROMO%' then l_extendedprice * (1 - l_discount) else 0 end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue from lineitem, part where l_partkey = p_partkey and l_shipdate >= date '1995-09-01' and l_shipdate < date '1995-09-01' + interval '1' month""",
        """select s_suppkey, s_name, s_address, s_phone, total_revenue from supplier, (select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) as total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate < date '1996-01-01' + interval '3' month group by l_suppkey) revenue0 where s_suppkey = supplier_no and total_revenue = ( select max(total_revenue) from (select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) as total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate < date '1996-01-01' + interval '3' month group by l_suppkey) ) order by s_suppkey""",
        """select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt from partsupp, part where p_partkey = ps_partkey and p_brand <> 'Brand#45' and p_type not like 'MEDIUM POLISHED%' and p_size in (49, 14, 23, 45, 19, 3, 36, 9) and ps_suppkey not in ( select s_suppkey from supplier where s_comment like '%Customer%Complaints%' ) group by p_brand, p_type, p_size order by supplier_cnt desc, p_brand, p_type, p_size""",
        """select sum(l_extendedprice) / 7.0 as avg_yearly from lineitem, part where p_partkey = l_partkey and p_brand = 'Brand#23' and p_container = 'MED BOX' and l_quantity < ( select 0.2 * avg(l_quantity) from lineitem where l_partkey = p_partkey )""",
        """select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from customer, orders, lineitem where o_orderkey in ( select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300 ) and c_custkey = o_custkey and o_orderkey = l_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate""",
        """select sum(l_extendedprice* (1 - l_discount)) as revenue from lineitem, part where ( p_partkey = l_partkey and p_brand = 'Brand#12' and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') and l_quantity >= 1 and l_quantity <= 1 + 10 and p_size between 1 and 5 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#23' and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') and l_quantity >= 10 and l_quantity <= 10 + 10 and p_size between 1 and 10 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' ) or ( p_partkey = l_partkey and p_brand = 'Brand#34' and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') and l_quantity >= 20 and l_quantity <= 20 + 10 and p_size between 1 and 15 and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON' )""",
        """select s_name, s_address from supplier, nation where s_suppkey in ( select ps_suppkey from partsupp where ps_partkey in ( select p_partkey from part where p_name like 'forest%' ) and ps_availqty > ( select 0.5 * sum(l_quantity) from lineitem where l_partkey = ps_partkey and l_suppkey = ps_suppkey and l_shipdate >= date '1994-01-01' and l_shipdate < date '1994-01-01' + interval '1' year ) ) and s_nationkey = n_nationkey and n_name = 'CANADA' order by s_name""",
        """select s_name, count(*) as numwait from supplier, lineitem l1, orders, nation where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey and o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate and exists ( select * from lineitem l2 where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey ) and not exists ( select * from lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey and l3.l_receiptdate > l3.l_commitdate ) and s_nationkey = n_nationkey and n_name = 'SAUDI ARABIA' group by s_name order by numwait desc, s_name""",
        """select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal from ( select substr(c_phone, 1, 2) as cntrycode, c_acctbal from customer where substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') and c_acctbal > ( select avg(c_acctbal) from customer where c_acctbal > 0.00 and substr(c_phone, 1, 2) in ('13', '31', '23', '29', '30', '18', '17') ) and not exists ( select * from orders where o_custkey = c_custkey ) ) as custsale group by cntrycode order by cntrycode"""
    ]

def get_db_connection(service_name):
    """Establishes and returns a database connection."""
    dsn = f"{DB_HOST}:{DB_PORT}/{service_name}"
    return oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=dsn, mode=oracledb.SYSDBA)

def run_benchmark_on_pdb(pdb_name, service_name, agent, use_agent, queries):
    """
    Runs the full TPC-H benchmark on a specified PDB and logs results.
    """
    print(f"\n--- Starting Benchmark on {pdb_name} ---")
    print(f"Reminder for RQ3: Start monitoring system-level metrics (CPU, Memory, TPS) for this run.")
    
    results = []
    action_to_hint = { 0: "-- No Hint", 1: "/*+ USE_HASH */", 2: "/*+ USE_NL */", 3: "/*+ INDEX(lineitem lineitem_pk) */" }
    
    try:
        connection = get_db_connection(service_name)
        cursor = connection.cursor()
        print(f"-> Successfully connected to {pdb_name}")

        for i, query_text in enumerate(queries):
            print(f"  -> Running Query #{i+1}...")
            
            # --- This is a placeholder for getting the real-time state ---
            state = np.random.rand(10).astype(np.float32) 
            
            action = 0 # Default action is no hint
            if use_agent and agent:
                action = agent.act(state)

            hint = action_to_hint[action]
            tuned_query = f"SELECT {hint} {query_text[6:]}"

            start_time = time.time()
            cursor.execute(tuned_query)
            cursor.fetchall() # Fetch all results to ensure query completes
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            results.append({
                "PDB_NAME": pdb_name,
                "QUERY_NO": i + 1,
                "EXECUTION_SECONDS": execution_time
            })
            
    except Exception as e:
        print(f"An error occurred during benchmark on {pdb_name}: {e}")
    finally:
        if 'connection' in locals() and connection.is_healthy():
            connection.close()
            print(f"-> Connection to {pdb_name} closed.")
            
    return results


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    oracledb.init_oracle_client(config_dir=ORACLE_CONFIG_DIR)

    # 1. Load the trained agent
    print("--- 1. Loading Trained RL Agent ---")
    state_size = 10 # Must match the state size used during training
    action_size = 4  # Must match the action size used during training
    agent = DQNAgent(state_size, action_size)
    try:
        agent.model.load_weights(SAVED_MODEL_PATH)
        print(f"-> Successfully loaded trained model from '{SAVED_MODEL_PATH}'")
    except Exception as e:
        print(f"Could not load model weights: {e}. The agent will use random weights.")
        agent = None # Set agent to None if loading fails

    # 2. Get the queries
    tpc_h_queries = get_tpc_h_queries()
    
    # 3. Run benchmarks
    all_results = []
    # Run on Control PDB (without the agent)
    control_results = run_benchmark_on_pdb("PDB_CONTROL", PDB_CONTROL_SERVICE, agent=None, use_agent=False, queries=tpc_h_queries)
    all_results.extend(control_results)
    
    # Run on Tuned PDB (with the agent)
    tuned_results = run_benchmark_on_pdb("PDB_PERF_TUNE", PDB_TUNED_SERVICE, agent=agent, use_agent=True, queries=tpc_h_queries)
    all_results.extend(tuned_results)
    
    # 4. Process and Save Results
    print("\n--- 4. Processing Final Results ---")
    if not all_results:
        print("No results were collected. Exiting.")
    else:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(FINAL_RESULTS_CSV, index=False)
        print(f"-> Final performance data saved to '{FINAL_RESULTS_CSV}'")

        # 5. Generate Final Table and Chart for RQ1
        pivot_df = results_df.pivot(index='QUERY_NO', columns='PDB_NAME', values='EXECUTION_SECONDS')
        pivot_df.columns = ['PDB_CONTROL (Mean QET)', 'PDB_PERF_TUNE (Mean QET)']
        pivot_df = pivot_df.dropna() # Drop queries that might have failed on one PDB

        pivot_df['Performance Improvement (%)'] = ((pivot_df['PDB_CONTROL (Mean QET)'] - pivot_df['PDB_PERF_TUNE (Mean QET)']) / pivot_df['PDB_CONTROL (Mean QET)']) * 100
        
        print("\n--- FINAL RESULTS FOR RQ1 ---")
        print("\n[Table 4.2: Comparative Mean Query Execution Times (seconds)]")
        print(pivot_df.to_string())

        # Perform Paired T-Test
        if len(pivot_df) > 1:
            ttest_result = stats.ttest_rel(pivot_df['PDB_PERF_TUNE (Mean QET)'], pivot_df['PDB_CONTROL (Mean QET)'])
            print("\n[Table 4.3: Results of Paired Samples t-Test]")
            print(f"t-statistic: {ttest_result.statistic:.4f}")
            print(f"p-value: {ttest_result.pvalue:.4f}")
            print(f"Degrees of Freedom: {len(pivot_df)-1}")
            if ttest_result.pvalue < 0.05:
                print("Interpretation: Since the p-value is less than 0.05, the result is statistically significant.")
            else:
                print("Interpretation: Since the p-value is not less than 0.05, the result is not statistically significant.")

        # Generate and Save Bar Chart
        plot_df = results_df[results_df['QUERY_NO'] <= 6] # Plot first 6 queries for readability
        plt.figure(figsize=(15, 8))
        sns.barplot(data=plot_df, x='QUERY_NO', y='EXECUTION_SECONDS', hue='PDB_NAME', palette='viridis')
        plt.title('Figure 4.4: Final Comparative Query Execution Times', fontsize=16)
        plt.xlabel('TPC-H Query Number', fontsize=12)
        plt.ylabel('Mean Execution Time (seconds)', fontsize=12)
        plt.savefig(FINAL_FIGURE_PATH)
        print(f"\n-> Final comparative chart saved to '{FINAL_FIGURE_PATH}'")