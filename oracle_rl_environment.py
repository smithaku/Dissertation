# Import necessary libraries
import numpy as np
import pandas as pd
import oracledb  # The Python driver for Oracle
import gymnasium as gym
from gymnasium import spaces
import os # Required for setting the config directory

# --- The Custom Environment Class ---

class OracleTuningEnv(gym.Env):
    """
    A custom Gymnasium environment for reinforcement learning-based
    Oracle database query tuning.
    """
    def __init__(self, db_connection_string, tpc_h_queries):
        """
        Initializes the environment.
        This is where we define the state and action spaces.
        """
        super(OracleTuningEnv, self).__init__()

        # --- 1. Database Connection ---
        self.db_connection_string = db_connection_string
        self.connection = None
        self.cursor = None
        
        # --- 2. TPC-H Queries ---
        self.tpc_h_queries = tpc_h_queries
        self.current_query_id = 0
        self.current_query_text = ""

        # --- 3. Action Space Definition ---
        self.action_space = spaces.Discrete(4) 
        self._action_to_hint = {
            0: "-- No Hint",
            1: "/*+ USE_HASH */",
            2: "/*+ USE_NL */",
            3: "/*+ INDEX(lineitem lineitem_pk) */"
        }

        # --- 4. State Space Definition ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def _get_state(self):
        """
        A helper method to retrieve the current state of the database.
        """
        # Placeholder for real state-gathering logic
        return np.random.rand(10).astype(np.float32)

    def step(self, action):
        """
        Executes one step in the environment.
        """
        hint = self._action_to_hint[action]
        tuned_query = f"SELECT {hint} {self.current_query_text[6:]}"

        # Simulate performance measurement
        baseline_qet = np.random.uniform(0.5, 2.0)
        tuned_qet = baseline_qet * np.random.uniform(0.7, 1.5) 

        reward = (baseline_qet - tuned_qet) / baseline_qet
        
        done = True
        next_state = self._get_state()
        info = {}

        return next_state, reward, done, info

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        if not self.connection:
            # **FINAL UPDATE:** Path updated based on your screenshot.
            oracle_config_dir = r"E:\app\oracle\product\19.3.0\db_home\network\admin"
            
            # Initialize the Oracle client with the correct config directory
            oracledb.init_oracle_client(config_dir=oracle_config_dir)

            # Now, the connect call will work
            self.connection = oracledb.connect(self.db_connection_string, mode=oracledb.SYSDBA)
            self.cursor = self.connection.cursor()
            print("--- Database Connection Established (as SYSDBA) ---")

        self.current_query_id = np.random.choice(len(self.tpc_h_queries))
        self.current_query_text = self.tpc_h_queries[self.current_query_id]
        
        print(f"\n--- New Episode: Tuning Query #{self.current_query_id + 1} ---")

        initial_state = self._get_state()
        info = {}

        return initial_state, info

    def close(self):
        """
        Closes the database connection to clean up resources.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("--- Database Connection Closed ---")