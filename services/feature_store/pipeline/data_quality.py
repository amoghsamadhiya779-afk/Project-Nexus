#!/usr/bin/env python3
"""
=============================================================================
Data Quality & Contract Validator (Great Expectations style)
Prevents pipeline corruption by asserting strict schema, null-value, and 
distribution boundary rules on offline datasets before feature materialization.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from shared.utils.config import config

class DataContractValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []

    def expect_column_to_exist(self, col: str):
        if col not in self.df.columns:
            self.errors.append(f"❌ Missing required column: '{col}'")
        else:
            print(f"✅ Column '{col}' exists.")

    def expect_column_values_to_not_be_null(self, col: str, threshold_pct: float = 0.01):
        if col in self.df.columns:
            null_pct = self.df[col].isnull().sum() / len(self.df)
            if null_pct > threshold_pct:
                self.errors.append(f"❌ Column '{col}' exceeds null threshold: {null_pct*100:.2f}% nulls (Max {threshold_pct*100}%)")
            else:
                print(f"✅ Column '{col}' null rate ({null_pct*100:.2f}%) is within bounds.")

    def expect_column_values_to_be_between(self, col: str, min_val: float, max_val: float):
        if col in self.df.columns:
            outliers = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            if len(outliers) > 0:
                self.errors.append(f"❌ Column '{col}' has {len(outliers)} values outside bounds [{min_val}, {max_val}].")
            else:
                print(f"✅ Column '{col}' all values within [{min_val}, {max_val}].")

    def validate(self) -> bool:
        print("\n--- Nexus Data Contract Validation Report ---")
        if not self.errors:
            print("🚀 STATUS: PASSED. Dataset is clean and ready for materialization.")
            return True
        else:
            print("🛑 STATUS: FAILED. Data quality breaches detected:")
            for err in self.errors:
                print(f"   {err}")
            return False

if __name__ == "__main__":
    print("[*] Loading offline Feature Store dataset for Data Quality checks...")
    try:
        parquet_path = f"{config.BASE_DATA_DIR}/features/offline/historical_interactions.parquet"
        df = pd.read_parquet(parquet_path)
        
        validator = DataContractValidator(df)
        
        # Data Contract Rules
        validator.expect_column_to_exist("user_id")
        validator.expect_column_to_exist("item_id")
        validator.expect_column_to_exist("event_type")
        validator.expect_column_to_exist("price")
        
        validator.expect_column_values_to_not_be_null("user_id")
        validator.expect_column_values_to_not_be_null("price")
        
        # Prices should be positive, nobody gets paid to buy items
        validator.expect_column_values_to_be_between("price", 0.0, 50000.0)
        
        passed = validator.validate()
        
    except Exception as e:
        print(f"[❌] Validation failed to execute: {e}")