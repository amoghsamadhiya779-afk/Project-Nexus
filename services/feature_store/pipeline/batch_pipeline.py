"""
services/feature_store/pipeline/batch_pipeline.py
===================================================
Batch feature materialisation pipeline.

Reads from offline store (Parquet) → computes aggregations →
writes to both online (Redis) and offline (Parquet) stores.

Designed to run on schedule via Dagster or cron.
For large-scale, swap DuckDB → Spark (same API surface).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from loguru import logger

from services.feature_store.core.feature_view import (
    FeatureRegistry, FeatureView, WindowAggregation, Aggregation
)
from services.feature_store.storage.stores import MaterialisationEngine
from shared.utils.config import settings
from shared.monitoring.metrics import feature_store_metrics


AGGREGATION_SQL: Dict[Aggregation, str] = {
    Aggregation.SUM:   "SUM",
    Aggregation.AVG:   "AVG",
    Aggregation.COUNT: "COUNT",
    Aggregation.MAX:   "MAX",
    Aggregation.MIN:   "MIN",
    Aggregation.LAST:  "LAST_VALUE",
    Aggregation.P50:   "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col})",
    Aggregation.P90:   "PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY {col})",
    Aggregation.P99:   "PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {col})",
    Aggregation.STD:   "STDDEV",
}


class BatchPipeline:
    """
    Orchestrates batch feature computation and materialisation.

    Flow:
    1. For each FeatureView, load source data from Parquet
    2. Compute window aggregations via DuckDB SQL
    3. Write online features to Redis
    4. Write offline features to Parquet partitions
    """

    def __init__(
        self,
        engine:   MaterialisationEngine,
        registry: FeatureRegistry,
        offline_path: str = None,
    ):
        self._engine      = engine
        self._registry    = registry
        self._offline_path = offline_path or settings.offline_store_path
        self._duckdb      = duckdb.connect(database=":memory:")

    async def run_all(self, as_of: Optional[datetime] = None) -> Dict[str, Any]:
        """Run materialisation for all registered feature views."""
        as_of    = as_of or datetime.utcnow()
        results  = {}
        views    = self._registry.list_views()

        logger.info(f"Starting batch materialisation for {len(views)} views (as_of={as_of})")

        for view_name in views:
            try:
                stats = await self.run_view(view_name, as_of)
                results[view_name] = {"status": "ok", **stats}
                feature_store_metrics.materialise_total.labels(
                    feature_view=view_name, status="ok"
                ).inc()
            except Exception as e:
                logger.error(f"Materialisation failed for {view_name}: {e}")
                results[view_name] = {"status": "error", "error": str(e)}
                feature_store_metrics.materialise_total.labels(
                    feature_view=view_name, status="error"
                ).inc()

        logger.info(f"Batch materialisation complete: {results}")
        return results

    async def run_view(
        self,
        view_name: str,
        as_of:     Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Materialise a single feature view."""
        as_of = as_of or datetime.utcnow()
        view  = self._registry.get_view(view_name)
        if view is None:
            raise ValueError(f"View '{view_name}' not registered")

        logger.info(f"Materialising: {view_name}")

        # Load source data
        df = self._load_source(view, as_of)
        if df.empty:
            logger.warning(f"No source data for {view_name}")
            return {"rows": 0}

        # Compute window aggregations
        if view.window_aggregations:
            agg_df = self._compute_window_aggregations(view, df, as_of)
            entity_key = view.entities[0].join_key
            df = df.merge(agg_df, on=entity_key, how="left")

        # Materialise to stores
        online_written, offline_written = await self._engine.materialise_batch(
            view, df, ttl_seconds=int(view.ttl.total_seconds())
        )

        return {
            "rows":            len(df),
            "online_written":  online_written,
            "offline_written": offline_written,
            "as_of":           as_of.isoformat(),
        }

    def _load_source(self, view: FeatureView, as_of: datetime) -> pd.DataFrame:
        """Load source data from Parquet offline store."""
        source_path = Path(self._offline_path) / view.source.name
        if not source_path.exists():
            logger.warning(f"Source path not found: {source_path}")
            return pd.DataFrame()

        # Use DuckDB for efficient Parquet reads
        parquet_glob = str(source_path / "**/*.parquet")
        query = f"""
            SELECT *
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            WHERE {view.source.timestamp_field} <= '{as_of.isoformat()}'
        """
        try:
            df = self._duckdb.execute(query).df()
            logger.debug(f"Loaded {len(df):,} rows from {source_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load source {view.source.name}: {e}")
            return pd.DataFrame()

    def _compute_window_aggregations(
        self,
        view:  FeatureView,
        df:    pd.DataFrame,
        as_of: datetime,
    ) -> pd.DataFrame:
        """
        Compute time-window aggregations using DuckDB.
        This is the core logic of a feature store's compute engine.
        """
        entity_key = view.entities[0].join_key
        ts_col     = view.source.timestamp_field
        results    = []

        # Register the dataframe in DuckDB
        self._duckdb.register("source_df", df)

        for wa in view.window_aggregations:
            window_start = as_of - wa.window
            feature_col  = wa.feature.name

            if feature_col not in df.columns:
                logger.debug(f"Column {feature_col} not in source, skipping")
                continue

            agg_fn = AGGREGATION_SQL.get(wa.aggregation, "COUNT")
            if "{col}" in agg_fn:
                agg_expr = agg_fn.format(col=feature_col)
            else:
                agg_expr = f"{agg_fn}({feature_col})"

            query = f"""
                SELECT
                    {entity_key},
                    {agg_expr} AS {wa.feature_name}
                FROM source_df
                WHERE {ts_col} >= '{window_start.isoformat()}'
                  AND {ts_col} <= '{as_of.isoformat()}'
                GROUP BY {entity_key}
            """
            try:
                agg_result = self._duckdb.execute(query).df()
                results.append(agg_result)
            except Exception as e:
                logger.warning(f"Aggregation failed for {wa.feature_name}: {e}")

        if not results:
            return pd.DataFrame({entity_key: df[entity_key].unique()})

        # Merge all aggregation results
        merged = results[0]
        for r in results[1:]:
            merged = merged.merge(r, on=entity_key, how="outer")

        self._duckdb.unregister("source_df")
        return merged
