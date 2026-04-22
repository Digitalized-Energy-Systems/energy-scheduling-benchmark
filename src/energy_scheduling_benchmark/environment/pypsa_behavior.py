"""PyPSA-backed mango :class:`~mango.simulation.environment.Behavior`.

Plays the same role as ``PowerSystemsBehavior`` in
``mango-energy-environments`` but uses a PyPSA ``Network`` as the
underlying power-system data model.

PyPSA stores components in pandas DataFrames on the ``Network`` object:

* ``n.generators`` — dispatchable and non-dispatchable generators
* ``n.loads`` — power demands
* ``n.storage_units`` — batteries / storage
* ``n.buses``, ``n.lines``, ``n.snapshots`` — topology & time

This behavior classifies generators into ``THERMAL`` and ``RENEWABLE``
via their ``carrier`` attribute.  Carriers can be customised by passing
``renewable_carriers``/``thermal_carriers`` to the constructor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from mango.simulation.environment import Behavior, Environment
from mango.util.clock import Clock

logger = logging.getLogger(__name__)

__all__ = [
    "THERMAL",
    "RENEWABLE",
    "LOAD",
    "STORAGE",
    "ComponentRef",
    "PowerUpdateInfo",
    "PyPSABehavior",
    "calculate_initial_time",
    "get_possible_components",
    "get_components_by_type",
]


THERMAL = "thermal"
RENEWABLE = "renewable"
LOAD = "load"
STORAGE = "storage"

_DEFAULT_RENEWABLE_CARRIERS = frozenset(
    {"wind", "solar", "pv", "onwind", "offwind", "hydro", "ror", "biomass"}
)

# PyPSA column names
_COL_P_NOM = "p_nom"
_COL_P_SET = "p_set"
_COL_P_MAX_PU = "p_max_pu"
_COL_P_MIN_PU = "p_min_pu"
_COL_MARGINAL_COST = "marginal_cost"
_COL_CARRIER = "carrier"


@dataclass(frozen=True)
class PowerUpdateInfo:
    """Agent event emitted when a component's power value changes."""


@dataclass(frozen=True)
class ComponentRef:
    """Reference to a single component in a PyPSA network.

    Parameters
    ----------
    element_type:
        One of :data:`THERMAL`, :data:`RENEWABLE`, :data:`LOAD`, :data:`STORAGE`.
    component_id:
        The PyPSA component name (row label of the underlying DataFrame).
    """

    element_type: str
    component_id: str

    def __iter__(self):
        yield self.element_type
        yield self.component_id


class PyPSABehavior(Behavior):
    """Mango :class:`~mango.simulation.environment.Behavior` backed by a PyPSA network.

    Parameters
    ----------
    net:
        A ``pypsa.Network`` with buses, generators, loads and (optionally)
        storage units populated.
    timeseries:
        Mapping of :class:`ComponentRef` (or ``(element_type, component_id)``
        tuples) to :class:`pandas.Series` with a ``DatetimeIndex``.  Values
        are MW set-points (or per-unit availability for renewables; see
        :meth:`_apply_timeseries_update`).
    relevant_types:
        Element types to manage.  Defaults to all four supported types.
    renewable_carriers:
        PyPSA ``carrier`` values considered renewable.  Generators whose
        carrier is in this set are classified as :data:`RENEWABLE`; all
        other generators are :data:`THERMAL`.
    start_datetime:
        Reference datetime for converting timeseries timestamps to
        simulation seconds.  Defaults to the earliest timestamp in
        *timeseries* (or ``datetime.now`` if empty).
    """

    def __init__(
        self,
        net,
        *,
        timeseries: dict[ComponentRef | tuple, pd.Series] | None = None,
        relevant_types: list[str] | None = None,
        renewable_carriers: Iterable[str] | None = None,
        start_datetime: datetime | None = None,
    ) -> None:
        self._net = net
        self._timeseries: dict[ComponentRef, pd.Series] = {
            (k if isinstance(k, ComponentRef) else ComponentRef(*k)): v
            for k, v in (timeseries or {}).items()
        }
        self._relevant_types: list[str] = relevant_types or [
            THERMAL,
            RENEWABLE,
            LOAD,
            STORAGE,
        ]
        self._renewable_carriers: frozenset[str] = frozenset(
            renewable_carriers
            if renewable_carriers is not None
            else _DEFAULT_RENEWABLE_CARRIERS
        )

        if start_datetime is not None:
            self._start_dt: datetime = start_datetime
        elif self._timeseries:
            self._start_dt = self._earliest_timestamp()
        else:
            self._start_dt = datetime.now(UTC).replace(tzinfo=None)

        # aid -> {name: callable}
        self._observers: dict[str, dict[str, Callable[[], Any]]] = {}
        self._actions: dict[str, dict[str, Callable]] = {}
        self._ref_to_aid: dict[ComponentRef, str] = {}
        self._ref_to_agent: dict[ComponentRef, Any] = {}

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_network(
        cls,
        net,
        *,
        auto_timeseries: bool = True,
        timeseries: dict | None = None,
        relevant_types: list[str] | None = None,
        renewable_carriers: Iterable[str] | None = None,
        start_datetime: datetime | None = None,
    ) -> PyPSABehavior:
        """Build a behavior directly from a :class:`pypsa.Network`.

        When *auto_timeseries* is ``True`` (default) and no explicit
        ``timeseries`` mapping is given, timeseries are extracted from the
        network's ``generators_t`` / ``loads_t`` / ``storage_units_t``
        attributes via
        :func:`energy_scheduling_benchmark.networks.extract_timeseries`.

        Use this when you loaded a network with
        :func:`~energy_scheduling_benchmark.networks.load_scenario` or
        :func:`~energy_scheduling_benchmark.networks.load_example` —
        the network already carries its own snapshots and time-dependent
        data.
        """
        if timeseries is None and auto_timeseries:
            from energy_scheduling_benchmark.networks import extract_timeseries

            timeseries = extract_timeseries(net, renewable_carriers=renewable_carriers)

        return cls(
            net=net,
            timeseries=timeseries,
            relevant_types=relevant_types,
            renewable_carriers=renewable_carriers,
            start_datetime=start_datetime,
        )

    @classmethod
    def from_scenario(cls, scenario, **kwargs) -> PyPSABehavior:
        """Build a behavior from a :class:`~energy_scheduling_benchmark.networks.ScenarioData` bundle."""
        return cls(
            net=scenario.net,
            timeseries=scenario.timeseries,
            start_datetime=kwargs.pop("start_datetime", scenario.start),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def net(self):
        return self._net

    @property
    def start_datetime(self) -> datetime:
        return self._start_dt

    # ------------------------------------------------------------------
    # Behavior lifecycle
    # ------------------------------------------------------------------

    def initialize(self, environment: Environment, clock: Clock) -> None:
        """Schedule all timeseries values on their owning agents."""
        count = 0
        for ref, series in self._timeseries.items():
            if not self._type_is_relevant(ref.element_type):
                continue
            agent = self._ref_to_agent.get(ref)
            if agent is None:
                logger.debug("No agent installed for %s; skipping timeseries", ref)
                continue
            for ts, value in series.items():
                t_s = self._ts_to_seconds(ts)
                agent.schedule_timestamp_task(
                    self._update_coro(ref, float(value), environment),
                    timestamp=t_s,
                )
                count += 1
        logger.debug("PyPSABehavior: %d timeseries tasks scheduled", count)

    def install(self, agent, **kwargs) -> None:
        """Register observers/actions for *agent* bound to a :class:`ComponentRef`.

        Expected kwargs
        ---------------
        id:
            A :class:`ComponentRef` or ``(element_type, component_id)`` tuple.
        """
        raw = kwargs["id"]
        ref = raw if isinstance(raw, ComponentRef) else ComponentRef(*raw)

        self._ref_to_aid[ref] = agent.aid
        self._ref_to_agent[ref] = agent
        self._observers[agent.aid] = self._build_observers(ref)
        self._actions[agent.aid] = self._build_actions(ref)

    # ------------------------------------------------------------------
    # Observer / action interface
    # ------------------------------------------------------------------

    def observe(self, agent_id: str, name: str = "active_power") -> Any:
        """Return the named observation for *agent_id*.

        Built-in observer names:

        - ``"statics"``            – full row dict of the component DataFrame.
        - ``"max_active_power"``   – current maximum active power (MW).
        - ``"active_power"``       – current active power set-point (MW).
        - ``"cost"``               – marginal cost (for generators only).
        """
        fn = self._observers.get(agent_id, {}).get(name)
        if fn is None:
            logger.warning("No observer %r for agent %r", name, agent_id)
            return None
        return fn()

    def act(self, agent_id: str, action: str, *args: Any, **kwargs: Any) -> None:
        fn = self._actions.get(agent_id, {}).get(action)
        if fn is None:
            logger.warning("No action %r for agent %r", action, agent_id)
            return
        fn(*args, **kwargs)

    def has_action(self, agent_id: str, action: str) -> bool:
        return action in self._actions.get(agent_id, {})

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def get_components_by_type(self, types: list[str]) -> list[ComponentRef]:
        refs: list[ComponentRef] = []
        for et in types:
            refs.extend(self._refs_for_type(et))
        return refs

    def get_possible_components(self) -> list[ComponentRef]:
        return self.get_components_by_type(self._relevant_types)

    def calculate_initial_time(self) -> datetime:
        return self._earliest_timestamp()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _type_is_relevant(self, element_type: str) -> bool:
        return element_type in self._relevant_types

    def _refs_for_type(self, element_type: str) -> list[ComponentRef]:
        df, predicate = self._dataframe_and_predicate(element_type)
        if df is None or df.empty:
            return []
        mask = (
            df.index.to_series().apply(predicate)
            if predicate
            else pd.Series(True, index=df.index)
        )
        return [ComponentRef(element_type, str(idx)) for idx in df.index[mask]]

    def _dataframe_and_predicate(self, element_type: str):
        """Return ``(df, predicate)`` for a given element type.

        The predicate, if not ``None``, filters rows of the underlying PyPSA
        DataFrame (used for THERMAL vs RENEWABLE distinction on ``n.generators``).
        """
        if element_type == THERMAL:
            df = getattr(self._net, "generators", None)
            if df is None:
                return None, None
            renewables = self._renewable_carriers

            def predicate(idx):
                carrier = (
                    str(df.at[idx, _COL_CARRIER]) if _COL_CARRIER in df.columns else ""
                )
                return carrier.lower() not in renewables

            return df, predicate

        if element_type == RENEWABLE:
            df = getattr(self._net, "generators", None)
            if df is None:
                return None, None
            renewables = self._renewable_carriers

            def predicate(idx):
                carrier = (
                    str(df.at[idx, _COL_CARRIER]) if _COL_CARRIER in df.columns else ""
                )
                return carrier.lower() in renewables

            return df, predicate

        if element_type == LOAD:
            return getattr(self._net, "loads", None), None

        if element_type == STORAGE:
            return getattr(self._net, "storage_units", None), None

        return None, None

    def _dataframe_for(self, element_type: str):
        df, _ = self._dataframe_and_predicate(element_type)
        return df

    def _build_observers(self, ref: ComponentRef) -> dict[str, Callable[[], Any]]:
        et, cid = ref
        df = self._dataframe_for(et)
        if df is None:
            return {}

        def statics() -> dict:
            return df.loc[cid].to_dict()

        def max_active_power() -> float:
            row = df.loc[cid]
            if et == LOAD:
                return float(row.get(_COL_P_SET, 0.0))
            return float(row.get(_COL_P_NOM, row.get(_COL_P_SET, float("nan"))))

        def active_power() -> float:
            return float(df.at[cid, _COL_P_SET]) if _COL_P_SET in df.columns else 0.0

        def cost() -> float:
            return (
                float(df.at[cid, _COL_MARGINAL_COST])
                if _COL_MARGINAL_COST in df.columns
                else 0.0
            )

        return {
            "statics": statics,
            "max_active_power": max_active_power,
            "active_power": active_power,
            "cost": cost,
        }

    def _build_actions(self, ref: ComponentRef) -> dict[str, Callable]:
        et, cid = ref
        if et == LOAD:
            return {}

        df = self._dataframe_for(et)
        if df is None:
            return {}

        def regulate(active_power_mw: float) -> None:
            df.at[cid, _COL_P_SET] = float(active_power_mw)

        return {"regulate": regulate}

    def _apply_timeseries_update(
        self, ref: ComponentRef, value: float, environment: Environment
    ) -> None:
        et, cid = ref
        df = self._dataframe_for(et)
        if df is None:
            return

        if et == RENEWABLE:
            # Treat value as per-unit availability of installed capacity
            nominal = float(df.at[cid, _COL_P_NOM]) if _COL_P_NOM in df.columns else 1.0
            df.at[cid, _COL_P_NOM] = value * nominal
        elif et == LOAD:
            df.at[cid, _COL_P_SET] = float(value)
        else:  # THERMAL or STORAGE
            df.at[cid, _COL_P_NOM] = float(value)

        aid = self._ref_to_aid.get(ref)
        if aid is not None:
            environment.emit_agent_event(PowerUpdateInfo(), aid)

    async def _update_coro(
        self,
        ref: ComponentRef,
        value: float,
        environment: Environment,
    ) -> None:
        self._apply_timeseries_update(ref, value, environment)

    def _ts_to_seconds(self, ts) -> float:
        if isinstance(ts, datetime):
            return (ts - self._start_dt).total_seconds()
        if hasattr(ts, "to_pydatetime"):
            return (ts.to_pydatetime() - self._start_dt).total_seconds()
        return float(ts)

    def _earliest_timestamp(self) -> datetime:
        earliest: datetime | None = None
        for series in self._timeseries.values():
            if series.empty:
                continue
            first = series.index[0]
            dt = first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
            if earliest is None or dt < earliest:
                earliest = dt
        return earliest or datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Functional wrappers (mirror the Julia API)
# ---------------------------------------------------------------------------


def calculate_initial_time(behavior: PyPSABehavior) -> datetime:
    return behavior.calculate_initial_time()


def get_possible_components(behavior: PyPSABehavior) -> list[ComponentRef]:
    return behavior.get_possible_components()


def get_components_by_type(
    behavior: PyPSABehavior, types: list[str]
) -> list[ComponentRef]:
    return behavior.get_components_by_type(types)
