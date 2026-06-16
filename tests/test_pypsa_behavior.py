"""Unit tests for :class:`PyPSABehavior`."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from mango import RoleAgent
from mango.simulation.environment import DefaultEnvironment
from mango.simulation.world import create_world, discrete_step_until

from energy_scheduling_benchmark.environment import (
    LOAD,
    RENEWABLE,
    STORAGE,
    THERMAL,
    ComponentRef,
    PowerUpdateInfo,
    PyPSABehavior,
    calculate_initial_time,
    get_components_by_type,
    get_possible_components,
)


class TestComponentRef:
    def test_equality(self):
        assert ComponentRef(THERMAL, "g0") == ComponentRef(THERMAL, "g0")
        assert ComponentRef(THERMAL, "g0") != ComponentRef(LOAD, "g0")

    def test_unpack(self):
        et, cid = ComponentRef(THERMAL, "g0")
        assert et == THERMAL
        assert cid == "g0"

    def test_hashable(self):
        d = {ComponentRef(THERMAL, "g0"): 1}
        assert d[ComponentRef(THERMAL, "g0")] == 1


class TestPyPSABehaviorUnit:
    def test_get_components_by_type_splits_thermal_and_renewable(
        self, five_bus_pypsa_net
    ):
        net, ts, start = five_bus_pypsa_net
        behavior = PyPSABehavior(net=net, timeseries=ts, start_datetime=start)

        thermal = behavior.get_components_by_type([THERMAL])
        renewable = behavior.get_components_by_type([RENEWABLE])

        assert all(c.element_type == THERMAL for c in thermal)
        assert all(c.element_type == RENEWABLE for c in renewable)
        assert {c.component_id for c in thermal} == {"thermal0", "thermal1"}
        assert {c.component_id for c in renewable} == {"wind0"}

    def test_get_possible_components_returns_all_relevant(self, five_bus_pypsa_net):
        net, ts, start = five_bus_pypsa_net
        behavior = PyPSABehavior(
            net=net,
            timeseries=ts,
            start_datetime=start,
            relevant_types=[THERMAL, LOAD],
        )
        comps = behavior.get_possible_components()
        assert {c.element_type for c in comps} == {THERMAL, LOAD}

    def test_start_datetime_inferred_from_timeseries(self, pypsa_net_with_timeseries):
        net, g0, l0, ts, start = pypsa_net_with_timeseries
        behavior = PyPSABehavior(net=net, timeseries=ts)
        assert behavior.start_datetime == start

    def test_calculate_initial_time_wrapper(self, pypsa_net_with_timeseries):
        net, g0, l0, ts, start = pypsa_net_with_timeseries
        behavior = PyPSABehavior(net=net, timeseries=ts, start_datetime=start)
        assert calculate_initial_time(behavior) == start

    def test_get_components_by_type_wrapper(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        comps = get_components_by_type(behavior, [LOAD])
        assert ComponentRef(LOAD, l0) in comps

    def test_get_possible_components_wrapper(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net, relevant_types=[THERMAL])
        comps = get_possible_components(behavior)
        assert all(c.element_type == THERMAL for c in comps)

    def test_timeseries_key_normalisation(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        index = pd.date_range("2024-01-01", periods=1, freq="h")
        ts_ref = {ComponentRef(THERMAL, g0): pd.Series([1.0], index=index)}
        ts_tup = {(THERMAL, g0): pd.Series([1.0], index=index)}
        b_ref = PyPSABehavior(
            net=net, timeseries=ts_ref, start_datetime=datetime(2024, 1, 1)
        )
        b_tup = PyPSABehavior(
            net=net, timeseries=ts_tup, start_datetime=datetime(2024, 1, 1)
        )
        assert b_ref._timeseries.keys() == b_tup._timeseries.keys()


class TestObserversAndActions:
    async def test_statics_observer_returns_dict(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        result = behavior.observe(agent.aid, "statics")
        assert isinstance(result, dict)
        assert "p_nom" in result

    async def test_active_power_observer(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.observe(agent.aid, "active_power") == pytest.approx(5.0)

    async def test_max_active_power_observer(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.observe(agent.aid, "max_active_power") == pytest.approx(10.0)

    async def test_regulate_action_for_thermal(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="gen-agent")
        world.environment.install(agent, id=ComponentRef(THERMAL, g0))

        assert behavior.has_action(agent.aid, "regulate")
        behavior.act(agent.aid, "regulate", 3.0)
        assert net.generators.at[g0, "p_set"] == pytest.approx(3.0)

    async def test_no_regulate_action_for_load(self, simple_pypsa_net):
        net, g0, l0 = simple_pypsa_net
        behavior = PyPSABehavior(net=net)
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)
        agent = RoleAgent()
        world.register(agent, suggested_aid="load-agent")
        world.environment.install(agent, id=ComponentRef(LOAD, l0))

        assert not behavior.has_action(agent.aid, "regulate")


class TestTimeseriesScheduling:
    async def test_renewable_timeseries_fires_events(self, five_bus_pypsa_net):
        net, timeseries, start = five_bus_pypsa_net
        behavior = PyPSABehavior(
            net=net,
            timeseries=timeseries,
            start_datetime=start,
            relevant_types=[THERMAL, RENEWABLE, LOAD, STORAGE],
        )
        environment = DefaultEnvironment(behavior=behavior)
        world = create_world(start_time=0.0, environment=environment)

        events: list[PowerUpdateInfo] = []

        class Capture(RoleAgent):
            def on_agent_event(self, event):
                super().on_agent_event(event)
                if isinstance(event, PowerUpdateInfo):
                    events.append(event)

        agent = Capture()
        world.register(agent, suggested_aid="wind0")
        world.environment.install(agent, id=ComponentRef(RENEWABLE, "wind0"))

        async with world:
            await discrete_step_until(world, 3 * 24 * 3600.0)

        # 72 hourly points → 72 events.
        assert len(events) == 72
