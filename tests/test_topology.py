"""--topology: graph construction, routing and graph-dependent parameters."""

from __future__ import annotations

import networkx as nx
import pytest

from energy_scheduling_benchmark import LOAD, THERMAL, ComponentRef
from energy_scheduling_benchmark.networks import build_toy_network
from energy_scheduling_benchmark.scenarios._common import (
    build_world,
    install_topology,
    participant_graph,
    register_component_agent,
    run_scenario_main,
)
from energy_scheduling_benchmark.scenarios.exact_diffusion import perron_scales


def _forwarded(argv):
    seen = {}

    async def fake_execute_test_case(**kwargs):
        seen.update(kwargs)

    run_scenario_main(
        fake_execute_test_case, doc=None, default_name_base="x", argv=argv
    )
    return seen


class TestTopologyFlag:
    def test_default_is_complete(self):
        assert _forwarded([])["topology"] == "complete"

    def test_forwarded(self):
        assert _forwarded(["--topology", "ring"])["topology"] == "ring"

    @pytest.mark.parametrize("kind", ["grid", "star"])
    def test_unknown_rejected(self, kind):
        with pytest.raises(SystemExit):
            _forwarded(["--topology", kind])


def _thermal_refs(scenario):
    return [
        ComponentRef(THERMAL, str(g))
        for g in scenario.net.generators.index
        if "hydro" not in str(g)
    ]


class TestParticipantGraph:
    def test_complete_has_diameter_one(self):
        scenario = build_toy_network(periods=24)
        refs = _thermal_refs(scenario)
        pg = participant_graph(scenario, "complete", refs)
        assert pg.diameter == 1 and pg.max_degree == len(refs) - 1

    def test_ring_diameter(self):
        scenario = build_toy_network(periods=24)
        refs = _thermal_refs(scenario)
        pg = participant_graph(scenario, "ring", refs)
        assert pg.diameter == len(refs) // 2
        assert set(pg.degrees.values()) == {2}


class TestPerronScales:
    def test_one_for_symmetric_rules_and_regular_graphs(self):
        assert set(perron_scales(nx.path_graph(4), "mean_metropolis").values()) == {1.0}
        assert set(perron_scales(nx.cycle_graph(5), "averaging").values()) == {1.0}

    def test_irregular_graph_with_averaging(self):
        p = perron_scales(nx.path_graph(4), "averaging")
        assert sum(p.values()) == pytest.approx(4.0)
        # Averaging weights on the path: nodes with more neighbours count more.
        assert p[1] > p[0]


class TestInstallTopology:
    async def test_ring_routes_over_hops(self):
        scenario = build_toy_network(periods=24)
        world, _ = build_world(scenario, delay_s=0.1)
        from mango import RoleAgent

        refs = _thermal_refs(scenario)
        agents = []
        for ref in refs:
            agent = RoleAgent()
            register_component_agent(world, agent, ref)
            agents.append(agent)
        load = ComponentRef(LOAD, str(scenario.net.loads.index[0]))
        load_agent = RoleAgent()
        register_component_agent(world, load_agent, load)
        install_topology(
            world, participant_graph(scenario, "ring", refs, attached=[load]), agents
        )

        sim = world.communication_sim
        a0, a1 = agents[0].aid, agents[1].aid
        assert sim.hops(a0, a0) == 1  # unknown/self: one hop
        hops = [sim.hops(a.aid, load_agent.aid) for a in agents]
        assert min(hops) == 1 and max(hops) > 1
        assert sim.hops(a0, a1) >= 1

    async def test_complete_keeps_one_hop(self):
        scenario = build_toy_network(periods=24)
        world, _ = build_world(scenario, delay_s=0.1)
        from mango import RoleAgent

        agents = []
        for ref in _thermal_refs(scenario):
            agent = RoleAgent()
            register_component_agent(world, agent, ref)
            agents.append(agent)
        install_topology(
            world,
            participant_graph(scenario, "complete", _thermal_refs(scenario)),
            agents,
        )
        sim = world.communication_sim
        assert all(sim.hops(agents[0].aid, a.aid) == 1 for a in agents[1:])


async def test_consensus_runs_on_ring(tmp_path):
    from energy_scheduling_benchmark.scenarios.consensus import execute_test_case

    await execute_test_case(
        scenario=build_toy_network(periods=24),
        name_base=str(tmp_path / "c"),
        simulate_days=1,
        topology="ring",
        strict=True,
    )


class TestHopsAndDeadlines:
    def test_hops_from_leader(self):
        scenario = build_toy_network(periods=24)
        refs = _thermal_refs(scenario)
        load = ComponentRef(LOAD, str(scenario.net.loads.index[0]))
        ring = participant_graph(scenario, "ring", refs, attached=[load])
        assert ring.hops_from(load) > 1
        assert (
            participant_graph(scenario, "complete", refs, attached=[load]).hops_from(
                load
            )
            == 1
        )

    def test_for_hops_scales_default_deadlines(self):
        from energy_scheduling_benchmark.scenarios._common import loss_tolerance

        lt = loss_tolerance(0.02, loss_rate=0.05)
        scaled = lt.for_hops(3)
        assert scaled.load_timeout == pytest.approx(3 * lt.load_timeout)
        assert scaled.request_timeout == pytest.approx(3 * lt.request_timeout)
        assert scaled.round_timeout == lt.round_timeout  # peer rounds: neighbours
        assert lt.for_hops(3, coordinator=True).round_timeout == pytest.approx(
            3 * lt.round_timeout
        )
        assert lt.for_hops(1) is lt

    def test_for_hops_keeps_explicit_timeouts(self):
        from energy_scheduling_benchmark.scenarios._common import loss_tolerance

        lt = loss_tolerance(0.02, loss_rate=0.05, round_timeout=1.0, load_timeout=2.0)
        scaled = lt.for_hops(4, coordinator=True)
        assert scaled.round_timeout == 1.0 and scaled.load_timeout == 2.0


async def test_admm_on_lossy_ring_converges(tmp_path, monkeypatch):
    """Review fix: on a ring the coordinator's answers take hops x delay; with
    hop-scaled deadlines ADMM no longer drops every far participant and stops
    after --max-missing-rounds."""
    import energy_scheduling_benchmark.scenarios.admm as admm_mod

    coordinators = []
    real = admm_mod.create_consensus_target_reach_admm_coordinator

    def spy(**kwargs):
        coordinators.append(real(**kwargs))
        return coordinators[-1]

    monkeypatch.setattr(admm_mod, "create_consensus_target_reach_admm_coordinator", spy)
    await admm_mod.execute_test_case(
        scenario=build_toy_network(periods=24),
        name_base=str(tmp_path / "admm"),
        simulate_days=1,
        topology="ring",
        loss_rate=0.01,
        seed=1,
    )
    assert coordinators[0].converged
