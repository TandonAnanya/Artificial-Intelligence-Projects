import time

def test_hybrid_wumpus():
    """
    Tests Wumpus world with a second layout
    """
    from wumpus import wscenario_4x4_HybridWumpusAgent, world_scenario_hybrid_wumpus_agent_from_layout

    s = world_scenario_hybrid_wumpus_agent_from_layout('wumpus_4x4_2')
    s.run()
    final_score = s.agent.performance_measure
    time_step = s.env.time_step
    assert final_score == 960
    assert time_step == 30

def main():
    test_hybrid_wumpus()


if __name__ == "__main__":
    main()
