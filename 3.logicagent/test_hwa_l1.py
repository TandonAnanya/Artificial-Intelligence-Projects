import time

def test_hybrid_wumpus():
    """
    Tests Wumpus world with the book layouts
    """
    from wumpus import wscenario_4x4_HybridWumpusAgent, world_scenario_hybrid_wumpus_agent_from_layout
    s = wscenario_4x4_HybridWumpusAgent()
    s.run()
    final_score = s.agent.performance_measure
    time_step = s.env.time_step
    assert final_score == 983
    assert time_step == 17

def main():
    test_hybrid_wumpus()


if __name__ == "__main__":
    main()
