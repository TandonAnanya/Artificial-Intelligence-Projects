from wumpus_kb import *

axiom_tests = [
    {
        'name': 'axiom_generator_percept_sentence',
        'callable': axiom_generator_percept_sentence,
        'tests': [
            { 
                'input': [0, (True, True, False, True, False)], 
                'inferTrue': [
                    'Stench0', 'Breeze0', '~Glitter0', 'Bump0', '~Scream0',
                ],
                'inferFalse': ['Stench1', '~Stench1'],
            }
        ],
        'score': 1,
    },
    {
        'name': 'axiom_generator_initial_location_assertions',
        'callable': axiom_generator_initial_location_assertions,
        'tests': [
            { 
                'input': [3, 2], 
                'inferTrue': ['~P3_2', '~W3_2', '~P3_2 | ~W3_2', '~P3_2 & ~W3_2'],
                'inferFalse': ['~P2_3', '~W2_3', 'P2_3', 'W2_3'],
            },
            {
                'input': [1, 1],
                'inferTrue': ['~P1_1', '~W1_1'],
            }
        ],
        'score': 0.5,
    },
    {
        'name': 'axiom_generator_pits_and_breezes',
        'callable': axiom_generator_pits_and_breezes,
        'tests': [
            {
                'input': [2, 2, 1, 4, 1, 4], # Breeze somewhere in the middle
                'inferTrue': [
                    'B2_2 >> (P2_1 | P1_2 | P3_2 | P2_3 | P2_2)',
                    '(P2_1 | P1_2 | P3_2 | P2_3 | P2_2) >> B2_2',
                ]
            },
            {
                'input': [1, 1, 1, 4, 1, 4], # Breeze at south-west corner
                'inferTrue': [
                    'B1_1 >> (P2_1 | P1_2 | P1_1)',
                    '(P2_1 | P1_2 | P1_1) >> B1_1',
                ]
            },
            {
                'input': [4, 1, 1, 4, 1, 4], # Breeze at south-east corner
                'inferTrue': [
                    'B4_1 >> (P3_1 | P4_2 | P4_1)',
                    '(P3_1 | P4_2 | P4_1) >> B4_1',
                ]
            },
            {
                'input': [1, 4, 1, 4, 1, 4], # Breeze at north-west corner
                'inferTrue': [
                    'B1_4 >> (P1_3 | P2_4 | P1_4)',
                    '(P1_3 | P2_4 | P1_4) >> B1_4',
                ]
            },
            {
                'input': [4, 4, 1, 4, 1, 4], # Breeze at north-east corner
                'inferTrue': [
                    'B4_4 >> (P3_4 | P4_3 | P4_4)',
                    '(P3_4 | P4_3 | P4_4) >> B4_4',
                ]
            },
            {
                'input': [1, 3, 1, 4, 1, 4], # Breeze at west edge
                'inferTrue': [
                    'B1_3 >> (P1_2 | P1_4 | P1_3 | P2_3)',
                    '(P1_2 | P1_4 | P1_3 | P2_3) >> B1_3',
                ]
            },
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_wumpus_and_stench',
        'callable': axiom_generator_wumpus_and_stench,
        'tests': [
            {
                'input': [2, 2, 1, 4, 1, 4], # Stench somewhere in the middle
                'inferTrue': [
                    'S2_2 >> (W2_1 | W1_2 | W3_2 | W2_3 | W2_2)',
                    '(W2_1 | W1_2 | W3_2 | W2_3 | W2_2) >> S2_2',
                ],
            },
            {
                'input': [1, 1, 1, 4, 1, 4], # Stench at south-west corner
                'inferTrue': [
                    'S1_1 >> (W2_1 | W1_2 | W1_1)',
                    '(W2_1 | W1_2 | W1_1) >> S1_1',
                ]
            },
            {
                'input': [4, 1, 1, 4, 1, 4], # Stench at south-east corner
                'inferTrue': [
                    'S4_1 >> (W3_1 | W4_2 | W4_1)',
                    '(W3_1 | W4_2 | W4_1) >> S4_1',
                ]
            },
            {
                'input': [1, 4, 1, 4, 1, 4], # Stench at north-west corner
                'inferTrue': [
                    'S1_4 >> (W1_3 | W2_4 | W1_4)',
                    '(W1_3 | W2_4 | W1_4) >> S1_4',
                ]
            },
            {
                'input': [4, 4, 1, 4, 1, 4], # Stench at north-east corner
                'inferTrue': [
                    'S4_4 >> (W3_4 | W4_3 | W4_4)',
                    '(W3_4 | W4_3 | W4_4) >> S4_4',
                ]
            },
            {
                'input': [1, 3, 1, 4, 1, 4], # Stench at west edge
                'inferTrue': [
                    'S1_3 >> (W1_2 | W1_4 | W1_3 | W2_3)',
                    '(W1_2 | W1_4 | W1_3 | W2_3) >> S1_3',
                ]
            },
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_at_least_one_wumpus',
        'callable': axiom_generator_at_least_one_wumpus,
        'tests': [
            {
                'input': [1, 2, 1, 2],
                'inferTrue': ['W1_1 | W1_2 | W2_1 | W2_2'],
                'inferFalse': ['W1_3', 'W1_4', 'W1_1 & W1_2 & W2_1 & W2_2']
            },
            {
                'input': [1, 2, 1, 3],
                'inferTrue': ['W1_1 | W1_2 | W1_3 | W2_1 | W2_2 | W2_3'],
                'inferFalse': ['W3_2', 'W3_1', 'W1_1 & W1_2 & W1_3 & W2_1 & W2_2 & W2_3']
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_at_most_one_wumpus',
        'callable': axiom_generator_at_most_one_wumpus,
        'tests': [
            {
                'input': [1, 2, 1, 2],
                'inferTrue': [
                    '~W1_1 | ~W1_2', '~(W1_1 & W1_2)', 
                    '(~W1_1 | ~W2_1)', '(~W1_1 | ~W2_2)', '(~W1_2 | ~W2_1)', '(~W1_2 | ~W2_2)', 
                    '(~W2_1 | ~W2_2)',
                ],
                'inferFalse': [
                    '~W1_3 | ~W1_2', '~W1_4 | ~W1_3'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_only_in_one_location',
        'callable': axiom_generator_only_in_one_location,
        'tests': [
            {
                'input': [2, 2, 1, 2, 1, 3, 6],
                'inferTrue': [
                    'L2_2_6 >> L2_2_6', 'L2_2_6 >> ~L1_1_6', 'L2_2_6 >> ~L1_2_6', 'L2_2_6 >> ~L1_3_6', 'L2_2_6 >> ~L2_1_6',
                    'L2_2_6 >> ~L2_3_6', 'L2_2_6 | L1_1_6 | L1_2_6 | L1_3_6 | L2_1_6 | L2_3_6'
                ],
                'inferFalse': [
                    'L2_2_6 >> L4_3_6', 'L2_2_6 >> L2_2_5', 'L2_2_6 >> L2_2_0',
                    'L4_3_6', 'L2_2_5', 'L2_2_0'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_only_one_heading',
        'callable': axiom_generator_only_one_heading,
        'tests': [
            {
                'input': ['west', 2],
                'inferTrue': [
                    'HeadingWest2 >> ~HeadingNorth2', 'HeadingWest2 >> ~HeadingSouth2', 'HeadingWest2 >> HeadingWest2', 'HeadingWest2 >> ~HeadingEast2', 'HeadingNorth2 | HeadingWest2 | HeadingEast2 | HeadingSouth2'
                ],
                'inferFalse': [
                    'HeadingWest2 >> HeadingNorth0', '~HeadingNorth0', '~HeadingWest0', 'HeadingWest0',
                    'HeadingWest1', '~HeadingEast1', 'HeadingWest2 >> HeadingWest3', '~HeadingWest3'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_have_arrow_and_wumpus_alive',
        'callable': axiom_generator_have_arrow_and_wumpus_alive,
        'tests': [
            {
                'input': [5],
                'inferTrue': ['HaveArrow5', 'WumpusAlive5'],
            }
        ],
        'score': 0.5
    },
    {
        'name': 'axiom_generator_location_OK',
        'callable': axiom_generator_location_OK,
        'tests': [
            {
                'input': [2, 3, 5],
                'inferTrue': [
                    'OK2_3_5 >> ~P2_3',
                    'OK2_3_5 >> ~W2_3 | ~WumpusAlive5',
                    '(~P2_3 & ~(W2_3 & WumpusAlive5)) >> OK2_3_5'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_breeze_percept_and_location_property',
        'callable': axiom_generator_breeze_percept_and_location_property,
        'tests': [
            {
                'input': [2, 3, 5],
                'inferTrue': [
                    'L2_3_5 >> (Breeze5 >> B2_3)',
                    'L2_3_5 >> (B2_3 >> Breeze5)',
                ],
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_stench_percept_and_location_property',
        'callable': axiom_generator_stench_percept_and_location_property,
        'tests': [
            {
                'input': [2, 3, 5],
                'inferTrue': [
                    'L2_3_5 >> (Stench5 >> S2_3)',
                    'L2_3_5 >> (S2_3 >> Stench5)',
                ],
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_at_location_ssa',
        'callable': axiom_generator_at_location_ssa,
        'tests': [
            {
                'input': [4, 1, 2, 1, 2, 1, 3],
                'inferTrue': [
                    '(L2_2_4 & HeadingWest4 & Forward4) >> L1_2_5',
                    '(L1_3_4 & HeadingSouth4 & Forward4) >> L1_2_5',
                    '(L1_1_4 & HeadingNorth4 & Forward4) >> L1_2_5',
                    '(L1_2_4 & Grab4 & ~Forward4) >> L1_2_5',
                    '(L1_2_4 & Shoot4 & ~Forward4) >> L1_2_5',
                    '(L1_2_4 & TurnLeft4 & ~Forward4) >> L1_2_5',
                    '(L1_2_4 & TurnRight4 & ~Forward4) >> L1_2_5',
                    '(L1_2_4 & Wait4 & ~Forward4) >> L1_2_5',
                    '(L1_2_4 & Bump5 & ~Forward4) >> L1_2_5',
                ],
                'inferFalse': [
                    '(L2_2_4 & HeadingEast4 & Forward4) >> L1_2_5',
                    '(L2_2_4 & Grab4) >> ~L2_2_5'
                ]
            },
            {
                'input': [4, 2, 2, 1, 4, 1, 4],
                'inferTrue': [
                    '(L3_2_4 & HeadingWest4 & Forward4) >> L2_2_5',
                    '(L2_3_4 & HeadingSouth4 & Forward4) >> L2_2_5',
                    '(L2_1_4 & HeadingNorth4 & Forward4) >> L2_2_5',
                    '(L1_2_4 & HeadingEast4 & Forward4) >> L2_2_5',
                ]
            }
        ],
        'score': 4
    },
    {
        'name': 'axiom_generator_have_arrow_ssa',
        'callable': axiom_generator_have_arrow_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HaveArrow6 >> (HaveArrow5 & ~Shoot5)',
                    '(HaveArrow5 & ~Shoot5) >> HaveArrow6'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_wumpus_alive_ssa',
        'callable': axiom_generator_wumpus_alive_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'WumpusAlive6 >> (WumpusAlive5 & ~Scream6)',
                    '(WumpusAlive5 & ~Scream6) >> WumpusAlive6'
                ]
            }
        ],
        'score': 1
    },
    {
        'name': 'axiom_generator_heading_north_ssa',
        'callable': axiom_generator_heading_north_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HeadingNorth6 >> ((HeadingNorth5 & ~TurnLeft5 & ~TurnRight5) | (HeadingEast5 & TurnLeft5) | (HeadingWest5 & TurnRight5))',
                    '(HeadingNorth5 & ~TurnLeft5 & ~TurnRight5) >> HeadingNorth6',
                    '(HeadingEast5 & TurnLeft5) >> HeadingNorth6',
                    '(HeadingWest5 & TurnRight5) >> HeadingNorth6',
                ]
            }
        ],
        'score': 0.75,
    },
    {
        'name': 'axiom_generator_heading_east_ssa',
        'callable': axiom_generator_heading_east_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HeadingEast6 >> ((HeadingEast5 & ~TurnLeft5 & ~TurnRight5) | (HeadingSouth5 & TurnLeft5) | (HeadingNorth5 & TurnRight5))',
                    '(HeadingEast5 & ~TurnLeft5 & ~TurnRight5) >> HeadingEast6',
                    '(HeadingSouth5 & TurnLeft5) >> HeadingEast6',
                    '(HeadingNorth5 & TurnRight5) >> HeadingEast6'
                ]
            }
        ],
        'score': 0.75
    },
    {
        'name': 'axiom_generator_heading_south_ssa',
        'callable': axiom_generator_heading_south_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HeadingSouth6 >> ((HeadingSouth5 & ~TurnLeft5 & ~TurnRight5) | (HeadingWest5 & TurnLeft5) | (HeadingEast5 & TurnRight5))',
                    '(HeadingSouth5 & ~TurnLeft5 & ~TurnRight5) >> HeadingSouth6',
                    '(HeadingWest5 & TurnLeft5) >> HeadingSouth6',
                    '(HeadingEast5 & TurnRight5) >> HeadingSouth6',
                ]
            }
        ],
        'score': 0.75
    },
    {
        'name': 'axiom_generator_heading_west_ssa',
        'callable': axiom_generator_heading_west_ssa,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HeadingWest6 >> ((HeadingWest5 & ~TurnLeft5 & ~TurnRight5) | (HeadingNorth5 & TurnLeft5) | (HeadingSouth5 & TurnRight5))',
                    '(HeadingWest5 & ~TurnLeft5 & ~TurnRight5) >> HeadingWest6',
                    '(HeadingNorth5 & TurnLeft5) >> HeadingWest6',
                    '(HeadingSouth5 & TurnRight5) >> HeadingWest6',
                ]
            }
        ],
        'score': 0.75
    },
    {
        'name': 'axiom_generator_heading_only_north',
        'callable': axiom_generator_heading_only_north,
        'tests': [
            {
                'input': [4],
                'inferTrue': [
                    'HeadingNorth4 >> (~HeadingSouth4 & ~HeadingEast4 & ~HeadingWest4)',
                    '(~HeadingSouth4 & ~HeadingEast4 & ~HeadingWest4) >> HeadingNorth4'
                ]
            }
        ],
        'score': 0.5
    },
    {
        'name': 'axiom_generator_heading_only_east',
        'callable': axiom_generator_heading_only_east,
        'tests': [
            {
                'input': [4],
                'inferTrue': [
                    'HeadingEast4 >> (~HeadingSouth4 & ~HeadingNorth4 & ~HeadingWest4)',
                    '(~HeadingSouth4 & ~HeadingNorth4 & ~HeadingWest4) >> HeadingEast4'
                ]
            }
        ],
        'score': 0.5
    },
    {
        'name': 'axiom_generator_heading_only_south',
        'callable': axiom_generator_heading_only_south,
        'tests': [
            {
                'input': [5],
                'inferTrue': [
                    'HeadingSouth5 >> (~HeadingEast5 & ~HeadingNorth5 & ~HeadingWest5)',
                    '(~HeadingEast5 & ~HeadingNorth5 & ~HeadingWest5) >> HeadingSouth5'
                ]
            }
        ],
        'score': 0.5
    },
    {
        'name': 'axiom_generator_heading_only_west',
        'callable': axiom_generator_heading_only_west,
        'tests': [
            {
                'input': [10],
                'inferTrue': [
                    'HeadingWest10 >> (~HeadingEast10 & ~HeadingNorth10 & ~HeadingSouth10)',
                    '(~HeadingEast10 & ~HeadingNorth10 & ~HeadingSouth10) >> HeadingWest10'
                ]
            }
        ],
        'score': 0.5
    },
    {
        'name': 'axiom_generator_only_one_action_axioms',
        'callable': axiom_generator_only_one_action_axioms,
        'tests': [
            {
                'input': [4],
                'inferTrue': [
                    '(Grab4 | Shoot4 | Climb4 | TurnLeft4 | TurnRight4 | Forward4 | Wait4)',
                    '~Grab4 | ~Shoot4', '~Grab4 | ~Climb4', '~Grab4 | ~TurnLeft4', '~Grab4 | ~TurnRight4',
                    '~Grab4 | ~Forward4', '~Grab4 | ~Wait4', '~Shoot4 | ~Climb4', '~Shoot4 | ~TurnLeft4',
                    '~Shoot4 | ~TurnRight4', '~Shoot4 | ~Forward4', '~Shoot4 | ~Wait4', '~Climb4 | ~TurnLeft4',
                    '~Climb4 | ~TurnRight4', '~Climb4 | ~Forward4', '~Climb4 | ~Wait4', '~TurnLeft4 | ~TurnRight4',
                    '~TurnLeft4 | ~Forward4', '~TurnLeft4 | ~Wait4', '~TurnRight4 | ~Forward4', 
                    '~TurnRight4 | ~Wait4', '~Forward4 | ~Wait4'
                ]
            }
        ],
        'score': 2
    }
]
