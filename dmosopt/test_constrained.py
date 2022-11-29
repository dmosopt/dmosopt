import numpy as np
from constrained_sampling import ParamSpacePoints

if __name__ == "__main__":
    Space = {
        "gc": [0.01, 50],
        "soma_gnabar": [0.1, 50],
        "soma_gl": [0.001, 0.6],
        "dend_gcabar": [0.01, 10],
        "dend_gkcbar": [0.1, 10],
        "dend_gkahpbar": [0.001, 0.6],
        "dend_gl": [0.001, 0.6],
    }

    SpaceCons = {
        "soma_gkdrbar": {
            "abs": [0.0, 60.0],
            "lb": [("gc", "+ 5")],  # ('soma_gl', '* 2')],
            "ub": [("gc", "+ 10")],
            "method": ("uniform", None, None),
        },
        "soma_gkahpbar": {
            "abs": [0.001, 0.6],
            "method": ("normal", 0, 200),
        },
    }

    parents_params = np.array(
        [
            "dend_gcabar",
            "dend_gkahpbar",
            "dend_gkcbar",
            "dend_gl",
            "gc",
            "soma_gkahpbar",
            "soma_gkdrbar",
            "soma_gl",
            "soma_gnabar",
        ]
    )

    parents_val = np.array(
        [
            [
                2.50750000e00,
                9.08500000e-02,
                7.52500000e00,
                2.10650000e-01,
                1.75065000e01,
                2.89968936e-01,
                2.61441123e01,
                4.50250000e-01,
                7.58500000e00,
            ],
            [
                8.50150000e00,
                2.70550000e-01,
                8.51500000e00,
                9.08500000e-02,
                1.25075000e01,
                2.89968936e-01,
                2.03855014e01,
                5.70050000e-01,
                3.25350000e01,
            ],
            [
                9.50050000e00,
                1.50750000e-01,
                3.56500000e00,
                4.50250000e-01,
                4.25015000e01,
                2.89968936e-01,
                4.81822941e01,
                2.10650000e-01,
                1.25750000e01,
            ],
            [
                6.50350000e00,
                2.10650000e-01,
                9.50500000e00,
                5.70050000e-01,
                2.50950000e00,
                2.89968936e-01,
                1.06595366e01,
                9.08500000e-02,
                2.25550000e01,
            ],
            [
                5.50450000e00,
                5.70050000e-01,
                5.54500000e00,
                3.30450000e-01,
                2.25055000e01,
                2.89968936e-01,
                2.81312857e01,
                3.30450000e-01,
                2.59500000e00,
            ],
            [
                4.50550000e00,
                3.09500000e-02,
                4.55500000e00,
                2.70550000e-01,
                2.75045000e01,
                2.89968936e-01,
                3.36952325e01,
                2.70550000e-01,
                4.75050000e01,
            ],
            [
                3.50650000e00,
                3.90350000e-01,
                5.95000000e-01,
                3.09500000e-02,
                4.75005000e01,
                2.89968936e-01,
                5.58042778e01,
                5.10150000e-01,
                2.75450000e01,
            ],
            [
                5.09500000e-01,
                4.50250000e-01,
                6.53500000e00,
                1.50750000e-01,
                7.50850000e00,
                2.89968936e-01,
                1.54715004e01,
                3.90350000e-01,
                3.75250000e01,
            ],
            [
                1.50850000e00,
                3.30450000e-01,
                1.58500000e00,
                5.10150000e-01,
                3.75025000e01,
                2.89968936e-01,
                4.44470781e01,
                3.09500000e-02,
                1.75650000e01,
            ],
            [
                7.50250000e00,
                5.10150000e-01,
                2.57500000e00,
                3.90350000e-01,
                3.25035000e01,
                2.89968936e-01,
                4.10182644e01,
                1.50750000e-01,
                4.25150000e01,
            ],
        ]
    )

    parents_test = {
        "local_random": None,
        "params": parents_params,
        "values": parents_val,
        "pop_size": 10,
        "crossover_rate": 0.01,
        "mutation_rate": 0.2,
        "di_crossover": 1,
        "di_mutation": 20,
        "nchildren": 20,
        "feasibility_model": None,
        "cross_constrained": False,
        #        'ranks': None,
        "ranks": np.array([5, 4, 7, 2, 3, 8, 0, 1, 6, 9]),
        "toursize": 2,
    }

    Space.update(SpaceCons)
    #    print(Space)

    #    sampled_points = ParamSpacePoints(10, Space)
    sampled_points = ParamSpacePoints(10, Space, Method=None, parents=parents_test)
    print(sampled_points.param_keys)
    print(sampled_points.param_arr)
