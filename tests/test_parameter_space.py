from dmosopt.datatypes import ParameterSpace


def example_usage():
    # Define parameter space using nested dictionary
    config = {
        "Ra": [100, 200],
        "soma": {
            "gkabar_kap": [0.001, 0.1],
            "gkdrbar_kdr": [0.001, 0.1],
            "gbar_nax": [0.001, 0.2],
        },
        "axon": {
            "gbar_nax": [0.001, 0.2],
            "gkdrbar_kdr": [0.001, 0.1],
        },
    }

    # Create parameter space from config
    param_space = ParameterSpace.from_dict(config)

    # Example parameter point
    params = {
        "Ra": 100,
        "soma": {"gkabar_kap": 0.05, "gkdrbar_kdr": 0.02, "gbar_nax": 0.15},
        "axon": {"gbar_nax": 0.1, "gkdrbar_kdr": 0.01},
    }

    # Test conversions
    flat_params = param_space.flatten(params)
    reconstructed_params = param_space.unflatten(flat_params)

    # Print parameter space structure
    for param_range in param_space._flat_ranges:
        print(f"Parameter {param_range.name}:")
        print(f"  Range: [{param_range.lower}, {param_range.upper}]")

    print(f"flat parameter array: {flat_params}")
    print(f"reconstructed parameter array: {reconstructed_params}")

    print(param_space)


example_usage()
