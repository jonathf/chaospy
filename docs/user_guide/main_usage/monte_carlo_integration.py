from problem_formulation import joint

sobol_samples = joint.sample(10000, rule="sobol")
antithetic_samples = joint.sample(10000, antithetic=True, seed=1234)
halton_samples = joint.sample(10000, rule="halton")
