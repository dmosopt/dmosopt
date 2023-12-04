# Results

All results of the optimization are stored in the HDF5 result file under the unique `opt_id` namespace.

The structure of the file is as follows:

```
/{opt_id}
 ├─ optimizer_params
 │  └─ 0 ... {num_epochs - 1}
 │     ├─ crossover_prob
 │     ├─ di_crossover
 │     ├─ ...
 │     
 ├─ 0               // problem_id
 │  ├─ epochs       // epoch
 |  ├─ features     // features (if enabled)
 │  ├─ objectives   // objective values
 |  ├─ constraints  // constraints (if enabled)
 │  ├─ parameters   // a column for each parameter
 │  └─ predictions  // surrogate predications (if saved)
 ├─ objective_enum
 ├─ objective_spec 
 ├─ ...

```
