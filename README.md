# Sampling Toolbox

A python script to run relative information gain (RIG) permutation tests for boolean features and binary target variables.

## How to use
Instantiate a permutation test object by running
* `test_object = PermutationObject()`

Then run the following to set the parameters for the synthetic data
* `test_object.set_data_parameters(nrows = <number of rows>, class_weights = [<minority class fraction>]`)

The class weights takes a python list For a binary target, provide only the minority class weight. For multiclass provide all class weights/fractions.

Create the synthetic data
* `test_object.create_sythetic_data()`

Run the permutation tests to calculate the null distribution for RIG given the specified number of rows and observations of the minority class
* `test_object.calculate_null_rigs(permutations = <number of permutations>`)
