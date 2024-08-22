Installation
==============


## Pre-requisite:

- [Filterpy](https://filterpy.readthedocs.io/en/latest/)
- numpy
- scipy

## To use our method, users need to provide:
- the input models, which will be converted to surrogate models using the method above
- the name of connecting variables, which are the selected statistically-related variables in the input models
- (optional) the parameters reflecting the coupling strength (how tightly the models are coupled together), by default is the reciprocal of the number of models sharing the same connecting variable. If the user can provide additional information about the coupling strength, it can be changed.