Concepts
=========

[Bayesian metamodeling](https://www.biorxiv.org/content/10.1101/2021.03.29.437574v1.full.pdf) is a modeling approach that divides-and-conquers the problem of modeling a big and complex system into computing a number of smaller models of different types, followed by assembling these models into a complete map of the system. A collection of heterogeneous input models are integrated together to compute a metamodel. Given the input models, Bayesian metamodeling proceeds through **three major steps**:

- convert the input models into probabilistic surrogate models;   
- couple the surrogate models through subsets of statistically related variables;   
- update the input models by computing the probability density functions(PDFs) of free parameters for each input model in the context of all other input models.   
    
Bayesian metamodeling decentralizes computing and expertise; in addition, it often produces a more accurate, precise, and complete metamodel that contextualizes input models as well as resolves conflicting information. 

### Terms in Bayesian metamodeling:
- **Input model:** The input model refers to mathematical descriptions of the entities, dynamics and functions of a complex biological system. In principle, each input model can be based on any type of data, at any scales or any level of granularity.

- **Model variables:** Model variables are the representation of an attribute or quantity of a model state and may change during the simulation, including independent variables (i.e., regressors, features, or inputs) and dependent variables (i.e., response variables, regressands, outcomes, labels, predictions, or outputs). 

- **Model parameters:** Model parameters are constants or distributions of quantities fit to input information in each simulated model and change only when the model behaviour is adjusted, including free parameters (i.e., degrees of freedom) and fixed parameters (i.e., constants or hyper parameters). 

- **Surrogate model:** We create a common representation for different input models by converting them into probabilistic models (i.e., surrogate models) according to the framework of Bayesian metamodeling. Formally, a surrogate model specifies a PDF over some input model variables and additional necessary variables. The surrogate model quantifies the input model uncertainty through marginal PDFs and encodes the statistical dependencies between its variables through joint PDFs. 

- **Metamodel:** The joint PDF defined over all the model variables.