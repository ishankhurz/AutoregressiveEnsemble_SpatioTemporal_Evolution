## Autoregressive-ensemble approach for trajectory prediction in spatiotemporal systems

Project summary: Systems governed by partial differential equations are commonplace in scientific fields. We propose a machine learning ensemble approach for reduced error accumulation while using an autoregressive strategy. Specifically, multiple models with randomly initialized weights are trained in parallel for the same task of predicting the next state of the system given information from previous time steps. During inference, the model predictions are averaged and fed back into the model to obtain predictions for the future timesteps and the ground truth for only the first K time-steps is required to predict the full trajectory, saving computation time. 

Manuscript in preparation (authors: Ishan Khurjekar, Indrashish Saha, Somdatta Goswami, Lori Graham Brady)


## Training: 
1. Data: Tuples of {x_{i}(t-k:k), e_{i:k}(t-k:t+1), y_{i}(t+1)} where x is input field, e is external time-dependent variable, and y is output field
2. Collect data tuples for all t on the simulation trajectory (t = K, K + 1 .... T)
3. Procedure: Train individual models for the same task and with same data in parallel with random initialization of weights


## Inference: 
1. First K time-step ground truth fields as prompt to the model
2. Average individual model predictions for time step t and store final field prediction at time t  
3. Field prediction at time t goes back in as model input for the next time-step (t+1)
4. Repeat steps 2-3 till the end of simulation trajectory  (t = K , K + 1, ..... T)


## Datasets:
1. Computational mechanics: 2-Phase microstructure [1]
2. Biochemical system: Gray-Scott reaction diffusion [2]
3. Atmospherical dynamics: Planet Shallow Water Equation [3]


## Dataset references
[1] DOI to be updated

[2] Gray, Peter, and Stephen K. Scott. "Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Oscillations and instabilities in the system A+ 2B→ 3B; B→ C." Chemical Engineering Science 39.6 (1984): 1087-1097.

[3] McCabe, Michael, et al. "Towards stability of autoregressive neural operators." arXiv preprint arXiv:2306.10619 (2023).
  
