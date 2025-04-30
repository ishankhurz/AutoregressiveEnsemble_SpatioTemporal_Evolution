## Autoregressive  approach for trajectory prediction in spatiotemporal systems using UNet ensemble

Summary: We propose an ensemble based approach for predicting evolution of physical fields in scientific systems with both spatially and temporally dependent variables

TO-DO


Training: 
Data: Tuples of {x_{i}(t-k:k), e_{i:k}(t-k:t+1), y_{i}(t+1)} where x is input field, e is external time-dependent variable, and y is output field
Collect data tuples for all t on the simulation trajectory
Procedure: Train individual models for the same task and with same data in parallel with random initialization of weights


Inference Procedure: 
1. First K time-step ground truth fields as prompt to the model
2. Average individual model predictions for time step t and store final field prediction at time t
3. Field prediction at time t goes back in as model input for the next time-step (t+1)
4. Repeat steps 2-3 till the end of simulation trajectory


Datasets:
1. Computational mechanics: 2-Phase microstructure
2. Biochemical system: Gray-Scott reaction diffusion
3. Atmospherical dynamics: Planet Shallow Water Equation 
  
