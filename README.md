# greedyGPCS
Greedy Gaussian Process-Based Covariance Steering (GPCS)

## Setup

```sh
#!bash
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ python -m ipykernel install --user --name=env
`````````

## Train Stochastic Variational Gaussian Process

1. Generate some training data: [generate_data_simple_car.ipynb](generate_data_simple_car.ipynb)
2. Run stochastic gradient descent on GPyTorch-based GP model: [train_gp_simple_car.ipynb](train_gp_simple_car.ipynb)

## Run Covariance Steering

### SVGP Model

Run the notebook: [CS_simple_car_GP.ipynb](CS_simple_car_GP.ipynb)
<p float="left">
  <img src="figs/gp_inputs.png" width="200" />
  <img src="figs/gp_position_uncertainties.png" width="350" />
  <img src="figs/gp_model_states.png" width="200" /> 
</p>

### Exact Model
Run the notebook: [CS_simple_car_exact.ipynb](CS_simple_car_exact.ipynb)
<p float="left">
  <img src="figs/exact_inputs.png" width="200" />
  <img src="figs/exact_position_uncertainties.png" width="350" />
  <img src="figs/exact_model_states.png" width="200" /> 
</p>
