# Dependencies

* Python >= 3.6
* torchonn-pyutils >= 0.0.1.
* pytorch-onn >= 0.0.1.
* Python libraries listed in `requirements.txt`
* NVIDIA GPUs and CUDA >= 11.5

# Usage

* Black-Scholes weigh-domain `> cd ./PINNs/Black_Scholes > python -u main.py -m both -c configs_weight/Ours.ini`
* Black-Scholes phase-domain `> cd ./PINNs/Black_Scholes > python -u main.py -m both -c configs_phase/Ours.ini -y configs_phase/tonn.yml`
* 20-dim HJB weigh-domain `> cd ./PINNs/HJB_20d_FD > python -u main.py -m both -c configs_weight/Ours.ini`
* 20-dim HJB phase-domain `> cd ./PINNs/HJB_20d_FD > python -u main.py -m both -c configs_phase/Ours.ini -y configs_phase/tonn.yml`
