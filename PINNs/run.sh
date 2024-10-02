pip install -r requirements.txt

##### PINNs dnn #####
CUDA_VISIBLE_DEVICES=0 python -u main.py -m both -c pinns_dnn.ini
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -m both -c pinns_dnn.ini >/dev/null 2>&1 &

##### PINNs tensorized_dnn #####
CUDA_VISIBLE_DEVICES=0 python -u main.py -m both -c pinns_tt.ini
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -m both -c pinns_tt.ini >/dev/null 2>&1 &

##### PINNs MZI ONN #####
CUDA_VISIBLE_DEVICES=0 python -u main.py -m both -c pinns_onn.ini -y tonn.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -m both -c pinns_onn.ini -y tonn.yml >/dev/null 2>&1 &

##### PINNs MZI TONN #####
CUDA_VISIBLE_DEVICES=0 python -u main.py -m both -c pinns_tonn.ini -y tonn.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -m both -c pinns_tonn.ini -y tonn.yml >/dev/null 2>&1 &



##### pretrain #####
## PINNs MZI ONN ##
# pretrain
CUDA_VISIBLE_DEVICES=2 python -u main.py -m both -c pinns_onn.ini -y pretrain.yml
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -m both -c pinns_onn.ini -y pretrain.yml >/dev/null 2>&1 &

## PINNs MZI TONN ##
# pretrain
CUDA_VISIBLE_DEVICES=2 python -u main.py -m both -c pinns_tonn.ini -y pretrain.yml
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -m both -c pinns_tonn.ini -y pretrain.yml >/dev/null 2>&1 &



##### PINNs MRR ONN
# pretrain
CUDA_VISIBLE_DEVICES=2 python -u main.py -m both -c pinns_mrr.ini
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py -m both -c pinns_mrr.ini >/dev/null 2>&1 &

# tailored
CUDA_VISIBLE_DEVICES=2 python -u mrr_training.py -m both -c pinns_mrr.ini
CUDA_VISIBLE_DEVICES=2 nohup python -u mrr_training.py -m both -c pinns_mrr.ini >/dev/null 2>&1 &