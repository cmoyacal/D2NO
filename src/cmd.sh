#!/usr/bin/env bash


# base model l2 errors = [0.42, 0.49]


# python main.py --num-rounds 100

# 0: 0.39403385, 1: 0.455519

#python main.py --num-rounds 100 

# I have added a simple scheduler [lr, 0.5*lr @ 50, 0.5*0.5*lr @ 75]
# 0: 0.42104712, 1: 0.48970258

#python main.py --num-rounds 100 --lr 0.005

# 0: 0.35987815, 1: 0.44044

#python main.py --num-rounds 80 --lr 0.005 --local-epochs 2 --gamma 0.5 --branch-depth-1 3

# 0: 0.32724702, 1: 0.41396


#python main.py --num-rounds 40 --lr 0.0025 --local-epochs 4 --gamma 0.5 --branch-depth-1 3 --num-experiments 1000 \
#		--trunk-depth 4

# 0: 0.65343714, 1: 0.778

#python main.py --num-rounds 50 --lr 0.0025 --local-epochs 4 --gamma 0.5 \
		#--branch-depth-0 3 --branch-depth-1 4 --num-experiments 1000 --trunk-depth 3

# 0: 0.43982047, 1: 0.47

#python main.py --num-rounds 200 --lr 0.005 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.2944089, 1: 0.410


# python main.py --num-rounds 50 --lr 0.006 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.40304577, 1: 0.4733705


# python main.py --num-rounds 50 --lr 0.0065 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.5779041, 1: 0.61743


# python main.py --num-rounds 50 --lr 0.0045 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.40761486, 1: 0.48078


# python main.py --num-rounds 50 --lr 0.0035 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.33509305, 1: 0.44221

# python main.py --num-rounds 50 --lr 0.003 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.3638877, 1: 0.42419


# python main.py --num-rounds 50 --lr 0.0025 --local-epochs 1 --gamma 0.5 --branch-depth-1 3

# 0: 0.30608332, 1: 0.381240

# python main.py --num-rounds 50 --lr 0.0025 --local-epochs 1 --gamma 0.5 --branch-depth-1 3 --num-experiments 1100

# 0: 0.45320827, 1: 0.4866


python main.py --num-rounds 50 --lr 0.0025 --local-epochs 1 --gamma 0.5 --branch-depth-1 3 --branch-depth-0 3 --num-experiments 1100 --trunk-depth 2 
