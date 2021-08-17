#!/bin/bash

## IF
#python main.py -M IF -D kddcup99 -L neptune -I simple_inc -P 10

#python main.py -M IF -D cifa10 -L 1 -I simple_inc -P 10

#python main.py -M IF -D cifa100 -L 1 -I simple_inc -P 10

#python main.py -M IF -D cardioForAnomalies -L 1 -I simple_inc -P 10
#
#python main.py -M IF -D defaultCreditCard -L 1 -I simple_inc -P 10
#
#python main.py -M IF -D DryBean -L SEKER -I simple_inc -P 10
#
#python main.py -M IF -D fashion_mnist -L 1 -I simple_inc -P 10
#
#python main.py -M IF -D kddcup_rev -L neptune -I simple_inc -P 10
#
#python main.py -M IF -D mnist -L 9 -I simple_inc -P 10
#
#python main.py -M IF -D aps_failure -L pos -I simple_inc -P 10


## OCSVM
#python main.py -M OCSVM -D kddcup99 -L neptune -I simple_inc -P 10

#python main.py -M OCSVM -D cifa10 -L 1 -I simple_inc -P 10
#
#python main.py -M IF -D cifa100 -L 1 -I simple_inc -P 10
#
#python main.py -M OCSVM -D cardioForAnomalies -L 1 -I simple_inc -P 10
#
#python main.py -M OCSVM -D defaultCreditCard -L 1 -I simple_inc -P 10

python main.py -M OCSVM -D DryBean -L SEKER -I simple_inc -P 10

python main.py -M OCSVM -D fashion_mnist -L 1 -I simple_inc -P 10

python main.py -M OCSVM -D kddcup_rev -L neptune -I simple_inc -P 10

python main.py -M OCSVM -D mnist -L 1 -I simple_inc -P 10

#python main.py -M OCSVM -D aps_failure -L pos -I simple_inc -P 10
