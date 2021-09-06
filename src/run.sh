#!/bin/bash

#python main.py -M IF -D mnist -L 9 -I simple_inc -P 50
#
#python main.py -M IF -D aps_failure -L pos -I simple_inc -P 500
#
#python main.py -M IF -D defaultCreditCard -L 1 -I simple_inc -P 250
#
#python main.py -M IF -D cifa10 -L 1 -I simple_inc -P 50
#
#python main.py -M IF -D cifa100 -L 1 -I simple_inc -P 25
#
#python main.py -M IF -D cardioForAnomalies -L 1 -I simple_inc -P 500
#
#python main.py -M IF -D defaultCreditCard -L 1 -I simple_inc -P 500
#
#python main.py -M IF -D DryBean -L SEKER -I simple_inc -P 100
#
#python main.py -M IF -D fashion_mnist -L 1 -I simple_inc -P 50
#
#python main.py -M IF -D kddcup99 -L neptune -I simple_inc -P 5000
#
#python main.py -M IF -D kddcup_rev -L neptune -I simple_inc -P 10000

## OCSVM
#python main.py -M OCSVM -D kddcup99 -L neptune -I simple_inc -P 10000 
#python main.py -M OCSVM -D kddcup_rev -L neptune -I simple_inc -P 20000

#python main.py -M OCSVM -D cifa10 -L 1 -I simple_inc -P 100
#
#python main.py -M OCSVM -D cifa100 -L 1 -I simple_inc -P 50
#
#python main.py -M OCSVM -D cardioForAnomalies -L 1 -I simple_inc -P 1000
#
#python main.py -M OCSVM -D defaultCreditCard -L 1 -I simple_inc -P 500

#python main.py -M OCSVM -D DryBean -L SEKER -I simple_inc -P 250

#python main.py -M OCSVM -D fashion_mnist -L 1 -I simple_inc -P 100

#python main.py -M OCSVM -D mnist -L 1 -I simple_inc -P 100

#python main.py -M OCSVM -D aps_failure -L pos -I simple_inc -P 1000

## AE
python main.py -M AutoEncoder -D mnist -L 9 -I simple_inc -P 50 -T image
#
#python main.py -M AutoEncoder -D aps_failure -L pos -I simple_inc -P 500
#
#python main.py -M AutoEncoder -D defaultCreditCard -L 1 -I simple_inc -P 250
#
#python main.py -M AutoEncoder -D cifa10 -L 1 -I simple_inc -P 50
#
#python main.py -M AutoEncoder -D cifa100 -L 1 -I simple_inc -P 25
#
#python main.py -M AutoEncoder -D cardioForAnomalies -L 1 -I simple_inc -P 500
#
#python main.py -M AutoEncoder -D defaultCreditCard -L 1 -I simple_inc -P 500
#
#python main.py -M AutoEncoder -D DryBean -L SEKER -I simple_inc -P 100
#
#python main.py -M AutoEncoder -D fashion_mnist -L 1 -I simple_inc -P 50
#
#python main.py -M AutoEncoder -D kddcup99 -L neptune -I simple_inc -P 5000
#
#python main.py -M AutoEncoder -D kddcup_rev -L neptune -I simple_inc -P 10000