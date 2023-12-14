#!/bin/bash

python mava/systems/ff_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env/scenario=small-4ag,tiny-4ag,tiny-2ag
