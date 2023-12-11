#!/bin/bash

python mava/systems/ff_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=lbf env.scenario=2s-8x8-2p-2f-coop,8x8-2p-2f-coop,2s-10x10-3p-3f,10x10-3p-3f,15x15-4p-3f,15x15-4p-5f,15x15-3p-5f && python mava/systems/ff_mappo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=lbf env.scenario=2s-8x8-2p-2f-coop,8x8-2p-2f-coop,2s-10x10-3p-3f,10x10-3p-3f,15x15-4p-3f,15x15-4p-5f,15x15-3p-5f && python mava/systems/rec_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=lbf env.scenario=2s-8x8-2p-2f-coop,8x8-2p-2f-coop,2s-10x10-3p-3f,10x10-3p-3f,15x15-4p-3f,15x15-4p-5f,15x15-3p-5f && python mava/systems/rec_mappo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=lbf env.scenario=2s-8x8-2p-2f-coop,8x8-2p-2f-coop,2s-10x10-3p-3f,10x10-3p-3f,15x15-4p-3f,15x15-4p-5f,15x15-3p-5f