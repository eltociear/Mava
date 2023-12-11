#!/bin/bash
python mava/systems/ff_ippo.py -m env.scenario=2s-8x8-2p-2f-coop system.actor_lr=0.004 system.critic_lr=0.004 

python mava/systems/ff_ippo.py -m env.scenario=2s-8x8-2p-2f-coop
