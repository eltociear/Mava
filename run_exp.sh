python mava/systems/ff_ippo_rware.py -m system.seed=1,2,3,4,5,6,7,8,9 env/rware_scenario=small-4ag,tiny-2ag,tiny-4ag && python mava/systems/ff_mappo_rware.py -m system.seed=1,2,3,4,5,6,7,8,9 env/rware_scenario=small-4ag,tiny-2ag,tiny-4ag && python mava/systems/rec_ippo_rware.py -m system.seed=1,2,3,4,5,6,7,8,9 env/rware_scenario=small-4ag,tiny-2ag,tiny-4ag && python mava/systems/rec_mappo_rware.py -m system.seed=1,2,3,4,5,6,7,8,9 env/rware_scenario=small-4ag,tiny-2ag,tiny-4ag