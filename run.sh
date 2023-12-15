python mava/systems/ff_ippo.py -m system.seed=0,1,2,3 env/scenario=2s-8x8-2p-2f-coop,15x15-4p-3f &&
python mava/systems/ff_mappo.py -m system.seed=0,1,2,3 env/scenario=2s-8x8-2p-2f-coop,15x15-4p-3f &&
python mava/systems/rec_mappo.py -m system.seed=0,1,2,3 env/scenario=2s-8x8-2p-2f-coop,15x15-4p-3f &&
python mava/systems/rec_ippo.py -m system.seed=0,1,2,3 env/scenario=2s-8x8-2p-2f-coop,15x15-4p-3f
