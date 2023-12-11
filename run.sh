python mava/systems/ff_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=rware env/scenario=tiny-2ag,tiny-4ag,small-4ag && \
python mava/systems/ff_mappo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=rware env/scenario=tiny-2ag,tiny-4ag,small-4ag && \
python mava/systems/rec_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=rware env/scenario=tiny-2ag,tiny-4ag,small-4ag && \
python mava/systems/rec_mappo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env=rware env/scenario=tiny-2ag,tiny-4ag,small-4ag