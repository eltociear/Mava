# Python code to generate the command lines for running the script with different configurations

# Configurations
batch_sizes = [8192, 1024, 256]
num_minibatches = [128, 32, 8]
rollout_lengths = [4, 300, 1000]
rb_sizes = [1000000, 3000]
lrs = [3e-4, 1e-3]
update_batch_sizes = [2, 8]

# Generating the command lines
commands = []
for batch_size in batch_sizes:
    for num_minibatch in num_minibatches:
        for rollout_length in rollout_lengths:
            for rb_size in rb_sizes:
                for lr in lrs:
                    for update_batch_size in update_batch_sizes:
                        command = f"python mava/systems/ma_isac_continuous.py -m system.batch_size={batch_size} system.num_minibatches={num_minibatch} system.rollout_length={rollout_length} system.rb_size={rb_size} system.lr={lr} system.update_batch_size={update_batch_size}"
                        print(command)


