Experiment setup:

python 3.6

tensorflow 1.8 cpu


file names with 0, 0_1, 0_2, 0_3, 0_4 are experiments where weights and bias are defined in float32, and would be casted bfloat16 for bfloat16-required calculations

file names with 1,1_1,1_2,1_3 are experiments where weights and bias are defined in bfloat16, and
would be casted to float32 for any float32-required calculations


