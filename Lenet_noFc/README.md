running envrionment:
python3.6
tensorflow 1.8 cpu

files here are experiments in Lenet wihout the fully connected layers.

files with 0, 0_1, 0_2, 0_3 are experiments where all the weights and bias are defined in float 32 and would be casted bfloat16 when necessary



files with 1, 1_1,1_2 are experiments where not all the weights and bias are defined in float 32, most of them are defined in bfloat16 and would be casted to float32 when necessary.

