This is purely for proof of concept / educational purposes. Refer to the README at the root of the repository for optimisation related information.

To load the weights:
- Copy the GPT implementation and load the pretrained model from [here](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L72). This helps to get everything to linear layers
- Run the following (note the tranposes)
```
import numpy as np
root = "r-nn/pysrc/weights"
for n, p in model.named_parameters():
    f_name = n.replace('transformer.', '')+"_" +"_".join([str(x) for x in list(p.shape)])+".npy"
    f_path = f"{root}/{f_name}"
    if len(p.shape) == 2:
        p = p.T
    arr = p.detach().numpy().flatten()
    np.save(f_path, arr)

f_name = "lm_head.weight_"+"_".join([str(x) for x in list(model.lm_head.weight.shape)])+".npy"
f_path = f"{root}/{f_name}"
arr = model.lm_head.weight.T.detach().numpy().flatten()
np.save(f_path, arr)
```