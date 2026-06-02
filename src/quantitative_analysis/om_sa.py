from matplotlib import pyplot as plt
import numpy as np
#import torch
import os
from collections import Counter


beta = 0.99; F = 512; imgClass = 9; idx = list(range(20))


#model = torch.load("/Users/francescoaldoventurelli/qml/projet_one/model_16_features_trained_20_epochs", weights_only=False, map_location=torch.device('cpu'))
solutions = []
valid_idx = []


### WHEN YOU USE beta > 0.7 YOU HAVE SAVED THE BITSTRING INSIDE
## THE .npz file

#for x in idx:
#    fmap_path = os.path.join(
#        os.getcwd(),
#        "SA_fullresnet",
#        f"class{imgClass}",
#        f"features_selected_Img_class_{imgClass}_idx_{x}_beta_{beta}_model{F}.npy"
#    )
for x in idx:
    fmap_path = os.path.join(
        os.getcwd(),
        "SA_fullresnet",
        f"class{imgClass}",
        f"Img_class_{imgClass}_idx_{x}_beta_{beta}_model{F}.npz"
    )

    if not os.path.exists(fmap_path):
        print(f"[WARNING] Missing file for idx {x}: {fmap_path}")
        continue

    fm = np.load(fmap_path)
    bitstring = fm["selected_features"]
    print(f"[INFO] Loaded idx {x}, shape: {bitstring.shape}")

    solutions.append(bitstring)
    valid_idx.append(x)

if len(solutions) == 0:
    raise ValueError("No valid feature maps found. Cannot proceed.")

flat = np.concatenate(solutions, axis=0)
print(f"[INFO] Concatenated shape: {flat.shape}")
print(f"[INFO] Used indices: {valid_idx}")

hist, bins = np.histogram(flat, range=(0, F), bins=F)
hist = hist / hist.sum()

plt.hist(flat, bins=F, range=(0, F))
plt.xlabel("Feature value")
plt.ylabel("Count")
plt.title("Raw Histogram")
plt.show()

np.save(
    os.path.join(os.getcwd(), "histos", f"histClass{imgClass}-mostFrequent_SA_F{F}_beta{beta}.npy"),
    hist
)