from qubo_utils import *
import torch
import os
from feature_extraction_utils import FeatureExtractor, grad_cam, cosine_similarity
import torch.nn as nn
from torchvision import models


class ReducedResNet18(nn.Module):
    def __init__(self, n_features: int, num_classes=10):
        self.n_features = n_features
        super().__init__()
        original_resnet = models.resnet18(weights="DEFAULT")
        self.features = nn.Sequential(
            original_resnet.conv1,
            original_resnet.bn1,
            original_resnet.relu,
            original_resnet.maxpool,
            original_resnet.layer1,
            nn.ReLU(inplace=False),
            original_resnet.layer2,
            nn.ReLU(inplace=False),
            original_resnet.layer3,
            nn.ReLU(inplace=False),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, self.n_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(inplace=False),
        )

        self.classifier = nn.Linear(self.n_features * 14**2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


similarity = False
penalty_lambda = 0
n_features_constrain = 5
beta = .6
F = 16

model = ReducedResNet18(n_features=F, num_classes=10)

device = torch.device("cpu")
# model = OriginalResNet18(num_classes=10, reduced_features=F)
model.to(device)
if F == 16:
    checkpoint = torch.load(
    os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep17_fullResnetMod16.pth"),
        map_location=device,
        weights_only=False)
elif F == 24:
    checkpoint = torch.load(
    os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod24.pth"),
        map_location=device,
        weights_only=False)
elif F == 32:
    checkpoint = torch.load(
    os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep6_fullResnetMod32.pth"),
        map_location=device,
        weights_only=False)
elif F == 512:
    checkpoint = torch.load(
        os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod512.pth"),
        map_location=device,
        weights_only=False)
else:
    raise ValueError("Not implemented")

test_set = torch.load(os.path.join(os.getcwd(),
                                       "data",
                                       "data",
                                       "test_dataset.pt"),
                          map_location=device,
                          weights_only=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

"""ref_labels = np.arange(10)
class_inputs = []

for ref_label in ref_labels:
    inputs=[]
    for i in range(len(test_loader)):
        img,label=test_set[i]
        if label==ref_label:
            inputs.append(img)

    class_inputs.append(inputs)"""

def extraction(input_tensor: torch.Tensor):
    extractor = FeatureExtractor()
    extractor.set_model(model)

    extractor.set_target_layer(model.reduce[-1])
    extractor.set_input_tensor(input_tensor, see_picture=False)

    extractor.register_hooks()

    features, gradients, pred = extractor.extract_features()

    # we consider only nonzero features because are the only ones that contribute to the gradcam (and the cosine similarity is ill-defined for zeros)
    # Compute per-channel spatial mean on the first sample
    channel_means = features[0].mean(dim=(1, 2))  # Shape: [C]
    # Get mask of non-zero mean channels
    # nonzero_mask = channel_means != 0  # Shape: [C]
    # # Apply mask to features and gradients
    # features = features[:, nonzero_mask, :, :]
    # gradients = gradients[:, nonzero_mask, :, :]
    zero_mask=(channel_means==0)
    gradients[:,zero_mask,:,:]=-10**-10

    return features, gradients


if __name__ == '__main__':
    for index_class in range(10):

        #index_class = int(sys.argv[1])
        #n_samples = len(class_inputs[index_class])
        n_samples = 20

        for idx in range(n_samples):
            beta_true = 0.7 ## this is only to use the same input tensor
            filepath = os.path.join(
                'selected_fmaps',
                f'class{index_class}',
                f'Img_class_{index_class}_idx_{idx}_beta_{beta_true}_model{F}.npz'
            )

            # Skip if the file does not exist
            try:
                img_tensor = np.load(filepath)['input_tensor']
            except FileNotFoundError:
                print(f"Skipping missing file: {filepath}")
                continue

            #img_tensor = class_inputs[index_class][idx]
            input_tensor = torch.from_numpy(img_tensor)
            features, gradients = extraction(input_tensor)
            gradcam, alpha = grad_cam(features=features.cpu(), gradients=gradients.cpu(), subset_features=None)
            sorted_indices = np.argsort(-alpha)
            cos_sim = cosine_similarity(features=features.clone().squeeze().detach().cpu().numpy())
            Qubo = QuboXFeatureSelection()
            Qubo.set_hyperparameters(alpha.copy(),
                                     cosine_similarity=cos_sim,
                                     penalty_lambda=penalty_lambda,
                                     n_features_constraint=n_features_constrain,
                                     beta=beta,
                                     only_positive=True,
                                     num_reads=1000,
                                     similarity_condition=similarity)

            Qubo.solve(method='gurobi')
            optimal_solutions = Qubo.optimal_features.copy()

            target_dir = f"F{F}"
            if not os.path.exists(os.path.join(os.getcwd(), 'gurobi_results', target_dir)):
                os.makedirs(os.path.join(os.getcwd(), 'gurobi_results', target_dir))
            np.savez(os.path.join(os.getcwd(), "gurobi_results",
                                  target_dir, f"optimal_solutions_{index_class}_idx_{idx}_beta_{beta}.npz"),
                nonzero_indices=Qubo.idxs_alpha_nozero,
                cosine_similarity=Qubo.cosine_similarity,
                alpha=Qubo.alpha,
                input_tensor=input_tensor.numpy(),
                solutions=optimal_solutions,
                )