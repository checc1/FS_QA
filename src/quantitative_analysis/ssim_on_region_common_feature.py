import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.feature_extraction_utils2 import FeatureExtractor
import torch.nn as nn
from torchvision import transforms, models
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.transforms.functional import crop
import torchvision


def grad_cam(features, gradients, subset_features=None,
             with_normalization=None):
    avg_gradients = gradients.mean(dim=(2, 3), keepdim=True)

    if subset_features is not None:
        subset_features = np.asarray(subset_features, dtype=np.int64)
        subset_features = torch.as_tensor(subset_features, dtype=torch.long)
        gradcam = torch.relu(avg_gradients[:, subset_features] * features[:, subset_features]).sum(dim=1, keepdim=True)
        alpha = avg_gradients.squeeze()[subset_features]

    else:
        gradcam = torch.relu(avg_gradients[:, :] * features[:, :]).sum(dim=1, keepdim=True)
        alpha = avg_gradients
    raw_gradcam = torch.relu(
        avg_gradients[:, subset_features] * features[:, subset_features]
    ).sum(dim=1, keepdim=True)
    #print(raw_gradcam.mean(), raw_gradcam.std())
    #gradcam = gradcam / (gradcam.mean() + 1e-8)
    #gradcam = torch.clamp(gradcam, 0, 1)

    if with_normalization == True:
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        return gradcam.detach().cpu().numpy(), alpha.squeeze().detach().numpy()
    else:
        return gradcam.detach().cpu().numpy(), alpha.squeeze().detach().numpy()

F = 16
beta = 0.7
## H = b * linear term + (1-b) * quadratic term

datatest_path = os.path.join(os.getcwd(), 'data', 'test_set.pth')
class ReducedResNet18(nn.Module):
    def __init__(self, num_classes=10, reduced_features=F):
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
            nn.Conv2d(256, reduced_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_features),
            nn.ReLU(inplace=False),
        )

        self.classifier = nn.Linear(reduced_features * 14**2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



device = torch.device("cpu")
model = ReducedResNet18(num_classes=10, reduced_features=F)
model.to(device)
checkpoint = torch.load(
    os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep17_fullResnetMod16.pth"),
    #os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod24.pth"),
    #os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod512.pth"),
    map_location=device
)

test_set = torch.load(os.path.join(os.getcwd(), "data", "data", "test_dataset.pth"), map_location=device, weights_only=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
model.load_state_dict(checkpoint["model_state_dict"])

TARGET_LAYER = model.reduce[-1]
#TARGET_LAYER = model.features[-1]
CLASS_NAMES = ['Airplane', 'Bird', 'Car', 'Cat', 'Deer',
               'Dog', 'Horse', 'Monkey', 'Ship', 'Truck']

NUM_CLASSES = len(CLASS_NAMES)
model_dict = dict(
    type='resnet',
    arch=model,
    layer_name=TARGET_LAYER,
    input_size=(224, 224)
)

model.eval()

def roi(class1: int, class2: int, index: int = None) -> tuple[list, list]:
    bitString1 = np.load(
        os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class1}_beta{beta}.npy'),
        allow_pickle=True)
    bitString2 = np.load(
        os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class2}_beta{beta}.npy'),
        allow_pickle=True)
    bitString1 = [np.asarray(x, dtype=np.int64) for x in bitString1]
    bitString2 = [np.asarray(x, dtype=np.int64) for x in bitString2]
    length = min(len(bitString1), len(bitString2))

    if index is None:
        index = np.random.randint(0, length)
    else:
        if index >= length or index < 0:
            raise ValueError(f"Provided index {index} is out of bounds for max length {length}")

    data1 = np.load(
        f"/Users/francescoaldoventurelli/qml/projet_one/AnnealingWORK/selected_fmaps/class{class1}/Img_class_{class1}_idx_{index}_beta_{beta}_model{F}.npz",
        allow_pickle=True)
    data2 = np.load(
        f"/Users/francescoaldoventurelli/qml/projet_one/AnnealingWORK/selected_fmaps/class{class2}/Img_class_{class2}_idx_{index}_beta_{beta}_model{F}.npz",
        allow_pickle=True)
    input_tensor1 = data1["input_tensor"]
    input_tensor2 = data2["input_tensor"]
    # print(input_tensor1.shape)
    # print(input_tensor2.shape)
    extractor1 = FeatureExtractor()
    extractor1.set_model(model)
    extractor1.set_target_layer(TARGET_LAYER)
    extractor1.set_input_tensor(torch.from_numpy(input_tensor1), see_picture=False)
    extractor1.register_hooks()

    try:
        features1, gradients1, pred1 = extractor1.extract_features()
    except Exception as e:
        extractor1.remove_hooks()  # Clean up hooks even if it fails
        raise RuntimeError(f"Feature extraction failed for Class {class1} at index {index}. Error: {e}")
    finally:
        extractor1.remove_hooks()

    cam_gc_Bitstring1, _ = grad_cam(
        features=features1.cpu(),
        gradients=gradients1.cpu(),
        subset_features=bitString1[index],
        with_normalization=True,
    )

    extractor2 = FeatureExtractor()
    extractor2.set_model(model)
    extractor2.set_target_layer(TARGET_LAYER)
    extractor2.set_input_tensor(torch.from_numpy(input_tensor2), see_picture=False)
    extractor2.register_hooks()

    try:
        features2, gradients2, pred2 = extractor2.extract_features()
    except Exception as e:
        extractor2.remove_hooks()  # Clean up hooks even if it fails
        raise RuntimeError(f"Feature extraction failed for Class {class2} at index {index}. Error: {e}")
    finally:
        extractor2.remove_hooks()

    cam_gc_Bitstring2, _ = grad_cam(
        features=features2.cpu(),
        gradients=gradients2.cpu(),
        subset_features=bitString2[index],
        with_normalization=True,
    )

    camwBitstring1 = torch.nn.functional.interpolate(
        torch.from_numpy(cam_gc_Bitstring1),
        size=input_tensor1.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    camwBitstring2 = torch.nn.functional.interpolate(
        torch.from_numpy(cam_gc_Bitstring2),
        size=input_tensor2.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    masked1 = torch.from_numpy(input_tensor1) * camwBitstring1
    masked2 = torch.from_numpy(input_tensor2) * camwBitstring2

    return [masked1, camwBitstring1, input_tensor1], [masked2, camwBitstring2, input_tensor2]


def plot():
    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(np.permute_dims(img1, (1, 2, 0)))
    axs[0,1].imshow(mask1.squeeze(0).detach().cpu().permute(1, 2, 0).numpy())
    axs[0,2].imshow(cam1.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    axs[1,0].imshow(np.permute_dims(img2, (1, 2, 0)))
    axs[1,1].imshow(mask2.squeeze(0).detach().cpu().permute(1, 2, 0).numpy())
    axs[1,2].imshow(cam2.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def filtering_cam(inputTensor: torch.Tensor, threshold: float):
    cam_thresholded = inputTensor.clone()
    cam_thresholded[cam_thresholded < threshold] = 0
    return cam_thresholded


def plot_all(img1, img2, mask1, mask2, cam1_thresholded,
             cam2_thresholded, ssim_val: str):
    fig, axs = plt.subplots(2, 4)
    axs[0,0].imshow(np.permute_dims(img1, (1, 2, 0)))
    axs[0,1].imshow(mask1.squeeze(0).detach().cpu().permute(1, 2, 0).numpy())
    axs[0,2].imshow(cam1.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    axs[0,3].imshow(cam1_thresholded.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    axs[1,0].imshow(np.permute_dims(img2, (1, 2, 0)))
    axs[1,1].imshow(mask2.squeeze(0).detach().cpu().permute(1, 2, 0).numpy())
    axs[1,2].imshow(cam2.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    axs[1,3].imshow(cam2_thresholded.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cmap='inferno')
    num = np.random.randint(0, 19)
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    #plt.suptitle(fr"$ssim = {ssim_val}$")
    plt.tight_layout()
    plt.savefig(f"comparison_mask_{class1}-{class2}-index{num}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

def _calculateSSIM(inputTensor: torch.Tensor,
                   targetTensor: torch.Tensor,) -> torch.Tensor:
    ### data range is the range of the image
    ssim = StructuralSimilarityIndexMeasure(data_range=float(max(inputTensor.max(), targetTensor.max()) - min(inputTensor.min(), targetTensor.min())))
    s = ssim(inputTensor, targetTensor)
    return s


def _calculateSSIM_masked(inputTensor: torch.Tensor, targetTensor: torch.Tensor) -> torch.Tensor:
    data_range = float(max(inputTensor.max(), targetTensor.max()) - min(inputTensor.min(), targetTensor.min()))
    ssim_value, ssim_map = structural_similarity_index_measure(
        inputTensor,
        targetTensor,
        data_range=data_range,
        return_full_image=True
    )

    roi_mask = (inputTensor > 0) | (targetTensor > 0)
    masked_ssim_values = ssim_map[roi_mask]

    if masked_ssim_values.numel() == 0:
        return torch.tensor(0.0)

    return masked_ssim_values.mean()


#

## just for the threshold
# :TODO: prendere features di classi che non si overlappano!
class1 = 3
class2 = 7

threshold1 = 0.4
threshold2 = 0.4

#threshold_c = float(c01.max() - 0.4*c02.max())
ssim_list = []
bitString1 = np.load(
        os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class1}_beta{beta}.npy'),
        allow_pickle=True)
bitString2 = np.load(
    os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class2}_beta{beta}.npy'),
    allow_pickle=True)
bitString1 = [np.asarray(x, dtype=np.int64) for x in bitString1]
bitString2 = [np.asarray(x, dtype=np.int64) for x in bitString2]
max_available_length = min(len(bitString1), len(bitString2))
iterations = min(20, max_available_length)

print(f"Class{class1} bitstring", bitString1)
print(f"Class{class2} bitstring", bitString2)
index_to_choose = 17
#index_to_choose = int(np.random.randint(0, max_available_length))


def roi_common(class1: int, class2: int, index: int = None):
    bitString1 = np.load(
        os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class1}_beta{beta}.npy'),
        allow_pickle=True)
    bitString2 = np.load(
        os.path.join(os.getcwd(), 'feature_maps_subsetQA', f'selected_maps_class{class2}_beta{beta}.npy'),
        allow_pickle=True)
    bitString1 = [np.asarray(x, dtype=np.int64) for x in bitString1]
    bitString2 = [np.asarray(x, dtype=np.int64) for x in bitString2]
    length = min(len(bitString1), len(bitString2))

    if index is None:
        index = np.random.randint(0, length)
    else:
        if index >= length or index < 0:
            raise ValueError(f"Provided index {index} is out of bounds for max length {length}")

    data1 = np.load(
        f"/Users/francescoaldoventurelli/qml/projet_one/AnnealingWORK/selected_fmaps/class{class1}/Img_class_{class1}_idx_{index}_beta_{beta}_model{F}.npz",
        allow_pickle=True)
    data2 = np.load(
        f"/Users/francescoaldoventurelli/qml/projet_one/AnnealingWORK/selected_fmaps/class{class2}/Img_class_{class2}_idx_{index}_beta_{beta}_model{F}.npz",
        allow_pickle=True)
    input_tensor1 = data1["input_tensor"]
    input_tensor2 = data2["input_tensor"]
    # print(input_tensor1.shape)
    # print(input_tensor2.shape)
    extractor1 = FeatureExtractor()
    extractor1.set_model(model)
    extractor1.set_target_layer(TARGET_LAYER)
    extractor1.set_input_tensor(torch.from_numpy(input_tensor1), see_picture=False)
    extractor1.register_hooks()

    try:
        features1, gradients1, pred1 = extractor1.extract_features()
    except Exception as e:
        extractor1.remove_hooks()  # Clean up hooks even if it fails
        raise RuntimeError(f"Feature extraction failed for Class {class1} at index {index}. Error: {e}")
    finally:
        extractor1.remove_hooks()

    bitStringCommon = np.intersect1d(bitString1[index], bitString2[index])
    print("Shared FM bitstring", bitStringCommon)

    cam_gc_Bitstring1, _ = grad_cam(
        features=features1.cpu(),
        gradients=gradients1.cpu(),
        subset_features=bitStringCommon,
        with_normalization=True,
    )

    extractor2 = FeatureExtractor()
    extractor2.set_model(model)
    extractor2.set_target_layer(TARGET_LAYER)
    extractor2.set_input_tensor(torch.from_numpy(input_tensor2), see_picture=False)
    extractor2.register_hooks()

    try:
        features2, gradients2, pred2 = extractor2.extract_features()
    except Exception as e:
        extractor2.remove_hooks()  # Clean up hooks even if it fails
        raise RuntimeError(f"Feature extraction failed for Class {class2} at index {index}. Error: {e}")
    finally:
        extractor2.remove_hooks()

    cam_gc_Bitstring2, _ = grad_cam(
        features=features2.cpu(),
        gradients=gradients2.cpu(),
        subset_features=bitStringCommon,
        with_normalization=True,
    )

    camwBitstring1 = torch.nn.functional.interpolate(
        torch.from_numpy(cam_gc_Bitstring1),
        size=input_tensor1.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    camwBitstring2 = torch.nn.functional.interpolate(
        torch.from_numpy(cam_gc_Bitstring2),
        size=input_tensor2.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    masked1 = torch.from_numpy(input_tensor1) * camwBitstring1
    masked2 = torch.from_numpy(input_tensor2) * camwBitstring2

    return [masked1, camwBitstring1, input_tensor1], [masked2, camwBitstring2, input_tensor2]

##### ROI WITH FM in common
b1, b2 = roi_common(class1, class2, index_to_choose)

mask1, cam1, t1 = b1
mask2, cam2, t2 = b2
c1_thr, c2_thr = filtering_cam(cam1, threshold1), filtering_cam(cam2, threshold1)
#print(c1_thr.max())
#print(c1_thr.min())

#ssim_val = _calculateSSIM_masked(c1_thr, c2_thr).item()
#plot_all(t1, t2, mask1, mask2, c1_thr, c2_thr, f"{ssim_val:.4f}")
### try the patch-wise method and calculate the ssim of the sample patches
## cc-correlation coefficient (PCC)
## PSNR peak signal-to-noise ratio
## take similar images and do the analysis (no average) - show one or 2 images, not too much

class Cropper():
    def __init__(self, input_t: torch.Tensor, cam_thresholded: torch.Tensor, inner_threshold: float) -> None:
        self.input_t = input_t
        self.cam_thresholded = cam_thresholded
        self.inner_threshold = inner_threshold

    @staticmethod
    def get_max(x: dict) -> tuple:
        maxkey = max(x, key=x.get)
        return maxkey, x[maxkey]

    @staticmethod
    def get_min(x: dict) -> tuple:
        minkey = min(x, key=x.get)
        return minkey, x[minkey]

    def _get_max_pixels(self):
        dict_row = {}
        dict_col = {}
        batch_idx = 0
        channel_idx = 0

        for i in range(self.cam_thresholded.shape[2]):
            max_row = 0
            for j in range(self.cam_thresholded.shape[3]):
                if self.cam_thresholded[batch_idx, channel_idx, i, j] > self.inner_threshold:
                    max_row += 1
            if max_row > 0:
                dict_row[i] = max_row
        for j in range(self.cam_thresholded.shape[3]):
            max_col = 0
            for i in range(self.cam_thresholded.shape[2]):
                if self.cam_thresholded[batch_idx, channel_idx, i, j] > self.inner_threshold:
                    max_col += 1
            if max_col > 0:
                dict_col[j] = max_col
        return dict_row, dict_col

    @staticmethod
    def keep_largest_component(x: torch.Tensor) -> torch.Tensor:
        batch_idx, channel = 0, 0
        image = x[batch_idx, channel, :, :]
        labeled_matrix = torch.zeros_like(image)
        current_label = 1

        for i in range(1, image.shape[0]):
            for j in range(1, image.shape[1]):
                if image[i, j] > 0:
                    # Check if neighbor to the left or top already has a cluster ID assigned
                    left_neighbor_label = labeled_matrix[i, j - 1].item()
                    top_neighbor_label = labeled_matrix[i - 1, j].item()

                    if left_neighbor_label > 0:
                        labeled_matrix[i, j] = left_neighbor_label
                    elif top_neighbor_label > 0:
                        labeled_matrix[i, j] = top_neighbor_label
                    else:
                        # Found a brand new isolated highlighted component!
                        labeled_matrix[i, j] = current_label
                        current_label += 1

        unique_labels = torch.unique(labeled_matrix)
        unique_labels = unique_labels[unique_labels != 0]  # drop background 0s

        if len(unique_labels) <= 1:
            return x

        largest_label = None
        max_pixels = 0
        for label in unique_labels:
            pixel_count = (labeled_matrix == label).sum().item()
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                largest_label = label

        cleaned_image = image.clone()
        cleaned_image[labeled_matrix != largest_label] = 0
        return cleaned_image.unsqueeze(0).unsqueeze(0)

    def _crop(self, return_input_t: bool = False) -> list:
        batch_idx = 0
        channel_idx = 0

        active_indices = (self.cam_thresholded[batch_idx, channel_idx] > self.inner_threshold).nonzero()
        if active_indices.numel() == 0:
            print("Warning: No pixels found above the inner_threshold! Returning original CAM.")
            return [self.cam_thresholded, self.cam_thresholded]

        min_row_index = active_indices[:, 0].min().item()
        max_row_index = active_indices[:, 0].max().item()
        min_col_index = active_indices[:, 1].min().item()
        max_col_index = active_indices[:, 1].max().item()
        crop_height = max_row_index - min_row_index + 1
        crop_width = max_col_index - min_col_index + 1

        #print(
        #    f"Bounding box calculated -> Row: {min_row_index} to {max_row_index} (H: {crop_height}), Col: {min_col_index} to {max_col_index} (W: {crop_width})")


        cam_cropped = crop(self.cam_thresholded, min_row_index, min_col_index, crop_height, crop_width)
        #cam_cropped = self.keep_largest_component(cam_cropped)

        if return_input_t:
            input_tensor_cropped = crop(torch.from_numpy(self.input_t), min_row_index, min_col_index, crop_height, crop_width)
            return [cam_cropped, self.cam_thresholded, input_tensor_cropped]
        else:
            return [cam_cropped, self.cam_thresholded]


c1 = Cropper(t1, c1_thr, 0.3)
croppedCAM1, CAM1, INPUT_TCROPPED1 = c1._crop(return_input_t=True)
c2 = Cropper(t2, c2_thr, 0.3)
croppedCAM2, CAM2, INPUT_TCROPPED2 = c2._crop(return_input_t=True)
mask_cropped1 = INPUT_TCROPPED1 * croppedCAM1
mask_cropped2 = INPUT_TCROPPED2 * croppedCAM2

#img = torchvision.transforms.ToPILImage()(torch.from_numpy(t1))
#print(img.mode) is RGB so 24 bit per pixel
index = np.random.randint(100)
fig, axs = plt.subplots(2, 4)
axs[0,0].imshow(np.permute_dims(t1, (1,2,0)), cmap='inferno')
axs[0,1].imshow(CAM1.squeeze(0).permute(1, 2, 0).numpy(), cmap="inferno")
axs[0,2].imshow(croppedCAM1.squeeze(0).permute(1, 2, 0).numpy(), cmap='inferno')
axs[0,3].imshow(mask_cropped1.squeeze(0).permute(1, 2, 0).numpy())
axs[1,0].imshow(np.permute_dims(t2, (1,2,0)), cmap='inferno')
axs[1,1].imshow(CAM2.squeeze(0).permute(1, 2, 0).numpy(), cmap="inferno")
axs[1,2].imshow(croppedCAM2.squeeze(0).permute(1, 2, 0).numpy(), cmap='inferno')
axs[1, 3].imshow(mask_cropped2.squeeze(0).permute(1, 2, 0).numpy())
for ax in axs.flatten():
    ax.axis('off')
plt.savefig(f"all_images-classes{class1}{class2}-{index}-F{F}-beta{beta}-commonF.pdf", bbox_inches='tight', dpi=300)
plt.show()

x1, y1, _ = mask_cropped1.squeeze(0).permute(1,2,0).numpy().shape
x2, y2, _ = mask_cropped2.squeeze(0).permute(1,2,0).numpy().shape
minx, miny = min(x1, x2), min(y1, y2)
mask_min1 = mask_cropped1.squeeze(0).permute(1,2,0).numpy()[:minx, :miny]
mask_min2 = mask_cropped2.squeeze(0).permute(1,2,0).numpy()[:minx, :miny]
fig, axs = plt.subplots(1, 2)
axs[0].imshow(mask_min1)
axs[1].imshow(mask_min2)
for ax in axs:
    ax.axis('off')
plt.savefig(f"comparison_2masks_index-classes{class1}{class2}-{index}-F{F}-beta{beta}-commonF.pdf", bbox_inches='tight', dpi=300)
plt.show()

t_mask1 = torch.from_numpy(mask_min1)
t_mask2 = torch.from_numpy(mask_min2)
t_mask1 = t_mask1.permute(2, 0, 1)
t_mask2 = t_mask2.permute(2, 0, 1)
t_mask1 = t_mask1.unsqueeze(0)
t_mask2 = t_mask2.unsqueeze(0)
#print("SSIM", _calculateSSIM(t_mask1, t_mask2))
print("T mask 1", t_mask1.shape)
print("T mask 2", t_mask2.shape)

"""plt.imshow(t_mask1.squeeze(0).permute(1,2,0).numpy())
plt.show()
plt.imshow(t_mask2.squeeze(0).permute(1,2,0).numpy())
plt.show()"""
#combined_active_mask = (t_mask1 > 0) & (t_mask2 > 0)
#print(combined_active_mask.shape)
#active_pixels_1 = t_mask1.reshape(1, 3, -1)
#active_pixels_2 = t_mask1.reshape(1, 3, -1)
#print("pixels", active_pixels_1.shape)
#active_pixels_1 = t_mask1[combined_active_mask, :, :, :]
#active_pixels_2 = t_mask2[combined_active_mask]


#pcc = torch.corrcoef(torch.stack([active_pixels_1, active_pixels_2]))[0, 1]
#ssim_active = custom_ssim(active_pixels_1, active_pixels_2)
#print("True SSIM on Active Region Only:", ssim_active.item())
print("SSIM on Active Region Only:", _calculateSSIM(t_mask1, t_mask2))



