from typing import Literal
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

def get_labels(mask : np.ndarray) -> np.ndarray:
    """
    Gets fully a list of fully connected components from a bitmask.
    """
    return np.asarray(ski.measure.label(mask))

def plot_labels(labels : np.ndarray) -> None:
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.nipy_spectral # type: ignore
    im = ax.imshow(labels, cmap=cmap)
    
    regions = ski.measure.regionprops(labels)
    for region in regions:
        y, x = region.centroid
        ax.text(x, y, str(region.label), color='white', fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_title('Labeled regions with label numbers')
    ax.axis('off')
    plt.show()


def extract(stack : np.ndarray, mask : np.ndarray, apply_filter : bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts the stack corresponding to the mask.
    """
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    
    if apply_filter:
        stack=stack*mask[..., np.newaxis]
    
    return stack[x0:x1, y0:y1, :], mask[x0:x1, y0:y1]


def extract_stacks(stack : np.ndarray, labels : np.ndarray, apply_filter : bool = False) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Extracts the stacks corresponding to the fully connected components in the mask.
    """
    stacks = []
    masks = []
    indices = []
    for i in np.unique(labels):
        if i == 0:
            continue
        mask = labels == i
        extracted_stack, extracted_mask = extract(stack, mask, apply_filter)
        indices.append(i)
        stacks.append(extracted_stack)
        masks.append(extracted_mask)
        
    sorted_indices = np.argsort([mask.sum() for mask in masks])[::-1]
    stacks = [stacks[i] for i in sorted_indices]
    masks = [masks[i] for i in sorted_indices]
    indices = [indices[i] for i in sorted_indices]
    
    return indices, stacks, masks

def normalize_stack(stack : np.ndarray, mode = Literal["minmax", "zscore"]) -> np.ndarray:
    """
    applies normalization to each channel in a stack
    """
    if mode == "minmax":
        return (stack - stack.min(axis=(0, 1))) / (stack.max(axis=(0, 1)) - stack.min(axis=(0, 1)) + 1e-8)
    elif mode == "zscore":
        return (stack - stack.mean(axis=(0, 1))) / (stack.std(axis=(0, 1)) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    

def manders_overlap_coeff(stacks: list[np.ndarray], masks: list[np.ndarray], ch1: int, ch2: int) -> list[float]:
    """
    Gathers statistics from the stacks.
    """
    moc = []
    for stack, mask in zip(stacks, masks):
        moc.append(ski.measure.manders_overlap_coeff(stack[..., ch1], stack[..., ch2], mask))

    return moc

def pearson_corr_coeff(stacks: list[np.ndarray], masks: list[np.ndarray], ch1: int, ch2: int) -> list[float]:
    """
    Gathers statistics from the stacks.
    """
    pearson = []
    for stack, mask in zip(stacks, masks):
        pearson.append(ski.measure.pearson_corr_coeff(stack[..., ch1], stack[..., ch2], mask)[0])

    return pearson