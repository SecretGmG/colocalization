from typing import Dict, Literal
import itertools
import numpy as np
import skimage as ski
import pandas as pd

def get_labels(mask : np.ndarray) -> np.ndarray:
    """
    Gets fully a list of fully connected components from a bitmask.
    """
    return np.asarray(ski.measure.label(mask))

def plot_labels(labels : np.ndarray) -> None:
    # Visualize labeled image with label numbers at centroids
    import matplotlib.pyplot as plt
    
    # Assuming 'labels' is your labeled image (e.g., from skimage.measure.label)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.nipy_spectral # type: ignore
    im = ax.imshow(labels, cmap=cmap)
    
    # Compute region properties
    regions = ski.measure.regionprops(labels)
    for region in regions:
        # Place label number at centroid
        y, x = region.centroid
        ax.text(x, y, str(region.label-1), color='white', fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_title('Labeled regions with label numbers')
    ax.axis('off')
    plt.show()


def extract(stack : np.ndarray, mask : np.ndarray, apply_filter : bool = False) -> np.ndarray:
    """
    Extracts the stack corresponding to the mask.
    """
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    
    if apply_filter:
        stack=stack*mask[..., np.newaxis]
    
    return stack[x0:x1, y0:y1, :]


def extract_stacks(stack : np.ndarray, labels : np.ndarray, apply_filter : bool = False) -> list[np.ndarray]:
    """
    Extracts the stacks corresponding to the fully connected components in the mask.
    """
    stacks = []
    for i in range(1, labels.max() + 1):
        mask = labels == i
        extracted_stack = extract(stack, mask, apply_filter)
        stacks.append(extracted_stack)
    return stacks

def normalize_stack(stack : np.ndarray, mode = Literal["minmax", "zscore"]) -> np.ndarray:
    """
    Normalizes a stack to [0,1].
    """
    if mode == "minmax":
        return (stack - stack.min()) / (stack.max() - stack.min())
    elif mode == "zscore":
        return (stack - stack.mean()) / (stack.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    

def gather_coefficients(stacks: list[np.ndarray]) -> Dict[str, Dict[tuple[int, int], np.ndarray]]:
    """
    Gathers statistics from the stacks.
    """
    combos = list(itertools.combinations(range(stacks[0].shape[-1]), 2))
    moc = {(i, j): np.ndarray([]) for i, j in combos}
    pearson = {(i, j): np.ndarray([]) for i, j in combos}

    for i, stack in enumerate(stacks):
        for i, j in combos:
            moc[(i, j)] = np.append(moc[(i, j)], ski.measure.manders_overlap_coeff(stack[..., i], stack[..., j]))
            pearson[(i, j)] = np.append(pearson[(i, j)], ski.measure.pearson_corr_coeff(stack[..., i], stack[..., j]))

    return {"MOC": moc, "Pearson": pearson}