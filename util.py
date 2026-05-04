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


def extract_stacks(stack : np.ndarray, labels : np.ndarray, apply_filter : bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Extracts the stacks corresponding to the fully connected components in the mask.
    """
    stacks = []
    masks = []
    for i in range(1, labels.max() + 1):
        if i == 0:
            continue
        mask = labels == i
        extracted_stack, extracted_mask = extract(stack, mask, apply_filter)
        stacks.append(extracted_stack)
        masks.append(extracted_mask)
        
    return stacks, masks

def normalize_stack(stack : np.ndarray, mode : Literal["minmax", "zscore"] = "minmax") -> np.ndarray:
    """
    applies normalization to each channel in a stack
    """
    if mode == "minmax":
        return (stack - stack.min(axis=(0, 1))) / (stack.max(axis=(0, 1)) - stack.min(axis=(0, 1)) + 1e-8)
    elif mode == "zscore":
        return (stack - stack.mean(axis=(0, 1))) / (stack.std(axis=(0, 1)) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def filter_stacks(extracted_stacks, extracted_masks, stack_channel=0, figsize=(8, 4)):
    accepted_stacks = []
    current_idx = 0

    out = widgets.Output()
    status = widgets.HTML()

    accept_btn = widgets.Button(description="Accept", button_style="success")
    decline_btn = widgets.Button(description="Decline", button_style="danger")

    def show_stack(i):
        with out:
            clear_output(wait=True)

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(f"Stack {i}")

            axes[0].imshow(extracted_stacks[i][..., stack_channel], cmap="gray")
            axes[0].set_title("Stack")
            axes[0].axis("off")

            axes[1].imshow(extracted_masks[i], cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

    def finish():
        with out:
            clear_output(wait=True)
            print("Done.")
            print("Accepted indices:", accepted_stacks)

        accept_btn.disabled = True
        decline_btn.disabled = True
        status.value = "<b>Finished</b>"

    def next_stack():
        nonlocal current_idx

        if current_idx < len(extracted_stacks):
            show_stack(current_idx)
            status.value = f"<b>{current_idx + 1} / {len(extracted_stacks)}</b>"
        else:
            finish()

    def on_accept(_):
        nonlocal current_idx
        accepted_stacks.append(current_idx)
        current_idx += 1
        next_stack()

    def on_decline(_):
        nonlocal current_idx
        current_idx += 1
        next_stack()

    accept_btn.on_click(on_accept)
    decline_btn.on_click(on_decline)

    display(
        widgets.VBox(
            [
                status,
                widgets.HBox([accept_btn, decline_btn]),
                out,
            ]
        )
    )

    next_stack()
    
    return accepted_stacks


def manders_overlap_coeff(stacks: list[np.ndarray], masks: list[np.ndarray], ch1: int, ch2: int, threshold_ab: float, threshold_ba: float) -> np.ndarray:
    """
    returns the MOC and M1/M2 coefficients for each stack and mask. The thresholds are used to binarize the channels for the M1/M2 coefficients.
    """
    moc = []
    for stack, mask in zip(stacks, masks):
        moc.append([ski.measure.manders_overlap_coeff(stack[..., ch1], stack[..., ch2], mask),
        ski.measure.manders_coloc_coeff(stack[..., ch1], stack[..., ch2]>threshold_ab, mask),
        ski.measure.manders_coloc_coeff(stack[..., ch2], stack[..., ch1]>threshold_ba, mask)])

    return np.asarray(moc)

def pearson_corr_coeff(stacks: list[np.ndarray], masks: list[np.ndarray], ch1: int, ch2: int) -> np.ndarray:
    """
    Gathers statistics from the stacks.
    """
    pearson = []
    for stack, mask in zip(stacks, masks):
        pearson.append(ski.measure.pearson_corr_coeff(stack[..., ch1], stack[..., ch2], mask)[0])

    return np.asarray(pearson)