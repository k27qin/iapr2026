#########################################################################################################
################################### This are functions for lab 01.   ####################################
###################################          DO NOT MODIFY!          ####################################
#########################################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Callable
from datetime import datetime


def show_introduction_figure():
    # Define path
    path_he = "../data/data_lab_01/tcga_ag.png"
    # Check if folder and image exist
    assert os.path.exists(path_he), "Image not found, please check directory structure"

    # Load image
    img_he = np.array(Image.open(path_he))

    # Display image
    plt.imshow(img_he)
    plt.axis('off')

    # Annotations
    plt.annotate('Tumor', xy=(1250, 540), xytext=(2240, 540), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Other', xy=(1840, 387), xytext=(2240, 387), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.tight_layout()
    return img_he


# Plot color space distribution 
def plot_colors_histo(
    img: np.ndarray,
    func: Callable,
    labels: list[str],
):
    """
    Plot the original image (top) as well as the channel's color distributions (bottom).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    func: Callable
        A callable function that extracts D channels from the input image
    labels: list of str
        List of D labels indicating the name of the channel
    """

    # Extract colors
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    # Use random seed to downsample image colors (increase run speed - 10%)
    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1
    
    # Plot base image
    ax = fig.add_subplot(gs[:2, :])
    ax.imshow(img)
    # Remove axis
    ax.axis('off')
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])

    # Plot channel distributions
    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("{} vs {}".format(labels[0], labels[1]))
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))
        
    plt.tight_layout()


# Plot color space distribution 
def plot_thresholded_image(
    img: np.ndarray,
    func: Callable,
    title: str,
):
    """
    Plot the original image and its thresholded version

    Args
    ----
    img: np.ndarray (M, N, 3)
        Input image of shape MxNx3.
    func: Callable
        Thresholded image.
    title: str
        Title of the plot
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img)
    axes[1].imshow(func(img), interpolation=None)
    [a.axis('off') for a in axes]
    plt.suptitle(title)
    plt.tight_layout()


def plot_images(
    imgs: np.ndarray,
    sizes: list[int],
    title: str,
):
    """
    Plot multiple images. The title of each subplot is defined by the disk_size elements.

    Args
    ----
    imgs: np.ndarray (D, M, N)
        List of D images of size MxN.
    disk_sizes: list of int
        List of D int that are the size of the disk used for the operation
    title:
        The overall title of the figure
    """
    D = len(imgs)
    ncols = int(np.ceil(D/2))
    _, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(10, 4*ncols))
    
    # Remove axis
    axes = axes.ravel()
    [ax.axis('off') for ax in axes]
    
    for i in range(D):
        axes[i].imshow(imgs[i])
        axes[i].set_title("Size: {}".format(sizes[i]))
    
    plt.suptitle(title)
    plt.tight_layout()

def plot_close_open(img_th, apply_closing, apply_opening):
    disk_sizes = [1, 2, 5, 10]
    imgs_closing = []
    imgs_opening = []

    # Apply opening and closing to masked image 
    for d in disk_sizes:
        imgs_closing.append(apply_closing(img_th, d))
        imgs_opening.append(apply_opening(img_th, d))

    # Plot
    plot_images(imgs=imgs_closing, sizes=disk_sizes, title="Closing")
    plot_images(imgs=imgs_opening, sizes=disk_sizes, title="Opening")


def plot_remove_holes_objects(img_th, remove_holes, remove_objects):
    # Define area sizes
    sizes = [10, 50, 100, 500]
    imgs_holes = []
    imgs_objects = []

    # Remove holes and objects from masked image 
    for d in sizes:
        imgs_holes.append(remove_holes(img_th, d))
        imgs_objects.append(remove_objects(img_th, d))
        
    # Plot results    
    plot_images(imgs=imgs_holes, sizes=sizes, title="Remove small holes")
    plot_images(imgs=imgs_objects, sizes=sizes, title="Remove small objects")


def plot_morphology_best(
    img_source: np.ndarray,
    img_best: np.ndarray,
):
    """
    Plot the original images beside the best tumor estimation

    Args
    ----
    img_source: np.ndarray (M, N, 3)
        RGB source image.
    img_best: np.ndarray (M, N)
        Best thresholded image.
    """
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    # Remove axis
    axes = axes.ravel()
    [ax.axis('off') for ax in axes]
    axes[0].imshow(img_source)
    axes[1].imshow(img_best)
    axes[0].set_title("Source")
    axes[1].set_title("Tumor estimation")
    
    plt.tight_layout()


def plot_region_growing(
    seeds: list[tuple],
    img: np.ndarray,
    func: Callable,
    iters: list[int],
    **kwargs
):
    """
    Plot the region growing results based on seeds, function and iterations
    
    Args
    ----
    seeds: list of tuple
        List of seed points
    img: np.ndarray (M, N, C)
        RGB image of size M, N, C
    func: callable
        Region growing function
    iters: list of ints
        Number of iteration to plot
    """

    # Define plot size
    n = len(iters) + 1
    n_rows = np.ceil(n // 2).astype(int)
    _, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
    axes = axes.ravel()
    [a.axis('off') for a in axes]   

    # Reference image
    axes[0].imshow(img)
    axes[0].set_title("Input image")

    # Plot all iterations
    for i, it in enumerate(iters):
        t1 = datetime.now()
        img_rg = func(seeds=seeds, img=img, n_max=iters[i], **kwargs)
        # Compute time difference in seconds
        t2 = datetime.now()
        seconds = (t2 - t1).total_seconds()
        axes[i+1].imshow(img_rg)
        axes[i+1].set_title("RG {} iter in {:.2f} seconds".format(iters[i], seconds))
                            
    plt.tight_layout()
    return img_rg


def plot_tumor_region_growing(img_he, region_growing, **kwargs):
    # Set manual seeds (located inside tumor blobs)
    seeds = [
        (392, 280), (581, 199), (859, 119), (184, 777), (441, 718), (1039, 526), 
        (1255, 509), (1252, 990), (658, 1136), (732, 2000), (207, 1971), (192, 1741), (1218, 1007), (1314, 1367),
    ]

    return plot_region_growing(
        seeds=seeds,
        img=img_he,
        func=region_growing,
        iters=[20, 100, 2000],
        **kwargs
    )


def plot_final_comparison(img_he, img_th, img_best_morpho, img_grow):
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Remove axis
    axes = axes.ravel()
    [a.axis('off') for a in axes]

    # Original image
    axes[0].imshow(img_he)
    # Detections
    axes[1].imshow(img_th)
    axes[1].set_title("HSV Threshold")
    axes[2].imshow(img_best_morpho)
    axes[2].set_title("HSV Thresh + Morphology")
    axes[3].imshow(img_grow)
    axes[3].set_title("Region Growing")
    plt.tight_layout()
    

def show_exo2_figure():
    # Load image
    path_he2 = "../data/data_lab_01/tcga_blood.png"
    # Check if folder and image exist
    assert os.path.exists(path_he2), "Image not found, please check directory structure"
    img_he2 = np.array(Image.open(path_he2))

    # Display image
    plt.figure(figsize=(14, 7))
    plt.imshow(img_he2)
    plt.axis('off')
    plt.tight_layout()
    
    h, w = img_he2.shape[:2]
    plt.xlim(0, w + 280)
    ann = dict(
    arrowprops=dict(arrowstyle='-|>', color='black', lw=2, shrinkA=0, shrinkB=0),
    ha='left',
    va='center',
    annotation_clip=False
    )

    label_x = w + 40

    plt.annotate('Background', xy=(315, 150), xytext=(label_x, 150), **ann)
    plt.annotate('Blood',      xy=(206, 122), xytext=(label_x, 122), **ann)
    plt.annotate('Mucin',      xy=(512, 404), xytext=(label_x, 404), **ann)


    # Annotations
    #plt.annotate('Background', xy=(315, 150), xytext=(730, 70), arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.annotate('Blood', xy=(206, 122), xytext=(730, 190), arrowprops=dict(facecolor='black', shrink=0.05))
    #plt.annotate('Mucin', xy=(512, 404), xytext=(730, 560), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.tight_layout()

    return img_he2


def plot_results(
    img: np.ndarray,
    mask_blood: np.ndarray,
    mask_mucin: np.ndarray,
):
    """
    Plot the blood and mucin detection as well as the estimated area in pixels
    
    Args
    ----
    img: np.ndarray (M, N, C)
        Reference RGB input image
    mask_blood: np.ndarray (M, N)
        Estimation mask of the blood aggregates
    mask_mucin: np.ndarray (M, N)
        Estimation mask of the mucin area
    """
    
    _, axes = plt.subplots(1, 3, figsize=(16, 6))
    [a.axis("off") for a in axes]

    area_blood = np.sum(mask_blood)
    area_mucin = np.sum(mask_mucin)
    
    # Image
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    # Mask blood
    axes[1].imshow(img)
    axes[1].imshow(mask_blood, alpha=0.8)
    axes[1].set_title("Blood detection (area: {:.0f})".format(area_blood))
    # Mask mucin
    axes[2].imshow(img)
    axes[2].imshow(mask_mucin, alpha=0.8)
    axes[2].set_title("Mucin detection (area: {:.0f})".format(area_mucin))
    plt.tight_layout()

