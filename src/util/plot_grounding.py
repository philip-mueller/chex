from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch.nn.functional as F
import torch
import textwrap as twp

import wandb

from util.plot_utils import plot_img_with_bounding_boxes


def plot_grounding(model_output: 'ChEX', max_samples: int = 10):
    N = len(model_output.sample_id)
    if max_samples is None or max_samples > N:
        max_samples = N
    
    figs = []
    for i in range(max_samples):
        output_i = model_output[i]
        fig = plot_grounding_sample(output_i.x, output_i.sentences,
                                output_i.encoded_img.patch_features, 
                                output_i.encoded_sentences.sentence_features, output_i.encoded_sentences.sentence_mask,
                                output_i.grounding.boxes, output_i.grounding.multiboxes,
                                output_i.grounding.box_features)
        figs.append(wandb.Image(fig))
    return figs


def plot_grounding_sample(img, sentences,
                          patch_features, 
                          sentence_features, sentence_mask, 
                          boxes, multiboxes, box_features):
    n_sentences = len(sentences)
    assert sentence_mask.sum() == n_sentences, f'Expected {n_sentences} sentences, got {sentence_mask.sum()}'
    img_shape = img.shape
    img = img.numpy()

    sentence_features = sentence_features[sentence_mask]
    box_features = box_features[sentence_mask]
    if multiboxes is not None:
        boxes = multiboxes[sentence_mask].numpy()
        S, R, _ = boxes.shape
    else:
        boxes = boxes[sentence_mask].numpy()
        S, _ = boxes.shape
    
    # Similarities and neighbors
    # (S x S_r)
    sent_region_l2_pairwise = torch.cdist(sentence_features, box_features, p=2)
    # (S)
    sentence_region_l2 = sent_region_l2_pairwise.diagonal()
    # (S x S_r)
    sent_region_cos_pairwise = torch.nn.functional.cosine_similarity(sentence_features.unsqueeze(1), box_features.unsqueeze(0), dim=-1)
    # (S)
    sentence_region_cos = sent_region_cos_pairwise.diagonal()
    # (S)
    region_rank = (sent_region_cos_pairwise > sentence_region_cos.unsqueeze(-1)).sum(dim=-1) + 1
    sentence_rank = (sent_region_cos_pairwise > sentence_region_cos.unsqueeze(0)).sum(dim=0) + 1
    # (S)
    sentence_region_neighbor = sent_region_cos_pairwise.argmax(dim=-1) + 1
    region_sentence_neighbor = sent_region_cos_pairwise.argmax(dim=0) + 1

    *shape_patch, d = patch_features.shape
    # (S x H*W)
    sentence_patch_cos_pairwise = torch.nn.functional.cosine_similarity(sentence_features.unsqueeze(1), patch_features.view(-1, d).unsqueeze(0), dim=-1)
    # (H*W)
    patch_sentence_neighbor = sentence_patch_cos_pairwise.argmax(dim=0) + 1
    patch_sentence_neighbor = patch_sentence_neighbor.view(*shape_patch)
    patch_sentence_neighbor_upsampled = F.interpolate(patch_sentence_neighbor[None, None, ...].float(), size=img_shape, mode='nearest-exact').squeeze()
    upsample_factor = np.array(img_shape) / np.array(shape_patch)

    # To numpy
    sentence_region_l2 = sentence_region_l2.numpy()
    sentence_region_cos = sentence_region_cos.numpy()
    sentence_rank = sentence_rank.numpy()
    region_rank = region_rank.numpy()
    region_sentence_neighbor = region_sentence_neighbor.numpy()
    sentence_region_neighbor = sentence_region_neighbor.numpy()
    patch_sentence_neighbor = patch_sentence_neighbor.numpy()
    patch_sentence_neighbor_upsampled = patch_sentence_neighbor_upsampled.numpy()

    # IDs and colors, text wrap
    sentence_ids = list(range(1, n_sentences + 1))
    sentence_colors = list(mcolors.TABLEAU_COLORS.values())[:n_sentences] if n_sentences <= 10 else matplotlib.color_sequences['tab20'][:n_sentences]
    sentences = [twp.fill(s, 70) for s in sentences]
    cmap = mcolors.LinearSegmentedColormap.from_list("cmap_name", sentence_colors, N=n_sentences)

    # Plotting...
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1, 1])

    ax_boxes = fig.add_subplot(gs[0, 0])
    ax_boxes.set_xticks([])
    ax_boxes.set_yticks([])
    sentence_ids_array = np.array(sentence_ids)  # (S)
    if multiboxes is not None:
        # (S x R)
        sentence_ids_array = sentence_ids_array[:, None].repeat(R, axis=1)
    boxes_with_ids = np.concatenate([boxes, sentence_ids_array[:, None] - 1], axis=1) if multiboxes is None \
        else np.concatenate([boxes, sentence_ids_array[:, :, None] - 1], axis=2)
    
    if multiboxes is not None:
        boxes_with_ids = boxes_with_ids.reshape(-1, 5)
    plot_img_with_bounding_boxes(ax_boxes, class_names=sentence_ids, 
                                img=img, target_list=boxes_with_ids, plot_pred=False,
                                class_cmap=sentence_colors)
        
    boxes_upper_left = (boxes_with_ids[:, :2] - boxes_with_ids[:, 2:4] / 2) * img_shape[::-1]
    boxes_lower_right = (boxes_with_ids[:, :2] + boxes_with_ids[:, 2:4] / 2) * img_shape[::-1]
    ax_boxes.set_xlim(min(0, boxes_upper_left[:, 0].min()), max(img_shape[1], boxes_lower_right[:, 0].max()))
    ax_boxes.set_ylim(max(img_shape[0], boxes_lower_right[:, 1].max()), min(0, boxes_upper_left[:, 1].min()))

    ax_patch_neighbors = fig.add_subplot(gs[0, 1])
    ax_patch_neighbors.imshow(img, cmap='gray')
    ax_patch_neighbors.imshow(patch_sentence_neighbor_upsampled, cmap=cmap, vmin=1, vmax=n_sentences, alpha=0.6)
    ax_patch_neighbors.set_xticks([])
    ax_patch_neighbors.set_yticks([])
    for y, neighbors in enumerate(patch_sentence_neighbor):
        y = (y + 0.5) * upsample_factor[0] - 1
        for x, neighbor in enumerate(neighbors):
            x = (x + 0.5) * upsample_factor[1] - 1
            ax_patch_neighbors.text(x, y, neighbor, ha='center', va='center', color='w')

    ax_sent_barplots = fig.add_subplot(gs[0, -1])
    sentence_ids = np.array(sentence_ids)
    bar_pos = np.linspace(-0.3, 0.3, n_sentences)
    
    ax_sent_barplots.bar(bar_pos, sentence_region_cos, color=sentence_colors, width=0.6 / n_sentences)
    ax_sent_barplots.set_xticks([0])
    ax_sent_barplots.set_xticklabels(['cos'])

    tab_data = [
        *zip(sentence_ids, sentences, sentence_rank, region_rank, sentence_region_l2.round(2), sentence_region_cos.round(2), region_sentence_neighbor, sentence_region_neighbor)
    ]
    tab_colors = [
        [col, col, col, col, col, col, sentence_colors[neighbor_s - 1], sentence_colors[neighbor_r - 1]] 
        for col, neighbor_s, neighbor_r in zip(sentence_colors, region_sentence_neighbor, sentence_region_neighbor)
    ]

    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    ax_table.axis('tight')
    table = ax_table.table(
        cellText=tab_data, 
        cellLoc='center', 
        colLoc='center',
        loc='center',
        cellColours=tab_colors,
        colLabels=['ID', 'Sentence', 'Rank s', 'Rank r', 'L2', 'cos', '1-NN s', '1-NN r'],
        colWidths=[0.04, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    n_cols = 8

    # Apply custom cell renderer to each cell
    for i in range(len(sentences) + 1):
        cell = table.get_celld()[(i, 1)]
        cell.set_text_props(horizontalalignment='left')
        cell.PAD = 0.01
        lines = len(cell.get_text().get_text().splitlines())
        for j in range(n_cols):
            cell = table.get_celld()[(i, j)]
            cell.set_height(lines * 0.11)

    return fig

