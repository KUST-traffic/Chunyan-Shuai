"""
===========================
Creating annotated heatmaps
===========================

It is often desirable to show data which depends on two independent
variables as a color coded image plot. This is often referred to as a
heatmap. If the data is categorical, this would be called a categorical
heatmap.

Matplotlib's `~matplotlib.axes.Axes.imshow` function makes
production of such plots particularly easy.

The following examples show how to create a heatmap with annotations.
We will start with an easy example and expand it to be usable as a
universal function.
"""


##############################################################################
#
# A simple categorical heatmap
# ----------------------------
#
# We may start by defining some data. What we need is a 2D list or array
# which defines the data to color code. We then also need two lists or arrays
# of categories; of course the number of elements in those lists
# need to match the data along the respective axes.
# The heatmap itself is an `~matplotlib.axes.Axes.imshow` plot
# with the labels set to the categories we have.
# Note that it is important to set both, the tick locations
# (`~matplotlib.axes.Axes.set_xticks`) as well as the
# tick labels (`~matplotlib.axes.Axes.set_xticklabels`),
# otherwise they would become out of sync. The locations are just
# the ascending integer numbers, while the ticklabels are the labels to show.
# Finally we can label the data itself by creating a `~matplotlib.text.Text`
# within each cell showing the value of that cell.


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = ["Class 1", "Class 2", "Class 3", "Class 4",
              "Class 5", "Class 6", "Class 7","Class 8", "Class 9","sum"]
farmers = ["Class 1", "Class 2", "Class 3", "Class 4",
              "Class 5", "Class 6", "Class 7","Class 8", "Class 9","sum"]

#16 kernal with 70 iterations
harvest1 = np.array([[197, 1, 29, 0, 5, 12, 0, 0, 1],
                    [4,428,8,2,1,13,3,0,18],
                    [16,9,776,85,8,6,4,70,9],
                    [3,4,58,332,1,1,0,50,5],
                    [14,24,20,4,271,5,4,5,6],
                    [2,9,1,1,4,322,14,0,3],
                    [0,8,6,0,15,54,281,0,2],
                    [0,1,12,2,0,4,0,180,0],
                    [2,12,11,16,16,2,1,5,443]])
#        [238,496,921,442,321,419,307,310,487]
#32 kernal with 70 iterations
harvest2 =  np.array([[190,4,26,2,11,7,6,0,1],
                    [4,487,5,3,17,4,8,8,14],
                    [10,7,780,78,14,4,8,52,4],
                    [2,5,50,298,2,1,2,68,6],
                    [10,13,11,2,328,1,17,0,2],
                    [4,1,2,0,1,148,38,0,3],
                    [0,5,4,1,8,2,441,0,4],
                    [1,0,14,0,1,2,2,192,0],
                    [1,11,11,10,22,2,7,0,423]])
                    #[222,533,903,394,404,171,529,320,457]])
#48 kernal with 70 iterations
harvest3 =  np.array([[186,0,31,1,3,6,0,0,2],
                    [4,432,17,1,13,5,2,0,24],
                    [15,6,760,69,18,5,4,5,2],
                    [3,2,76,298,4,2,0,11,11],
                    [18,10,14,2,307,3,11,0,7],
                    [6,1,4,1,6,328,22,0,4],
                    [0,5,5,1,24,45,275,0,2],
                    [0,0,4,10,10,20,0,141,0],
                    [2,8,6,6,6,6,2,0,468]])
                    #[234,464,917,389,391,420,316,157,520]])
#64 kernal with 70 iterations
harvest4= np.array([[180,0,37,0,6,6,0,0,4],
                    [2,441,11,1,8,11,2,0,14],
                    [12,6,805,65,13,9,1,37,7],
                    [0,1,40,344,3,3,0,17,7],
                    [10,16,15,1,317,4,9,6,4],
                    [1,4,3,1,0,333,29,0,3],
                    [0,1,3,2,10,13,319,6,4],
                    [0,0,8,7,0,5,0,240,0],
                    [1,12,11,6,15,8,2,6,434]])
                    #[206,481,938,427,372,392,362,312,477]])

#precision=np.array([])
#50 iterations with 16 kernels
precision1=[0.83,0.87,0.79,0.75,0.88,0.86,0.83,0.74,0.95]
precision2=[0.86,0.91,0.78,0.76,0.84,0.9,0.82,0.77,0.94]
precision3=[0.85,0.93,0.88,0.71,0.84,0.8,0.88,0.58,0.93]
'''
harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])



fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(farmers)))
#ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()
'''

#############################################################################
# Using the helper function code style
# ------------------------------------
#
# As discussed in the :ref:`Coding styles <coding_styles>`
# one might want to reuse such code to create some kind of heatmap
# for different input data and/or on different axes.
# We create a function that takes the data and the row and column labels as
# input, and allows arguments that are used to customize the plot
#
# Here, in addition to the above we also want to create a colorbar and
# position the labels above of the heatmap instead of below it.
# The annotations shall get different colors depending on a threshold
# for better contrast against the pixel color.
# Finally, we turn the surrounding axes spines off and create
# a grid of white lines to separate the cells.


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)
    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
   # ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="minor", bottom=True, left=True)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None),fontsize=8, **kw)
            texts.append(text)

    return texts


##########################################################################
# The above now allows us to keep the actual plot creation pretty compact.
#
'''
fig, ax = plt.subplots()

im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="harvest [t/year]")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.show()
'''
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 12))

harvest1=np.transpose(harvest1)
im, cbar = heatmap(harvest1, vegetables, farmers, ax=ax1, cmap="Greens")
texts = annotate_heatmap(im, valfmt="{x}")

ax1.set_title("16 kernels and 70 iterations")

harvest2=np.transpose(harvest2)
im, cbar = heatmap(harvest2, vegetables, farmers, ax=ax2, cmap="Blues")
texts = annotate_heatmap(im, valfmt="{x}")
ax2.set_title("32 kernels and 50 iterations")

harvest3=np.transpose(harvest3)
im, cbar = heatmap(harvest3, vegetables, farmers, ax=ax3, cmap="Purples")
texts = annotate_heatmap(im, valfmt="{x}")
ax3.set_title("48 kernels and 40 iterations")

harvest4=np.transpose(harvest4)
im, cbar = heatmap(harvest4, vegetables, farmers, ax=ax4 , cmap="Wistia")
texts = annotate_heatmap(im, valfmt="{x}")
ax4.set_title("64 kernels and 40 iterations")

fig.tight_layout()
plt.show()
#eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
fig.savefig('.\heatmap.svg', format='svg',dpi=600)
fig.savefig('.\heatmap.png', format='png',dpi=600)
#############################################################################
# Some more complex heatmap examples
# ----------------------------------
#
# In the following we show the versatility of the previously created
# functions by applying it in different cases and using different arguments.
#