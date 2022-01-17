# dmosopt visualization routines
# Radar visualization code based on matplotlib gallery example:
# https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from dmosopt.datatypes import Struct

# Default figure configuration
default_fig_options = Struct(figFormat='png', lw=2, figSize=(15,8), fontSize=14, saveFig=None, showFig=True,
                             colormap='jet', saveFigDir=None)



def save_figure(file_name_prefix, fig=None, **kwargs):
    """

    :param file_name_prefix:
    :param fig: :class:'plt.Figure'
    :param kwargs: dict
    """
    fig_options = copy.copy(default_fig_options)
    fig_options.update(kwargs)
    fig_file_path = f'{file_name_prefix}.{fig_options.figFormat}'
    if fig_options.saveFigDir is not None:
        fig_file_path = f'{fig_options.saveFigDir}/{fig_file_path}'
    if fig is not None:
        fig.savefig(fig_file_path)
    else:
        plt.savefig(fig_file_path)


