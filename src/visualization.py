"""Shared point-cloud plotting utilities."""
from pathlib import Path
import numpy as np
from .io_utils import COLORS


def configure() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42})


def color(label: int) -> str:
    return COLORS[label+1] if -1 <= label <= 10 else '#606060'


def setup_3d(ax, xyz: np.ndarray) -> None:
    low,high = xyz.min(0),xyz.max(0)
    middle = (low+high)/2
    radius = max(float(np.ptp(xyz,axis=0).max())*.55,1e-6)
    ax.set(xlim=(middle[0]-radius,middle[0]+radius),ylim=(middle[1]-radius,middle[1]+radius),
           zlim=(middle[2]-radius,middle[2]+radius),xlabel='X',ylabel='Y',zlabel='Z')
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=18,azim=-55)


def save_figure(fig, destination: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(destination,dpi=180,facecolor='white')
    plt.close(fig)
