# %%
# Loading extension for reloading editable packages (pip install -e .)
%load_ext autoreload

# %%
# Reloading editable packages.
%autoreload
from lab1.main import get_results

# %%
def visualize_states(states_stats):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('_mpl-gallery')

    # make data:
    x = 0.5 + np.arange(8)
    y = states_stats

    # plot
    fig, ax = plt.subplots()

    ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


get_results()
visualize_states()
