from typing import List

import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def plot(
        frame_idx: int,
        scores: List[float],
        losses: List[float]
):
    """Plot the training progresses."""
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
    plt.pause(0.001)
