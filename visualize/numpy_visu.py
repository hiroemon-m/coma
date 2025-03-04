import numpy as np
import pandas as pd

path = "experiment_data/action_space/variable/space=10%/train_persona_ration.npy"

data = np.load(path)
visualize_data = pd.DataFrame(data[0])


print(visualize_data.sum())