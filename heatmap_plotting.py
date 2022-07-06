import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import voxel_labelling as vl
import pickle

fn_setup = r"./autoplanner_setups/jezero03_padnwall_1inVoxel_setup.pkl"
fn_plan = r"/home/tyler/MS_Thesis/ProgressDetection/linear_plans/landingPad_blastWall_by8.csv"

with open(fn_setup, 'r') as f_setup:
    print "Loading autoplanner setup file"
    (point_cloud_density, elements, voxel_reference) = pickle.load(f_setup)

print "Importing linear plan"
BIM_order = vl.import_LinearPlan(fn_plan, elements)

mymap = np.array([[248, 105, 107], [255, 235, 132], [99, 190, 123]])
mymap = mymap/255.0

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", mymap)

df = pd.read_csv("./Jezero03_testing_noon/array08/s15/jezero03_noon_array08_s15.csv")
prediction_board = df[df.columns[3:]].to_numpy()

# # # ind_first = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][0]
# # # ind_last = np.where(~np.all(np.isnan(prediction_board), axis=1))[0][-1]

ind_first = 100
ind_last = 130

sns.set(font_scale=1)
linewidths = 1
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

# fig, ax = plt.subplots()
ax = sns.heatmap(prediction_board, cmap=cmap, vmin=0, vmax=1,
                 cbar_kws={
                     #'label': 'Progress Probability',
                     'ticks': [0.0, 0.25, 0.5, 0.75, 1.0]},
                 antialiased=True, linecolor='face', linewidths=0.5)

labels = range(prediction_board.shape[1]+1)
# # # def format_fn(tick_val, tick_pos):
# # #     if int(tick_val) in range(prediction_board.shape[1]):
# # #         return labels[int(tick_val)]
# # #     else:
# # #         return ''

ax.xaxis.set_major_formatter(matplotlib.ticker.IndexFormatter(labels))
#ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.vlines(range(1,prediction_board.shape[1]), *ax.get_ylim(), colors=[0.8]*3, linewidths=linewidths)
step_inds = np.cumsum([len(step) for step in BIM_order])
ax.hlines(step_inds[:-1], *ax.get_xlim(), colors=[0.8]*3, linewidths=linewidths)
plt.ylim([ind_first, ind_last + 1])
#plt.xlim([13, 25])
#plt.xlabel("Step")
#plt.ylabel("Construction Element")
#plt.title("Prediction Heatmap for Step 23")

plt.show()