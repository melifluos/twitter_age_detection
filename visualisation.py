"""
Visualise age data
"""

import utils
from sklearn.manifold import TSNE
import seaborn as sb
import pandas as pd

model = TSNE(n_components=2, random_state=0)

x_path = 'resources/test/X.p'
y_path = 'resources/test/y.p'
emd_path = 'resources/test/test64.emd'

X, y = utils.read_data(x_path, y_path, threshold=10)

target = utils.read_target(y_path)
X1 = utils.read_embedding(emd_path, target, 64)
embedding = model.fit_transform(X1)

#sb.set_context("notebook", font_scale=1.1)
sb.set_style("ticks")

print X1.shape

df = pd.DataFrame(data=embedding, index=None, columns=['x', 'y'])
df['label'] = y

sb.lmplot('x', 'y',
          data=df,
          fit_reg=False,
          hue="label",
          scatter_kws={"marker": "D",
                       "s": 100})
