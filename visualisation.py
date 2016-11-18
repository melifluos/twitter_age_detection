"""
Visualise age data
"""

from sklearn.manifold import TSNE
import utils
import seaborn as sb
import pandas as pd

model = TSNE(n_components=2, random_state=0)

x_path = 'resources/test/X_large.p'
y_path = 'resources/test/y_large.p'
emd_path = 'resources/test/test64.emd'

X, y = utils.read_data(x_path, y_path, threshold=10)
embedding = model.fit_transform(X.toarray())
X1 = utils.read_embedding(emd_path)


sb.set_context("notebook", font_scale=1.1)
sb.set_style("ticks")

df = pd.DataFrame(data=X, header=['x1', 'x2'])
df['label'] = y

sb.lmplot('x', 'y',
          data=df,
          fit_reg=False,
          dropna=True,
          hue="label",
          scatter_kws={"marker": "D",
                       "s": 100})
