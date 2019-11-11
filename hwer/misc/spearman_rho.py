import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import spearmanr

x = np.arange(2, 20, 0.05)
y1 = np.log(x)
y2 = np.square(x)
y3 = np.exp(x)
y4 = np.sqrt(x)
y5 = np.exp(x) + (np.exp(x)/1.25)*np.sin(x*(2*np.pi))

ydf1 = pd.DataFrame({"x":x, "y":y1, "hue":["Log: Coerrelation = %.4f" % np.corrcoef(x, y1)[0][1]]*len(x)})
ydf2 = pd.DataFrame({"x":x, "y":y2, "hue":["Square: Coerrelation = %.4f" % np.corrcoef(x, y2)[0][1]]*len(x)})
ydf3 = pd.DataFrame({"x":x, "y":y3, "hue":["e^x: Coerrelation = %.4f" % np.corrcoef(x, y3)[0][1]]*len(x)})
ydf4 = pd.DataFrame({"x":x, "y":y4, "hue":["Sqrt: Coerrelation = %.4f" % np.corrcoef(x, y4)[0][1]]*len(x)})
ydf5 = pd.DataFrame({"x":x, "y":y5, "hue":["e^x + sine: Coerrelation = %.4f" % np.corrcoef(x, y5)[0][1]]*len(x)})

df = pd.concat((ydf1, ydf2, ydf3, ydf4, ydf5))



plt.figure(figsize=(8,8))
sns.lineplot(x="x",y="y",hue="hue", data=df)
plt.title(" With Pearson Correlation ")
plt.semilogy()
plt.show()


# ######

ydf1 = pd.DataFrame({"x":x, "y":y1, "hue":["Log: SpearmanR = %.4f" % spearmanr(x, y1)[0]]*len(x)})
ydf2 = pd.DataFrame({"x":x, "y":y2, "hue":["Square: SpearmanR = %.4f" % spearmanr(x, y2)[0]]*len(x)})
ydf3 = pd.DataFrame({"x":x, "y":y3, "hue":["e^x: SpearmanR = %.4f" % spearmanr(x, y3)[0]]*len(x)})
ydf4 = pd.DataFrame({"x":x, "y":y4, "hue":["Sqrt: SpearmanR = %.4f" % spearmanr(x, y4)[0]]*len(x)})
ydf5 = pd.DataFrame({"x":x, "y":y5, "hue":["e^x + sine: SpearmanR = %.4f" % spearmanr(x, y5)[0]]*len(x)})

df = pd.concat((ydf1, ydf2, ydf3, ydf4, ydf5))



plt.figure(figsize=(8,8))
sns.lineplot(x="x",y="y",hue="hue", data=df)
plt.title(" With SpearmanR ")
plt.semilogy()
plt.show()



