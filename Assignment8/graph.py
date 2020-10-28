import matplotlib.pyplot as plt
import numpy as np

loss = []
with open("loss", 'r') as f:
    content = f.readlines()[0]
    loss = eval(content)

x = np.array(loss)
y = [i for i in range(len(loss))]
plt.figure(figsize=(6, 4))
plt.plot(x, y, linewidth=1)
plt.xlabel("epochs")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
plt.ylabel("loss")
plt.title("loss")  # title：设置子图的标题。
# plt.savefig('quxiantu.png',dpi=120,bbox_inches='tight')
plt.show()
# plt.close()
