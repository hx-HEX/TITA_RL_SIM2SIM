import numpy as np
from PIL import Image

# 图像大小
w, h = 256, 256
img = np.zeros((h, w), dtype=np.uint8)

# 台阶层数
steps = 6              # 总层数（增加台阶数）
step_height = 255 // steps

# 普通台阶宽度（最后一层单独处理）
base_step_width = w // (steps + 1)

for i in range(steps):
    if i < steps - 1:
        # 前几层等宽
        img[:, i*base_step_width:(i+1)*base_step_width] = i * step_height
    else:
        # 最后一层更宽（占据剩余的宽度）
        img[:, i*base_step_width:w] = i * step_height

Image.fromarray(img).save("stair.png")
print("保存台阶高度图 stair.png")
