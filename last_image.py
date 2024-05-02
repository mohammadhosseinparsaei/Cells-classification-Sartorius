import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

directory = r'D:\Data for machine learning\Sartorius cell instance segmentation\LIVECell_dataset_2021\images\livecell_test_images'

fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, folder in enumerate(['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'RatC6', 'SHSY5Y', 'SkBr3', 'SKOV3']):
    folder_path = os.path.join(directory, folder)
    images = [img for img in os.listdir(folder_path) if img.endswith('.tif')]
    if images:
        last_image = images[-1]
        img = mpimg.imread(os.path.join(folder_path, last_image))
        ax = axes[i // 3, i % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(folder)
        ax.axis('off')

plt.tight_layout()
script_path = os.path.dirname(__file__)
plt.savefig(os.path.join(script_path, 'The last image of each cell.png'), dpi=300)
plt.show()
