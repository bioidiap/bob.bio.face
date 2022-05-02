import matplotlib.pyplot as plt

import bob.io.image

from bob.bio.face.preprocessor import FaceCrop
from bob.bio.face.utils import get_default_cropped_positions

src = bob.io.base.load("../img/cropping_example_source.png")
modes = ["legacy", "dnn", "pad"]
cropped_images = []


SIZE = 160
# Pick cropping mode
for mode in modes:
    if mode == "legacy":
        cropped_image_size = (SIZE, 4 * SIZE // 5)
    else:
        cropped_image_size = (SIZE, SIZE)

    annotation_type = "eyes-center"
    # Load default cropped positions
    cropped_positions = get_default_cropped_positions(
        mode, cropped_image_size, annotation_type
    )

    # Instanciate cropper and crop
    cropper = FaceCrop(
        cropped_image_size=cropped_image_size,
        cropped_positions=cropped_positions,
        fixed_positions={"reye": (480, 380), "leye": (480, 650)},
        color_channel="rgb",
    )

    cropped_images.append(cropper.transform([src])[0].astype("uint8"))


# Visualize cropped images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i, (img, label) in enumerate(
    zip([src] + cropped_images, ["original"] + modes)
):
    ax = axes[i // 2, i % 2]
    ax.axis("off")
    ax.imshow(bob.io.image.to_matplotlib(img))
    ax.set_title(label)
