# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## How to crop a face

# +
import bob.bio.face
import bob.io.image

# Loading Ada's images
image = bob.io.image.load("./img/838_ada.jpg")

# Setting Ada's eyes
annotations = dict()
annotations["reye"] = (265, 203)
annotations["leye"] = (278, 294)

# Final cropped size
cropped_image_size = (224, 224)

# Defining where we want the eyes to be located after the crop
cropped_positions = {"leye": (65, 150), "reye": (65, 77)}


face_cropper = bob.bio.face.preprocessor.FaceCrop(
    cropped_image_size=cropped_image_size,
    cropped_positions=cropped_positions,
    color_channel="rgb",
)

# Crops always a batch of images
cropped_image = face_cropper.transform([image], annotations=[annotations])


# +
import matplotlib.pyplot as plt

figure = plt.figure()
plt.subplot(121)
bob.io.image.imshow(image)
plt.subplot(122)
bob.io.image.imshow(cropped_image[0].astype("uint8"))
figure.show()
