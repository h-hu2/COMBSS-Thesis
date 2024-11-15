#%%
from PIL import Image

# Open the image
image = Image.open("ACC-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("ACC_LD-Case-2.png")

#%%
from PIL import Image

# Open the image
image = Image.open("F1-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("F1_LD-Case-2.png")

#%%
from PIL import Image

# Open the image
image = Image.open("MCC-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("MCC_LD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("PE-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("PE_LD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("SE-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("SE_LD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("SP-LD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("SP_LD-Case-2.png")


# %%
