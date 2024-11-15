#%%
from PIL import Image

# Open the image
image = Image.open("F1-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("F1_HD-Case-2.png")

#%%
from PIL import Image

# Open the image
image = Image.open("MCC-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("MCC_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("PE-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("PE_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("SE-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("SE_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("SP-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("SP_HD-Case-2.png")


# %%
