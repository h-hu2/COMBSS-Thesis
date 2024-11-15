#%%
from PIL import Image

# Open the image
image = Image.open("CH6-F1_HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-F1_HD-Case-2.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH6-MCC-HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-MCC_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-PE_HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-PE_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-SE_HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-SE_HD-Case-2.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-SP_HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-SP_HD-Case-2.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH6-TIME_HD-Case-2.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-TIME_HD-Case-2.png")
