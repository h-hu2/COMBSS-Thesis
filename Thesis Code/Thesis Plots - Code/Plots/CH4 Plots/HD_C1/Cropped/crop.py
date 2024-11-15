#%%
from PIL import Image

# Open the image
image = Image.open("CH4-F1_HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-F1_HD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH4-MCC-HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-MCC_HD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-PE_HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-PE_HD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-SE_HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-SE_HD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-SP_HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-SP_HD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH4-TIME_HD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-TIME_HD-Case-1.png")

# %%
