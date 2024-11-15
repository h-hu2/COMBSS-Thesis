#%%
from PIL import Image

# Open the image
image = Image.open("CH6-ACC_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-ACC_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH6-F1_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-F1_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH6-MCC-LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-MCC_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-PE_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-PE_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-SE_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-SE_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH6-SP_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-SP_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH6-TIME_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH6-TIME_LD-Case-1.png")

# %%
