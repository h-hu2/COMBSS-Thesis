#%%
from PIL import Image

# Open the image
image = Image.open("CH4-ACC_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-ACC_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH4-F1_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-F1_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH4-MCC-LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-MCC_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-PE_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-PE_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-SE_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-SE_LD-Case-1.png")
#%%
from PIL import Image

# Open the image
image = Image.open("CH4-SP_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-SP_LD-Case-1.png")

#%%
from PIL import Image

# Open the image
image = Image.open("CH4-TIME_LD-Case-1.png")

# Get bounding box of the non-transparent region
bbox = image.getbbox()

# Crop the image to the bounding box
cropped_image = image.crop(bbox)

# Save the cropped image
cropped_image.save("CH4-TIME_LD-Case-1.png")

# %%
