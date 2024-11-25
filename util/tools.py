import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# ==================================================================================================
def load_img(fname, dirname='Image Processing with Python course exercise dataset/chapter 1/'):
  try:
    # Load the image using skimage.io.imread
    img_path = dirname + fname
    Im_array = io.imread(img_path)
    
    # Check if the image has an alpha channel (4 channels)
    if Im_array.ndim == 3 and Im_array.shape[-1] == 4:
        # Remove the alpha channel by selecting the first three channels (RGB)
        Im_array = Im_array[..., :3]
    
    # Handle grayscale images (single channel)
    elif Im_array.ndim == 2:  # Grayscale image
        Im_array = np.expand_dims(Im_array, axis=-1)  # Convert to 3D (height, width, 1)
    
    return Im_array
  except Exception as e:
    print(f"Error loading image: {e}")
    return None
# --------------------------------------------------------------------------------------------------   
def show_image(image, title='Image', cmap_type='gray'):
  plt.imshow(image, cmap=cmap_type)
  plt.title(title)
  plt.axis('off')
  plt.show()
# --------------------------------------------------------------------------------------------------
def plot_comparison(original, filtered, title_filtered):
  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,6), sharex=True, sharey=True)
  ax1.imshow(original, cmap=plt.cm.gray)
  ax1.set.title('original')
  ax1.axis('off')
  ax2.imshow(filtered, cmap=plt.cm.gray)
  ax2.set_title(title_filtered)
  ax2.axis('off')
# --------------------------------------------------------------------------------------------------  
def get_mask(image):
  ''' Creates mask with three defect regions '''
  mask = np.zeros(image.shape[:-1])
  
  mask[101:106, 0:240] = 1
  
  mask[152:154, 0:60] = 1
  mask[153:155, 60:100] = 1    
  mask[154:156, 100:120] = 1
  mask[155:156, 120:140] = 1
  
  mask[212:217, 0:150] = 1
  mask[217:222, 150:256] = 1
  
  return mask
# --------------------------------------------------------------------------------------------------
def generate_dead_pixel_mask(image, dead_value=0):
  # Create a mask where dead pixels are 1 and valid pixels are 0
  mask = np.all(image == dead_value, axis=-1)  # Check if all channels (for RGB) are dead (e.g., all 0)
  return mask.astype(np.uint8)  # Convert mask to binary (0, 1)
# --------------------------------------------------------------------------------------------------
def show_image_contour(image, contours):
  plt.figure()
  for n, contour in enumerate(contours):
      plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
  plt.imshow(image, interpolation='nearest', cmap='gray_r')
  plt.title('Contours')
  plt.axis('off')
  plt.show()
# --------------------------------------------------------------------------------------------------
def show_image_with_corners(image, coords, title="Corners detected"):
  plt.figure(figsize=(12, 10))
  plt.imshow(image, interpolation='nearest', cmap='gray')
  plt.title(title)
  plt.plot(coords[:, 1], coords[:, 0], '+r', markersize=35)
  plt.axis('off')
  plt.show()
  plt.close()
# --------------------------------------------------------------------------------------------------
def show_detected_face(result, detected, title="Face image"):
  plt.figure()
  plt.imshow(result)
  img_desc = plt.gca()
  plt.set_cmap('gray')
  plt.title(title)
  plt.axis('off')

  for patch in detected:
      
    img_desc.add_patch(
        patches.Rectangle(
            (patch['c'], patch['r']),
            patch['width'],
            patch['height'],
            fill=False,
            color='r',
            linewidth=2)
    )
  plt.show()
# --------------------------------------------------------------------------------------------------
def getFaceRectangle(d):
    ''' Extracts the face from the image using the coordinates of the detected image '''
    # X and Y starting points of the face rectangle
    x, y  = d['r'], d['c']
    
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'],  d['c'] + d['height']
    group_image = load_img('face_det25.jpg')
    # Extract the detected face
    face= group_image[ x:width, y:height]
    return face
# --------------------------------------------------------------------------------------------------  
def mergeBlurryFace(original, gaussian_image):
     # X and Y starting points of the face rectangle
    x, y  = d['r'], d['c']
    # The width and height of the face rectangle
    width, height = d['r'] + d['width'],  d['c'] + d['height']
    
    original[ x:width, y:height] =  gaussian_image
    return original
# ==================================================================================================