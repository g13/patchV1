import os
#from selenium import webdriver
import imageio as iio
import time
from PIL import Image, ImageDraw, ImageFont

# Importing Image class from PIL module
 
# Opens a image in RGB mode
def add_text(txt, img_file, fontsize = 20, l = 0.1, t = 0.1):
    # Open an Image
    img = Image.open(img_file)
     
    # Call draw Method to add 2D graphics in an image
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype('arial.ttf', size=fontsize); 
    width, height = img.size
    # Add Text to an image
    #draw.text((l*width, t*height), txt, font = font, fill=(125, 125, 125, 50))
    draw.text((l*width, t*height), txt, fill=(125, 125, 125, 50))
    # Display edited image
     
    # Save the edited image
    fmt = img_file.split('.')[-1]
    output_fn = img_file[:-len(fmt)-1] + f'_add_text.{fmt}'
    img.save(output_fn)#
    return output_fn

def crop_image(img_file, l=0, t=0, r=0, b=0):
    im = Image.open(img_file)
     
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    left = width*l
    top = height*t
    right = width*(1-r)
    bottom = height*(1-b)
    # Cropped image of above dimension
    # (It will not change original image)
    cropped = im.crop((left, top, right, bottom))
    fmt = img_file.split('.')[-1]
    output_fn = img_file[:-len(fmt)-1] + f'_cropped.{fmt}'
    cropped.save(output_fn)
    return output_fn

#def html_to_png(inputFn, outputFn):
#    delay = 1
#    #Open a browser window...
#    #browser = webdriver.Firefox(executable_path=os.path.abspath("geckodriver"))
#    browser = webdriver.Chrome()
#    #..that displays the map...
#    browser.get('file://{path}/{html}'.format(path=os.getcwd(),html=inputFn))
#    #Give the map tiles some time to load
#    time.sleep(delay)
#    #Grab the screenshot
#    pic = os.getcwd() + '/' + outputFn
#    browser.save_screenshot(pic)
#    #Close the browser
#    browser.quit()
    
def create_gif(filenames, output_file, duration, crop = None, text = None, fontsize = 20, l = 0.1, t = 0.1):
    images = []
    if text is not None:
        if len(text) != len(filenames):
            raise Exception('text length not matching with number of images!')
    else:
        text = [None] * len(filenames)
    del_tmp = []
    for filename, txt in zip(filenames, text):
        if crop is not None:
            filename = crop(filename)
            del_tmp.append(filename)
        if txt is not None:
            filename = add_text(txt, filename, fontsize = fontsize, l = l, t = t)
            del_tmp.append(filename)
        images.append(iio.imread(filename))
    iio.mimwrite(output_file + '.gif', images, duration=duration)
    for f in del_tmp:
        if os.path.exists(f):
            os.remove(f) 
