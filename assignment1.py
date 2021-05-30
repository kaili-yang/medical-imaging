# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
import PIL.Image, PIL.ImageTk
import matplotlib.pyplot as plt
import nibabel as nib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from nibabel.viewers import OrthoSlicer3D

# =============================================================================
# Introduce
# Please uncomment the function before running
# =============================================================================

# Read all images in the folder
# file_list = ['cardiac_axial',
#               'cardiac_realtime',
#               'ct',
#               'fmri',
#               'meanpet',
#               'swi',
#               'T1_with_tumor',
#               'tof']

path = 'D:/1CSmaster/cs516/modalities/'
file_list = os.listdir(path) 
file_length = len(file_list)

# =============================================================================
# Part 1 & Part2
# =============================================================================

# Part 1(a)
def part1a():
    for i in range(file_length):
        title = file_list[i] 
        file = nib.load(path + title)
        img = file.get_fdata()
        # 3D
        middleZ =img.shape[2] // 2
        # 4D
        middleT = 0
        if len(img.shape)>3: 
            middleT = img.shape[3] // 2
        # part2
        michelson = michelson_contrast(img) # Michelson Contrast
        rms       = rms_contrast(img)       # RMS Contrast
        entropy   = entropy_contrast(img)   # Entropy Contrast
        title = title + '\n Cm=' + michelson + '\n RMS=' + rms + '\n Entropy=' + entropy
        # plot image
        plot_img(i, title, img, middleZ, middleT)

def plot_img(index, title, img, middleZ, middleT):
    plt.suptitle('assginment1 Part1&2',  y=1.2)
    plt.subplot(3, 3, int(index) + 1)
    plt.title(title, pad=10)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(wspace=1.0, hspace=2.4)
    if middleT == 0:  
        plt.imshow(img[:, :, middleZ], cmap='jet') # 3D
    else:
        plt.imshow(img[:, :, middleZ, middleT], cmap='jet') # 4D
     
def michelson_contrast(img):
    lmin = img.min() if img.min() > 0 else 0
    lmax = img.max() if img.min() > 0 else img.max() - img.min()
    michelson = (lmax - lmin) / (lmax + lmin)
    return str("%.5f" % michelson) # round to 5 decimal places, conver to String
        
def rms_contrast(img):
    rms = np.std(img, ddof = 1) 
    return str("%.5f" % rms)

def entropy_contrast(img):
    hist, edges = np.histogram(img.ravel())
    entropy_list = []
    for item in hist:
        probability = item / img.size
        if probability == 0:
            entropy = 0
        else:
            entropy = -1 * probability * (np.log(probability) / np.log(2))
        entropy_list.append(entropy)
    Centropy = np.sum(entropy_list)
    return str("%.5f" % Centropy)
    

# Part 1(b)
def part1b():
    # swi
    img_swi = nib.load(path+'swi.nii.gz')
    img_swi = img_swi.get_fdata()
    plt.suptitle('assginment1 Part2')
    plt.subplot(1,2,1)
    plt.title('SWI')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    middleZ = img_swi.shape[2] // 2
    radius = 50 
    minRange = middleZ - radius
    maxRange = middleZ + radius
    plt.imshow(np.min(img_swi[:, :, minRange:maxRange], axis=2), cmap='jet')
    
    # tof
    img_tof = nib.load(path+'tof.nii.gz')
    img_tof = img_tof.get_fdata()
    plt.subplot(1,2,2)
    plt.title('ToF')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    middleZ = img_tof.shape[2] // 2
    radius = 50
    minRange = middleZ - radius
    maxRange = middleZ + radius
    plt.imshow(np.max(img_tof[:, :, minRange:maxRange], axis=2), cmap='jet')

# part1a()
# part1b()

# =============================================================================
# Part 3 
# sip means signal patch
# nop means noise patch
# They are coordinate information of patches.
# patch width, shape = 30
# They are self-defined and can be modified.
# =============================================================================

sip = [
    {'x1': 130, 'x2': 160, 'y1': 130, 'y2': 160 }, # cardiac_axial
    {'x1': 70, 'x2': 100, 'y1': 85, 'y2': 115 }, # cardiac_realtime
    {'x1': 240, 'x2': 270, 'y1': 60, 'y2': 90 }, # ct
    {'x1': 70, 'x2': 100, 'y1': 40, 'y2': 70 }, # fmri
    {'x1': 120, 'x2': 150, 'y1': 90, 'y2': 120 }, # meanpet
    {'x1': 390, 'x2': 420, 'y1': 200, 'y2': 230 }, # swi
    {'x1': 130, 'x2': 160, 'y1': 130, 'y2': 150 }, # T1_with_tumor
    {'x1': 90, 'x2': 120, 'y1': 50, 'y2': 80 }] # tof

nop = [
    {'x1': 130, 'x2': 160, 'y1': 250, 'y2': 280 }, # cardiac_axial
    {'x1': 0, 'x2': 30, 'y1': 85, 'y2': 115 }, # cardiac_realtime
    {'x1': 470, 'x2': 500, 'y1': 200, 'y2': 230 }, # ct
    {'x1': 0, 'x2': 30, 'y1': 40, 'y2': 70 }, # fmri
    {'x1': 10, 'x2': 40, 'y1': 90, 'y2': 120 }, # meanpet
    {'x1': 470, 'x2': 500, 'y1': 200, 'y2': 230 }, # swi
    {'x1': 130, 'x2': 160, 'y1': 200, 'y2': 230 }, # T1_with_tumor
    {'x1': 10, 'x2': 40, 'y1': 10, 'y2': 40 }] # tof
    
def snr_hist():
    for i in range(file_length):
        title = file_list[i] 
        file = nib.load(path + title)
        img = file.get_fdata()
        signal_patch = img[sip[i]['x1'] : sip[i]['x2'], sip[i]['y1'] : sip[i]['y2']]
        noise_patch = img[nop[i]['x1'] : nop[i]['x2'], nop[i]['y1'] : nop[i]['y2']]
        signal = np.mean(signal_patch)
        noise = np.std(noise_patch)
        snr =  np.where(noise == 0, 0, signal/noise)
        gauss = np.random.normal(loc=0.0, scale=1.0, size=noise_patch.shape).ravel()
        plot_hist(i, title, img, snr, gauss)

def plot_hist(index, title, img, snr, gauss):
    plt.suptitle('assginment1 Part3',  y=1.2)
    plt.subplot(3, 3, int(index) + 1)
    plt.title(title + '\n SNR=' + str("%.4f" % snr), pad=10)
    plt.subplots_adjust(wspace=1.0, hspace=1.6)
    plt.hist(gauss, bins=100, color='blue', alpha=0.7,rwidth=0.9, density=False)
   
# snr_hist()

# =============================================================================
# Part 4
# =============================================================================
# set sigma
sigmas = [2, 4, 15]

def linear_filter():
    k = 0
    for i in range(file_length):
        for sigma in sigmas:
            k+=1 
            title = file_list[i] 
            file = nib.load(path + title)
            img = file.get_fdata()
            middleZ =img.shape[2] // 2  # 3D
            middleT = 0 # 4D
            if len(img.shape)>3: 
                middleT = img.shape[3] // 2
            img_freqs = np.fft.fftshift(np.fft.fftn(img))
            x = img.shape[0]
            y = img.shape[1]
            z = img.shape[2]
            if len(img.shape)>3:  # 4D
                d = img.shape[3]
                [caX, caY, caZ, caD] = np.mgrid[0:x, 0:y, 0:z, 0:d]
                xpr = caX - img.shape[0] // 2
                ypr = caY - img.shape[1] // 2
                zpr = caZ - img.shape[2] // 2
                dpr = caD - img.shape[3] // 2
            else: # 3D
                [caX, caY, caZ] = np.mgrid[0:x, 0:y, 0:z]
                xpr = caX - img.shape[0] // 2
                ypr = caY - img.shape[1] // 2
                zpr = caZ - img.shape[2] // 2
                dpr = 0
            gauss_filter = np.exp(-((xpr**2 + ypr**2 + zpr**2 + dpr**2) / (2*sigma**2))) / (2*np.pi*sigma**2)
            gauss_filter = gauss_filter / np.max(gauss_filter)
            filtered_freqs = img_freqs * gauss_filter
            filtered_img = np.abs(np.fft.ifftn(np.fft.fftshift(filtered_freqs)))
            title = title + '\n sigma=' + str(sigma)
            # plot image
            filter_img(k, title, filtered_img, middleZ, middleT)

def filter_img(index, title, img, middleZ, middleT):
    print(index)
    plt.suptitle('assginment1 Part4',  y=1.1)
    plt.subplot(3, 3, int(index))
    plt.title(title, pad=10)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(hspace=1.3)
    if middleT == 0:  
        plt.imshow(img[:, :, middleZ], cmap='gray') # 3D
    else:
        plt.imshow(img[:, :, middleZ, middleT], cmap='gray') # 4D

# linear_filter()

# =============================================================================
# Bonus
# =============================================================================
#
#=============================================================================
# class App:
#     def __init__(self, master):
#         
#         # Add a title 
#         master.title('Assignment1 Bonus')
#         
#         # Creating a Menu Bar
#         master.option_add('*tearOff', False)
#         menuBar = tk.Menu(master)
#         master.config(menu=menuBar)
#         
#         # Add menu items
#         fileMenu = tk.Menu(menuBar)
#         menuBar.add_cascade(label='File', menu=fileMenu)
#         fileMenu.add_command(label='Open', command = self.select_image)
#         fileMenu.add_separator()
#         fileMenu.add_command(label='Save', command=self.save_image) 
#         fileMenu.add_separator()
#         fileMenu.add_command(label='Exit', command=lambda: self._quit(master))
#         
#           # Main Frame
#         self.main_frame = ttk.Frame(master)
#         self.main_frame.pack()
#         
#         # initialize Image Frame, this panel will store our original and adjusted image
#         self.img_frame = ttk.Frame(self.main_frame)
#         self.img_frame.grid(column=0, row=0)
# 		
#         # Create a canvas that can fit the above image
#         self.canvas = tk.Canvas(self.img_frame, width=150, height=100)
#         self.canvas.pack()
#         
#         # Edit Frame, this panel stores the choices for user to select
#         self.edit_frame = ttk.Frame(self.main_frame)
#         self.edit_frame.grid(column=1, row=0)
#         
#         # Time series 
#         self.zSlice_frame = ttk.LabelFrame(self.edit_frame, text='zSlice')
#         self.zSlice_frame.grid(row=0, column=1)
#         self.zSlice = tk.DoubleVar()
#         self.zSlice.set(1.0)
#         ttk.Label(self.zSlice_frame, textvariable=self.zSlice).pack()
#         ttk.Scale(self.zSlice_frame, orient = tk.HORIZONTAL, length=100, variable=self.zSlice,from_=1, to = 100, command=lambda x: self.adjust_zSlice(self.zSlice.get())).pack()
#         
#         for child in self.edit_frame.winfo_children():
#             child.grid_configure(padx=5, pady=5)
#         
#         # Callback for the "zSlice Correct"
#     def adjust_zSlice(self, zSlice):
#      	# build a lookup table mapping the pixel values [0, 255] to their adjusted zSlice values
#         self.zSlice.set('%0.2f' % self.zSlice.get()) 
#         self.table = np.array([((i / 255.0) ** zSlice) * 255 for i in np.arange(0, 256)]).astype('uint8')
#      
#         # apply zSlice correction using the lookup table
#         self.adjusted = cv.LUT(self.original, self.table)
#         self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
#         # update the image panel
#         self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
#         
#     def _quit(self, win):
#         win.quit()      # win will exit when this function is called
#         win.destroy()
#         exit()
# 
#     # Load an image using OpenCV
#     def select_image(self):
#         # open a file chooser dialog and allow the user to select an input image
#         self.image_path = filedialog.askopenfilename(initialdir='',title='Choose an image',filetypes=(('GZ','*.gz'),))
# 
#         # ensure a file path was selected
#         if len(self.image_path) > 0:
#             self.original = nib.load(self.image_path)
#             # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
#             if self.original.shape == 3:
#                 self.height, self.width, self.z_axis= self.original.shape
#             if self.original.shape == 4:
#                 self.height, self.width, self.z_axis, self.d_axis= self.original.shape
#             # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage,
#             # convert the images to PIL format and then to ImageTk format
#             self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.original))
#         
#             # Add a PhotoImage to the Canvas
#             width, height, queue, time = img.dataobj.shape
#             OrthoSlicer3D(img.dataobj).show()
#             num = 1
#             for i in range(0, queue, 10):
#                 img_arr = img.dataobj[:, :, i]
#                 plt.subplot(5, 4, num)
#                 plt.imshow(img_arr, cmap='gray')
#                 num += 1
#             
#             plt.show()
#             # self.canvas.config(width=self.width, height=self.height)
#             # self.canvas.create_image(0, 0, image=self.photo, anchor='nw') 
#             # OrthoSlicer3D(self.photo.dataobj).show()
#     # ---------------------------------------------------------------
# 
#     def save_image(self):
#         # open a file chooser dialog and allow the user to select an input image
#         self.filename = filedialog.asksaveasfilename(initialdir='',title='Select an image',
#                                           filetypes=(('JPEG','*.jpg;*.jpeg'),
#                                                       ('GIF','*.gif'),
#                                                       ('PNG','*.png')))
#         cv.imwrite(self.filename, self.blur)
#         
# 
# def main():
#     # Create a window and pass it to the Application object
#     # Create instance
#     root = tk.Tk()
#     app = App(root)
#     #======================
#     # Start GUI
#     #======================
#     root.mainloop()
# 
# if __name__ == '__main__':
#     main()
# =============================================================================
        
