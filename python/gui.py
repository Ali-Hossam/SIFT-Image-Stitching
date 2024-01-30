import customtkinter as ctk
import os
import cv2
import time
from PIL import Image
from general_functions import concatenate_images
from panorama_stitching import Panorama
import threading
import numpy as np

upload_img = cv2.imread(os.getcwd()+"\\python\\upload.jpeg")
upload_img_size = (600, 360)
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme(os.getcwd()+"\\python\\dark-blue.json")  # Themes: "blue" (standard), "green", "dark-blue"

stitcher = Panorama()
is_done_stitching = False
final_image = np.zeros(2)

def stitch(images):
        global is_done_stitching, final_image
        final_image = (stitcher.get_panorama(images)*255).astype("uint8")
        is_done_stitching = True
    
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry(f"{720}x{620}")
        self.title("StitchMaster")
        self.resizable(False, False)
        
        # configure grid layout (4x4)
        self.grid_columnconfigure((0, 1, 2), weight=1)  # Center column
        self.grid_rowconfigure((0, 1, 2), weight=1)     # Center row

        # create images window frame
        self.images_frame = ctk.CTkFrame(self, corner_radius=20)
        self.images_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=20, pady=20)
        
        # add image to the frame
        self.images_frame_label = ctk.CTkLabel(self.images_frame, text="", corner_radius=20)
        self.images_frame_label.pack(padx=5, pady=35)
        self.add_image_to_frame(upload_img, "images_frame_label", upload_img_size)
        
        # bind images frame to upload function
        self.images_frame.bind("<Button-1>", self.upload_images)
        self.images_frame_label.bind("<Button-1>", self.upload_images)
        
        # add progress bar
        self.progress_bar = ctk.CTkProgressBar(self, height=1, corner_radius=10, 
                                               border_width=2, border_color='gray',
                                               determinate_speed=0.5)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky='nsew', columnspan=3, padx=20, pady=5)
        
        # add frame for swap button
        self.swap_frame = ctk.CTkFrame(self, corner_radius=20, fg_color='black')
        self.swap_frame.grid(row=2, column=1)
        
        # add two entries for swap button
        self.swap_entry1 = ctk.CTkEntry(self.swap_frame, width=90)
        self.swap_entry2 = ctk.CTkEntry(self.swap_frame, width=90)
        self.swap_entry1.grid(row=0, column=0, pady=10)
        self.swap_entry2.grid(row=0, column=1, pady=10)
        
        # create buttons
        self.start_button = ctk.CTkButton(self, text="Start", height=40, width=190, corner_radius=8)
        self.save_button = ctk.CTkButton(self, text="Save", height=40, width=190, corner_radius=8, state='disabled', fg_color="gray")
        self.swap_button = ctk.CTkButton(self.swap_frame, text="Swap", height=40, width=190, corner_radius=8)
        
        # bind swap button to swap images function
        self.swap_button.bind("<Button-1>", self.swap_images)
        
        # bind start button to start stitching function
        self.start_button.bind("<Button-1>", self.start_stitching)
        
        # bind save button to save function
        self.save_button.bind("<Button-1>", self.save_image)
        
        # grid buttons below the images frame
        self.start_button.grid(row=2, column=0, pady=0)
        self.save_button.grid(row=2, column=2, pady=0)
        self.swap_button.grid(row=1, columnspan=2)
        
        # additional widgets or configurations can be added here
        # configure row heights
        self.grid_rowconfigure(0, weight=10)  
        self.grid_rowconfigure(1, weight=1)  
        self.grid_rowconfigure(2, minsize=50, weight=2)   # Set the height of row 2 to 20 pixels

        # configure frames grid propagate
        self.images_frame.pack_propagate(False)
        
        # intialize images list
        self.images_list = []
        
    def upload_images(self, e):
        """
        Event handler for uploading images.

        Arguments:
        e : Event
            Event object.
        """
        # Store files paths
        images_path = list(ctk.filedialog.askopenfilenames())
        self.images_list = [cv2.imread(img_path) for img_path in images_path]
        if(len(images_path)):
            self.show_images()
    
    def show_images(self):
        all_images = concatenate_images(self.images_list)
        self.add_image_to_frame(all_images, "images_frame_label", size=upload_img_size)
            
    def add_image_to_frame(self, img, frame_label, size):
        """
        Adds an image (one frame from the video) to a label UI frame.

        Arguments:
        img : numpy.ndarray
            Image data in the form of a NumPy array.
        frame_label : str
            Name of the label UI frame to which the image will be added.
        size : tuple
            Size of the image.

        Notes:
        - Converts BGR image to RGB if the image has more than 2 dimensions.
        - Utilizes CTkImage and configures the specified label frame with the image.
        """
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ctk.CTkImage(img, size=size)
        vid_label_func = getattr(self, frame_label)
        vid_label_func.configure(image=img)
        setattr(self, frame_label, vid_label_func)

    def swap_images(self, e):
        """swaps two images in self.images_list using entries in entry1 and entry2"""
        num_of_images = len(self.images_list)
        if(num_of_images):
            num1 = int(self.swap_entry1.get())
            num2 = int(self.swap_entry2.get())
            
            if(0 < num1 < num_of_images+1 and 0 < num2 < num_of_images+1):
                # swap both images
                img1 = self.images_list[num1 - 1]
                img2 = self.images_list[num2 - 1]
                
                self.images_list[num1 - 1] = img2
                self.images_list[num2 - 1] = img1
                
                # show updated image
                self.show_images()
                
    def update_progress(self):
        """ updates progress bar"""
        while not is_done_stitching:
            current_value = self.progress_bar.get()
            self.progress_bar.step()
            self.progress_bar.update_idletasks()
            time.sleep(0.1)  # Simulate some work being done
        
        # set progress bar to maximum        
        self.progress_bar.set(1)
         
    def start_stitching(self, e):
        # Create a new thread for stitching the image
        stitching_thread = threading.Thread(target=stitch, args=(self.images_list,))
        
        # Start the thread
        stitching_thread.start()
        
        # update progress bar untill stitching finishes
        self.update_progress()
        
        # show final image        
        self.add_image_to_frame(final_image, "images_frame_label", upload_img_size)
        
        # enable save button
        self.save_button.configure(fg_color="#7e6aff", text_color="#4C067A", text="Save", state="normal")
    
    def save_image(self, e):
        # save image
        cv2.imwrite("panorama_img.png", img=final_image)
        
        # configure save button text, state, and color
        self.save_button.configure(fg_color="gray", text_color="gray28", text="Saved", state="disabled")

        # reset images frame
        self.add_image_to_frame(upload_img, "images_frame_label", upload_img_size)

if __name__ == "__main__":
    app = App()
    app.mainloop()
