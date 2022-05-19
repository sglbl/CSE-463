import tkinter as tk # To create the buttons and labels
import PointSaving   # Saving all the points

if __name__=='__main__':

    root = tk.Tk()
    root.title("Main Menu")
    root.geometry("600x400")
    
    topTitle = tk.Label(root, text="Suleyman's Homography App", bg="purple", fg="white", font=("Informal Roman", 40))
    topTitle.pack(ipadx=10, ipady=50)

    buttonCamera = tk.Button(root, text = 'Click to get image from Webcam', font=("Helvetica", 15), command = lambda : PointSaving.PointSaver(isWebCamMode = True ) )
    buttonFolder = tk.Button(root, text = 'Click to get image from Folder', font=("Helvetica", 15), command = lambda isWebCamMode = False : PointSaving.PointSaver(isWebCamMode) )
    
    # Set the position of button on the top of window
    buttonCamera.pack(ipadx=10, ipady=10, padx=10, pady=15)
    buttonFolder.pack(ipadx=10, ipady=10, padx=10, pady=15) 

    root.mainloop()