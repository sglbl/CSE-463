import tkinter as tk # To create the buttons and labels
import ImageSelecting   # Saving all the points

if __name__=='__main__':

    imageSelector = ImageSelecting.ImageSelector()
    rootWidget = tk.Tk()
    rootWidget.title("Main Menu")
    rootWidget.geometry("650x400")
    
    topTitle = tk.Label(rootWidget, text="Suleyman's Object Detection App", bg="purple", fg="white", font=("Informal Roman", 40))
    topTitle.pack(ipadx=10, ipady=50)

    buttonImage1 = tk.Button( rootWidget, text = 'Click to get image 1', font=("Helvetica", 15), 
                              command = lambda : imageSelector.selector(1) )
    buttonImage2 = tk.Button( rootWidget, text = 'Click to get image 2 to match', font=("Helvetica", 15), 
                              command = lambda : imageSelector.selector(2) )

    # Set the position of button on the top of window
    buttonImage1.pack(ipadx=10, ipady=10, padx=10, pady=15)
    buttonImage2.pack(ipadx=10, ipady=10, padx=10, pady=15) 

    rootWidget.mainloop()