import tkinter as tk # To create the buttons and labels
import Selecting   # Saving all the points

if __name__=='__main__':

    selectorObject = Selecting.Selector()
    rootWidget = tk.Tk()
    rootWidget.title("Main Menu")
    rootWidget.geometry("650x400")
    
    topTitle = tk.Label(rootWidget, text="Suleyman's Object Detection App", bg="purple", fg="white", font=("Informal Roman", 40))
    topTitle.pack(ipadx=10, ipady=50)

    buttonImage1 = tk.Button( rootWidget, text = 'Click to get image of left camera', font=("Helvetica", 15), 
                              command = lambda : selectorObject.imageSelector(1) )
    buttonImage2 = tk.Button( rootWidget, text = 'Click to get image of right camera', font=("Helvetica", 15), 
                              command = lambda : selectorObject.imageSelector(2) )

    # Set the position of button on the top of window
    buttonImage1.pack(anchor = tk.CENTER, pady=30)
    buttonImage2.pack(anchor = tk.CENTER, pady=20)

    rootWidget.mainloop()
