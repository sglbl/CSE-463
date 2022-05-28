import tkinter as tk # To create the buttons and labels
import Selecting   # Saving all the points

if __name__=='__main__':

    selectorObject = Selecting.Selector()
    rootWidget = tk.Tk()
    rootWidget.title("Main Menu")
    rootWidget.geometry("750x420")
    
    topTitle = tk.Label(rootWidget, text="Suleyman's Stereo Correspondence App", bg="purple", fg="white", font=("Informal Roman", 40))
    topTitle.pack(ipadx=10, ipady=50)

    selectedButtonNo = tk.IntVar()
    selectedButtonNo.set(1)
    R1 = tk.Radiobutton(rootWidget, text="Sift with BF", variable=selectedButtonNo, value=1, font=("Helvetica", 15), 
                    command = lambda : selectorObject.buttonSelector(selectedButtonNo))
    R1.pack( anchor = tk.W)

    R2 = tk.Radiobutton(rootWidget, text="ORB with Flann", variable=selectedButtonNo, value=2, font=("Helvetica", 15),
                    command = lambda : selectorObject.buttonSelector(selectedButtonNo))
    R2.pack( anchor = tk.W)

    R3 = tk.Radiobutton(rootWidget, text="Edge with Fast Brief NH ", variable=selectedButtonNo, value=3, font=("Helvetica", 15),
                    command = lambda : selectorObject.buttonSelector(selectedButtonNo))
    R3.pack( anchor = tk.W)

    buttonImage1 = tk.Button( rootWidget, text = 'Click to get image of left camera', font=("Helvetica", 15), 
                              command = lambda : selectorObject.imageSelector(1) )
    buttonImage2 = tk.Button( rootWidget, text = 'Click to get image of right camera', font=("Helvetica", 15), 
                              command = lambda : selectorObject.imageSelector(2) )

    # Set the position of button on the top of window
    buttonImage1.pack(anchor = tk.CENTER)
    buttonImage2.pack(anchor = tk.CENTER)

    rootWidget.mainloop()
