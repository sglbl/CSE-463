import tkinter as tk
import os

root = tk.Tk()
root.withdraw() # make invisible the tkiner window

def filePathFinder():
    currdir = os.getcwd()
    tempdir = tk.filedialog.askopenfilename(parent=root, initialdir=currdir, title='Please select a file')
    if len(tempdir) > 0:
        print (tempdir)
    return tempdir
