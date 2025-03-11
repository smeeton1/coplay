import tkinter as tk
# Function to close the window
def close_window():
    root.destroy()
# Function to move the button
def move_button(event):
 # Get current position
    x, y = button.winfo_x(), button.winfo_y()
    
    if event.keysym == 'Up':
        # Move up if within bounds
        if y > 0:
            button.place(y=y - 10)
    elif event.keysym == 'Down':
        # Move down if within bounds
        if y < root.winfo_height() - button.winfo_height():
            button.place(y=y + 10)
    elif event.keysym == 'Left':
        # Move left if within bounds
        if x > 0:
            button.place(x=x - 10)
    elif event.keysym == 'Right':
        # Move right if within bounds
        if x < root.winfo_width() - button.winfo_width():
            button.place(x=x + 10)
# Create the main window
root = tk.Tk()
# Set the title of the window
root.title("My First Window")
# Set the size of the window
root.geometry("400x300")
# Create a label widget
label = tk.Label(root, text="Hello, World!")
label.pack(pady=20)
# Create a button to close the window
button = tk.Button(root, text="Close", command=close_window)
button.place(x=150, y=100)  # Set initial position of the button
# Bind arrow key events to the move_button function
root.bind('<KeyPress>', move_button)
# Start the Tkinter event loop
root.mainloop()