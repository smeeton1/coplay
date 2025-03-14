import tkinter as tk
class App:
    def __init__(self, master):
        self.master = master
        master.title("Menu Screen")
        master.geometry("300x200")  # Set the size of the main window
        self.menu_frame = tk.Frame(master)
        self.menu_frame.pack()
        self.label = tk.Label(self.menu_frame, text="Menu Screen")
        self.label.pack(pady=20)
        self.start_button = tk.Button(self.menu_frame, text="Start Moving Button", command=self.show_moving_button)
        self.start_button.pack(pady=10)
        self.close_button = tk.Button(self.menu_frame, text="Close", command=master.quit)
        self.close_button.pack(pady=10)
        self.moving_button = None
        self.button_x = 100  # Initial x position of the button
        self.button_y = 100  # Initial y position of the button
        self.button_width = 100  # Width of the button
        self.button_height = 50  # Height of the button
    def show_moving_button(self):
        # Clear the menu frame
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
        # Create a moving button
        self.moving_button = tk.Button(self.menu_frame, text="Click Me", command=self.return_to_menu)
        self.moving_button.place(x=self.button_x, y=self.button_y, width=self.button_width, height=self.button_height)
        # Bind arrow keys to move the button
        self.master.bind("<KeyPress>", self.move_button)
    def move_button(self, event):
        step = 10  # Number of pixels to move
        # Get current position of the button
        current_x = self.moving_button.winfo_x()
        current_y = self.moving_button.winfo_y()
        if event.keysym == "Up":
            new_y = max(0, current_y - step)
            self.moving_button.place(x=current_x, y=new_y)
        elif event.keysym == "Down":
            new_y = min(self.master.winfo_height() - self.button_height, current_y + step)
            self.moving_button.place(x=current_x, y=new_y)
        elif event.keysym == "Left":
            new_x = max(0, current_x - step)
            self.moving_button.place(x=new_x, y=current_y)
        elif event.keysym == "Right":
            new_x = min(self.master.winfo_width() - self.button_width, current_x + step)
            self.moving_button.place(x=new_x, y=current_y)
    def return_to_menu(self):
        # Clear the moving button
        self.moving_button.destroy()
        # Recreate the menu
        self.show_menu()
    def show_menu(self):
        self.label = tk.Label(self.menu_frame, text="Menu Screen")
        self.label.pack(pady=20)
        self.start_button = tk.Button(self.menu_frame, text="Start Moving Button", command=self.show_moving_button)
        self.start_button.pack(pady=10)
        self.close_button = tk.Button(self.menu_frame, text="Close", command=self.master.quit)
        self.close_button.pack(pady=10)
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()