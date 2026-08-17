import tkinter as tk

class App:
    def __init__(self, master):
        self.master = master
        master.title("Menu Screen")
        master.geometry("300x200")

        self.menu_frame = tk.Frame(master)
        self.menu_frame.pack(fill="both", expand=True)

        self.label = tk.Label(self.menu_frame, text="Menu Screen")
        self.label.pack(pady=20)

        self.start_button = tk.Button(
            self.menu_frame,
            text="Start Moving Button",
            command=self.show_moving_button
        )
        self.start_button.pack(pady=10)

        self.close_button = tk.Button(
            self.menu_frame,
            text="Close",
            command=master.quit
        )
        self.close_button.pack(pady=10)

        self.moving_button = None

        self.button_x = 100
        self.button_y = 75
        self.button_width = 100
        self.button_height = 50

    def show_moving_button(self):
        # Clear the menu
        for widget in self.menu_frame.winfo_children():
            widget.destroy()

        # Create moving button
        self.moving_button = tk.Button(
            self.menu_frame,
            text="Click Me",
            command=self.return_to_menu
        )

        self.moving_button.place(
            x=self.button_x,
            y=self.button_y,
            width=self.button_width,
            height=self.button_height
        )

        self.master.bind("<KeyPress>", self.move_button)

    def move_button(self, event):
        step = 10

        current_x = self.moving_button.winfo_x()
        current_y = self.moving_button.winfo_y()

        frame_width = self.menu_frame.winfo_width()
        frame_height = self.menu_frame.winfo_height()

        if event.keysym == "Up":
            new_y = max(0, current_y - step)
            self.moving_button.place(x=current_x, y=new_y)

        elif event.keysym == "Down":
            new_y = min(
                frame_height - self.button_height,
                current_y + step
            )
            self.moving_button.place(x=current_x, y=new_y)

        elif event.keysym == "Left":
            new_x = max(0, current_x - step)
            self.moving_button.place(x=new_x, y=current_y)

        elif event.keysym == "Right":
            new_x = min(
                frame_width - self.button_width,
                current_x + step
            )
            self.moving_button.place(x=new_x, y=current_y)

    def return_to_menu(self):
        self.moving_button.destroy()
        self.show_menu()

    def show_menu(self):
        self.label = tk.Label(self.menu_frame, text="Menu Screen")
        self.label.pack(pady=20)

        self.start_button = tk.Button(
            self.menu_frame,
            text="Start Moving Button",
            command=self.show_moving_button
        )
        self.start_button.pack(pady=10)

        self.close_button = tk.Button(
            self.menu_frame,
            text="Close",
            command=self.master.quit
        )
        self.close_button.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
