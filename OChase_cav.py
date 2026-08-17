import tkinter as tk


class App:
    def __init__(self, master):
        self.master = master
        master.title("Menu Screen")
        master.geometry("300x200")

        self.canvas = tk.Canvas(master, width=300, height=200)
        self.canvas.pack(fill="both", expand=True)

        self.show_menu()

        self.button_width = 100
        self.button_height = 50
        self.button_x = 100
        self.button_y = 75

        self.moving_button = None

    def show_menu(self):
        self.canvas.delete("all")

        self.canvas.create_text(
            150, 40,
            text="Menu Screen",
            font=("Arial", 16)
        )

        # Start button
        self.start_button = tk.Button(
            self.canvas,
            text="Start Moving Button",
            command=self.show_moving_button
        )

        self.canvas.create_window(
            150, 90,
            window=self.start_button
        )

        # Close button
        self.close_button = tk.Button(
            self.canvas,
            text="Close",
            command=self.master.quit
        )

        self.canvas.create_window(
            150, 140,
            window=self.close_button
        )

    def show_moving_button(self):
        self.canvas.delete("all")

        self.moving_button = tk.Button(
            self.canvas,
            text="Click Me",
            command=self.return_to_menu
        )

        # Put the button on the canvas
        self.canvas.create_window(
            self.button_x,
            self.button_y,
            window=self.moving_button,
            width=self.button_width,
            height=self.button_height,
            anchor="nw",
            tags="moving_button"
        )

        # Allow arrow keys to move it
        self.master.bind("<KeyPress>", self.move_button)

        # Make sure the window receives keyboard focus
        self.master.focus_set()

    def move_button(self, event):
        step = 10

        # Get current position
        x = self.button_x
        y = self.button_y

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if event.keysym == "Up":
            y -= step

        elif event.keysym == "Down":
            y += step

        elif event.keysym == "Left":
            x -= step

        elif event.keysym == "Right":
            x += step

        # Keep the button inside the canvas
        x = max(0, min(x, canvas_width - self.button_width))
        y = max(0, min(y, canvas_height - self.button_height))

        self.button_x = x
        self.button_y = y

        # Move the button
        self.canvas.coords(
            "moving_button",
            x,
            y
        )

    def return_to_menu(self):
        self.master.unbind("<KeyPress>")

        self.moving_button.destroy()
        self.moving_button = None

        self.button_x = 100
        self.button_y = 75

        self.show_menu()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
