# Import the modules
import requests
import shutil
import tkinter as tk
from tkinter import messagebox

# Define the constants
LOCAL_VERSION = "1.0.0" # The current version of your application
GITHUB_VERSION_URL = "https://raw.githubusercontent.com/your_username/your_repository/main/version.txt" # The URL of the file on GitHub that contains the latest version number
GITHUB_FILE_URL = "https://raw.githubusercontent.com/your_username/your_repository/main/your_file.py" # The URL of the file on GitHub that you want to update
LOCAL_FILE_PATH = "your_file.py" # The path of the file on your local machine that you want to update

# Create a Tkinter window
window = tk.Tk()
window.title("Update Checker")
window.geometry("300x100")

# Create a label to display the status
status_label = tk.Label(window, text="Checking for updates...", font=("Arial", 12))
status_label.pack(pady=10)

# Define a function to check for updates
def check_for_updates():
    # Get the latest version number from GitHub
    response = requests.get(GITHUB_VERSION_URL)
    if response.status_code == 200:
        github_version = response.text.strip()
        # Compare the version numbers
        if github_version > LOCAL_VERSION:
            # Update the status label
            status_label.config(text="Update available: {}".format(github_version))
            # Ask the user if they want to download the update
            answer = messagebox.askyesno("Update", "A new version of your application is available. Do you want to download it?")
            if answer:
                # Download the updated file from GitHub and replace the old file
                response = requests.get(GITHUB_FILE_URL, stream=True)
                if response.status_code == 200:
                    with open(LOCAL_FILE_PATH, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                    # Update the status label
                    status_label.config(text="Update completed.")
                    # Show a message box to inform the user that they need to restart the application
                    messagebox.showinfo("Restart", "Please restart your application to apply the update.")
                else:
                    # Update the status label
                    status_label.config(text="Update failed: {}".format(response.status_code))
                    # Show a message box to inform the user that there was an error downloading the update
                    messagebox.showerror("Error", "There was an error downloading the update. Please try again later.")
        else:
            # Update the status label
            status_label.config(text="No updates available.")
    else:
        # Update the status label
        status_label.config(text="Update failed: {}".format(response.status_code))
        # Show a message box to inform the user that there was an error checking for updates
        messagebox.showerror("Error", "There was an error checking for updates. Please try again later.")

# Call the function to check for updates
check_for_updates()

# Start the main loop of the window
window.mainloop()
