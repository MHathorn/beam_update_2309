# Import the modules
import requests
import shutil
import tkinter as tk
from tkinter import messagebox
import os
from packaging import version # Import the packaging module

# Define the constants
LOCAL_VERSION_FILE = os.path.join(os.getcwd(), "unitac-backend", "version.txt") # The path of the file on your local machine that contains the current version of your application
GITHUB_VERSION_URL = "https://raw.githubusercontent.com/UNITAC-Hamburg/beam_update_2309/main/unitac-backend/version.txt" # The URL of the file on GitHub that contains the latest version number
GITHUB_FILE_URL = "https://raw.githubusercontent.com/UNITAC-Hamburg/beam_update_2309/main/unitac-backend/" # The URL of the directory on GitHub that contains the updated files
LOCAL_FILE_PATH = os.path.join(os.getcwd(), "unitac-backend") # The path of the directory on your local machine that contains the files to be updated

# Create a Tkinter window
window = tk.Tk()
window.title("Update Checker")
window.geometry("300x100")

# Create a label to display the status
status_label = tk.Label(window, text="Checking for updates...", font=("Arial", 12))
status_label.pack(pady=10)

# Define a function to check for updates
def check_for_updates():
    # Get the current local version from version.txt
    with open(LOCAL_VERSION_FILE, "r") as f:
        LOCAL_VERSION = f.read().strip()

    # Get the latest version number from GitHub
    response = requests.get(GITHUB_VERSION_URL)
    if response.status_code == 200:
        github_version = response.text.strip()
        # Parse the version strings into objects
        local_version = version.parse(LOCAL_VERSION)
        github_version = version.parse(github_version)
        # Compare the version objects
        if github_version > local_version:
            # Update the status label
            status_label.config(text="Update available: {}".format(github_version))
            # Ask the user if they want to download the update
            answer = messagebox.askyesno("Update", "A new version of your application is available. Do you want to download it?")
            if answer:
                # Download the updated files from GitHub and replace the old files
                for file in ["environment.yml", "version.txt", "init.py", "main.py"]:
                    response = requests.get(GITHUB_FILE_URL + file)
                    if response.status_code == 200:
                        with open(os.path.join(LOCAL_FILE_PATH, file), "w") as f:
                            f.write(response.text)
                    else:
                        # Update the status label
                        status_label.config(text="Update failed: {}".format(response.status_code))
                        # Show a message box to inform the user that there was an error downloading the update
                        messagebox.showerror("Error", "There was an error downloading the update. Please try again later.")
                        return # Exit the function if any file fails to download
                # Update the status label
                status_label.config(text="Update completed.")
                # Show a message box to inform the user that they need to restart the application
                messagebox.showinfo("Restart", "Please restart your application to apply the update.")
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
