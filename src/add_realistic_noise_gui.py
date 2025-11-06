"""
Realistic Astronomical Noise Generator - GUI Version
===================================================
Simple GUI to add realistic telescope noise to clean images.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from pathlib import Pathc
import threading
from add_realistic_noise import add_realistic_telescope_noise


class NoiseGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Astronomical Noise Generator")
        self.root.geometry("900x700")
        self.root.configure(bg="#0d1117")
        
        self.input_path = None
        self.output_path = None
        self.clean_image = None
        self.noisy_image = None
        
        self.setup_styles()
        self.create_ui()
    
    def setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        bg_dark = "#0d1117"
        bg_card = "#161b22"
        accent_blue = "#58a6ff"
        accent_green = "#3fb950"
        text_primary = "#e6edf3"
        text_secondary = "#8b949e"
        
        # Card frame
        style.configure("Card.TFrame", background=bg_card, relief="solid", borderwidth=1)
        
        # Labels
        style.configure("TLabel", background=bg_card, foreground=text_primary, 
                       font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=bg_dark, foreground=text_primary,
                       font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=bg_dark, foreground=text_secondary,
                       font=("Segoe UI", 10))
        
        # Buttons
        style.configure("Accent.TButton", background=accent_blue, foreground=bg_dark,
                       font=("Segoe UI", 10, "bold"), padding=(15, 8))
        style.configure("Success.TButton", background=accent_green, foreground=bg_dark,
                       font=("Segoe UI", 10, "bold"), padding=(15, 8))
        
        # Scale
        style.configure("TScale", background=bg_card, troughcolor="#21262d")
    
    def create_ui(self):
        """Create the user interface"""
        # Header
        header = tk.Frame(self.root, bg="#0d1117")
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title = ttk.Label(header, text="🔭 Astronomical Noise Generator", style="Title.TLabel")
        title.pack(anchor=tk.W)
        
        subtitle = ttk.Label(header, text="Add realistic telescope noise to clean images",
                            style="Subtitle.TLabel")
        subtitle.pack(anchor=tk.W)
        
        # Main content
        content = tk.Frame(self.root, bg="#0d1117")
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(content, style="Card.TFrame", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        ttk.Label(left_panel, text="📁 Input Image", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(left_panel, text="Select Clean Image", style="Accent.TButton",
                  command=self.select_input).pack(fill=tk.X, pady=(0, 5))
        
        self.input_label = ttk.Label(left_panel, text="No image selected", 
                                     foreground="#8b949e", wraplength=200)
        self.input_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Noise level
        ttk.Label(left_panel, text="🎚️ Noise Level", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        self.noise_var = tk.StringVar(value="medium")
        
        levels = [
            ("Light (Good conditions)", "light"),
            ("Medium (Typical amateur)", "medium"),
            ("Heavy (Poor conditions)", "heavy"),
            ("Extreme (Very poor)", "extreme")
        ]
        
        for text, value in levels:
            ttk.Radiobutton(left_panel, text=text, variable=self.noise_var,
                           value=value).pack(anchor=tk.W, pady=2)
        
        # Process button
        ttk.Label(left_panel, text="", font=("Segoe UI", 1)).pack(pady=10)
        ttk.Button(left_panel, text="Generate Noisy Image", style="Success.TButton",
                  command=self.generate_noise).pack(fill=tk.X, pady=(10, 5))
        
        ttk.Button(left_panel, text="Save Result", style="Accent.TButton",
                  command=self.save_result).pack(fill=tk.X)
        
        # Status
        self.status_label = ttk.Label(left_panel, text="Ready", 
                                      foreground="#3fb950", wraplength=200)
        self.status_label.pack(anchor=tk.W, pady=(20, 0))
        
        # Right panel - Image preview
        right_panel = tk.Frame(content, bg="#0d1117")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Before image
        before_frame = ttk.Frame(right_panel, style="Card.TFrame", padding="10")
        before_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(before_frame, text="Clean Image", 
                 font=("Segoe UI", 11, "bold"), foreground="#58a6ff").pack()
        
        self.before_canvas = tk.Label(before_frame, bg="#0d1117", 
                                      text="No image loaded\n\nSelect a clean image to begin",
                                      fg="#8b949e", font=("Segoe UI", 10))
        self.before_canvas.pack(expand=True, pady=10)
        
        # After image
        after_frame = ttk.Frame(right_panel, style="Card.TFrame", padding="10")
        after_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(after_frame, text="Noisy Image", 
                 font=("Segoe UI", 11, "bold"), foreground="#3fb950").pack()
        
        self.after_canvas = tk.Label(after_frame, bg="#0d1117",
                                     text="Generate noise to see result",
                                     fg="#8b949e", font=("Segoe UI", 10))
        self.after_canvas.pack(expand=True, pady=10)
    
    def select_input(self):
        """Select input image"""
        filepath = filedialog.askopenfilename(
            title="Select Clean Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.input_path = Path(filepath)
            self.input_label.config(text=f"📁 {self.input_path.name}", foreground="#58a6ff")
            self.load_preview()
    
    def load_preview(self):
        """Load and display preview of clean image"""
        try:
            img = Image.open(self.input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            self.clean_image = np.array(img)
            
            # Create thumbnail
            thumbnail = img.copy()
            thumbnail.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(thumbnail)
            self.before_canvas.config(image=photo, text="")
            self.before_canvas.image = photo
            
            self.status_label.config(text=f"Loaded: {img.size[0]}×{img.size[1]} pixels",
                                    foreground="#3fb950")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def generate_noise(self):
        """Generate noisy version of the image"""
        if self.clean_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        self.status_label.config(text="Generating noise...", foreground="#bc8cff")
        
        def process():
            try:
                # Add noise
                noise_level = self.noise_var.get()
                self.noisy_image = add_realistic_telescope_noise(
                    self.clean_image, 
                    noise_level=noise_level
                )
                
                # Update preview
                self.root.after(0, self.show_result)
                
            except Exception as ex:
                error_msg = str(ex)
                self.root.after(0, lambda msg=error_msg: self.on_error(msg))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def show_result(self):
        """Display the noisy result"""
        try:
            # Create thumbnail
            img = Image.fromarray(self.noisy_image)
            thumbnail = img.copy()
            thumbnail.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(thumbnail)
            self.after_canvas.config(image=photo, text="")
            self.after_canvas.image = photo
            
            self.status_label.config(text="✓ Noise added successfully!", 
                                    foreground="#3fb950")
            
        except Exception as e:
            self.on_error(str(e))
    
    def on_error(self, error):
        """Handle errors"""
        self.status_label.config(text="✗ Error", foreground="#f78166")
        messagebox.showerror("Error", f"Failed to generate noise:\n{error}")
    
    def save_result(self):
        """Save the noisy image"""
        if self.noisy_image is None:
            messagebox.showwarning("Warning", "Please generate a noisy image first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Noisy Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("TIFF files", "*.tif"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                img = Image.fromarray(self.noisy_image)
                img.save(filepath)
                self.status_label.config(text=f"✓ Saved to {Path(filepath).name}",
                                        foreground="#3fb950")
                messagebox.showinfo("Success", f"Noisy image saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")


def main():
    root = tk.Tk()
    app = NoiseGeneratorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
