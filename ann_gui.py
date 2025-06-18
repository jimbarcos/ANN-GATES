import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

import threading
import time
from neural_network import NeuralNetwork
from tkinter import filedialog, messagebox
import os

class ANNGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Logic Gates")
        self.root.geometry("1400x900")
        
        # Initialize neural network
        self.nn = NeuralNetwork(hidden_activation='ReLU', output_activation='Sigmoid')
        self.training_data = {}  # Initialize as empty dict instead of None
        self.current_gate = tk.StringVar(value="AND")
        self.is_training = False
        self.stop_training_flag = False
        self.errors_history = []

        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load default training data first
        self.create_default_training_data()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_training_tab()
        self.create_demonstration_tab()
    
    def create_training_tab(self):
        # Training tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training")
        
        # Left panel for controls with fixed width
        left_panel = ttk.Frame(self.training_frame, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)  # Prevent frame from shrinking
        
        # Gate selection
        ttk.Label(left_panel, text="Select Logic Gate:", font=('Arial', 12, 'bold')).pack(pady=5)
        gate_combo = ttk.Combobox(left_panel, textvariable=self.current_gate, 
                                 values=["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR"],
                                 font=('Arial', 10), width=20)
        gate_combo.pack(pady=5)
        
        # Training parameters
        ttk.Label(left_panel, text="Training Parameters:", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        
        ttk.Label(left_panel, text="Epochs:").pack()
        self.epochs_var = tk.IntVar(value=10000)
        ttk.Entry(left_panel, textvariable=self.epochs_var, width=20).pack(pady=2)
        
        ttk.Label(left_panel, text="Learning Rate:").pack()
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(left_panel, textvariable=self.lr_var, width=20).pack(pady=2)
        
        ttk.Label(left_panel, text="Target Error:").pack()
        self.error_var = tk.StringVar(value="1.00e-15")
        ttk.Entry(left_panel, textvariable=self.error_var, width=20).pack(pady=2)
        
        # Activation function selection with fixed width
        ttk.Label(left_panel, text="Hidden Layer Activation:").pack(pady=(10,0))
        self.activation_var = tk.StringVar(value="ReLU")
        activation_combo = ttk.Combobox(left_panel, textvariable=self.activation_var, 
                                       values=["ReLU", "Leaky ReLU", "Sigmoid", "Tansig", "Purelin"],
                                       font=('Arial', 10), state="readonly", width=20)
        activation_combo.pack(pady=2)
        activation_combo.bind('<<ComboboxSelected>>', self.on_activation_change)
        
        # Output layer activation function selection with fixed width
        ttk.Label(left_panel, text="Output Layer Activation:").pack(pady=(10,0))
        self.output_activation_var = tk.StringVar(value="Sigmoid")
        output_activation_combo = ttk.Combobox(left_panel, textvariable=self.output_activation_var, 
                                              values=["Sigmoid", "ReLU", "Leaky ReLU", "Tansig", "Purelin"],
                                              font=('Arial', 10), state="readonly", width=20)
        output_activation_combo.pack(pady=2)
        output_activation_combo.bind('<<ComboboxSelected>>', self.on_activation_change)
        
        # Control buttons frame with fixed width
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(pady=20, fill=tk.X)
        
        # Start Training button (highlighted)
        self.start_btn = tk.Button(button_frame, text="Start Training", command=self.start_training,
                                  bg="#4CAF50", fg="white", font=('Arial', 11, 'bold'),
                                  relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.start_btn.pack(pady=5, fill=tk.X)
        
        # Stop Training button
        self.stop_btn = tk.Button(button_frame, text="Stop Training", command=self.stop_training,
                                 bg="#FF5722", fg="white", font=('Arial', 10),
                                 relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.stop_btn.pack(pady=5, fill=tk.X)
        
        # Reset Network button
        self.reset_btn = tk.Button(button_frame, text="Reset Network", command=self.reset_network,
                                  bg="#2196F3", fg="white", font=('Arial', 10),
                                  relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.reset_btn.pack(pady=5, fill=tk.X)
        
        # Model Management section
        model_frame = ttk.LabelFrame(left_panel, text="Model Management", padding=5)
        model_frame.pack(pady=(10,5), fill=tk.X)
        
        # Save Model button
        self.save_btn = tk.Button(model_frame, text="Save Model", command=self.save_model,
                                 bg="#9C27B0", fg="white", font=('Arial', 10),
                                 relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.save_btn.pack(pady=2, fill=tk.X)
        
        # Load Model button
        self.load_btn = tk.Button(model_frame, text="Load Model", command=self.load_model,
                                 bg="#FF9800", fg="white", font=('Arial', 10),
                                 relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.load_btn.pack(pady=2, fill=tk.X)
        
        # Continue Training button
        self.continue_btn = tk.Button(model_frame, text="Continue Training", command=self.continue_training,
                                     bg="#607D8B", fg="white", font=('Arial', 10),
                                     relief=tk.RAISED, bd=2, cursor="hand2", width=25)
        self.continue_btn.pack(pady=2, fill=tk.X)
        
        # Training data display with fixed size
        ttk.Label(left_panel, text="Training Data:", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.data_text = tk.Text(left_panel, width=30, height=8, font=('Courier', 9))
        self.data_text.pack(pady=5)
        
        # Progress section with improved styling and fixed width
        progress_frame = ttk.LabelFrame(left_panel, text="Training Progress", padding=15)
        progress_frame.pack(pady=(20,15), fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to train")
        self.status_label = tk.Label(progress_frame, textvariable=self.status_var, 
                                   font=('Arial', 10, 'bold'), fg="#2E7D32")
        self.status_label.pack(pady=(0,5))
        
        # Epoch counter
        self.epoch_var = tk.StringVar(value="Epoch: 0")
        self.epoch_label = tk.Label(progress_frame, textvariable=self.epoch_var, 
                                  font=('Arial', 9), fg="#1976D2")
        self.epoch_label.pack(pady=(0,5))
        
        # Error rate with highlighting - increased height
        self.error_display_var = tk.StringVar(value="Error: Not started")
        self.error_label = tk.Label(progress_frame, textvariable=self.error_display_var, 
                                  font=('Arial', 10, 'bold'), fg="#D32F2F", 
                                  bg="#FFEBEE", relief=tk.RIDGE, bd=1, pady=8, height=2)
        self.error_label.pack(pady=(5,10), fill=tk.X)
        
        # Progress bar removed as requested
        
        # Right panel for visualization
        right_panel = ttk.Frame(self.training_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for training visualization
        self.fig_training, (self.ax_error, self.ax_weights) = plt.subplots(2, 1, figsize=(10, 12))
        self.canvas_training = FigureCanvasTkAgg(self.fig_training, right_panel)
        self.canvas_training.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.ax_error.set_title("Training Error Over Time", fontsize=14)
        self.ax_error.set_xlabel("Epoch (x100)")
        self.ax_error.set_ylabel("Mean Squared Error")
        self.ax_error.set_yscale('log')
        self.ax_error.grid(True, alpha=0.3)
        
        self.ax_weights.set_title("Network Weights and Biases", fontsize=14, y=0.9)
        self.ax_weights.axis('off')
        
        # Update training data display when gate changes
        gate_combo.bind('<<ComboboxSelected>>', self.update_training_data_display)
        # Initialize training data display after creating default data
        self.root.after(100, self.update_training_data_display)
        

    def create_demonstration_tab(self):
        # Demonstration tab
        self.demo_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.demo_frame, text="Network Visualization")
        
        # Control panel
        control_panel = ttk.Frame(self.demo_frame)
        control_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_panel, text="Test Inputs:", font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        ttk.Label(control_panel, text="Input 1:").pack(side=tk.LEFT, padx=(20,5))
        self.input1_var = tk.DoubleVar(value=0)
        input1_entry = ttk.Entry(control_panel, textvariable=self.input1_var, width=10)
        input1_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_panel, text="Input 2:").pack(side=tk.LEFT, padx=(10,5))
        self.input2_var = tk.DoubleVar(value=0)
        input2_entry = ttk.Entry(control_panel, textvariable=self.input2_var, width=10)
        input2_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_panel, text="Test", command=self.test_network).pack(side=tk.LEFT, padx=20)
        
        self.output_var = tk.StringVar(value="Output: Not tested")
        ttk.Label(control_panel, textvariable=self.output_var, font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=20)
        
        # Test all combinations button
        ttk.Button(control_panel, text="Test All Combinations", command=self.test_all_combinations).pack(side=tk.LEFT, padx=10)
        
        # Create matplotlib figure for network visualization
        self.fig_demo, self.ax_demo = plt.subplots(figsize=(14, 10))
        self.canvas_demo = FigureCanvasTkAgg(self.fig_demo, self.demo_frame)
        self.canvas_demo.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Draw initial network
        self.draw_network()
        
        # Bind Enter key to test
        input1_entry.bind('<Return>', lambda e: self.test_network())
        input2_entry.bind('<Return>', lambda e: self.test_network())
    
    def create_default_training_data(self):
        """Create default training data for all logic gates"""
        # Truth tables for each gate
        gates_data = {
            'AND': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [0, 0, 0, 1]},
            'OR': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [0, 1, 1, 1]},
            'XOR': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [0, 1, 1, 0]},
            'NOT': {'inputs': [[0], [1]], 'outputs': [1, 0]},  # Only uses first input
            'NAND': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [1, 1, 1, 0]},
            'NOR': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [1, 0, 0, 0]},
            'XNOR': {'inputs': [[0,0], [0,1], [1,0], [1,1]], 'outputs': [1, 0, 0, 1]}
        }
        
        self.training_data = gates_data
    
    def update_training_data_display(self, event=None):
        """Update the training data display"""
        gate = self.current_gate.get()
        if gate and gate in self.training_data:
            data = self.training_data[gate]
            self.data_text.delete(1.0, tk.END)
            
            # Display truth table
            if gate == "NOT":
                self.data_text.insert(tk.END, f"{gate} Gate Truth Table:\n")
                self.data_text.insert(tk.END, "Input | Output\n")
                self.data_text.insert(tk.END, "------|-------\n")
                if data and 'inputs' in data and 'outputs' in data:
                    for inp, out in zip(data['inputs'], data['outputs']):
                        self.data_text.insert(tk.END, f"  {inp[0]}   |   {out}\n")
            else:
                self.data_text.insert(tk.END, f"{gate} Gate Truth Table:\n")
                self.data_text.insert(tk.END, "I1 I2 | Output\n")
                self.data_text.insert(tk.END, "------|-------\n")
                if data and 'inputs' in data and 'outputs' in data:
                    for inp, out in zip(data['inputs'], data['outputs']):
                        self.data_text.insert(tk.END, f" {inp[0]} {inp[1]}  |   {out}\n")
    
    def on_activation_change(self, event=None):
        """Called when activation function is changed"""
        hidden_activation = self.activation_var.get()
        output_activation = self.output_activation_var.get()
        self.nn = NeuralNetwork(hidden_activation=hidden_activation, output_activation=output_activation)
        self.status_var.set(f"Network reset with {hidden_activation}/{output_activation} activation")
        self.epoch_var.set("Epoch: 0")
        self.error_display_var.set("Error: Not started")
        self.error_label.config(fg="#D32F2F", bg="#FFEBEE")
        self.errors_history = []
        self.is_training = False
        self.start_btn.config(state="normal")
        
        # Update network visualization
        self.draw_network()
    
    def start_training(self):
        """Start training the neural network"""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
        
        gate = self.current_gate.get()
        if not gate or gate not in self.training_data:
            messagebox.showerror("Error", f"No training data for gate: {gate}")
            return
        
        # Prepare training data
        data = self.training_data[gate]
        if gate == "NOT":
            # For NOT gate, use only first input and pad with zeros
            X = np.array([[inp[0], 0] for inp in data['inputs']])
        else:
            X = np.array(data['inputs'])
        y = np.array(data['outputs']).reshape(-1, 1)
        
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        target_error = float(self.error_var.get())
        
        self.is_training = True
        self.stop_training_flag = False  # Reset stop flag
        self.errors_history = []
        self.target_epochs = epochs  # Store target epochs for final display
        
        # Update UI for training state
        self.start_btn.config(state="disabled")
        self.status_var.set("Starting training...")
        self.error_display_var.set("Error: Initializing...")
        
        # Start training in separate thread
        training_thread = threading.Thread(
            target=self._train_network,
            args=(X, y, epochs, lr, target_error)
        )
        training_thread.daemon = True
        training_thread.start()
    
    def _train_network(self, X, y, epochs, lr, target_error):
        """Train network in separate thread"""
        def update_callback(epoch, error, W1, b1, W2, b2):
            # Check if training should stop
            if self.stop_training_flag:
                return False  # Signal to stop training
                
            self.errors_history.append(error)
            

            
            # Update GUI in main thread
            if not self.stop_training_flag:
                self.root.after(0, self._update_training_display, epoch, error, W1, b1, W2, b2)
            return True  # Continue training
        
        # Progress bar animation removed
        
        try:
            errors = self.nn.train(X, y, epochs, lr, target_error, update_callback, self)
            if not self.stop_training_flag:
                self.root.after(0, self._training_complete)
        except Exception as e:
            if not self.stop_training_flag:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.is_training = False
    
    def _update_training_display(self, epoch, error, W1, b1, W2, b2):
        """Update training display in main thread"""
        # Check if training was stopped
        if self.stop_training_flag or not self.is_training:
            return
            
        try:
            # Update status and progress
            self.status_var.set("Training in progress...")
            self.epoch_var.set(f"Epoch: {epoch:,}")
            
            # Update error with color coding
            if error < 1e-10:
                error_color = "#1B5E20"  # Dark green for very low error
                bg_color = "#E8F5E8"
            elif error < 1e-5:
                error_color = "#388E3C"  # Green for low error
                bg_color = "#F1F8E9"
            elif error < 1e-2:
                error_color = "#F57C00"  # Orange for medium error
                bg_color = "#FFF8E1"
            else:
                error_color = "#D32F2F"  # Red for high error
                bg_color = "#FFEBEE"
                
            self.error_display_var.set(f"Error: {error:.2e}")
            self.error_label.config(fg=error_color, bg=bg_color)
        except tk.TclError:
            # GUI element was destroyed, stop training
            self.stop_training_flag = True
            return
        
        try:
            # Update error plot
            if len(self.errors_history) > 1:
                self.ax_error.clear()
                self.ax_error.plot(self.errors_history, 'b-', linewidth=2)
                self.ax_error.set_title("Training Error Over Time", fontsize=14)
                self.ax_error.set_xlabel("Epoch (x100)")
                self.ax_error.set_ylabel("Mean Squared Error")
                self.ax_error.set_yscale('log')
                self.ax_error.grid(True, alpha=0.3)
            
            # Update weights display
            self.ax_weights.clear()
            self.ax_weights.set_title("Network Weights and Biases", fontsize=14, y=0.95)
            
            # Display weights and biases as text
            weights_text = f"Hidden Layer Weights (W1):\n{np.round(W1, 4)}\n\n"
            weights_text += f"Hidden Layer Biases (b1):\n{np.round(b1, 4)}\n\n"
            weights_text += f"Output Layer Weights (W2):\n{np.round(W2, 4)}\n\n"
            weights_text += f"Output Layer Biases (b2):\n{np.round(b2, 4)}\n\n"
            
            # Add current training results
            gate = self.current_gate.get()
            if gate and gate in self.training_data:
                data = self.training_data[gate]
                if gate == "NOT":
                    # For NOT gate, use only first input and pad with zeros
                    X = np.array([[inp[0], 0] for inp in data['inputs']])
                else:
                    X = np.array(data['inputs'])
                y_target = np.array(data['outputs']).reshape(-1, 1)
                
                # Get current predictions
                y_pred = self.nn.predict(X)
                individual_errors = np.abs(y_target - y_pred)
                
                weights_text += "Training Results:\n"
                weights_text += "Input → Target | Output | Error\n"
                weights_text += "--------------------------------\n"
                
                for i in range(len(X)):
                    if gate == "NOT":
                        input_str = f"[{X[i][0]:.0f}]"
                    else:
                        input_str = f"[{X[i][0]:.0f},{X[i][1]:.0f}]"
                    weights_text += f"{input_str:>6} → {y_target[i][0]:.0f} | {y_pred[i][0]:.6f} | {individual_errors[i][0]:.2e}\n"
                
                weights_text += f"\nMean Squared Error: {error:.2e}"
            
            self.ax_weights.text(0.05, 0.95, weights_text, transform=self.ax_weights.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
            self.ax_weights.axis('off')
            
            self.canvas_training.draw()
            

            
            # Update network visualization
            self.draw_network()
        except tk.TclError:
            # GUI element was destroyed, stop training
            self.stop_training_flag = True
            return
    
    def _training_complete(self):
        """Called when training is complete"""
        self.is_training = False
        self.status_var.set("Training completed!")
        
        # Update final epoch count to show actual completion
        actual_epochs = getattr(self, 'target_epochs', len(self.errors_history))
        self.epoch_var.set(f"Epoch: {actual_epochs:,}")
        
        # Show final error with success styling
        final_error = self.errors_history[-1] if self.errors_history else 0
        self.error_display_var.set(f"Final Error: {final_error:.2e}")
        self.error_label.config(fg="#1B5E20", bg="#E8F5E8")
        
        messagebox.showinfo("Training Complete", 
                          f"Neural network training completed successfully!\n"
                          f"Final error: {final_error:.2e}\n"
                          f"Total epochs: {actual_epochs:,}")
        
        # Enable testing
        self.start_btn.config(state="normal")
    
    def stop_training(self):
        """Stop training"""
        self.stop_training_flag = True
        self.is_training = False
        
        try:
            self.status_var.set("Training stopped by user")
            self.error_display_var.set("Error: Stopped")
            self.error_label.config(fg="#FF5722", bg="#FFEBEE")
            self.start_btn.config(state="normal")
        except tk.TclError:
            # GUI elements might be destroyed
            pass
    
    def reset_network(self):
        """Reset the neural network"""
        hidden_activation = self.activation_var.get()
        output_activation = self.output_activation_var.get()
        self.nn = NeuralNetwork(hidden_activation=hidden_activation, output_activation=output_activation)
        self.status_var.set("Network reset - ready to train")
        self.epoch_var.set("Epoch: 0")
        self.error_display_var.set("Error: Not started")
        self.error_label.config(fg="#D32F2F", bg="#FFEBEE")
        self.errors_history = []
        self.is_training = False
        self.start_btn.config(state="normal")
        
        # Clear plots
        self.ax_error.clear()
        self.ax_error.set_title("Training Error Over Time", fontsize=14)
        self.ax_error.set_xlabel("Epoch (x100)")
        self.ax_error.set_ylabel("Mean Squared Error")
        self.ax_error.set_yscale('log')
        self.ax_error.grid(True, alpha=0.3)
        
        self.ax_weights.clear()
        self.ax_weights.set_title("Network Weights and Biases", fontsize=14)
        self.ax_weights.axis('off')
        
        self.canvas_training.draw()
        self.draw_network()
    
    def test_network(self):
        """Test the network with given inputs"""
        input1 = self.input1_var.get()
        input2 = self.input2_var.get()
        
        # Test the network
        test_input = np.array([[input1, input2]])
        output = self.nn.predict(test_input)
        
        self.output_var.set(f"Output: {output[0][0]:.7f}")
        
        # Update network visualization with current values
        self.draw_network(test_input[0])
    
    def test_all_combinations(self):
        """Test all input combinations and display results"""
        results = []
        for i in [0, 1]:
            for j in [0, 1]:
                test_input = np.array([[i, j]])
                output = self.nn.predict(test_input)
                results.append(f"({i}, {j}) → {output[0][0]:.7f}")
        
        result_text = "All combinations:\n" + "\n".join(results)
        messagebox.showinfo("Test Results", result_text)
    
    def draw_network(self, test_input=None):
        """Draw the neural network architecture"""
        self.ax_demo.clear()
        self.ax_demo.set_xlim(0, 12)
        self.ax_demo.set_ylim(-1, 10)
        self.ax_demo.set_aspect('equal')
        self.ax_demo.axis('off')
        hidden_func = self.nn.hidden_activation if hasattr(self.nn, 'hidden_activation') else 'ReLU'
        output_func = self.nn.output_activation if hasattr(self.nn, 'output_activation') else 'Sigmoid'
        self.ax_demo.set_title(f"Neural Network Architecture (2-3-1) | Hidden: {hidden_func} | Output: {output_func}", 
                              fontsize=14, fontweight='bold')
        
        # Node positions
        input_pos = [(2, 7), (2, 3)]  # I1, I2
        hidden_pos = [(6, 8.5), (6, 5), (6, 1.5)]  # N1, N2, N3
        output_pos = [(10, 5)]  # N4 (Output)
        
        # Layer labels
        self.ax_demo.text(2, 9.5, 'Input Layer', ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue')
        self.ax_demo.text(6, 9.5, 'Hidden Layer', ha='center', va='center', fontsize=12, fontweight='bold', color='darkgreen')
        self.ax_demo.text(10, 9.5, 'Output Layer', ha='center', va='center', fontsize=12, fontweight='bold', color='darkred')
        
        # Draw input nodes
        for i, (x, y) in enumerate(input_pos):
            circle = patches.Circle((x, y), 0.4, color='lightblue', ec='black', linewidth=2)
            self.ax_demo.add_patch(circle)
            self.ax_demo.text(x, y, f'I{i+1}', ha='center', va='center', fontweight='bold', fontsize=12)
            if test_input is not None:
                self.ax_demo.text(x-1, y, f'{test_input[i]:.1f}', ha='center', va='center', 
                                color='red', fontweight='bold', fontsize=14)
        
        # Draw hidden nodes
        hidden_func = self.nn.hidden_activation if hasattr(self.nn, 'hidden_activation') else 'ReLU'
        for i, (x, y) in enumerate(hidden_pos):
            circle = patches.Circle((x, y), 0.4, color='lightgreen', ec='black', linewidth=2)
            self.ax_demo.add_patch(circle)
            self.ax_demo.text(x, y, f'N{i+1}', ha='center', va='center', fontweight='bold', fontsize=12)
            
            # Show bias
            bias = self.nn.b1[0][i]
            self.ax_demo.text(x, y+0.7, f'b:{bias:.2f}', ha='center', va='center', 
                            fontsize=9, color='blue', fontweight='bold')
            
            if test_input is not None and self.nn.a1 is not None:
                activation = self.nn.a1[0][i]
                self.ax_demo.text(x, y-0.8, f'{activation:.3f}', ha='center', va='center', 
                                color='green', fontsize=10, fontweight='bold')
        
        # Add activation function box for hidden layer (separate from neurons)
        self.ax_demo.add_patch(patches.Rectangle((4.5, -0.3), 3, 0.8, 
                                               facecolor='lightgreen', edgecolor='black', alpha=0.8))
        self.ax_demo.text(6, 0.1, f'Activation Function\n{hidden_func}', 
                        ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen')
        
        # Draw output node
        x, y = output_pos[0]
        circle = patches.Circle((x, y), 0.4, color='lightcoral', ec='black', linewidth=2)
        self.ax_demo.add_patch(circle)
        self.ax_demo.text(x, y, 'O', ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Show bias
        bias = self.nn.b2[0][0]
        self.ax_demo.text(x, y+0.7, f'b:{bias:.2f}', ha='center', va='center', 
                        fontsize=9, color='blue', fontweight='bold')
        
        if test_input is not None and self.nn.a2 is not None:
            output_val = self.nn.a2[0][0]
            self.ax_demo.text(x+1, y, f'{output_val:.3f}', ha='center', va='center', 
                            color='red', fontweight='bold', fontsize=14)
        
        # Add activation function box for output layer (separate from neuron)
        output_func = self.nn.output_activation if hasattr(self.nn, 'output_activation') else 'Sigmoid'
        self.ax_demo.add_patch(patches.Rectangle((8.5, -0.3), 3, 0.8, 
                                               facecolor='lightcoral', edgecolor='black', alpha=0.8))
        self.ax_demo.text(10, 0.1, f'Activation Function\n{output_func}', 
                        ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')
        
        # Draw connections with weights
        # Input to hidden connections
        for i, (x1, y1) in enumerate(input_pos):
            for j, (x2, y2) in enumerate(hidden_pos):
                self.ax_demo.plot([x1+0.4, x2-0.4], [y1, y2], 'k-', alpha=0.6, linewidth=1.5)
                # Show weight
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                weight = self.nn.W1[i][j]
                self.ax_demo.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center', 
                                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        
        # Hidden to output connections
        for i, (x1, y1) in enumerate(hidden_pos):
            x2, y2 = output_pos[0]
            self.ax_demo.plot([x1+0.4, x2-0.4], [y1, y2], 'k-', alpha=0.6, linewidth=1.5)
            # Show weight
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            weight = self.nn.W2[i][0]
            self.ax_demo.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center', 
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        
        # Add legend
        legend_elements = [
            patches.Circle((0, 0), 0.1, color='lightblue', label='Input Layer'),
            patches.Circle((0, 0), 0.1, color='lightgreen', label='Hidden Layer'),
            patches.Circle((0, 0), 0.1, color='lightcoral', label='Output Layer')
        ]
        self.ax_demo.legend(handles=legend_elements, loc='upper right' , bbox_to_anchor=(1.2, 0.98), fontsize=10)

        self.canvas_demo.draw()
    
    def save_model(self):
        """Save the current trained model"""
        if not hasattr(self.nn, 'W1') or self.nn.W1 is None:
            messagebox.showerror("Error", "No model to save. Please train a model first.")
            return
        
        # Create models directory if it doesn't exist
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Get current activation functions for filename
        hidden_act = self.activation_var.get().replace(" ", "")
        output_act = self.output_activation_var.get().replace(" ", "")
        gate = self.current_gate.get()
        
        # Default filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{gate}_{hidden_act}_{output_act}_{timestamp}.pkl"
        
        # Ask user for filename
        filepath = filedialog.asksaveasfilename(
            title="Save Neural Network Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=models_dir,
            initialfile=default_filename
        )
        
        if filepath:
            success = self.nn.save_model(filepath)
            if success:
                messagebox.showinfo("Success", f"Model saved successfully!\n\nFile: {os.path.basename(filepath)}\nLocation: {os.path.dirname(filepath)}")
            else:
                messagebox.showerror("Error", "Failed to save model. Please try again.")
    
    def load_model(self):
        """Load a previously saved model"""
        models_dir = "saved_models"
        initial_dir = models_dir if os.path.exists(models_dir) else "."
        
        filepath = filedialog.askopenfilename(
            title="Load Neural Network Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        
        if filepath:
            # Stop any ongoing training
            if self.is_training:
                self.stop_training()
            
            success = self.nn.load_model(filepath)
            if success:
                # Update GUI to match loaded model
                self.activation_var.set(self.nn.hidden_activation)
                self.output_activation_var.set(self.nn.output_activation)
                
                # Reset training state
                self.errors_history = []
                self.epoch_var.set("Epoch: 0")
                self.error_display_var.set("Model loaded successfully")
                self.error_label.config(fg="#1B5E20", bg="#E8F5E8")
                self.status_var.set("Model loaded - ready for testing or training")
                
                # Update visualizations
                self.draw_network()
                
                # Clear plots
                self.ax_error.clear()
                self.ax_error.set_title("Training Error Over Time", fontsize=14)
                self.ax_error.set_xlabel("Epoch (x100)")
                self.ax_error.set_ylabel("Mean Squared Error")
                self.ax_error.set_yscale('log')
                self.ax_error.grid(True, alpha=0.3)
                
                self.ax_weights.clear()
                self.ax_weights.set_title("Network Weights and Biases", fontsize=14, y=0.95)
                self.ax_weights.axis('off')
                self.canvas_training.draw()
                
                # Show model summary
                summary = self.nn.get_model_summary()
                summary_text = f"""Model Summary:
Architecture: {summary['architecture']}
Hidden Activation: {summary['hidden_activation']}
Output Activation: {summary['output_activation']}
Total Parameters: {summary['total_parameters']}

Model loaded successfully!
You can now test it or continue training."""
                
                messagebox.showinfo("Model Loaded", summary_text)
            else:
                messagebox.showerror("Error", "Failed to load model. Please check the file and try again.")
    
    def continue_training(self):
        """Continue training the loaded model"""
        if not hasattr(self.nn, 'W1') or self.nn.W1 is None:
            messagebox.showerror("Error", "No model loaded. Please load a model first or train a new one.")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return
        
        # Use current training parameters to continue training
        gate = self.current_gate.get()
        if gate not in self.training_data:
            messagebox.showerror("Error", f"No training data available for {gate}. Please select a valid gate.")
            return
        
        # Ask user if they want to continue with current parameters
        result = messagebox.askyesno("Continue Training", 
                                   f"Continue training with current parameters?\n\n"
                                   f"Gate: {gate}\n"
                                   f"Epochs: {self.epochs_var.get():,}\n"
                                   f"Learning Rate: {self.lr_var.get()}\n"
                                   f"Target Error: {self.error_var.get()}\n\n"
                                   f"The model will continue from its current state.")
        
        if result:
            # Start training (this will continue from current weights)
            self.start_training()
    
    def on_closing(self):
        """Handle window closing event"""
        if self.is_training:
            self.stop_training()
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    
    # Configure tkinter style
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    
    # Configure window icon and properties
    root.resizable(True, True)
    root.minsize(1200, 800)
    
    app = ANNGui(root)
    
    root.mainloop()

if __name__ == "__main__":
    main() 