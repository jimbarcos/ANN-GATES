import numpy as np
import pickle
import json
from datetime import datetime

class NeuralNetwork:
    def __init__(self, hidden_activation='ReLU', output_activation='Sigmoid'):
        # Network architecture: 2 inputs -> 3 hidden -> 1 output
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # He initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Store activations for visualization
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def sigmoid(self, x):
        # σ(x) = 1/(1+e^(-x))
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tansig(self, x):
        # tansig(x) = (e^x - e^(-x))/(e^x + e^(-x)) = tanh(x)
        return np.tanh(np.clip(x, -250, 250))
    
    def tansig_derivative(self, x):
        return 1 - x**2
    
    def purelin(self, x):
        # purelin(x) = x (linear activation)
        return x
    
    def purelin_derivative(self, x):
        return np.ones_like(x)
    
    def apply_activation(self, x, activation_type):
        if activation_type == 'ReLU':
            return self.relu(x)
        elif activation_type == 'Leaky ReLU':
            return self.leaky_relu(x)
        elif activation_type == 'Sigmoid':
            return self.sigmoid(x)
        elif activation_type == 'Tansig':
            return self.tansig(x)
        elif activation_type == 'Purelin':
            return self.purelin(x)
        else:
            return self.relu(x)  # default
    
    def apply_activation_derivative(self, x, activation_type):
        if activation_type == 'ReLU':
            return self.relu_derivative(x)
        elif activation_type == 'Leaky ReLU':
            return self.leaky_relu_derivative(x)
        elif activation_type == 'Sigmoid':
            return self.sigmoid_derivative(x)
        elif activation_type == 'Tansig':
            return self.tansig_derivative(x)
        elif activation_type == 'Purelin':
            return self.purelin_derivative(x)
        else:
            return self.relu_derivative(x)  # default
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.apply_activation(self.z1, self.hidden_activation)  # Using selected activation for hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.apply_activation(self.z2, self.output_activation)  # Using selected activation for output layer
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        if self.output_activation == 'Sigmoid':
            dZ2 = output - y  # For sigmoid with cross-entropy, this is the simplified derivative
        else:
            dZ2 = (output - y) * self.apply_activation_derivative(self.a2, self.output_activation)
        
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dZ1 = np.dot(dZ2, self.W2.T) * self.apply_activation_derivative(self.a1, self.hidden_activation)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs, learning_rate, target_error=1e-15, callback=None, gui_instance=None):
        errors = []
        final_epoch = 0
        for epoch in range(epochs):
            # Check if training should stop
            if gui_instance and gui_instance.stop_training_flag:
                print(f"Training stopped by user at epoch {epoch}")
                final_epoch = epoch
                break
                
            # Forward propagation
            output = self.forward(X)
            
            # Calculate error
            error = np.mean(np.square(y - output))
            errors.append(error)
            
            # Backward propagation
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            
            # Update weights
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # Apply NOT gate restrictions if needed (enforced from GUI callback)
            # This ensures input 2 weights remain zero for NOT gate
            
            # Callback for GUI updates
            if callback and epoch % 100 == 0:
                # Check callback return value for stop signal
                continue_training = callback(epoch, error, self.W1, self.b1, self.W2, self.b2)
                if continue_training is False:
                    print(f"Training stopped by callback at epoch {epoch}")
                    final_epoch = epoch
                    break
            
            # Check if target error is reached
            if error < target_error:
                print(f"Target error reached at epoch {epoch}")
                final_epoch = epoch
                break
            
            final_epoch = epoch
                
        return errors, final_epoch
    
    def predict(self, X):
        return self.forward(X)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Args:
            filepath (str): Path where to save the model (should end with .pkl)
        """
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'save_timestamp': datetime.now().isoformat(),
            'model_info': f"2-{self.hidden_size}-1 Neural Network with {self.hidden_activation}/{self.output_activation} activations"
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Args:
            filepath (str): Path to the saved model file
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore weights and biases
            self.W1 = model_data['W1']
            self.b1 = model_data['b1']
            self.W2 = model_data['W2']
            self.b2 = model_data['b2']
            
            # Restore activation functions
            self.hidden_activation = model_data['hidden_activation']
            self.output_activation = model_data['output_activation']
            
            # Restore architecture (for compatibility check)
            self.input_size = model_data['input_size']
            self.hidden_size = model_data['hidden_size']
            self.output_size = model_data['output_size']
            
            print(f"Model loaded successfully from {filepath}")
            print(f"Model info: {model_data.get('model_info', 'N/A')}")
            print(f"Saved on: {model_data.get('save_timestamp', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_summary(self):
        """
        Get a summary of the current model
        """
        total_params = (self.W1.size + self.b1.size + self.W2.size + self.b2.size)
        summary = {
            'architecture': f"{self.input_size}-{self.hidden_size}-{self.output_size}",
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'total_parameters': total_params,
            'weight_shapes': {
                'W1': self.W1.shape,
                'b1': self.b1.shape,
                'W2': self.W2.shape,
                'b2': self.b2.shape
            }
        }
        return summary 
    
    def enforce_not_gate_restrictions(self):
        """
        Enforce NOT gate restrictions by setting input 2 weights to zero
        This should be called from the GUI when NOT gate is selected
        """
        if hasattr(self, 'W1'):
            self.W1[1, :] = 0  # Set all weights from input 2 to hidden layer to 0 