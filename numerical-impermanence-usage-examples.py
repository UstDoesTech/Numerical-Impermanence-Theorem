import numpy as np
import matplotlib.pyplot as plt
from numerical_impermanence_framework import NumericalImpermanence, Context, apply_numerical_impermanence

# Example 1: Transforming a number with a Heisenberg-inspired context function
def heisenberg_context(value):
    """
    A context function inspired by the Heisenberg uncertainty principle.
    
    As the position (value) becomes more precisely defined,
    its momentum (represented as oscillation frequency) becomes more uncertain.
    """
    if isinstance(value, (int, float)):
        # Create an x-range centered around the value
        x = np.linspace(value - 5, value + 5, 1000)
        
        # Precision of position (inverse of spread)
        precision = 5.0
        
        # Position component: Gaussian centered at the value
        position = np.exp(-(x - value)**2 * precision)
        
        # Momentum component: oscillations with increasing uncertainty as precision increases
        # Higher precision leads to more rapid and irregular oscillations
        momentum = np.sin(x * value * precision) * np.cos(x**2 * 0.1)
        
        # Combined representation
        combined = position * (0.5 + 0.5 * momentum)
        
        return combined
    return value

# Example 2: A fractal-based context where numbers transform into self-similar patterns
def fractal_context(value):
    """
    A context where numbers transform into fractal-like patterns.
    
    The number's magnitude determines the complexity and self-similarity of the pattern.
    """
    if isinstance(value, (int, float)):
        # Generate x values
        x = np.linspace(-2, 2, 1000)
        
        # Base frequency determined by the value
        base_freq = max(0.1, abs(value))
        
        # Generate a fractal-like pattern using multiple frequency components
        # Each component is a fraction of the previous one (self-similarity)
        pattern = np.zeros_like(x)
        amplitude = 1.0
        
        # Add multiple frequency components
        for i in range(1, int(abs(value) * 5) + 3):
            # Each iteration adds a higher frequency component with smaller amplitude
            freq = base_freq * i
            phase = value * i * np.pi / 4
            pattern += amplitude * np.sin(freq * x + phase)
            amplitude *= 0.7  # Diminishing amplitude for higher frequencies
        
        # Normalize the pattern
        pattern = pattern / np.max(np.abs(pattern))
        
        # For negative values, invert the pattern
        if value < 0:
            pattern = -pattern
            
        return pattern
    return value

# Example 3: A relativistic context where numbers transform based on relative velocity
def relativistic_context(value):
    """
    A context where numbers transform based on relativistic principles.
    
    Numbers undergo Lorentz transformation-like effects, exhibiting length contraction
    and other relativistic phenomena.
    """
    if isinstance(value, (int, float)):
        # Relative velocity as a fraction of c (speed of light)
        # Bounded between 0 and 0.99c based on the value
        v_rel = min(0.99, abs(value) / (1 + abs(value)))
        
        # Lorentz factor
        gamma = 1 / np.sqrt(1 - v_rel**2)
        
        # Generate coordinate values (proper position)
        x_proper = np.linspace(-5, 5, 1000)
        
        # Apply length contraction to create the relativistic view
        x_relative = x_proper / gamma
        
        # Create a representation of the number in this relativistic context
        # A pulse centered at the value, but with relativistic effects
        center = value
        rest_width = 1.0
        
        # Width contracts with gamma
        contracted_width = rest_width / gamma
        
        # Create a contracted pulse
        rest_pulse = np.exp(-(x_proper - center)**2 / (2 * rest_width**2))
        contracted_pulse = np.exp(-(x_proper - center)**2 / (2 * contracted_width**2))
        
        # Return both the rest and contracted representations for comparison
        result = {
            'proper_frame': rest_pulse,
            'moving_frame': contracted_pulse,
            'x_values': x_proper,
            'gamma': gamma,
            'relative_velocity': v_rel
        }
        
        return result
    return value

# Example 4: A quantum field theory inspired context
def qft_context(value):
    """
    A context inspired by quantum field theory where numbers manifest as field excitations.
    
    The value determines the energy/amplitude of the field excitation.
    """
    if isinstance(value, (int, float)):
        # Generate spacetime coordinates
        x = np.linspace(-10, 10, 1000)  # Space
        
        # Field parameters based on the value
        mass = max(0.1, abs(value) * 0.5)  # Mass parameter
        amplitude = np.sign(value) * min(1.0, abs(value) / 5)  # Field amplitude
        
        # Create wave packet representing field excitation
        # Higher values create higher frequency excitations
        frequency = mass + abs(value) * 0.2
        envelope = np.exp(-x**2 / (2 * (5 / mass)**2))  # Localization envelope
        oscillation = np.sin(frequency * x)  # Oscillatory component
        
        # Combine to create a wave packet
        field = amplitude * envelope * oscillation
        
        # Add quantum fluctuations (noise proportional to inverse of value)
        noise_level = 0.05 / (0.1 + abs(value))
        fluctuations = np.random.normal(0, noise_level, len(x))
        field += fluctuations
        
        return field
    return value

# Example 5: A context based on cyclical time rather than linear time
def cyclical_time_context(value):
    """
    A context where numbers exist in cyclical rather than linear time.
    
    Numbers transform into repeating patterns that fold back on themselves.
    """
    if isinstance(value, (int, float)):
        # Cyclical parameter space (angle in radians)
        theta = np.linspace(0, 2*np.pi, 1000)
        
        # Modulo the value to fit within a cycle (e.g., clock arithmetic)
        # For this example, we'll use a cycle of length 10
        cycle_length = 10
        cyclic_value = value % cycle_length
        
        # Phase angle determined by the cyclic value
        phase = cyclic_value * 2 * np.pi / cycle_length
        
        # Generate a representation with multiple harmonics
        harmonics = 5
        pattern = np.zeros_like(theta)
        
        for n in range(1, harmonics + 1):
            # Each harmonic depends on the cyclic value
            amplitude = 1 / n * np.exp(-0.5 * abs(n - cycle_length/2 + cyclic_value))
            pattern += amplitude * np.sin(n * (theta + phase))
        
        # Normalize
        pattern = pattern / np.max(np.abs(pattern))
        
        # Create a representation showing both the linear and cyclical forms
        result = {
            'original_value': value,
            'cyclic_value': cyclic_value,
            'phase_angle': phase,
            'theta': theta,
            'pattern': pattern
        }
        
        return result
    return value

# Run examples and visualize the results
if __name__ == "__main__":
    # Create a plot for each example
    plt.figure(figsize=(15, 18))
    
    # Example 1: Heisenberg context
    plt.subplot(5, 1, 1)
    heisenberg_framework, heisenberg_number = apply_numerical_impermanence(
        initial_value=2.0,
        context_function=heisenberg_context,
        context_name="Heisenberg",
        context_description="Context inspired by Heisenberg uncertainty principle",
        visualize=False
    )
    
    # Get the transformed value
    _, context = heisenberg_number.at_time(1.0)
    transformed = heisenberg_number.value
    
    # Plot
    x = np.linspace(-3, 7, len(transformed))
    plt.plot(x, transformed)
    plt.title("Heisenberg Context: Position/Momentum Uncertainty")
    plt.ylabel("Amplitude")
    plt.axvline(x=2.0, color='r', linestyle='--', alpha=0.5, label="Original Value")
    plt.legend()
    
    # Example 2: Fractal context
    plt.subplot(5, 1, 2)
    fractal_framework, fractal_number = apply_numerical_impermanence(
        initial_value=3.0,
        context_function=fractal_context,
        context_name="Fractal",
        context_description="Context where numbers transform into self-similar patterns",
        visualize=False
    )
    
    # Get the transformed value
    _, context = fractal_number.at_time(1.0)
    transformed = fractal_number.value
    
    # Plot
    x = np.linspace(-2, 2, len(transformed))
    plt.plot(x, transformed)
    plt.title("Fractal Context: Self-Similar Patterns")
    plt.ylabel("Amplitude")
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Example 3: Relativistic context
    plt.subplot(5, 1, 3)
    relativistic_framework, relativistic_number = apply_numerical_impermanence(
        initial_value=5.0,
        context_function=relativistic_context,
        context_name="Relativistic",
        context_description="Context where numbers transform based on relativistic principles",
        visualize=False
    )
    
    # Get the transformed value
    _, context = relativistic_number.at_time(1.0)
    transformed = relativistic_number.value
    
    # Plot
    x = transformed['x_values']
    plt.plot(x, transformed['proper_frame'], label='Rest Frame')
    plt.plot(x, transformed['moving_frame'], label='Moving Frame')
    plt.title(f"Relativistic Context: γ = {transformed['gamma']:.2f}, v = {transformed['relative_velocity']:.2f}c")
    plt.ylabel("Amplitude")
    plt.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label="Original Value")
    plt.legend()
    
    # Example 4: QFT context
    plt.subplot(5, 1, 4)
    qft_framework, qft_number = apply_numerical_impermanence(
        initial_value=4.0,
        context_function=qft_context,
        context_name="QuantumField",
        context_description="Context where numbers manifest as quantum field excitations",
        visualize=False
    )
    
    # Get the transformed value
    _, context = qft_number.at_time(1.0)
    transformed = qft_number.value
    
    # Plot
    x = np.linspace(-10, 10, len(transformed))
    plt.plot(x, transformed)
    plt.title("Quantum Field Theory Context: Field Excitations")
    plt.ylabel("Field Amplitude")
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Example 5: Cyclical Time context
    plt.subplot(5, 1, 5)
    cyclical_framework, cyclical_number = apply_numerical_impermanence(
        initial_value=17.0,  # This will be 7.0 in the cycle of length 10
        context_function=cyclical_time_context,
        context_name="CyclicalTime",
        context_description="Context where numbers exist in cyclical rather than linear time",
        visualize=False
    )
    
    # Get the transformed value
    _, context = cyclical_number.at_time(1.0)
    transformed = cyclical_number.value
    
    # Plot
    theta = transformed['theta']
    plt.plot(theta, transformed['pattern'])
    plt.title(f"Cyclical Time Context: {transformed['original_value']} → {transformed['cyclic_value']} (mod 10)")
    plt.ylabel("Amplitude")
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# # Interactive transformation demonstration
# def demonstrate_interactive_transformation():
#     """Creates an interactive demonstration of a number transforming across user-defined contexts."""
#     from ipywidgets import interact, FloatSlider, Dropdown
    
#     # Initialize the framework
#     framework = NumericalImpermanence()
    
#     # Create a number
#     number = framework.create_number("interactive", 1.0, "classical")
    
#     # Define context functions
#     context_functions = {
#         "Heisenberg": heisenberg_context,
#         "Fractal": fractal_context,
#         "Relativistic": relativistic_context,
#         "Quantum Field": qft_context,
#         "Cyclical Time": cyclical_time_context
#     }
    
#     # Register all contexts
#     for name, func in context_functions.items():
#         context_id = name.lower().replace(" ", "_")
        
#         # Create context
#         context = Context(
#             name=name,
#             description=f"Context where numbers transform according to {name} principles",
#             properties={'user_defined': True}
#         )
        
#         # Register context
#         framework.register_context(context_id, context)
        
#         # Create wrapper for transformation function
#         def make_wrapper(f):
#             return lambda value, old_ctx, new_ctx: f(value)
        
#         # Register transformation
#         wrapper = make_wrapper(func)
#         framework.register_transformation("classical", context_id, wrapper)
    
#     # Define interactive update function
#     def update_plot(value, context_name):
#         # Transform the number
#         number.timeline = {0: (value, framework.contexts["classical"])}
#         context_id = context_name.lower().replace(" ", "_")
#         framework.transform_number("interactive", context_id, 1.0)
        
#         # Visualize
#         fig = framework.visualize_timeline("interactive", [0, 1], figsize=(12, 5))
#         plt.tight_layout()
#         plt.show()
    
#     # Create interactive controls
#     value_slider = FloatSlider(min=-10, max=10, step=0.1, value=1.0, description="Value:")
#     context_dropdown = Dropdown(
#         options=list(context_functions.keys()),
#         value="Heisenberg",
#         description="Context:"
#     )
    
#     # Display interactive widget
#     interact(update_plot, value=value_slider, context_name=context_dropdown)

# # Run the interactive demo if in a Jupyter environment
# # Note: This will only work in a Jupyter notebook
# try:
#     get_ipython()
#     print("Jupyter environment detected. Run demonstrate_interactive_transformation() for the interactive demo.")
# except:
#     print("Not running in Jupyter. Interactive demo unavailable.")
