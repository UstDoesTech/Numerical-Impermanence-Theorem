import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union, Optional
import math

class Context:
    """
    Represents a mathematical context in which numbers have meaning.
    
    A context defines how numbers are interpreted, manipulated, and related
    to each other within a specific mathematical framework.
    """
    
    def __init__(self, name: str, description: str, properties: Dict = None):
        """
        Initialize a mathematical context.
        
        Args:
            name: Short name of the context
            description: Longer description of the context
            properties: Dictionary of context-specific properties
        """
        self.name = name
        self.description = description
        self.properties = properties or {}
        
    def __repr__(self):
        return f"Context('{self.name}')"
    
    def __str__(self):
        return f"{self.name}: {self.description}"


class TemporalNumber:
    """
    Represents a number whose identity evolves with changing contexts.
    
    A temporal number N(t) is a function from time t to a value, where its
    meaning is tied to the context C(t) in which it is used.
    """
    
    def __init__(self, 
                 initial_value: Union[float, complex, np.ndarray], 
                 initial_context: Context,
                 identity_function: Callable = None):
        """
        Initialize a temporal number.
        
        Args:
            initial_value: The value in the initial context
            initial_context: The starting context
            identity_function: Optional function that maps (time, context) to value
        """
        self.timeline = {0: (initial_value, initial_context)}
        self.current_time = 0
        self._identity_function = identity_function
        
    @property
    def value(self) -> Union[float, complex, np.ndarray]:
        """Get the current value of the number."""
        return self.timeline[self.current_time][0]
    
    @property
    def context(self) -> Context:
        """Get the current context of the number."""
        return self.timeline[self.current_time][1]
    
    def at_time(self, t: float) -> Tuple[Union[float, complex, np.ndarray], Context]:
        """Get the value and context at a specific time."""
        if t in self.timeline:
            return self.timeline[t]
        
        # If we have an identity function, use it to calculate the value
        if self._identity_function:
            # Find the closest time before t
            times = sorted(self.timeline.keys())
            prev_t = max([time for time in times if time <= t], default=0)
            prev_value, prev_context = self.timeline[prev_t]
            
            # Calculate value at time t
            value = self._identity_function(t, prev_t, prev_value, prev_context)
            return value, prev_context
        
        # Otherwise, we need to interpolate
        times = sorted(self.timeline.keys())
        if t < times[0]:
            return self.timeline[times[0]]
        if t > times[-1]:
            return self.timeline[times[-1]]
        
        # Find the times that bracket t
        prev_t = max([time for time in times if time <= t])
        next_t = min([time for time in times if time >= t])
        
        # If t matches an exact time, return that value
        if prev_t == next_t:
            return self.timeline[prev_t]
        
        # Otherwise interpolate between the two times
        prev_value, prev_context = self.timeline[prev_t]
        next_value, next_context = self.timeline[next_t]
        
        # Linear interpolation for simple values
        if isinstance(prev_value, (int, float)) and isinstance(next_value, (int, float)):
            alpha = (t - prev_t) / (next_t - prev_t)
            value = prev_value * (1 - alpha) + next_value * alpha
            return value, prev_context  # Use previous context (contexts don't interpolate)
        
        # For complex values or arrays, interpolate carefully
        if isinstance(prev_value, complex) and isinstance(next_value, complex):
            alpha = (t - prev_t) / (next_t - prev_t)
            value = prev_value * (1 - alpha) + next_value * alpha
            return value, prev_context
        
        # For arrays, ensure they have same shape and do element-wise interpolation
        if isinstance(prev_value, np.ndarray) and isinstance(next_value, np.ndarray):
            if prev_value.shape == next_value.shape:
                alpha = (t - prev_t) / (next_t - prev_t)
                value = prev_value * (1 - alpha) + next_value * alpha
                return value, prev_context
        
        # Default fallback for other types
        return prev_value, prev_context
    
    def advance_time(self, t: float):
        """
        Move to a specific point in time.
        
        Args:
            t: The time to move to
        """
        self.current_time = t
        value, context = self.at_time(t)
        if t not in self.timeline:
            self.timeline[t] = (value, context)
    
    def transform(self, time: float, new_context: Context, transformation_func: Callable):
        """
        Transform the number into a new context at the specified time.
        
        Args:
            time: The time at which the transformation occurs
            new_context: The new context after transformation
            transformation_func: Function that maps (old_value, old_context, new_context) to new_value
        """
        # Get the value just before the transformation
        old_value, old_context = self.at_time(time - 1e-10)
        
        # Apply the transformation
        new_value = transformation_func(old_value, old_context, new_context)
        
        # Update the timeline
        self.timeline[time] = (new_value, new_context)
        
    def __repr__(self):
        return f"TemporalNumber({self.value}, {self.context})"


class ContextTransformer:
    """
    Manages transformations between different mathematical contexts.
    
    This class provides a collection of standard transformation functions
    and tools for defining custom transformations.
    """
    
    @staticmethod
    def classical_to_fuzzy(value: float, old_ctx: Context, new_ctx: Context) -> np.ndarray:
        """
        Transform a classical number to a fuzzy number.
        
        Args:
            value: The classical value (e.g., 1)
            old_ctx: The classical context
            new_ctx: The fuzzy context
            
        Returns:
            A fuzzy membership function (as a numpy array)
        """
        # Get fuzzy parameters from new context
        x_range = new_ctx.properties.get('x_range', np.linspace(-5, 5, 1000))
        fuzziness = new_ctx.properties.get('fuzziness', 0.5)
        
        # Create fuzzy membership function centered at the value
        membership = np.exp(-((x_range - value) ** 2) / (2 * fuzziness ** 2))
        
        return membership
    
    @staticmethod
    def fuzzy_to_quantum(value: np.ndarray, old_ctx: Context, new_ctx: Context) -> np.ndarray:
        """
        Transform a fuzzy number to a quantum state.
        
        Args:
            value: The fuzzy membership function
            old_ctx: The fuzzy context
            new_ctx: The quantum context
            
        Returns:
            A complex-valued quantum wavefunction
        """
        # Get quantum parameters from new context
        x_range = new_ctx.properties.get('x_range', old_ctx.properties.get('x_range'))
        phase_factor = new_ctx.properties.get('phase_factor', 5.0)
        
        # Extract the center of the fuzzy function
        center = x_range[np.argmax(value)]
        
        # Create a quantum wavefunction with phase
        amplitude = np.sqrt(value)  # Probability amplitude
        phase = np.exp(1j * phase_factor * (x_range - center))
        wavefunction = amplitude * phase
        
        # Normalize the wavefunction
        norm = np.sqrt(np.sum(np.abs(wavefunction) ** 2))
        if norm > 0:
            wavefunction = wavefunction / norm
            
        return wavefunction
    
    @staticmethod
    def quantum_to_intuitionistic(value: np.ndarray, old_ctx: Context, new_ctx: Context) -> Dict:
        """
        Transform a quantum state to an intuitionistic representation.
        
        Args:
            value: The quantum wavefunction
            old_ctx: The quantum context
            new_ctx: The intuitionistic context
            
        Returns:
            A dictionary representing the construction process
        """
        # Get the x range from the old context
        x_range = old_ctx.properties.get('x_range')
        
        # Find the "center" of the quantum state (where probability is highest)
        probability = np.abs(value) ** 2
        center_idx = np.argmax(probability)
        center = x_range[center_idx]
        
        # Create an intuitionistic construction
        construction = {
            'axioms': 1.0,  # Axioms are always fully established
            'natural_numbers': 0.9,  # Natural numbers are well-constructed
            'succession': 0.8,  # Succession operation is defined
            'zero': 0.7,  # Zero is constructed
        }
        
        # Add the construction of the specific number
        if abs(center - round(center)) < 0.1:  # If it's close to an integer
            number = round(center)
            construction[f'{number}'] = 0.6
            
            # Add properties of the number
            if number % 2 == 0:
                construction[f'{number}_is_even'] = 0.5
            else:
                construction[f'{number}_is_odd'] = 0.5
                
            if number > 0:
                construction[f'{number}_is_positive'] = 0.5
            elif number < 0:
                construction[f'{number}_is_negative'] = 0.5
                
            # Is it prime?
            is_prime = True
            if number > 1:
                for i in range(2, int(math.sqrt(number)) + 1):
                    if number % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    construction[f'{number}_is_prime'] = 0.5
        
        return construction
    
    @staticmethod
    def intuitionistic_to_non_euclidean(value: Dict, old_ctx: Context, new_ctx: Context) -> Dict:
        """
        Transform an intuitionistic representation to a non-Euclidean one.
        
        Args:
            value: The intuitionistic construction
            old_ctx: The intuitionistic context
            new_ctx: The non-Euclidean context
            
        Returns:
            A dictionary representing the number in non-Euclidean space
        """
        # Get non-Euclidean parameters
        curvature = new_ctx.properties.get('curvature', 1.0)
        
        # Find the constructed number from the intuitionistic representation
        number = None
        for key in value.keys():
            if key.isdigit() or (key[0] == '-' and key[1:].isdigit()):
                number = int(key)
                break
                
        if number is None:
            # Default to 0 if no specific number is found
            number = 0
            
        # Create a non-Euclidean representation
        non_euclidean = {
            'coordinates': (number, 0, curvature * number**2 / 10),  # (x, y, z) in curved space
            'geodesics': {},  # Paths to other points
            'metric': {}  # Distances to other points
        }
        
        # Calculate some geodesics to nearby points
        for i in range(-2, 3):
            if i != number:
                # Calculate geodesic to point i
                t = np.linspace(0, 1, 100)
                x = number * (1-t) + i * t
                y = np.zeros_like(t)
                z = curvature * x**2 / 10
                
                # Calculate the geodesic distance (arc length)
                dx = np.diff(x)
                dy = np.diff(y)
                dz = np.diff(z)
                segments = np.sqrt(dx**2 + dy**2 + dz**2)
                distance = np.sum(segments)
                
                non_euclidean['geodesics'][i] = (x, y, z)
                non_euclidean['metric'][i] = distance
                
        return non_euclidean
    
    @staticmethod
    def non_euclidean_to_alien(value: Dict, old_ctx: Context, new_ctx: Context) -> Dict:
        """
        Transform a non-Euclidean representation to an alien mathematical system.
        
        Args:
            value: The non-Euclidean representation
            old_ctx: The non-Euclidean context
            new_ctx: The alien context
            
        Returns:
            A dictionary representing the number in the alien system
        """
        # Get the base number from the coordinates
        x, y, z = value['coordinates']
        number = x  # The x-coordinate is our original number
        
        # Create an alien representation based on different mathematical primitives
        alien = {
            'process_identity': {
                'flux': number * 0.8,
                'cycle_frequency': number * 1.2,
                'stability': 1 / (1 + abs(number)),
            },
            'relational_web': {
                'connections': [
                    (number + i, number * i) for i in range(-2, 3) if i != 0
                ],
                'centrality': 1 / (1 + number**2),
            },
            'emergent_properties': {
                'coherence': np.sin(number * np.pi / 2),
                'resonance': np.cos(number * np.pi / 3),
                'interference_pattern': [np.sin(number * i * np.pi / 5) for i in range(10)]
            }
        }
        
        return alien


class NumericalImpermanence:
    """
    Main class implementing the Numerical Impermanence Theorem.
    
    This class coordinates context transformations and provides tools
    for tracking and visualizing how numbers transform across contexts.
    """
    
    def __init__(self):
        """Initialize the Numerical Impermanence framework."""
        self.contexts = {}
        self.numbers = {}
        self.transformations = {}
        self._register_standard_contexts()
        self._register_standard_transformations()
        
    def _register_standard_contexts(self):
        """Register standard mathematical contexts."""
        # Classical context
        classical = Context(
            name="Classical",
            description="Traditional mathematics with precise, well-defined numbers",
            properties={
                'x_range': np.linspace(-5, 5, 1000),
                'base': 10,
                'logic': 'boolean'
            }
        )
        
        # Fuzzy context
        fuzzy = Context(
            name="Fuzzy",
            description="Fuzzy logic where numbers have degrees of membership",
            properties={
                'x_range': np.linspace(-5, 5, 1000),
                'fuzziness': 0.5,
                'logic': 'fuzzy'
            }
        )
        
        # Quantum context
        quantum = Context(
            name="Quantum",
            description="Quantum mathematics where numbers exist in superposition",
            properties={
                'x_range': np.linspace(-5, 5, 1000),
                'phase_factor': 5.0,
                'logic': 'quantum'
            }
        )
        
        # Intuitionistic context
        intuitionistic = Context(
            name="Intuitionistic",
            description="Constructive mathematics where numbers exist through proof",
            properties={
                'logic': 'intuitionistic',
                'construction_steps': ['axioms', 'natural_numbers', 'succession', 'zero']
            }
        )
        
        # Non-Euclidean context
        non_euclidean = Context(
            name="NonEuclidean",
            description="Mathematics in curved space where distances and relationships change",
            properties={
                'curvature': 1.0,
                'geometry': 'curved',
                'logic': 'classical'
            }
        )
        
        # Alien mathematics context
        alien = Context(
            name="Alien",
            description="Speculative alien mathematical system with different primitives",
            properties={
                'primitives': ['process', 'relation', 'emergence'],
                'logic': 'non-human'
            }
        )
        
        # Register contexts
        self.contexts = {
            'classical': classical,
            'fuzzy': fuzzy,
            'quantum': quantum,
            'intuitionistic': intuitionistic,
            'non_euclidean': non_euclidean,
            'alien': alien
        }
        
    def _register_standard_transformations(self):
        """Register standard transformation functions between contexts."""
        self.transformations = {
            ('classical', 'fuzzy'): ContextTransformer.classical_to_fuzzy,
            ('fuzzy', 'quantum'): ContextTransformer.fuzzy_to_quantum,
            ('quantum', 'intuitionistic'): ContextTransformer.quantum_to_intuitionistic,
            ('intuitionistic', 'non_euclidean'): ContextTransformer.intuitionistic_to_non_euclidean,
            ('non_euclidean', 'alien'): ContextTransformer.non_euclidean_to_alien
        }
        
    def register_context(self, context_id: str, context: Context):
        """
        Register a new mathematical context.
        
        Args:
            context_id: Unique identifier for the context
            context: The Context object
        """
        self.contexts[context_id] = context
        
    def register_transformation(self, from_context: str, to_context: str, transformation_func: Callable):
        """
        Register a transformation function between contexts.
        
        Args:
            from_context: Source context ID
            to_context: Target context ID
            transformation_func: Function that transforms numbers between these contexts
        """
        self.transformations[(from_context, to_context)] = transformation_func
        
    def create_number(self, number_id: str, value: Union[float, complex, np.ndarray], context_id: str):
        """
        Create a new temporal number in the specified context.
        
        Args:
            number_id: Unique identifier for the number
            value: Initial value of the number
            context_id: Initial context ID
            
        Returns:
            The created TemporalNumber object
        """
        if context_id not in self.contexts:
            raise ValueError(f"Context '{context_id}' not found")
            
        number = TemporalNumber(value, self.contexts[context_id])
        self.numbers[number_id] = number
        return number
    
    def transform_number(self, number_id: str, to_context_id: str, time: float):
        """
        Transform a number from its current context to a new one.
        
        Args:
            number_id: ID of the number to transform
            to_context_id: Target context ID
            time: The time at which the transformation occurs
            
        Returns:
            The transformed TemporalNumber object
        """
        if number_id not in self.numbers:
            raise ValueError(f"Number '{number_id}' not found")
            
        if to_context_id not in self.contexts:
            raise ValueError(f"Context '{to_context_id}' not found")
            
        number = self.numbers[number_id]
        from_context_id = number.context.name.lower()
        
        # Find transformation function
        transform_key = (from_context_id, to_context_id)
        if transform_key not in self.transformations:
            raise ValueError(f"No transformation defined from '{from_context_id}' to '{to_context_id}'")
            
        transformation_func = self.transformations[transform_key]
        
        # Apply transformation
        number.transform(time, self.contexts[to_context_id], transformation_func)
        return number
    
    def transform_across_contexts(self, number_id: str, context_sequence: List[str], 
                                  start_time: float = 0, end_time: float = 1):
        """
        Transform a number across a sequence of contexts over time.
        
        Args:
            number_id: ID of the number to transform
            context_sequence: List of context IDs to transform through
            start_time: Starting time for the transformation sequence
            end_time: Ending time for the transformation sequence
            
        Returns:
            The transformed TemporalNumber object
        """
        if number_id not in self.numbers:
            raise ValueError(f"Number '{number_id}' not found")
            
        number = self.numbers[number_id]
        current_context = number.context.name.lower()
        
        # Check if the first context matches the current context
        if context_sequence and context_sequence[0].lower() != current_context:
            context_sequence = [current_context] + context_sequence
            
        # Check that all contexts exist
        for ctx in context_sequence:
            if ctx not in self.contexts:
                raise ValueError(f"Context '{ctx}' not found")
                
        # Calculate time points for each transformation
        time_points = np.linspace(start_time, end_time, len(context_sequence))
        
        # Apply transformations sequentially
        for i in range(1, len(context_sequence)):
            from_ctx = context_sequence[i-1]
            to_ctx = context_sequence[i]
            
            # Skip if same context
            if from_ctx == to_ctx:
                continue
                
            # Find transformation function
            transform_key = (from_ctx, to_ctx)
            if transform_key not in self.transformations:
                raise ValueError(f"No transformation defined from '{from_ctx}' to '{to_ctx}'")
                
            # Apply transformation
            transformation_func = self.transformations[transform_key]
            number.transform(time_points[i], self.contexts[to_ctx], transformation_func)
            
        return number
    
    def visualize_timeline(self, number_id: str, times: List[float], figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize a number at specific points in time.
        
        Args:
            number_id: ID of the number to visualize
            times: List of time points to visualize
            figsize: Size of the figure
            
        Returns:
            Matplotlib figure
        """
        if number_id not in self.numbers:
            raise ValueError(f"Number '{number_id}' not found")
            
        number = self.numbers[number_id]
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        n_times = len(times)
        
        # Determine layout based on number of subplots
        if n_times <= 3:
            rows, cols = 1, n_times
        else:
            cols = min(3, n_times)
            rows = (n_times + cols - 1) // cols
            
        # Create subplots
        for i, t in enumerate(times):
            ax = fig.add_subplot(rows, cols, i+1)
            
            # Get value and context at time t
            value, context = number.at_time(t)
            context_name = context.name
            
            # Customize plot based on context type
            self._plot_in_context(ax, value, context, t)
            
            ax.set_title(f"t = {t:.2f}, Context: {context_name}")
            
        plt.tight_layout()
        return fig
    
    def _plot_in_context(self, ax, value, context, time):
        """Helper method to plot a value in a specific context."""
        context_name = context.name.lower()
        
        if context_name == 'classical':
            # Plot a vertical line at the value
            if isinstance(value, (int, float)):
                ax.axvline(x=value, color='red', linestyle='-', linewidth=2)
                ax.set_xlim(value - 3, value + 3)
                ax.set_ylim(0, 1.2)
                ax.text(value, 1.1, f"{value}", ha='center', va='bottom', fontsize=12)
                ax.set_xlabel("Number Line")
                ax.set_yticks([])
                
        elif context_name == 'fuzzy':
            # Plot the fuzzy membership function
            if isinstance(value, np.ndarray):
                x_range = context.properties.get('x_range', np.linspace(-5, 5, len(value)))
                ax.plot(x_range, value, 'b-')
                ax.set_xlabel("Number Line")
                ax.set_ylabel("Membership Degree")
                ax.set_ylim(0, 1.1)
                
        elif context_name == 'quantum':
            # Plot the real and imaginary parts of the wavefunction
            if isinstance(value, np.ndarray):
                x_range = context.properties.get('x_range', np.linspace(-5, 5, len(value)))
                
                # Plot real part
                ax.plot(x_range, np.real(value), 'b-', label='Real')
                
                # Plot imaginary part
                ax.plot(x_range, np.imag(value), 'r-', label='Imaginary')
                
                # Plot probability density
                ax.plot(x_range, np.abs(value)**2, 'g--', label='Probability')
                
                ax.set_xlabel("Number Line")
                ax.set_ylabel("Amplitude")
                ax.legend()
                
        elif context_name == 'intuitionistic':
            # Plot the construction process
            if isinstance(value, dict):
                steps = list(value.keys())
                values = list(value.values())
                
                # Sort by value (construction sequence)
                sorted_idx = np.argsort(values)[::-1]
                steps = [steps[i] for i in sorted_idx]
                values = [values[i] for i in sorted_idx]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(steps))
                ax.barh(y_pos, values, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(steps)
                ax.set_xlabel("Degree of Construction")
                ax.set_xlim(0, 1.1)
                
        elif context_name == 'noneuclidean':
            # Plot in 3D space
            if isinstance(value, dict) and 'coordinates' in value:
                # Convert 2D axis to 3D
                ax.remove()
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                
                # Get coordinates
                x, y, z = value['coordinates']
                
                # Plot the point
                ax.scatter([x], [y], [z], color='red', s=100)
                
                # Plot geodesics
                for target, (geo_x, geo_y, geo_z) in value.get('geodesics', {}).items():
                    ax.plot(geo_x, geo_y, geo_z, 'g-')
                    
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
        elif context_name == 'alien':
            # Create a concept map visualization
            if isinstance(value, dict):
                # Extracting key components
                process = value.get('process_identity', {})
                relations = value.get('relational_web', {})
                emergent = value.get('emergent_properties', {})
                
                # Plot as a network/concept map
                ax.axis('off')
                ax.text(0.5, 0.9, "Alien Mathematical Representation", 
                       ha='center', fontsize=12, fontweight='bold')
                
                # Plot process identity
                ax.text(0.25, 0.7, "Process Identity", ha='center', fontsize=10)
                for i, (k, v) in enumerate(process.items()):
                    y_pos = 0.65 - i*0.05
                    ax.text(0.25, y_pos, f"{k}: {v:.2f}", ha='center', fontsize=8)
                    
                # Plot relational web
                ax.text(0.75, 0.7, "Relational Web", ha='center', fontsize=10)
                connections = relations.get('connections', [])
                for i, (a, b) in enumerate(connections[:5]):
                    y_pos = 0.65 - i*0.05
                    ax.text(0.75, y_pos, f"({a:.1f}, {b:.1f})", ha='center', fontsize=8)
                    
                # Plot emergent properties
                ax.text(0.5, 0.4, "Emergent Properties", ha='center', fontsize=10)
                for i, (k, v) in enumerate(emergent.items()):
                    if not isinstance(v, list):
                        y_pos = 0.35 - i*0.05
                        ax.text(0.5, y_pos, f"{k}: {v:.2f}", ha='center', fontsize=8)
                        
        else:
            # Generic plot for unknown contexts
            ax.text(0.5, 0.5, f"Value in {context_name} context\n{value}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
    
    def animate_transformation(self, number_id: str, start_time: float = 0, end_time: float = 1, 
                              frame_count: int = 100, interval: int = 50, figsize: Tuple[int, int] = (10, 6)):
        """
        Create an animation of a number transforming over time.
        
        Args:
            number_id: ID of the number to animate
            start_time: Starting time for the animation
            end_time: Ending time for the animation
            frame_count: Number of frames in the animation
            interval: Milliseconds between frames
            figsize: Size of the figure
            
        Returns:
            Matplotlib animation
        """
        if number_id not in self.numbers:
            raise ValueError(f"Number '{number_id}' not found")
            
        number = self.numbers[number_id]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define update function for animation
        def update(frame):
            ax.clear()
            
            # Calculate time for this frame
            t = start_time + (end_time - start_time) * frame / (frame_count - 1)
            
            # Get value and context
            value, context = number.at_time(t)
            
            # Plot the value in its context
            self._plot_in_context(ax, value, context, t)
            
            # Add title
            ax.set_title(f"t = {t:.2f}, Context: {context.name}")
            
        # Create animation
        ani = FuncAnimation(fig, update, frames=frame_count, interval=interval)
        
        return ani


# Example usage functions
def create_standard_example():
    """Create a standard example transforming the number 1 across contexts."""
    # Initialize the framework
    framework = NumericalImpermanence()
    
    # Create the number 1 in classical context
    one = framework.create_number("one", 1.0, "classical")
    
    # Transform it across all standard contexts
    context_sequence = ["classical", "fuzzy", "quantum", 
                        "intuitionistic", "non_euclidean", "alien"]
    framework.transform_across_contexts("one", context_sequence)
    
    return framework

def create_custom_context_example():
    """Create an example with a custom context and transformation."""
    # Initialize the framework
    framework = NumericalImpermanence()
    
    # Define a custom context
    hyperbolic = Context(
        name="Hyperbolic",
        description="Hyperbolic geometry with negative curvature",
        properties={
            'curvature': -1.0,
            'geometry': 'hyperbolic',
            'logic': 'classical'
        }
    )
    
    # Register the context
    framework.register_context("hyperbolic", hyperbolic)
    
    # Define a custom transformation
    def classical_to_hyperbolic(value, old_ctx, new_ctx):
        # Create a hyperbolic representation of the number
        x = value  # The original value becomes the x-coordinate
        y = 0      # Place it on the x-axis
        
        # In hyperbolic space, z = -k * (x² + y²) for negative curvature
        curvature = new_ctx.properties.get('curvature', -1.0)
        z = curvature * (x**2 + y**2) / 10
        
        # Create a dictionary with hyperbolic coordinates and properties
        hyperbolic_value = {
            'coordinates': (x, y, z),
            'geodesics': {},
            'metric': {}
        }
        
        # Calculate some geodesics (simplified)
        for i in range(-2, 3):
            if i != x:
                t = np.linspace(0, 1, 100)
                geo_x = x * (1-t) + i * t
                geo_y = np.zeros_like(t)
                geo_z = curvature * geo_x**2 / 10
                
                hyperbolic_value['geodesics'][i] = (geo_x, geo_y, geo_z)
                
                # Calculate approximate distance (simplified)
                hyperbolic_value['metric'][i] = abs(i - x) * (1 + 0.2 * abs(i - x))
                
        return hyperbolic_value
    
    # Register the transformation
    framework.register_transformation("classical", "hyperbolic", classical_to_hyperbolic)
    
    # Create a number and transform it
    pi = framework.create_number("pi", 3.14159, "classical")
    framework.transform_number("pi", "hyperbolic", 1.0)
    
    return framework

def custom_context_function_example(ctx_function):
    """
    Example using a user-provided context function.
    
    Args:
        ctx_function: A function that takes a value and returns a transformed value
        
    Returns:
        The framework with the transformation applied
    """
    # Initialize the framework
    framework = NumericalImpermanence()
    
    # Create a custom context
    custom = Context(
        name="Custom",
        description="User-defined custom context",
        properties={
            'x_range': np.linspace(-5, 5, 1000),
            'custom_param': 1.0
        }
    )
    
    # Register the context
    framework.register_context("custom", custom)
    
    # Create a wrapper for the user function to match our transform signature
    def custom_transform(value, old_ctx, new_ctx):
        return ctx_function(value)
    
    # Register the transformation
    framework.register_transformation("classical", "custom", custom_transform)
    
    # Create a number and transform it
    num = framework.create_number("custom_number", 1.0, "classical")
    framework.transform_number("custom_number", "custom", 1.0)
    
    return framework


# Example for allowing a user to plug in custom context transformations
def apply_numerical_impermanence(
    initial_value: float,
    context_function: Callable,
    context_name: str = "Custom",
    context_description: str = "User-defined context transformation",
    visualize: bool = True
):
    """
    Apply the Numerical Impermanence Theorem with a user-defined context function.
    
    Args:
        initial_value: The starting value in classical context
        context_function: Function that transforms a value according to the custom context
        context_name: Name for the custom context
        context_description: Description of the custom context
        visualize: Whether to visualize the transformation
        
    Returns:
        A tuple of (framework, transformed_number)
    """
    # Initialize framework
    framework = NumericalImpermanence()
    
    # Create custom context
    custom = Context(
        name=context_name,
        description=context_description,
        properties={
            'x_range': np.linspace(-5, 5, 1000),
            'user_defined': True
        }
    )
    
    # Register context
    context_id = context_name.lower()
    framework.register_context(context_id, custom)
    
    # Create wrapper for the transformation function
    def transform_wrapper(value, old_ctx, new_ctx):
        return context_function(value)
    
    # Register transformation
    framework.register_transformation("classical", context_id, transform_wrapper)
    
    # Create and transform number
    number_id = f"number_{initial_value}"
    number = framework.create_number(number_id, initial_value, "classical")
    framework.transform_number(number_id, context_id, 1.0)
    
    # Visualize if requested
    if visualize:
        fig = framework.visualize_timeline(number_id, [0, 1])
        plt.show()
    
    return framework, number


# Example usage
if __name__ == "__main__":
    # Create a standard example transforming the number 1
    framework = create_standard_example()
    
    # Visualize at key points in time
    time_points = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig = framework.visualize_timeline("one", time_points)
    plt.show()
    
    # Animate the transformation
    ani = framework.animate_transformation("one")
    plt.show()
    
    # Example with a custom context function
    def custom_sqrt_context(value):
        """A context where numbers are transformed into their square roots and patterns."""
        if isinstance(value, (int, float)):
            x = np.linspace(0, value*2, 1000)
            y = np.sin(np.sqrt(value) * x) * np.exp(-x/10)
            return y
        return value
    
    # Apply with custom function
    custom_framework, custom_number = apply_numerical_impermanence(
        initial_value=4.0,
        context_function=custom_sqrt_context,
        context_name="SquareRootContext",
        context_description="Numbers transform into wave patterns based on their square roots"
    )
