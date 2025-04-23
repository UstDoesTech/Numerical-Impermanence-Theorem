import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create a figure with subplots for all five transformations
plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1])

# Set up axes for each transformation
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3], projection='3d')
ax5 = plt.subplot(gs[4])

# Global settings
plt.subplots_adjust(hspace=0.4)
x = np.linspace(-2, 4, 1000)

#=================================================================================
# 1. Classical → Fuzzy Logic
#=================================================================================
ax1.set_title("Classical → Fuzzy Logic Transformation", fontsize=14)
ax1.set_xlim(-2, 4)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('Number Line', fontsize=12)
ax1.set_ylabel('Membership Value', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add reference points
ax1.axhline(y=0, color='white', linestyle='-', alpha=0.2)
ax1.axhline(y=1, color='white', linestyle='-', alpha=0.2)
ax1.axvline(x=1, color='white', linestyle='--', alpha=0.2)

# Plot labels for key points
ax1.text(1.1, -0.05, "1", fontsize=10, color='white')
ax1.text(-1.9, 1.03, "Full Membership", fontsize=10, color='white')
ax1.text(-1.9, 0.03, "No Membership", fontsize=10, color='white')

# Initial lines
classical_line, = ax1.plot([], [], '#ff9500', linewidth=2, label='Classical 1')
fuzzy_line, = ax1.plot([], [], '#00b4d8', linewidth=2, label='Fuzzy 1')
ax1.legend(loc='upper right')

def classical_one(x):
    """Classical representation of the number 1"""
    return np.where(np.abs(x - 1) < 0.01, 1, 0)  # Dirac delta approximation

def fuzzy_one(x, fuzziness):
    """Fuzzy representation of the number 1 with variable fuzziness"""
    return np.exp(-(x - 1)**2 / (2 * fuzziness**2))

#=================================================================================
# 2. Fuzzy Logic → Quantum Computing
#=================================================================================
ax2.set_title("Fuzzy Logic → Quantum Computing Transformation", fontsize=14)
ax2.set_xlim(-2, 4)
ax2.set_ylim(-0.6, 1.1)
ax2.set_xlabel('Number Line', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True, alpha=0.3)

# Add reference lines
ax2.axhline(y=0, color='white', linestyle='-', alpha=0.2)
ax2.axvline(x=1, color='white', linestyle='--', alpha=0.2)

# Labels
ax2.text(1.1, -0.05, "1", fontsize=10, color='white')
ax2.text(-1.9, 0.03, "Zero Amplitude", fontsize=10, color='white')

# Initial lines for fuzzy and quantum states
fuzzy2_line, = ax2.plot([], [], '#00b4d8', linewidth=2, label='Fuzzy State')
quantum_line, = ax2.plot([], [], '#ff006e', linewidth=2, label='Quantum State')
quantum_prob_line, = ax2.plot([], [], '#8338ec', linewidth=1, linestyle='--', 
                            label='Probability Density')
ax2.legend(loc='upper right')

def quantum_state(x, phase, width):
    """Quantum wavefunction representation"""
    psi = np.exp(-(x - 1)**2 / (2 * width**2)) * np.exp(1j * phase * (x - 1))
    return psi

#=================================================================================
# 3. Quantum Computing → Intuitionistic Mathematics
#=================================================================================
ax3.set_title("Quantum Computing → Intuitionistic Mathematics", fontsize=14)
ax3.set_xlim(-0.1, 4.1)
ax3.set_ylim(-0.1, 1.1)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Construction Steps', fontsize=12)
ax3.set_ylabel('Degree of Existence', fontsize=12)

# Steps in constructing the number 1
steps = ["Axioms", "Natural\nNumbers", "Successor\nOperation", "Zero", "One"]
ax3.set_xticks(np.arange(len(steps)))
ax3.set_xticklabels(steps)

# Create the bars for the construction process
quantum_bars = ax3.bar(np.arange(len(steps)), np.zeros(len(steps)), 
                     color='#ff006e', alpha=0.7, width=0.4)
construct_bars = ax3.bar(np.arange(len(steps))+0.4, np.zeros(len(steps)), 
                       color='#06d6a0', alpha=0.7, width=0.4)

# Add text for current paradigm
paradigm_text = ax3.text(2, 1.05, "", ha='center', fontsize=12, color='white',
                        bbox=dict(facecolor='black', alpha=0.8, boxstyle='round'))

#=================================================================================
# 4. Intuitionistic Mathematics → Non-Euclidean Framework
#=================================================================================
ax4.set_title("Intuitionistic → Non-Euclidean Transformation", fontsize=14)
ax4.set_xlim(-3, 3)
ax4.set_ylim(-3, 3)
ax4.set_zlim(0, 5)
ax4.set_xlabel('X', fontsize=10)
ax4.set_ylabel('Y', fontsize=10)
ax4.set_zlabel('Z', fontsize=10)
ax4.view_init(elev=30, azim=45)

# Create flat number line with points
x_line = np.linspace(-2, 2, 100)
y_line = np.zeros_like(x_line)
z_line = np.zeros_like(x_line)
number_line, = ax4.plot(x_line, y_line, z_line, 'white', linewidth=2)

# Create points for key numbers
x_points = [-2, -1, 0, 1, 2]
y_points = [0, 0, 0, 0, 0]
z_points = [0, 0, 0, 0, 0]
numbers_scatter = ax4.scatter(x_points, y_points, z_points, 
                             color=['#ffd166', '#ffd166', '#ffd166', '#ff006e', '#ffd166'], 
                             s=100, zorder=10)

# Generate surface data for curved space
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
Z = np.zeros_like(X)  # Start with flat space

# Create the surface (initially flat)
surface = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)

# Add a text annotation for curvature
curvature_text = ax4.text2D(0.05, 0.95, "", transform=ax4.transAxes, fontsize=10,
                          bbox=dict(facecolor='black', alpha=0.8, boxstyle='round'))

#=================================================================================
# 5. Non-Euclidean → Alien Mathematics
#=================================================================================
ax5.set_title("Non-Euclidean → Alien Mathematics Transformation", fontsize=14)
ax5.set_xlim(-10, 10)
ax5.set_ylim(-10, 10)
ax5.set_aspect('equal')
ax5.axis('off')

# Initialize the concept map components
concept_map_nodes = []
concept_map_edges = []

# Create initial node positions in a circular pattern
num_nodes = 20
node_types = ["quantity", "relation", "process", "entity"]
node_colors = {'quantity': '#ffd166', 'relation': '#06d6a0', 
              'process': '#118ab2', 'entity': '#ef476f'}
node_positions = []

for i in range(num_nodes):
    angle = 2 * np.pi * i / num_nodes
    r = 7 * np.random.rand() ** 0.5  # Distribute nodes within a circle
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    node_positions.append((x, y))

# Create the "1" node at a special position
one_position = (0, 0)  # Start at center
one_node = plt.Circle(one_position, 0.8, color='red')
ax5.add_patch(one_node)
one_text = ax5.text(0, 0, "1", fontsize=14, ha='center', va='center', color='white')

# Create random nodes of different types
for i in range(num_nodes):
    node_type = node_types[i % len(node_types)]
    size = 0.3 + 0.2 * np.random.rand()  # Random size
    node = plt.Circle(node_positions[i], size, color=node_colors[node_type], alpha=0.7)
    ax5.add_patch(node)
    concept_map_nodes.append(node)

# Add explanation text
alien_text = ax5.text(0, -9, "", ha='center', fontsize=12, color='white',
                     bbox=dict(facecolor='black', alpha=0.8, boxstyle='round'))

#=================================================================================
# Animation Function
#=================================================================================
def update(frame):
    """Update the animation for all five transformations."""
    t = frame / 100  # Normalize to 0-1
    
    #-----------------------------------------------------------------------
    # 1. Classical → Fuzzy Logic update
    #-----------------------------------------------------------------------
    classical_intensity = max(0, 1 - 2*t)  # Fades out by t=0.5
    fuzzy_intensity = min(1, 2*t)  # Fades in by t=0.5
    
    # Update classical representation (Dirac delta function)
    classical_y = classical_one(x) * classical_intensity
    classical_line.set_data(x, classical_y)
    
    # Update fuzzy representation
    fuzziness = 0.1 + t * 0.4  # Increases fuzziness over time
    fuzzy_y = fuzzy_one(x, fuzziness) * fuzzy_intensity
    fuzzy_line.set_data(x, fuzzy_y)
    
    #-----------------------------------------------------------------------
    # 2. Fuzzy Logic → Quantum Computing update
    #-----------------------------------------------------------------------
    # Update fuzzy state (fading out)
    fuzzy_fade = max(0, 1 - 2*t)
    fuzzy2_line.set_data(x, fuzzy_one(x, 0.3) * fuzzy_fade)
    
    # Update quantum state (fading in with oscillations)
    quantum_fade = min(1, 2*t)
    phase = 10 * t  # Increasing phase factor
    width = 0.3 - 0.1 * t  # Slightly decreasing width
    
    psi = quantum_state(x, phase, width) * quantum_fade
    quantum_line.set_data(x, np.real(psi))  # Real part of wavefunction
    
    # Probability density
    prob = np.abs(psi)**2
    quantum_prob_line.set_data(x, prob)
    
    #-----------------------------------------------------------------------
    # 3. Quantum Computing → Intuitionistic update
    #-----------------------------------------------------------------------
    # Update the quantum bars (decreasing)
    quantum_values = []
    for i in range(len(steps)):
        if i == 0:  # Axioms remain
            val = 1
        else:
            # Quantum values fade differently for each step
            val = max(0, 1 - 4*(t - 0.1*i))
        quantum_values.append(val)
    
    # Update the construction bars (increasing)
    construction_values = []
    for i in range(len(steps)):
        # Construction happens step by step
        val = min(1, max(0, 4*(t - 0.1*i)))
        construction_values.append(val)
    
    # Set the new bar heights
    for i, bar in enumerate(quantum_bars):
        bar.set_height(quantum_values[i])
    
    for i, bar in enumerate(construct_bars):
        bar.set_height(construction_values[i])
    
    # Update paradigm text
    if t < 0.33:
        paradigm_text.set_text("Quantum Paradigm: Probabilistic Existence")
    elif t < 0.66:
        paradigm_text.set_text("Transitional Paradigm")
    else:
        paradigm_text.set_text("Intuitionistic Paradigm: Constructive Existence")
    
    #-----------------------------------------------------------------------
    # 4. Intuitionistic → Non-Euclidean update
    #-----------------------------------------------------------------------
    # Calculate curvature based on time
    curvature = 3 * t
    
    # Update the surface with increasing curvature
    Z = curvature * ((X - 0)**2 + (Y - 0)**2) / 5
    
    # Remove old surface and create new one
    ax4.collections.remove(surface)
    surface = ax4.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
    
    # Update the number line and points to follow the curved surface
    z_line = curvature * ((x_line - 0)**2 + (y_line - 0)**2) / 5
    number_line.set_data_3d(x_line, y_line, z_line)
    
    # Update points on the number line
    z_points = curvature * ((np.array(x_points) - 0)**2 + (np.array(y_points) - 0)**2) / 5
    numbers_scatter._offsets3d = (x_points, y_points, z_points)
    
    # Update curvature text
    if t < 0.33:
        curvature_text.set_text("Flat Space: Euclidean Geometry")
    elif t < 0.66:
        curvature_text.set_text("Curved Space: Emerging Non-Euclidean Geometry")
    else:
        curvature_text.set_text("Strongly Curved Space: Full Non-Euclidean Geometry")
    
    #-----------------------------------------------------------------------
    # 5. Non-Euclidean → Alien Mathematics update
    #-----------------------------------------------------------------------
    # Transition of the "1" node
    if t < 0.5:
        # Phase 1: "1" remains central but starts to change
        one_color = (1, 0, 0, 1)  # Pure red
        one_size = 0.8
        one_pos = (0, 0)
        one_label = "1"
    else:
        # Phase 2: "1" transforms into a different concept
        red = max(0, 1 - (t - 0.5) * 2)
        green = min(1, (t - 0.5) * 2)
        blue = min(0.5, (t - 0.5))
        one_color = (red, green, blue, 1)
        
        # Size fluctuates
        one_size = 0.8 + 0.3 * np.sin(t * 20)
        
        # Position shifts
        one_pos = (3 * np.sin(t * 5), 2 * np.cos(t * 7))
        
        # Label changes
        if t > 0.7:
            one_label = "⊗" if t > 0.85 else "≈"
        else:
            one_label = "1"
    
    # Update the "1" node
    one_node.center = one_pos
    one_node.set_radius(one_size)
    one_node.set_color(one_color)
    one_text.set_position(one_pos)
    one_text.set_text(one_label)
    
    # Make other nodes dynamic in the alien system
    for i, node in enumerate(concept_map_nodes):
        # Basic position
        base_x, base_y = node_positions[i]
        
        # Add dynamic movement based on time
        angle = t * 5 + i * 0.1
        radius = 1 + 0.5 * np.sin(t * 3 + i)
        
        # More chaotic movement as t increases
        if t > 0.5:
            chaos_factor = (t - 0.5) * 2
            base_x += chaos_factor * np.sin(angle * 3 + i) * 0.5
            base_y += chaos_factor * np.cos(angle * 2 + i * 0.7) * 0.5
        
        # Update node position
        node.center = (base_x, base_y)
        
        # Pulsating size for process nodes
        if i % 4 == 2:  # process nodes
            node.set_radius(0.3 + 0.2 * np.sin(t * 10 + i))
    
    # Update explanation text
    if t < 0.25:
        alien_text.set_text("Non-Euclidean Mathematics: Geometric Identity")
    elif t < 0.5:
        alien_text.set_text("Early Transition: Emerging Alien Concepts")
    elif t < 0.75:
        alien_text.set_text("Advanced Transition: 1 Becoming Process-Oriented")
    else:
        alien_text.set_text("Alien Mathematics: Identity Through Dynamic Relations")
    
    # Return all updated artists
    return [classical_line, fuzzy_line, fuzzy2_line, quantum_line, quantum_prob_line, 
           paradigm_text, number_line, numbers_scatter, curvature_text, surface,
           one_node, one_text, alien_text] + quantum_bars + construct_bars + concept_map_nodes

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=100, blit=False)

# Add overall title
fig.suptitle('Numerical Impermanence: Evolution of "1" Across Mathematical Contexts', 
            fontsize=16, y=0.98)

# Show the animation
plt.tight_layout()
plt.subplots_adjust(top=0.92) # Adjust for main title
plt.show()

# Uncomment to save as a video (requires ffmpeg)
# ani.save('numerical_impermanence.mp4', writer='ffmpeg', fps=24, dpi=150)
