import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, RegularPolygon
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

# Classical vs. Fuzzy Number 1 Visualization
# This code visualizes the evolution of the concept of the number 1 from a classical binary perspective
# to a fuzzy perspective, illustrating the transition from a strict definition to a more nuanced understanding.

# # Set up the figure
# fig, ax = plt.subplots(figsize=(10, 6))

# # X-axis range for the number line
# x = np.linspace(-2, 4, 1000)

# def classical_one(x):
#     """Classical representation of the number 1"""
#     return np.where(x == 1, 1, 0)  # Binary: either 1 or not 1

# def fuzzy_one(x, fuzziness):
#     """Fuzzy representation of the number 1 with variable fuzziness"""
#     # As fuzziness increases, the membership function spreads out
#     return np.exp(-(x - 1)**2 / (2 * fuzziness**2))

# # Initial plot
# ax.set_xlim(-2, 4)
# ax.set_ylim(-0.1, 1.1)
# ax.set_xlabel('Number Line', fontsize=12)
# ax.set_ylabel('Membership Degree', fontsize=12)
# ax.set_title('Evolution from Classical to Fuzzy Number 1', fontsize=14)
# ax.grid(True, alpha=0.3)

# # Add reference points
# ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# ax.axhline(y=1, color='k', linestyle='-', alpha=0.3)
# ax.axvline(x=1, color='k', linestyle='--', alpha=0.3)

# # Plot labels for key points
# ax.text(1.1, -0.05, "1", fontsize=12)
# ax.text(-0.1, 1.03, "1.0", fontsize=12)
# ax.text(-0.1, 0.03, "0.0", fontsize=12)

# # Initial lines - we'll update these in the animation
# classical_line, = ax.plot(x, np.zeros_like(x), 'r-', label='Classical 1')
# fuzzy_line, = ax.plot(x, np.zeros_like(x), 'b-', label='Fuzzy 1')
# current_frame_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, 
#                            bbox=dict(facecolor='white', alpha=0.8))

# # Legend
# ax.legend(loc='upper right')

# def update(frame):
#     """Update function for animation"""
#     # Frame determines the transition from classical to fuzzy
#     # 0 = fully classical, 100 = fully fuzzy
#     transition = frame / 100.0
    
#     # Calculate y values for this frame
#     classical_y = classical_one(x) * (1 - transition)
    
#     # Start with almost no fuzziness, then increase
#     fuzziness = 0.01 + transition * 0.5
#     fuzzy_y = fuzzy_one(x, fuzziness) * transition
    
#     # Combined value shows the transition
#     combined_y = classical_y + fuzzy_y
    
#     # Update the plots
#     classical_line.set_ydata(classical_y)
#     fuzzy_line.set_ydata(fuzzy_y)
    
#     # Update text
#     if transition < 0.1:
#         current_frame_text.set_text("Classical Context: Binary Membership")
#     elif transition < 0.5:
#         current_frame_text.set_text("Transitional Context: Emerging Fuzziness")
#     else:
#         current_frame_text.set_text("Fuzzy Context: Graded Membership")
    
#     return classical_line, fuzzy_line, current_frame_text

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(101), blit=True, interval=50)

# plt.tight_layout()
# plt.show()

# # To save the animation (optional)
# ani.save('classical_to_fuzzy.gif', writer='pillow', fps=20)

# # Fuzzy to Quantum Number 1 Visualization
# # This code visualizes the evolution of the concept of the number 1 from a fuzzy perspective
# # to a quantum perspective, illustrating the transition from a fuzzy definition to a quantum state.

# # Set up the figure with two subplots
# fig = plt.figure(figsize=(15, 7))
# ax1 = fig.add_subplot(121)  # 2D plot for fuzzy membership
# ax2 = fig.add_subplot(122, projection='3d')  # 3D plot for quantum state

# # X-axis range
# x = np.linspace(0, 2, 1000)
# theta = np.linspace(0, 2*np.pi, 100)

# def fuzzy_one(x, fuzziness=0.2):
#     """Fuzzy representation of the number 1"""
#     return np.exp(-(x - 1)**2 / (2 * fuzziness**2))

# def quantum_state(theta, psi_0=0.0, psi_1=1.0):
#     """Quantum state representation on Bloch sphere"""
#     # psi_0 and psi_1 are probability amplitudes for |0⟩ and |1⟩
#     # We normalize them so |psi_0|^2 + |psi_1|^2 = 1
#     norm = np.sqrt(np.abs(psi_0)**2 + np.abs(psi_1)**2)
#     psi_0 /= norm
#     psi_1 /= norm
    
#     # Convert to Bloch sphere coordinates
#     # State |0⟩ is at north pole, |1⟩ is at south pole
#     # For the animation, we'll move from |1⟩ to a superposition
#     x = np.sin(theta) * np.real(psi_0 * np.conj(psi_1))
#     y = np.sin(theta) * np.imag(psi_0 * np.conj(psi_1))
#     z = np.cos(theta) * (np.abs(psi_0)**2 - np.abs(psi_1)**2)
    
#     return x, y, z

# # Setup for 2D fuzzy plot
# ax1.set_xlim(0, 2)
# ax1.set_ylim(-0.1, 1.1)
# ax1.set_xlabel('Number Line', fontsize=12)
# ax1.set_ylabel('Membership Degree', fontsize=12)
# ax1.set_title('Fuzzy Representation of 1', fontsize=14)
# ax1.grid(True, alpha=0.3)
# fuzzy_line, = ax1.plot(x, fuzzy_one(x), 'b-', label='Fuzzy 1')
# ax1.legend(loc='upper right')

# # Setup for 3D quantum plot
# # Draw Bloch sphere
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
# y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
# z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
# sphere = ax2.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.2)

# # Initial quantum state vector (|1⟩)
# x_q, y_q, z_q = quantum_state(theta, 0, 1)
# quantum_line, = ax2.plot([0, 0], [0, 0], [0, -1], 'r-', linewidth=3, label='|1⟩')
# ax2.set_xlim(-1.2, 1.2)
# ax2.set_ylim(-1.2, 1.2)
# ax2.set_zlim(-1.2, 1.2)
# ax2.set_title('Quantum State Representation', fontsize=14)

# # Add basis state labels
# ax2.text(0, 0, 1.3, '|0⟩', fontsize=14)
# ax2.text(0, 0, -1.3, '|1⟩', fontsize=14)
# ax2.text(1.3, 0, 0, '|+⟩', fontsize=14)
# ax2.text(-1.3, 0, 0, '|-⟩', fontsize=14)
# ax2.text(0, 1.3, 0, '|i+⟩', fontsize=14)
# ax2.text(0, -1.3, 0, '|i-⟩', fontsize=14)

# # Add text for current state
# state_text = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes, fontsize=12,
#                       bbox=dict(facecolor='white', alpha=0.8))

# def update(frame):
#     """Update function for animation"""
#     # Frame determines transition stage (0-100)
#     transition = frame / 100.0
    
#     # Update fuzzy representation (gradually making it more "quantum-like")
#     fuzzy_values = fuzzy_one(x) * (1 - transition) + \
#                    (np.cos(10 * (x - 1))**2 * np.exp(-(x - 1)**2 / 0.5)) * transition
#     fuzzy_line.set_ydata(fuzzy_values)
    
#     # Update quantum representation (transitioning from |1⟩ to superposition of |0⟩ and |1⟩)
#     # As transition progresses, we introduce more of the |0⟩ state and phase
#     psi_0 = transition * np.exp(1j * np.pi/4)  # Adding |0⟩ with phase
#     psi_1 = 1.0 * np.exp(1j * transition * np.pi/2)  # |1⟩ with changing phase
    
#     x_q, y_q, z_q = quantum_state(theta, psi_0, psi_1)
#     quantum_line.set_data_3d([0, x_q[-1]], [0, y_q[-1]], [0, z_q[-1]])
    
#     # Update state text
#     if transition < 0.05:
#         state_text.set_text("|ψ⟩ ≈ |1⟩")
#     elif transition < 0.5:
#         state_text.set_text(f"|ψ⟩ ≈ {transition:.2f}e^(iπ/4)|0⟩ + {1-transition:.2f}e^(i{transition*np.pi/2:.2f})|1⟩")
#     else:
#         state_text.set_text("|ψ⟩: Superposition state with phase")
    
#     return fuzzy_line, quantum_line, state_text

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(101), blit=False, interval=50)

# plt.tight_layout()
# plt.show()

# # To save the animation (optional)
# ani.save('fuzzy_to_quantum.gif', writer='pillow', fps=20)

# Quantum to Intuition Number 1 Visualization
# This code visualizes the evolution of the concept of the number 1 from a quantum perspective
# to an intuitive perspective, illustrating the transition from a quantum state to a more abstract understanding.

# # Set up the figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# # For quantum representation
# x = np.linspace(0, 2, 1000)

# def quantum_probability(x, phase=0):
#     """Quantum probability density for position near 1"""
#     return 0.5 + 0.5 * np.cos(10 * (x - 1)) * np.exp(-(x - 1)**2 / 0.2)

# # Setup quantum plot
# ax1.set_xlim(0, 2)
# ax1.set_ylim(-0.1, 1.1)
# ax1.set_xlabel('Position', fontsize=12)
# ax1.set_ylabel('Probability Density', fontsize=12)
# ax1.set_title('Quantum Representation', fontsize=14)
# ax1.grid(True, alpha=0.3)
# quantum_line, = ax1.plot(x, quantum_probability(x), 'b-')

# # Intuitionistic representation - we'll use a proof tree
# ax2.set_xlim(-1, 11)
# ax2.set_ylim(-1, 11)
# ax2.axis('off')
# ax2.set_title('Intuitionistic Representation', fontsize=14)

# # Nodes for the proof tree
# nodes = {
#     'axioms': (5, 10),
#     'natural_numbers': (5, 8),
#     'succession': (3, 6),
#     'zero': (7, 6),
#     'one_existence': (5, 4),
#     'one_properties': (5, 2),
#     'one_value': (5, 0)
# }

# # Initial node colors (green = constructed, red = not yet constructed)
# node_colors = {node: 'red' for node in nodes}
# node_colors['axioms'] = 'green'  # Axioms are initially available

# # Add explanation text
# explanation_text = ax2.text(5, -0.5, "", ha='center', fontsize=12, 
#                            bbox=dict(facecolor='white', alpha=0.8))

# # Draw nodes and edges
# node_circles = {}
# node_labels = {}
# edges = []

# # Create initial nodes
# for node, (x, y) in nodes.items():
#     circle = plt.Circle((x, y), 0.5, color=node_colors[node], alpha=0.7)
#     ax2.add_patch(circle)
#     node_circles[node] = circle
    
#     # Add labels with smaller font and wrapped text
#     node_text = node.replace('_', ' ').title()
#     label = ax2.text(x, y, node_text, ha='center', va='center', fontsize=9)
#     node_labels[node] = label

# # Create edges
# edge_pairs = [
#     ('axioms', 'natural_numbers'),
#     ('natural_numbers', 'succession'),
#     ('natural_numbers', 'zero'),
#     ('succession', 'one_existence'),
#     ('zero', 'one_existence'),
#     ('one_existence', 'one_properties'),
#     ('one_properties', 'one_value')
# ]

# for start, end in edge_pairs:
#     x1, y1 = nodes[start]
#     x2, y2 = nodes[end]
#     arrow = FancyArrowPatch((x1, y1-0.5), (x2, y2+0.5), 
#                            arrowstyle='->', color='gray', 
#                            connectionstyle='arc3,rad=0.1', 
#                            mutation_scale=15)
#     ax2.add_patch(arrow)
#     edges.append(arrow)

# # Ensure all artists are initialized
# quantum_line, = ax1.plot(x, quantum_probability(x), 'b-')
# explanation_text = ax2.text(5, -0.5, "", ha='center', fontsize=12, 
#                             bbox=dict(facecolor='white', alpha=0.8))
# node_circles = {}
# for node, (x, y) in nodes.items():
#     circle = plt.Circle((x, y), 0.5, color='red', alpha=0.7)
#     ax2.add_patch(circle)
#     node_circles[node] = circle

# def update(frame):
#     """Update function for animation"""
#     # Frame determines transition stage (0-100)
#     transition = frame / 100.0
    
#     # Update quantum representation (gradually fading out)
#     quantum_opacity = max(0, 1 - transition * 2)  # Fade out quantum representation
#     quantum_values = quantum_probability(x) * quantum_opacity
#     quantum_line.set_ydata(quantum_values)
#     quantum_line.set_alpha(quantum_opacity)
    
#     # Update intuitionistic representation (constructing the number 1 step by step)
#     construction_progress = transition * 7  # 7 steps in our construction
    
#     # Define the order of construction
#     construction_order = [
#         'axioms',  # Already green at the start
#         'natural_numbers',
#         'succession',
#         'zero',
#         'one_existence',
#         'one_properties',
#         'one_value'
#     ]
    
#     # Update colors based on construction progress
#     for i, node in enumerate(construction_order):
#         if i <= construction_progress:
#             node_circles[node].set_color('green')
#         else:
#             node_circles[node].set_color('red')
    
#     # Update explanation text based on construction stage
#     explanations = [
#         "Begin with mathematical axioms",
#         "Construct the concept of natural numbers",
#         "Define the successor operation",
#         "Construct zero as the first natural number",
#         "Construct one as the successor of zero",
#         "Derive the properties of one",
#         "The number 1 now exists as a fully constructed entity"
#     ]
    
#     stage = min(int(construction_progress), 6)
#     explanation_text.set_text(explanations[stage])
    
#     # Return all updated artists
#     return [quantum_line, explanation_text] + list(node_circles.values())

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(101), blit=False, interval=80)

# plt.tight_layout()
# plt.show()

# # To save the animation (optional)
# ani.save('quantum_to_intuitionistic.gif', writer='pillow', fps=20)

# Intuition to Non-Euclidean Number 1 Visualization
# This code visualizes the evolution of the concept of the number 1 from an intuitionistic perspective
# to a non-Euclidean perspective, illustrating the transition from a constructed number to a more abstract understanding.

# # Set up the figure
# fig = plt.figure(figsize=(15, 7))
# ax1 = fig.add_subplot(121)  # 2D plot for intuitionistic view
# ax2 = fig.add_subplot(122, projection='3d')  # 3D plot for non-Euclidean view

# # Setup for intuitionistic view (number line with construction elements)
# ax1.set_xlim(-3, 3)
# ax1.set_ylim(-1.5, 1.5)
# ax1.set_xlabel('Number Line', fontsize=12)
# ax1.set_title('Intuitionistic Number Line', fontsize=14)
# ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
# ax1.grid(True, alpha=0.3)

# # Key points on the intuitionistic number line
# points_x = [-2, -1, 0, 1, 2]
# points_y = [0, 0, 0, 0, 0]
# points_labels = ['-2', '-1', '0', '1', '2']

# # Highlight the number 1 specially
# ax1.plot([1], [0], 'ro', markersize=10)
# ax1.text(1, 0.2, '1', fontsize=14, ha='center')

# # Plot all points
# for i, (x, y, label) in enumerate(zip(points_x, points_y, points_labels)):
#     if x != 1:  # We already plotted 1 specially
#         ax1.plot([x], [y], 'bo', markersize=8)
#         ax1.text(x, y+0.2, label, fontsize=12, ha='center')

# # Construction arrows showing the generation of numbers
# arrows = []
# for i in range(len(points_x)-1):
#     arrow = ax1.annotate("", xy=(points_x[i+1], points_y[i+1]), 
#                         xytext=(points_x[i], points_y[i]),
#                         arrowprops=dict(arrowstyle="->", color='green'))
#     arrows.append(arrow)

# # Add text for construction process
# construction_text = ax1.text(0, -1, "Constructed by succession", fontsize=12, ha='center',
#                            bbox=dict(facecolor='white', alpha=0.8))

# # Setup for non-Euclidean view
# # We'll create a curved surface and place our number line on it
# x = np.linspace(-3, 3, 50)
# y = np.linspace(-3, 3, 50)
# X, Y = np.meshgrid(x, y)

# # Function to generate curved surface with variable curvature
# def curved_surface(X, Y, curvature):
#     return curvature * (X**2 + Y**2) / 10

# # Initial surface (flat)
# Z = curved_surface(X, Y, 0)
# surface = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, linewidth=0, antialiased=True)

# # Place points on the surface
# surface_points_x = np.array(points_x)
# surface_points_y = np.zeros_like(surface_points_x)
# surface_points_z = curved_surface(surface_points_x, surface_points_y, 0)

# # Plot points on the surface
# surface_points = ax2.scatter(surface_points_x, surface_points_y, surface_points_z, 
#                             color='blue', s=50, zorder=10)

# # Highlight the number 1 on the surface
# surface_one = ax2.scatter([1], [0], [curved_surface(1, 0, 0)], 
#                          color='red', s=100, zorder=11)

# # Add a line connecting the points on the surface (our number line)
# line_x = np.linspace(-2, 2, 100)
# line_y = np.zeros_like(line_x)
# line_z = curved_surface(line_x, line_y, 0)
# surface_line = ax2.plot(line_x, line_y, line_z, 'k-', linewidth=2)[0]

# # Setup for geodesic paths (shortest paths in curved space)
# def geodesic_path(start, end, curvature, points=20):
#     """Calculate an approximation of a geodesic path in our curved space"""
#     # This is a simplified model - in a truly non-Euclidean space,
#     # we would compute actual geodesics, but this gives the visual idea
#     t = np.linspace(0, 1, points)
#     x = np.linspace(start[0], end[0], points)
    
#     # For curved spaces, the shortest path bulges away from a straight line
#     # The amount of bulge depends on curvature
#     if curvature > 0:  # Positive curvature (sphere-like)
#         y = np.sin(t * np.pi) * curvature * abs(end[0] - start[0]) / 10
#     else:  # Negative curvature (hyperbolic-like)
#         y = np.zeros_like(x)  # Simplified for visualization
    
#     z = curved_surface(x, y, curvature)
#     return x, y, z

# # Calculate geodesic from 0 to 1
# geo_x, geo_y, geo_z = geodesic_path([0, 0], [1, 0], 0, 50)
# geodesic = ax2.plot(geo_x, geo_y, geo_z, 'g-', linewidth=3)[0]

# # Distance label
# distance_text = ax2.text2D(0.05, 0.95, "Distance from 0 to 1: 1.00", transform=ax2.transAxes,
#                           fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# # Setup 3D view
# ax2.set_xlabel('X', fontsize=12)
# ax2.set_ylabel('Y', fontsize=12)
# ax2.set_zlabel('Z', fontsize=12)
# ax2.set_title('Non-Euclidean Number Space', fontsize=14)
# ax2.view_init(elev=30, azim=-60)

# # Context text to explain what's happening
# context_text = fig.text(0.5, 0.01, "", ha='center', fontsize=14, 
#                       bbox=dict(facecolor='white', alpha=0.8))

# def update(frame):
#     """Update function for animation"""
#     # Frame determines transition stage (0-100)
#     transition = frame / 100.0
    
#     # Gradually increase curvature of the surface
#     curvature = transition * 4  # Max curvature = 4
    
#     # Update surface with new curvature
#     Z = curved_surface(X, Y, curvature)
#     surface.remove()
#     new_surface = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, linewidth=0, antialiased=True)
    
#     # Update points on the surface
#     surface_points_z = curved_surface(surface_points_x, surface_points_y, curvature)
#     surface_points._offsets3d = (surface_points_x, surface_points_y, surface_points_z)
    
#     # Update the special point for 1
#     surface_one._offsets3d = ([1], [0], [curved_surface(1, 0, curvature)])
    
#     # Update the number line on the surface
#     line_z = curved_surface(line_x, line_y, curvature)
#     surface_line.set_data_3d(line_x, line_y, line_z)
    
#     # Update geodesic path from 0 to 1
#     geo_x, geo_y, geo_z = geodesic_path([0, 0], [1, 0], curvature, 50)
#     geodesic.set_data_3d(geo_x, geo_y, geo_z)
    
#     # Calculate the geodesic distance (arc length along the curve)
#     # This is a simplification - in a real non-Euclidean space, 
#     # we would integrate along the geodesic
#     dx = np.diff(geo_x)
#     dy = np.diff(geo_y)
#     dz = np.diff(geo_z)
#     segments = np.sqrt(dx**2 + dy**2 + dz**2)
#     distance = np.sum(segments)
    
#     # Update distance text
#     distance_text.set_text(f"Distance from 0 to 1: {distance:.2f}")
    
#     # Update context text
#     if transition < 0.25:
#         context_text.set_text("Intuitionistic Context: 1 exists through construction from 0")
#     elif transition < 0.5:
#         context_text.set_text("Early Non-Euclidean Context: Space begins to curve")
#     elif transition < 0.75:
#         context_text.set_text("Moderate Curvature: Distance between numbers changes")
#     else:
#         context_text.set_text("Strong Non-Euclidean Context: The identity of 1 depends on geometry")
    
#     return [new_surface, surface_points, surface_one, surface_line, geodesic, distance_text, context_text]

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(101), blit=False, interval=80)

# plt.tight_layout()
# plt.show()

# # To save the animation (optional)
# ani.save('intuitionistic_to_non_euclidean.gif', writer='pillow', fps=20)

# Non-Euclidean to Speculative Alien Number 1 Visualization
# This code visualizes the evolution of the concept of the number 1 from a non-Euclidean perspective
# to a speculative alien perspective, illustrating the transition from a geometric understanding to an abstract one.

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Setup for non-Euclidean view (left panel)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal')
ax1.set_title('Non-Euclidean Mathematics', fontsize=14)
ax1.grid(True, alpha=0.3)

# Create a hyperbolic disk (Poincaré model)
circle = Circle((0, 0), 4, fill=False, color='black')
ax1.add_patch(circle)

# Create a grid of points in the hyperbolic disk
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
r = np.linspace(1, 3.5, 4)
points = []

for radius in r:
    for angle in theta:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append((x, y))

# Add special point for 1 (at position (1, 0))
points.append((1, 0))

# Create circles at each point
circles = []
for x, y in points:
    # Make "1" red and larger
    if (x, y) == (1, 0):
        circle = Circle((x, y), 0.3, color='red', alpha=0.8)
    else:
        circle = Circle((x, y), 0.2, color='blue', alpha=0.6)
    ax1.add_patch(circle)
    circles.append(circle)

# Label for the number 1
one_label = ax1.text(1, 0, "1", fontsize=14, ha='center', va='center', color='white')

# Add hyperbolic lines (geodesics)
def hyperbolic_line(start, end, resolution=100):
    """Create a geodesic in the Poincaré disk model between two points"""
    # Convert to complex numbers for easier math
    z1 = complex(start[0], start[1])
    z2 = complex(end[0], end[1])
    
    # If points are on same radius, the geodesic is a straight line
    if abs(np.angle(z1) - np.angle(z2)) < 1e-10 or abs(abs(np.angle(z1) - np.angle(z2)) - np.pi) < 1e-10:
        t = np.linspace(0, 1, resolution)
        x = start[0] * (1-t) + end[0] * t
        y = start[1] * (1-t) + end[1] * t
        return x, y
    
    # Otherwise, it's an arc of a circle perpendicular to the boundary
    # This is a simplified approach for visualization purposes
    center_angle = (np.angle(z1) + np.angle(z2)) / 2
    if abs(np.angle(z1) - np.angle(z2)) > np.pi:
        center_angle += np.pi
    
    # Find a point outside the disk that both points are equidistant from
    outer_radius = 6  # Beyond the disk boundary
    center_x = outer_radius * np.cos(center_angle)
    center_y = outer_radius * np.sin(center_angle)
    
    # Calculate angles from this center to our points
    angle1 = np.arctan2(start[1] - center_y, start[0] - center_x)
    angle2 = np.arctan2(end[1] - center_y, end[0] - center_x)
    
    # Ensure we draw the shorter arc
    if abs(angle1 - angle2) > np.pi:
        if angle1 > angle2:
            angle2 += 2 * np.pi
        else:
            angle1 += 2 * np.pi
    
    angles = np.linspace(angle1, angle2, resolution)
    radius = np.sqrt((start[0] - center_x)**2 + (start[1] - center_y)**2)
    
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    
    return x, y

# Add some geodesics connecting to the number 1
geodesics = []
for i, point in enumerate(points[:-1]):  # Exclude the last point which is 1
    if i % 3 == 0:  # Only draw some geodesics to avoid clutter
        x, y = hyperbolic_line(point, (1, 0))
        line, = ax1.plot(x, y, 'g-', alpha=0.5)
        geodesics.append(line)

# Setup for alien mathematics (right panel)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_aspect('equal')
ax2.set_title('Alien Mathematical Framework', fontsize=14)
ax2.axis('off')

# Create a complex network visualization to represent alien mathematics
# We'll use a dynamic system of interacting agents/nodes

# Generate initial node positions in a pattern
num_nodes = 30
node_positions = []
node_types = []
node_sizes = []

# Create pattern of nodes
for i in range(num_nodes):
    angle = 2 * np.pi * i / num_nodes
    r = 4 * np.random.rand() ** 0.5  # Square root to distribute more evenly
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    node_positions.append((x, y))
    
    # Assign different types (representing different mathematical concepts)
    if i % 5 == 0:
        node_types.append("process")  # Process-oriented concepts
        node_sizes.append(0.4)
    elif i % 5 == 1:
        node_types.append("relation")  # Relational concepts
        node_sizes.append(0.3)
    elif i % 5 == 2:
        node_types.append("entity")  # Entity-like concepts
        node_sizes.append(0.35)
    else:
        node_types.append("emergent")  # Emergent concepts
        node_sizes.append(0.25)

# Colors for different types
type_colors = {
    "process": 'blue',
    "relation": 'green',
    "entity": 'purple',
    "emergent": 'orange'
}

# Create nodes
alien_nodes = []
for i, ((x, y), type_, size) in enumerate(zip(node_positions, node_types, node_sizes)):
    if type_ == "process":
        shape = RegularPolygon((x, y), 3, radius=size, orientation=np.pi/4, 
                              color=type_colors[type_], alpha=0.8)
    elif type_ == "relation":
        shape = RegularPolygon((x, y), 4, radius=size, 
                              color=type_colors[type_], alpha=0.8)
    elif type_ == "entity":
        shape = Circle((x, y), size, color=type_colors[type_], alpha=0.8)
    else:
        shape = RegularPolygon((x, y), 5, radius=size, 
                              color=type_colors[type_], alpha=0.8)
    
    ax2.add_patch(shape)
    alien_nodes.append(shape)

# Create links between nodes (representing alien mathematical relationships)
alien_links = []
for i in range(num_nodes):
    # Connect each node to several others based on mathematical affinity
    num_connections = np.random.randint(1, 4)
    connected = set()
    
    for _ in range(num_connections):
        # Find a node to connect to
        while True:
            j = np.random.randint(0, num_nodes)
            if j != i and j not in connected:
                connected.add(j)
                break
        
        # Create link with style based on relationship type
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[j]
        
        # Different link styles based on the types of nodes being connected
        if node_types[i] == node_types[j]:
            # Same type - solid line
            link, = ax2.plot([x1, x2], [y1, y2], '-', color='gray', alpha=0.4, linewidth=1)
        else:
            # Different types - dashed line
            link, = ax2.plot([x1, x2], [y1, y2], '--', color='gray', alpha=0.4, linewidth=1)
        
        alien_links.append(link)

# Mark the concept that most closely relates to our "1"
# We'll use a special highlighted node that starts at (1,0) position but eventually
# gets integrated into the alien system
one_concept_position = [1, 0]
one_concept = Circle(one_concept_position, 0.4, color='red', alpha=0.9)
ax2.add_patch(one_concept)
one_concept_label = ax2.text(one_concept_position[0], one_concept_position[1], 
                           "1", fontsize=14, ha='center', va='center', color='white')

# Add connections from "1" to other nodes
one_connections = []
for i in range(5):  # Connect to a few random nodes
    j = np.random.randint(0, num_nodes)
    x1, y1 = one_concept_position
    x2, y2 = node_positions[j]
    link, = ax2.plot([x1, x2], [y1, y2], '-', color='red', alpha=0.5, linewidth=1.5)
    one_connections.append(link)

# Add legend for alien mathematics
legend_items = [
    Circle((0, 0), 0.1, color=type_colors["entity"]),
    RegularPolygon((0, 0), 3, radius=0.1, orientation=np.pi/4, color=type_colors["process"]),
    RegularPolygon((0, 0), 4, radius=0.1, color=type_colors["relation"]),
    RegularPolygon((0, 0), 5, radius=0.1, color=type_colors["emergent"]),
]
legend_labels = ["Entity-like", "Process-oriented", "Relational", "Emergent"]

# Create custom legend
legend = ax2.legend(legend_items, legend_labels, loc='upper right', title="Alien Mathematical Concepts")

# Add explanation text
explanation_text = fig.text(0.5, 0.02, "", ha='center', fontsize=14, 
                          bbox=dict(facecolor='white', alpha=0.8))

def update(frame):
    """Update function for animation"""
    # Frame determines transition stage (0-100)
    transition = frame / 100.0
    
    # 1. Update Non-Euclidean side - gradually fade it out
    for circle in circles:
        circle.set_alpha(max(0.1, 0.8 - transition * 0.8))
    
    for geo in geodesics:
        geo.set_alpha(max(0.1, 0.5 - transition * 0.5))
    
    # Fade out the label for 1
    one_label.set_alpha(max(0.1, 1.0 - transition * 0.9))
    
    # 2. Update Alien Mathematics side
    # Have nodes move in a complex pattern
    for i, node in enumerate(alien_nodes):
        # Original position
        orig_x, orig_y = node_positions[i]
        
        # Add oscillatory motion based on node type and transition
        if node_types[i] == "process":
            dx = 0.3 * np.sin(transition * 10 + i * 0.5) * transition
            dy = 0.3 * np.cos(transition * 8 + i * 0.7) * transition
        elif node_types[i] == "relation":
            dx = 0.2 * np.sin(transition * 12 + i * 0.3) * transition
            dy = 0.2 * np.cos(transition * 9 + i * 0.6) * transition
        elif node_types[i] == "entity":
            dx = 0.15 * np.sin(transition * 7 + i * 0.4) * transition
            dy = 0.15 * np.cos(transition * 11 + i * 0.2) * transition
        else:
            dx = 0.25 * np.sin(transition * 9 + i * 0.8) * transition
            dy = 0.25 * np.cos(transition * 10 + i * 0.5) * transition
        
        # Update position
        new_x = orig_x + dx
        new_y = orig_y + dy
        
        # Update node shape
        if isinstance(node, Circle):
            node.center = (new_x, new_y)
        else:  # RegularPolygon
            node.xy = (new_x, new_y)
        
        # Update node position for connection updates
        node_positions[i] = (new_x, new_y)
    
    # Update connections between alien nodes
    for i, link in enumerate(alien_links):
        # Find the nodes this link connects
        # This is a simplification - in a real implementation, 
        # we would store this information more efficiently
        if i < len(alien_links) // 2:
            idx1 = (i * 2) % num_nodes
            idx2 = (i * 3 + 1) % num_nodes
        else:
            idx1 = (i + 7) % num_nodes
            idx2 = (i * 2 + 3) % num_nodes
        
        x1, y1 = node_positions[idx1]
        x2, y2 = node_positions[idx2]
        link.set_data([x1, x2], [y1, y2])
    
    # Transform the concept of "1" into an alien mathematical concept
    # It starts at (1,0) but gradually moves to a new position and changes in nature
    target_x = 2 * np.sin(transition * 5)
    target_y = -1 + transition * 3
    
    one_x = 1 * (1 - transition) + target_x * transition
    one_y = 0 * (1 - transition) + target_y * transition
    
    # Update the visualization of "1"
    one_concept.center = (one_x, one_y)
    one_concept_label.set_position((one_x, one_y))
    
    # Gradually change label from "1" to an alien symbol
    if transition < 0.3:
        one_concept_label.set_text("1")
    elif transition < 0.6:
        one_concept_label.set_text("⊕")  # A more abstract symbol
    else:
        one_concept_label.set_text("⋇")  # Even more alien looking
    
    # Update connections from "1" to other nodes
    for i, link in enumerate(one_connections):
        j = (i * 5 + 3) % num_nodes
        x2, y2 = node_positions[j]
        link.set_data([one_x, x2], [one_y, y2])
        
        # Change connection style as transition progresses
        if transition > 0.5:
            link.set_linestyle('--')
        if transition > 0.8:
            link.set_color(type_colors[node_types[j]])
    
    # Update explanation
    if transition < 0.25:
        explanation_text.set_text("Initial Context: Number 1 exists in a non-Euclidean geometry")
    elif transition < 0.5:
        explanation_text.set_text("Transition: The concept of 1 begins to transform into an alien mathematical framework")
    elif transition < 0.75:
        explanation_text.set_text("Alien Context: 1 becomes a process-relationship nexus rather than a fixed entity")
    else:
        explanation_text.set_text("Complete Transformation: The original identity of 1 is unrecognizable in the alien framework")
    
    # Return all updated artists
    return circles + geodesics + [one_label, one_concept, one_concept_label, explanation_text] + alien_nodes + alien_links + one_connections

# Create animation
ani = FuncAnimation(fig, update, frames=range(101), blit=False, interval=80)

plt.tight_layout()
plt.show()

# To save the animation (optional)
ani.save('non_euclidean_to_alien.gif', writer='pillow', fps=20)