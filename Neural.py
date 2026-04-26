import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Neural Network Visualizer")

# --- Function to generate decision boundary visualization ---
def plot_decision_boundary(X, y, model, ax, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())

# --- Function to draw the neural network graph ---
def draw_neural_network(layer_sizes, ax):
    num_layers = len(layer_sizes)
    v_spacing = 0.3
    h_spacing = 0.5
    
    # Generate node positions
    pos = {}
    for i, n in enumerate(layer_sizes):
        layer_width = n * v_spacing
        layer_pos = h_spacing * i
        
        for j in range(n):
            node_key = f"{i}_{j}"
            x_pos = layer_pos
            y_pos = -layer_width / 2.0 + j * v_spacing + v_spacing / 2.0
            pos[node_key] = (x_pos, y_pos)
            
    # Add nodes and edges to the graph
    G = nx.Graph()
    for i, n in enumerate(layer_sizes):
        if i == 0:
            color = 'lightgreen'  # Input layer
        elif i == num_layers - 1:
            color = 'lightcoral'  # Output layer
        else:
            color = 'lightskyblue' # Hidden layer
            
        for j in range(n):
            node_key = f"{i}_{j}"
            G.add_node(node_key, pos=pos[node_key], color=color)
            
            # Connect to nodes in the next layer
            if i < num_layers - 1:
                next_layer_size = layer_sizes[i + 1]
                for k in range(next_layer_size):
                    next_node_key = f"{i+1}_{k}"
                    G.add_edge(node_key, next_node_key)
                    
    # Draw nodes and edges
    node_colors = [data['color'] for node, data in G.nodes(data=True)]
    edge_alpha = 0.5
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, node_color=node_colors, edgecolors='black')
    
    # Label the layers
    layer_names = ["Input", "Hidden", "Output"]
    for i, n in enumerate(layer_sizes):
        layer_pos = h_spacing * i
        ax.text(layer_pos, max([p[1] for p in pos.values()]) + 0.1, f"**{layer_names[i if i<2 else 2]} Layer**\n({n} nodes)", 
                horizontalalignment='center', fontweight='bold', fontsize=12)

    ax.axis('off')
    ax.set_title("Neural Network Architecture", fontsize=16)

# --- Sidebar ---
st.sidebar.header("Network Parameters")

# 1. Dataset Selection (Optional - currently fixed to 'Moons')

# 2. Network Architecture
num_hidden_neurons = st.sidebar.slider("Number of Neurons (Hidden Layer)", min_value=1, max_value=50, value=10)

# 3. Activation Function
activation_function = st.sidebar.selectbox("Activation Function", ("relu", "tanh", "logistic", "identity"))

# 4. Learning Rate
learning_rate_type = st.sidebar.selectbox("Learning Rate Schedule", ("constant", "invscaling", "adaptive"))
initial_learning_rate = st.sidebar.number_input("Initial Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, format="%.5f")

# 5. Solver (Optimizer)
solver = st.sidebar.selectbox("Solver (Optimizer)", ("adam", "sgd", "lbfgs"))

# 6. Maximum Iterations (Epochs)
max_iter = st.sidebar.number_input("Maximum Iterations", min_value=10, max_value=10000, value=500)

st.sidebar.markdown("---")
st.sidebar.markdown("Amitesh Ranjan | (25212011110)")

# --- Main Page Content ---
st.title("Interactive Neural Network Visualizer")
st.markdown("""
This app visualizes a neural network trained on the 'Moons' dataset. You can change the network's parameters on the sidebar to see how the architecture and decision boundary are affected.
""")

col1, col2 = st.columns([1, 1])

with col1:
    # --- Generate data ---
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Train Neural Network ---
    hidden_layer_sizes = (num_hidden_neurons,)  # Tuple for hidden layers

    with st.spinner(f"Training MLPClassifier with {num_hidden_neurons} neurons, {activation_function} activation, learning rate {initial_learning_rate}, solver {solver}..."):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                              activation=activation_function,
                              learning_rate=learning_rate_type,
                              learning_rate_init=initial_learning_rate,
                              solver=solver,
                              max_iter=max_iter,
                              random_state=42)
        model.fit(X_train, y_train)

    # --- Metrics ---
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    st.subheader("Model Performance")
    st.write(f"**Training Accuracy:** {train_accuracy:.4f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.4f}")

    # --- Visualize Decision Boundary ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(X_train, y_train, model, ax1, title=f"Decision Boundary on Training Data")
    st.pyplot(fig1)

with col2:
    # --- Visualize Neural Network Architecture ---
    # Input nodes: 2 (for 2D features x, y)
    # Output nodes: 1 (for binary classification - predict class 0 or 1)
    layer_sizes = [2, num_hidden_neurons, 1] 

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    draw_neural_network(layer_sizes, ax2)
    st.pyplot(fig2)

# Add a section to explain parameters and their effects
with st.expander("Explanation of Parameters", expanded=False):
    st.markdown("""
    * **Number of Neurons (Hidden Layer):** The total nodes in the one hidden layer. More neurons can capture complex patterns but also risk overfitting.
    * **Activation Function:** The non-linear function used in the hidden layer neurons.
        * `relu` (Rectified Linear Unit): Default for many networks, computes `max(0, x)`. Simple and effective.
        * `tanh`: S-shaped function, outputs between -1 and 1. Zero-centered.
        * `logistic` (Sigmoid): S-shaped, outputs between 0 and 1. Used in logistic regression.
        * `identity`: Linear activation, outputs `x`. Turns the layer into linear regression.
    * **Learning Rate Schedule:** How the learning rate is adjusted during training.
        * `constant`: Keeps learning rate fixed.
        * `invscaling`: Decreases learning rate inverse-exponentially.
        * `adaptive`: Keeps learning rate fixed unless training loss isn't improving, then decreases it.
    * **Initial Learning Rate:** The starting step size for the optimizer. Small values are stable but learn slowly; large values learn quickly but can be unstable.
    * **Solver (Optimizer):** The algorithm used to find optimal weights.
        * `adam`: Stochasitc gradient descent method. Good default for large datasets/networks.
        * `sgd`: Classic stochastic gradient descent.
        * `lbfgs`: Optimizer in the family of quasi-Newton methods. Can converge faster for small datasets.
    * **Maximum Iterations:** Number of epochs (full passes through the training data). Too low can result in underfitting.
    """)