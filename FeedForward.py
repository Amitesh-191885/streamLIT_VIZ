import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Feedforward Neural Network",
    page_icon="🧠",
    layout="wide",
)

ACTIVATION_OPTIONS = ["linear", "relu", "leaky_relu", "sigmoid", "tanh", "softmax", "step"]
ACTIVATION_FORMULAS = {
    "linear": "f(x) = x",
    "relu": "f(x) = max(0, x)",
    "leaky_relu": "f(x) = x if x ≥ 0 else 0.01x",
    "sigmoid": "f(x) = 1 / (1 + e^-x)",
    "tanh": "f(x) = tanh(x)",
    "softmax": "f(x) = e^xᵢ / Σe^xⱼ",
    "step": "f(x) = 1 if x ≥ 0 else 0",
}


def apply_activation(z: np.ndarray, fn: str) -> np.ndarray:
    if fn == "relu":
        return np.maximum(0, z)
    elif fn == "leaky_relu":
        return np.where(z >= 0, z, 0.01 * z)
    elif fn == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif fn == "tanh":
        return np.tanh(z)
    elif fn == "softmax":
        z_shifted = z - np.max(z)
        e = np.exp(z_shifted)
        return e / e.sum()
    elif fn == "step":
        return np.where(z >= 0, 1.0, 0.0)
    else:
        return z.copy()


def he_init(fan_in: int, fan_out: int, seed: int = 42, layer_idx: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + layer_idx * 1000)
    scale = np.sqrt(2.0 / fan_in)
    return rng.normal(0, scale, (fan_out, fan_in))


def forward_pass(input_vals: np.ndarray, layers, weights, biases):
    layer_outputs = []
    fn = layers[0]["activation"]
    a = apply_activation(input_vals, fn)
    layer_outputs.append({"z": input_vals.copy(), "a": a.copy(), "activation": fn})

    for i in range(1, len(layers)):
        W = weights[i - 1]
        b = biases[i - 1]
        z = W @ layer_outputs[-1]["a"] + b
        fn = layers[i]["activation"]
        a = apply_activation(z, fn)
        layer_outputs.append({"z": z, "a": a, "activation": fn})

    return layer_outputs


def build_weights(layers, seed):
    weights = []
    biases = []
    for i in range(1, len(layers)):
        fan_in = layers[i - 1]["neurons"]
        fan_out = layers[i]["neurons"]
        W = he_init(fan_in, fan_out, seed=seed, layer_idx=i)
        b = np.zeros(fan_out)
        weights.append(W)
        biases.append(b)
    return weights, biases


def layer_output_to_df(layer_out, layer_cfg, layer_idx) -> pd.DataFrame:
    n = layer_cfg["neurons"]
    prefix = "x" if layer_idx == 0 else "a"
    neuron_labels = [f"{prefix}{j+1}" for j in range(n)]
    df = pd.DataFrame({
        "Neuron": neuron_labels,
        "Pre-activation z": np.round(layer_out["z"], 6),
        "Post-activation a": np.round(layer_out["a"], 6),
    })
    return df


def weights_to_df(W: np.ndarray, fan_in: int, fan_out: int, from_idx: int, to_idx: int) -> pd.DataFrame:
    from_labels = [f"n{i+1} (L{from_idx+1})" for i in range(fan_in)]
    to_labels = [f"n{j+1} (L{to_idx+1})" for j in range(fan_out)]
    df = pd.DataFrame(np.round(W, 6), index=to_labels, columns=from_labels)
    return df


st.title("Feedforward Neural Network")
st.caption("Configure layers, set activation functions, and run the forward pass using NumPy computation.")

if "layers" not in st.session_state:
    st.session_state.layers = [
        {"neurons": 3, "activation": "linear", "label": "Input"},
        {"neurons": 4, "activation": "relu", "label": "Hidden 1"},
        {"neurons": 4, "activation": "relu", "label": "Hidden 2"},
        {"neurons": 2, "activation": "softmax", "label": "Output"},
    ]

if "weight_seed" not in st.session_state:
    st.session_state.weight_seed = 42

layers = st.session_state.layers

col_arch, col_main = st.columns([1, 2], gap="large")

with col_arch:
    st.subheader("Network Architecture")

    st.markdown("##### Input Layer")
    c1, c2 = st.columns(2)
    layers[0]["neurons"] = c1.number_input(
        "Neurons", min_value=1, max_value=16, value=layers[0]["neurons"], key="in_neurons"
    )
    layers[0]["activation"] = c2.selectbox(
        "Activation", ACTIVATION_OPTIONS, index=ACTIVATION_OPTIONS.index(layers[0]["activation"]), key="in_act"
    )
    st.caption(ACTIVATION_FORMULAS[layers[0]["activation"]])

    n_hidden = len(layers) - 2
    st.markdown(f"##### Hidden Layers ({n_hidden})")

    i = 1
    while i < len(layers) - 1:
        with st.container():
            hc1, hc2, hc3 = st.columns([2, 2, 1])
            layers[i]["neurons"] = hc1.number_input(
                f"H{i} Neurons", min_value=1, max_value=16,
                value=layers[i]["neurons"], key=f"h{i}_neurons"
            )
            layers[i]["activation"] = hc2.selectbox(
                f"H{i} Activation", ACTIVATION_OPTIONS,
                index=ACTIVATION_OPTIONS.index(layers[i]["activation"]), key=f"h{i}_act"
            )
            if hc3.button("✕", key=f"del_{i}", help="Remove this layer"):
                layers.pop(i)
                for j, lyr in enumerate(layers):
                    if j == 0:
                        lyr["label"] = "Input"
                    elif j == len(layers) - 1:
                        lyr["label"] = "Output"
                    else:
                        lyr["label"] = f"Hidden {j}"
                st.rerun()
            st.caption(ACTIVATION_FORMULAS[layers[i]["activation"]])
        i += 1

    if st.button("+ Add Hidden Layer"):
        insert_at = len(layers) - 1
        prev_n = layers[insert_at - 1]["neurons"]
        layers.insert(insert_at, {"neurons": prev_n, "activation": "relu", "label": f"Hidden {insert_at}"})
        for j, lyr in enumerate(layers):
            if j == 0:
                lyr["label"] = "Input"
            elif j == len(layers) - 1:
                lyr["label"] = "Output"
            else:
                lyr["label"] = f"Hidden {j}"
        st.rerun()

    st.markdown("##### Output Layer")
    oc1, oc2 = st.columns(2)
    out_idx = len(layers) - 1
    layers[out_idx]["neurons"] = oc1.number_input(
        "Neurons", min_value=1, max_value=16,
        value=layers[out_idx]["neurons"], key="out_neurons"
    )
    layers[out_idx]["activation"] = oc2.selectbox(
        "Activation", ACTIVATION_OPTIONS,
        index=ACTIVATION_OPTIONS.index(layers[out_idx]["activation"]), key="out_act"
    )
    st.caption(ACTIVATION_FORMULAS[layers[out_idx]["activation"]])

    st.divider()

    st.markdown("##### Weight Initialization")
    seed_col, rand_col = st.columns([2, 1])
    new_seed = seed_col.number_input("Random seed", min_value=0, max_value=99999,
                                      value=st.session_state.weight_seed, step=1)
    st.session_state.weight_seed = new_seed
    if rand_col.button("Random"):
        st.session_state.weight_seed = int(np.random.randint(0, 99999))
        st.rerun()

    st.caption("He (Kaiming) initialization: W ~ N(0, √(2/fan_in)). Biases initialized to 0.")

with col_main:
    st.subheader("Input Values")
    n_inputs = layers[0]["neurons"]

    if "input_vals" not in st.session_state or len(st.session_state.input_vals) != n_inputs:
        st.session_state.input_vals = [0.5, -0.3, 1.0] + [0.0] * max(0, n_inputs - 3)
        st.session_state.input_vals = st.session_state.input_vals[:n_inputs]

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    if preset_col1.button("All zeros"):
        st.session_state.input_vals = [0.0] * n_inputs
        st.rerun()
    if preset_col2.button("All ones"):
        st.session_state.input_vals = [1.0] * n_inputs
        st.rerun()
    if preset_col3.button("Unit vector"):
        st.session_state.input_vals = [1.0 if j == 0 else 0.0 for j in range(n_inputs)]
        st.rerun()
    if preset_col4.button("Random input"):
        rng = np.random.default_rng()
        st.session_state.input_vals = list(np.round(rng.uniform(-1, 1, n_inputs), 4))
        st.rerun()

    input_cols = st.columns(min(n_inputs, 8))
    for j in range(n_inputs):
        col_j = input_cols[j % len(input_cols)]
        st.session_state.input_vals[j] = col_j.number_input(
            f"x{j+1}", value=float(st.session_state.input_vals[j]),
            step=0.1, format="%.4f", key=f"inp_{j}"
        )

    input_arr = np.array(st.session_state.input_vals, dtype=float)

    st.divider()

    run_col, _ = st.columns([1, 3])
    run_clicked = run_col.button("▶ Run Forward Pass", type="primary", use_container_width=True)

    if run_clicked or st.session_state.get("last_run"):
        if run_clicked:
            st.session_state.last_run = True

        weights, biases = build_weights(layers, st.session_state.weight_seed)
        layer_outputs = forward_pass(input_arr, layers, weights, biases)

        st.subheader("Forward Pass Results")

        tab_labels = [f"L{i+1}: {lyr['label']}" for i, lyr in enumerate(layers)]
        tabs = st.tabs(tab_labels)

        for i, tab in enumerate(tabs):
            with tab:
                lyr = layers[i]
                lo = layer_outputs[i]

                st.markdown(f"**Activation:** `{lyr['activation']}` — {ACTIVATION_FORMULAS[lyr['activation']]}")

                df = layer_output_to_df(lo, lyr, i)

                if i == 0:
                    st.markdown("*Input layer: z = raw input, a = activation(z)*")
                    display_df = df.drop(columns=["Pre-activation z"])
                    display_df.columns = ["Neuron", "Value a"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    if i == len(layers) - 1:
                        best_idx = int(np.argmax(lo["a"]))
                        st.success(
                            f"**Predicted class: #{best_idx + 1}** "
                            f"(neuron a{best_idx+1} has highest activation: {lo['a'][best_idx]:.6f})"
                        )

                    if i > 0:
                        with st.expander("Weight matrix W (this layer ← previous layer)"):
                            W = weights[i - 1]
                            b = biases[i - 1]
                            w_df = weights_to_df(W, layers[i-1]["neurons"], lyr["neurons"], i-1, i)
                            st.dataframe(w_df, use_container_width=True)
                            bias_df = pd.DataFrame({
                                "Neuron": [f"n{j+1}" for j in range(lyr["neurons"])],
                                "Bias b": np.round(b, 6),
                            })
                            st.markdown("**Bias vector b:**")
                            st.dataframe(bias_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Network Summary")

        summary_rows = []
        total_params = 0
        for i, lyr in enumerate(layers):
            params = 0
            if i > 0:
                params = lyr["neurons"] * layers[i-1]["neurons"] + lyr["neurons"]
                total_params += params
            summary_rows.append({
                "Layer": f"L{i+1}",
                "Name": lyr["label"],
                "Neurons": lyr["neurons"],
                "Activation": lyr["activation"],
                "Trainable Parameters": str(params) if i > 0 else "—",
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.metric("Total trainable parameters", total_params)

        st.divider()
        st.subheader("Weight Statistics (per layer)")

        stats_rows = []
        for i, W in enumerate(weights):
            flat = W.flatten()
            stats_rows.append({
                "Weight Matrix": f"W{i+1} (L{i+1}→L{i+2})",
                "Shape": f"{W.shape[0]} × {W.shape[1]}",
                "Count": flat.size,
                "Mean": round(float(np.mean(flat)), 6),
                "Std Dev": round(float(np.std(flat)), 6),
                "Min": round(float(np.min(flat)), 6),
                "Max": round(float(np.max(flat)), 6),
            })
        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    else:
        st.info("Configure your network architecture and click **Run Forward Pass** to compute activations.")
