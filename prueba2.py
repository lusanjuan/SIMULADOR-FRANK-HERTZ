import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# ---------------- CONFIGURACIÓN DE PÁGINA Y ESTILO ----------------
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

st.markdown("""
    <style>
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 80% !important;
        }

        .stApp {
            background-color: #1b2d40;
        }

        h1, h2, h3, label, .stMarkdown, .css-qrbaxs {
            color: white !important;
            font-family: 'Segoe UI', sans-serif;
        }

        .stButton > button {
            background-color: #0a141a;
            color: white !important;
            border: 1px solid #444;
            padding: 0.5em 1em;
            border-radius: 8px;
            font-weight: bold;
        }

        .stButton > button:hover {
            background-color: #444444;
            color: white !important;
        }

        .stSlider {
            background-color: #0a141a;
            padding: 1rem;
            border: 1px solid #9da9b0;
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        .stSlider label {
            color: white !important;
            font-weight: bold;
        }

        .stSlider .css-14rggix {
            background-color: #1f1f1f !important;
            border-radius: 5px;
        }

        .stSlider .css-1c5cd5h {
            background-color: #00ffcc !important;
        }

        .stSlider .css-1cpxqw2 {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------- TÍTULO Y DESCRIPCIÓN ----------------
st.markdown("<h1>Experimento de Franck y Hertz</h1>", unsafe_allow_html=True)
st.markdown("""
Este simulador te permite explorar cómo los electrones colisionan con átomos a medida que aumenta el voltaje. 
Observa los picos de corriente cuando los electrones pierden energía por excitación atómica.
""")


# ---------------- PARTE 1: SLIDERS Y GRÁFICO ----------------
sliders_col, grafico_col = st.columns([1, 2])

with sliders_col:
    pot_excitacion = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
    voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 0.0, 0.1)
    num_electrones = st.slider("Número de electrones", 5, 100, 20)

def corriente_simulada(V, e_exc, num_electrones, A=1.0, B=0.7, alpha=1.2, phi=0):
    V = np.array(V)
    corriente = A * V**alpha * (1 - B * np.sin(np.pi * V / e_exc + phi)**2)
    corriente[V <= 0] = 0
    return corriente * num_electrones

voltaje = np.linspace(0, 50, 500)
corriente_actual = corriente_simulada([voltaje_max], pot_excitacion, num_electrones)[0]
with sliders_col:
    st.markdown(f"Corriente estimada para {voltaje_max:.1f} V: **{corriente_actual:.3f}** (u.a.)")
st.markdown("**Nota:** la corriente simulada es cualitativa y escalada al número de electrones.")

corrientes_visibles = corriente_simulada(voltaje, pot_excitacion, num_electrones)
corrientes_visibles[voltaje > voltaje_max] = np.nan

with grafico_col:
    fig, ax = plt.subplots()
    ax.plot(voltaje, corrientes_visibles, color="blue", label="Corriente simulada")
    ax.axvline(voltaje_max, color="red", linestyle="--", label=f"Voltaje actual: {voltaje_max:.1f} V")
    ax.set_xlabel("Voltaje (V)")
    ax.set_ylabel("Corriente (u.a.)")
    ax.set_title("Corriente vs Voltaje")
    ax.legend()
    st.pyplot(fig)


# ---------------- PARTE 2: SIMULACIÓN VISUAL ----------------
st.markdown("<h2>Simulación visual del experimento</h2>", unsafe_allow_html=True)

ancho, altura = 10, 5
dt = 0.1
np.random.seed(1)

posiciones = np.zeros((num_electrones, 2))
posiciones[:, 1] = np.random.uniform(0.5, altura - 0.5, num_electrones)

velocidades = np.zeros((num_electrones, 2))
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

cooldown = np.zeros(num_electrones)

num_atom = 70
pos_atoms_x = np.linspace(1, 13, num_atom)
pos_atoms_y = np.random.uniform(0, altura, num_atom)
atoms = np.column_stack((pos_atoms_x, pos_atoms_y))

if "animando" not in st.session_state:
    st.session_state.animando = False

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✅ Comenzar animación"):
        st.session_state.animando = True
with btn_col2:
    if st.button("🛑 Detener animación"):
        st.session_state.animando = False

grafico = st.empty()

# ---------------- LOOP DE ANIMACIÓN ----------------
trails = [[] for _ in range(num_electrones)]
trail_length = 6  # Cantidad de puntos para dejar rastro

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
canvas = st.empty()

while st.session_state.animando:
    posiciones[:, 0] += velocidades[:, 0] * dt

    for i in range(num_electrones):
        if cooldown[i] > 0:
            cooldown[i] -= 1
            continue
        for atom in atoms:
            dist = np.linalg.norm(posiciones[i] - atom)
            if dist < 0.3:
                energia_cinetica = 0.5 * 9.1e-31 * (velocidades[i, 0] / 1e-6) ** 2
                energia_exc = pot_excitacion * 1.6e-19
                n = int(energia_cinetica // energia_exc)
                if n >= 1:
                    prob = min(1.0, (energia_cinetica / energia_exc - 1) * 0.2)
                    if np.random.rand() < prob:
                        energia_perdida = n * energia_exc
                        energia_restante = energia_cinetica - energia_perdida
                        velocidades[i, 0] = np.sqrt(2 * energia_restante / 9.1e-31) * 1e-6
                        cooldown[i] = 10
                break

    reiniciar = posiciones[:, 0] > ancho
    posiciones[reiniciar, 0] = 0
    posiciones[reiniciar, 1] = np.random.uniform(0.5, altura - 0.5, np.sum(reiniciar))
    velocidades[reiniciar, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

    # Acá dibujamos con matplotlib sin parpadeo
    ax.clear()
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_facecolor('#111122')
    ax.set_title("Simulación de electrones", color='white')
    ax.tick_params(colors='white')
    ax.scatter(atoms[:, 0], atoms[:, 1], c='orange', s=20) #label="Átomos"
    ax.scatter(posiciones[:, 0], posiciones[:, 1], c='cyan', s=10) #, label="Electrones"
    #ax.legend(facecolor='gray')
    
    canvas.pyplot(fig)
    time.sleep(0.03)
