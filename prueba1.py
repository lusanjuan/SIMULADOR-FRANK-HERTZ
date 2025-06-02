import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Configuración inicial de la página
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

# Fondo y estilo
st.markdown("""
    <style>
        .stApp {
            background-color: #050314;
        }
        h1 {
            color: #1E90FF;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Título
st.markdown("""
    <h1>Experimento de Franck y Hertz</h1>
""", unsafe_allow_html=True)

# Descripción
st.markdown("""
Este simulador te permite explorar cómo los electrones colisionan con átomos a medida que aumenta el voltaje. 
Observa los picos de corriente cuando los electrones pierden energía por excitación atómica. 
""")

# Parámetros ajustables
e_exc = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 0.0, 0.1)
num_electrones = st.slider("Número de electrones", 5, 100, 20)

# Función de corriente simulada

def corriente_simulada(V, e_exc):
    corriente = []
    for v in V:
        base = v ** 1.2
        residuo = v % e_exc
        if residuo < 0.3:
            caida = 0.4
        else:
            caida = 1.0
        corriente.append(base * caida)
    return np.array(corriente)

# Gráfico de corriente vs voltaje
voltajes = np.linspace(0, 50, 500)
corrientes = corriente_simulada(voltajes, e_exc)
corriente_actual = corriente_simulada([voltaje_max], e_exc)[0]
st.write(f" Corriente estimada para {voltaje_max:.1f} V: **{corriente_actual:.3f}** (unidades arbitrarias)")

corrientes_visibles = corriente_simulada(voltajes, e_exc)
corrientes_visibles[voltajes > voltaje_max] = np.nan

fig, ax = plt.subplots()
ax.plot(voltajes, corrientes_visibles, color="blue", label="Corriente simulada hasta voltaje actual")
ax.axvline(voltaje_max, color="red", linestyle="--", label=f"Voltaje actual: {voltaje_max:.1f} V")
ax.set_xlabel("Voltaje (V)")
ax.set_ylabel("Corriente (u.a.)")
ax.set_title("Corriente vs Voltaje")
ax.legend()
st.pyplot(fig)

# ---------- Simulación Visual -----------
st.title("Simulación visual del experimento de Franck y Hertz")
ancho = 10
altura = 5
dt = 0.1

# Posiciones y velocidades iniciales
np.random.seed(1)
posiciones = np.zeros((num_electrones, 2))
posiciones[:, 1] = np.random.uniform(0.5, altura - 0.5, num_electrones)
velocidades = np.zeros((num_electrones, 2))
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

# Posiciones de átomos
num_atom = 10
pos_atoms_x = np.linspace(2, 8, num_atom)
pos_atoms_y = np.random.uniform(1, altura - 1, num_atom)
atoms = np.column_stack((pos_atoms_x, pos_atoms_y))

# Controles de animación
if "animando" not in st.session_state:
    st.session_state.animando = False

col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Comenzar animación"):
        st.session_state.animando = True
with col2:
    if st.button("🛑 Detener animación"):
        st.session_state.animando = False

# Contenedor gráfico
grafico = st.empty()

while st.session_state.animando:
    posiciones[:, 0] += velocidades[:, 0] * dt

    # Colisiones con átomos
    for i in range(num_electrones):
        for atom in atoms:
            dist = np.linalg.norm(posiciones[i] - atom)
            if dist < 0.3:
                velocidades[i, 0] *= 0.5  # pierde velocidad

    # Reposicionar electrones que llegan al final
    reiniciar = posiciones[:, 0] > ancho
    posiciones[reiniciar, 0] = 0
    posiciones[reiniciar, 1] = np.random.uniform(0.5, altura - 0.5, np.sum(reiniciar))
    velocidades[reiniciar, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 4))
    tubo = patches.Rectangle((0, 0), ancho, altura, linewidth=1.5, edgecolor='white', facecolor='black')
    ax.add_patch(tubo)

    for atom in atoms:
        circulo = plt.Circle((atom[0], atom[1]), 0.15, color='orange')
        ax.add_patch(circulo)

    ax.scatter(posiciones[:, 0], posiciones[:, 1], color='cyan', label='Electrones')
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_facecolor('#111122')
    ax.set_title("Movimiento de electrones en el tubo", color='white')
    ax.set_xlabel("Distancia (cm)", color='white')
    ax.set_ylabel("Altura (cm)", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#222244', edgecolor='white')

    grafico.pyplot(fig)
    time.sleep(0.1)