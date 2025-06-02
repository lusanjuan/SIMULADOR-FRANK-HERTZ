import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# LO QUE SE VE EN LA PESTAÑA
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

# FONDO Y ESTILO
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

# TITULO
st.markdown("""
    <h1>Experimento de Franck y Hertz</h1>
""", unsafe_allow_html=True)

# CUERPO DEL TEXTO
st.markdown("""
Este simulador te permite explorar cómo los electrones colisionan con átomos a medida que aumenta el voltaje. 
Observa los picos de corriente cuando los electrones pierden energía por excitación atómica. 
""")

# SLIDERS
pot_excitacion = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 0.0, 0.1)
num_electrones = st.slider("Número de electrones", 5, 100, 20)

# CORRIENTE EN FUNCION DEL VOLTAJE Y DEL NUMERO DE ELECTRONES 
def corriente_simulada(V, e_exc, num_electrones):
    corriente = []
    for v in V:
        base = v ** 1.2
        residuo = v % e_exc
        caída = 0.4 if residuo < 0.3 else 1.0
        corriente.append(base * caída)
    return np.array(corriente) * num_electrones

# GRAFICO DE LA CORRIENTE EN FUNCION DEL VOLTAJE
voltajes = np.linspace(0, 50, 500)
corrientes = corriente_simulada(voltajes, pot_excitacion, num_electrones)
corriente_actual = corriente_simulada([voltaje_max], pot_excitacion, num_electrones)[0]
st.write(f" Corriente estimada para {voltaje_max:.1f} V: **{corriente_actual:.3f}** (unidades arbitrarias)")
corrientes_visibles = corriente_simulada(voltajes, pot_excitacion, num_electrones)
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
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6  # v = sqrt(2qV/m)

# Cooldown por electrón (para evitar múltiples colisiones seguidas)
cooldown = np.zeros(num_electrones)

# NUMERO Y POSICION DE LOS ATOMOS
num_atom = 70
pos_atoms_x = np.linspace(1, 13, num_atom)
pos_atoms_y = np.random.uniform(0, altura , num_atom)
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

    for i in range(num_electrones):
        if cooldown[i] > 0:
            cooldown[i] -= 1
            continue

        for atom in atoms:
            dist = np.linalg.norm(posiciones[i] - atom)
            if dist < 0.3:
                energia_cinetica = 0.5 * 9.1e-31 * (velocidades[i, 0] / 1e-6) ** 2
                energia_exc_julios = pot_excitacion * 1.6e-19
                n = int(energia_cinetica // energia_exc_julios)

                if n >= 1:
                    # Probabilidad simple de colisión inelástica
                    prob_colision = min(1.0, (energia_cinetica / energia_exc_julios - 1) * 0.2)
                    if np.random.rand() < prob_colision:
                        energia_perdida = n * energia_exc_julios
                        energia_restante = energia_cinetica - energia_perdida
                        nueva_velocidad = np.sqrt(2 * energia_restante / 9.1e-31) * 1e-6
                        velocidades[i, 0] = nueva_velocidad
                        cooldown[i] = 10
                break  # una sola colisión por frame

    # Reposicionar electrones que llegan al final
    reiniciar = posiciones[:, 0] > ancho
    posiciones[reiniciar, 0] = 0
    posiciones[reiniciar, 1] = np.random.uniform(0.5, altura - 0.5, np.sum(reiniciar))
    velocidades[reiniciar, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

    # Graficar
    fig, ax = plt.subplots(figsize=(13, 7)) #ANCHO Y ALTO DEL TUBO 
    tubo = patches.Rectangle((0, 0), ancho, altura, linewidth=1.5, edgecolor='white', facecolor='black')
    ax.add_patch(tubo)

#ATOMOS DE HG
    for atom in atoms:
        circulo = plt.Circle((atom[0], atom[1]), 0.10, color='orange')
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
