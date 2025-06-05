import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# ---------------- CONFIGURACIÓN DE PÁGINA Y ESTILO ----------------
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

st.markdown("""
    <style>
        .block-container { padding-left: 1rem; padding-right: 1rem; max-width: 80% !important; }
        .stApp { background-color: #1b2d40; }
        h1, h2, h3, label, .stMarkdown, .css-qrbaxs {
            color: white !important; font-family: 'Segoe UI', sans-serif;
        }
        .stButton > button {
            background-color: #0a141a; color: white !important; border: 1px solid #444;
            padding: 0.5em 1em; border-radius: 8px; font-weight: bold;
        }
        .stButton > button:hover { background-color: #444444; color: white !important; }
        .stSlider {
            background-color: #0a141a; padding: 1rem;
            border: 1px solid #9da9b0; border-radius: 10px; margin-bottom: 1rem;
        }
        .stSlider label { color: white !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------- TÍTULO Y DESCRIPCIÓN ----------------
st.markdown("<h1>Experimento de Franck y Hertz</h1>", unsafe_allow_html=True)
# ---------------- EXPLICACIÓN TEÓRICA ----------------
st.markdown("""
### 🔬 Fundamento físico del experimento

El experimento de Franck y Hertz demuestra que los electrones al colisionar con átomos (como los de mercurio) pueden perder energía en cantidades discretas, evidenciando que los niveles de energía en los átomos están cuantizados.

- Un **filamento caliente** (cátodo) emite electrones por **emisión termoiónica**, un proceso explicado por la ley de Richardson-Dushman. A mayor temperatura, más electrones adquieren suficiente energía térmica para escapar del metal.

- Los electrones son acelerados hacia una rejilla (ánodo) mediante un **voltaje acelerador**. Durante su recorrido, pueden colisionar con átomos de mercurio.

- Si la energía cinética de un electrón es igual o superior a la **energía de excitación** del átomo (por ejemplo, 4.9 eV para Hg), el electrón puede excitar al átomo y pierde esa energía.

- Después del ánodo, un pequeño **voltaje de frenado** puede impedir que los electrones lleguen al colector, afectando la corriente detectada.

Al aumentar el voltaje acelerador, se observan picos y valles en la corriente, lo cual refleja los momentos en los que los electrones pierden energía por colisiones inelásticas con los átomos.

---
""")

# ---------------- SLIDERS Y GRÁFICO ----------------
sliders_col, grafico_col = st.columns([1, 2])
with sliders_col:
    pot_excitacion = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
    voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 0.0, 0.1)
    voltaje_frenado = st.slider("Voltaje de frenado (V)", 0.0, 10.0, 0.0, 0.1)
    temp_filamento = st.slider("Temperatura del filamento (K)", 1000, 3000, 2000, 100)


# ---------------- EMISIÓN TERMOIÓNICA (SUAVIZADA) ----------------
T_min, T_max = 1000, 3000
p = 2.2  # Exponente ajustable para suavizar el crecimiento
T_norm = max(0, (temp_filamento - T_min) / (T_max - T_min))  # entre 0 y 1
num_electrones = int(800 * T_norm**p)
num_electrones = max(10, min(num_electrones, 800))


# ---------------- CORRIENTE SIMULADA ----------------
def corriente_simulada(V, e_exc, num_electrones, A=1.0, B=0.7, alpha=1.2, phi=0):
    V = np.array(V)
    corriente = A * V**alpha * (1 - B * np.sin(np.pi * V / e_exc + phi)**2)
    corriente[V <= 0] = 0
    return corriente * num_electrones

voltaje = np.linspace(0, 50, 500)
corriente_actual = corriente_simulada([voltaje_max], pot_excitacion, num_electrones)[0]
with sliders_col:
    st.markdown(f"Corriente estimada: **{corriente_actual:.2f} u.a.**")
    st.markdown(f"Electrones emitidos: **{num_electrones}**")
st.markdown("**Nota:** la corriente es una simulación cualitativa proporcional al número de electrones emitidos.")

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

# ---------------- SIMULACIÓN VISUAL ----------------
st.markdown("<h2>Simulación visual del experimento</h2>", unsafe_allow_html=True)
ancho, altura = 10, 5
dt = 0.1

posiciones = np.zeros((num_electrones, 2))
posiciones[:, 1] = np.random.uniform(0.5, altura - 0.5, num_electrones)
velocidades = np.zeros((num_electrones, 2))
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6
cooldown = np.zeros(num_electrones)

num_atom = 70
pos_atoms_x = np.linspace(1, 8, num_atom)
pos_atoms_y = np.random.uniform(0, altura, num_atom)
atoms = np.column_stack((pos_atoms_x, pos_atoms_y))

x_catodo, x_filamento, x_anodo, x_frenado = 0.2, 0.3, 8.0, 9.5

if "animando" not in st.session_state:
    st.session_state.animando = False

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✅ Comenzar animación"):
        st.session_state.animando = True
with btn_col2:
    if st.button("🛑 Detener animación"):
        st.session_state.animando = False

fig, ax = plt.subplots(figsize=(8, 4))
canvas = st.empty()

while st.session_state.animando:
    posiciones[:, 0] += velocidades[:, 0] * dt

    for i in range(num_electrones):
        if cooldown[i] > 0:
            cooldown[i] -= 1
            continue
        for atom in atoms:
            if np.linalg.norm(posiciones[i] - atom) < 0.3:
                ec = 0.5 * 9.1e-31 * (velocidades[i, 0] / 1e-6)**2
                ee = pot_excitacion * 1.6e-19
                if ec >= ee:
                    prob = min(1.0, (ec / ee - 1) * 0.2)
                    if np.random.rand() < prob:
                        velocidades[i, 0] = np.sqrt(2 * (ec - ee) / 9.1e-31) * 1e-6
                        cooldown[i] = 10
                break

    energia_final = 0.5 * 9.1e-31 * (velocidades[:, 0] / 1e-6)**2
    energia_umbral = voltaje_frenado * 1.6e-19
    bloqueados = energia_final < energia_umbral
    posiciones[bloqueados, 0] = np.minimum(posiciones[bloqueados, 0], x_frenado)
    velocidades[bloqueados, 0] = 0

    reiniciar = posiciones[:, 0] > ancho
    posiciones[reiniciar, 0] = 0
    posiciones[reiniciar, 1] = np.random.uniform(0.5, altura - 0.5, np.sum(reiniciar))
    velocidades[reiniciar, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

    ax.clear()
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_facecolor('#111122')
    ax.set_title("Simulación de electrones", color='white')
    ax.tick_params(colors='white')
    ax.add_patch(patches.Rectangle((x_catodo, 0), 0.05, altura, color='gray'))
    ax.plot([x_filamento]*2, [0.5, altura - 0.5], color='orange', linewidth=2)
    ax.add_patch(patches.Rectangle((x_anodo, 0), 0.05, altura, color='green'))
    ax.axvspan(x_anodo + 0.1, ancho, color='red', alpha=0.1)

    ax.scatter(atoms[:, 0], atoms[:, 1], c='orange', s=35)
    ax.scatter(posiciones[:, 0], posiciones[:, 1], c='cyan', s=8)
    canvas.pyplot(fig)
    time.sleep(0.03)
