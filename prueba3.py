import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

# ---------------- CONFIGURACIÓN DE PÁGINA Y ESTILO ----------------
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

#------------------------------EXPLICACION TEORICA------------------------------
st.markdown("""
<h1>Experimento de Franck y Hertz</h1>
<h3>🔬 Fundamento físico</h3>

<p>
El experimento de Franck y Hertz demuestra que los electrones al colisionar con átomos (como los de mercurio) pueden perder energía en cantidades discretas, evidenciando que los niveles de energía en los átomos están cuantizados.

Un filamento caliente (cátodo) emite electrones por emisión termoiónica, un proceso explicado por la ley de Richardson-Dushman. A mayor temperatura, más electrones adquieren suficiente energía térmica para escapar del metal.

Los electrones son acelerados hacia una rejilla (ánodo) mediante un voltaje acelerador. Durante su recorrido, pueden colisionar con átomos de mercurio.

Si la energía cinética de un electrón es igual o superior a la energía de excitación del átomo (por ejemplo, 4.9 eV para Hg), el electrón puede excitar al átomo y pierde esa energía.

Después del ánodo, un pequeño voltaje de frenado puede impedir que los electrones lleguen al colector, afectando la corriente detectada.

Al aumentar el voltaje acelerador, se observan picos y valles en la corriente, lo cual refleja los momentos en los que los electrones pierden energía por colisiones inelásticas con los átomos.

</p>

<p>
El filamento caliente (cátodo) emite electrones por <strong>emisión termoiónica</strong>, descrita por la ley de Richardson-Dushman:
</p>
<p style="text-align: center">
J = A T² e^(−ϕ / kT)
</p>
<p>
Donde <em>T</em> es la temperatura, <em>ϕ</em> el trabajo de función, y <em>A</em> una constante. A mayor T, mayor flujo de electrones.
</p>
""", unsafe_allow_html=True)

# ------------------------------SLIDERS-------------------------------
sliders, grafico = st.columns([1, 2])
with sliders:
    pot_excitacion = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
    voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 8.0, 0.1)
    voltaje_frenado = st.slider("Voltaje de frenado (V)", 0.0, 10.0, 1.0, 0.1)
    temp_filamento = st.slider("Temperatura del filamento (K)", 1000, 3000, 2000, 100)

# ------------------------------CÁLCULO DEL FLUJO DE ELECTRONES-------------------------------
phi = 4.5  # eV
k = 8.617e-5  # eV/K
A = 1.0e6  # valor arbitrario
J = A * temp_filamento**2 * np.exp(-phi / (k * temp_filamento))
flujo_electrones = int(J * 1e-7)  # Escalado
flujo_electrones = max(1, min(flujo_electrones, 100))

# Mostrar
with sliders:
    st.markdown(f"Flujo de electrones: **{flujo_electrones} e⁻/frame**")

# ------------------------------GRÁFICO DE CORRIENTE SIMULADA-------------------------------
def corriente_simulada(V, e_exc, escala):
    V = np.array(V)
    A, B, alpha = 1.0, 0.7, 1.2
    corriente = A * V**alpha * (1 - B * np.sin(np.pi * V / e_exc)**2)
    corriente[V <= 0] = 0
    return corriente * escala

voltaje = np.linspace(0, 50, 500)
corriente = corriente_simulada(voltaje, pot_excitacion, flujo_electrones)
corriente[voltaje > voltaje_max] = np.nan

with grafico:
    fig, ax = plt.subplots()
    ax.plot(voltaje, corriente, color="cyan")
    ax.axvline(voltaje_max, color="red", linestyle="--")
    ax.set_xlabel("Voltaje (V)")
    ax.set_ylabel("Corriente (u.a.)")
    ax.set_title("Corriente vs Voltaje")
    st.pyplot(fig)

#  ------------------------------SIMULACIÓN VISUAL-------------------------------
st.markdown("## Simulación visual")
ancho, altura = 10, 5
x_catodo, x_filamento, x_anodo, x_colector = 0.3, 0.2, 8.0, 10 

if "animando" not in st.session_state:
    st.session_state.animando = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Comenzar"):
        st.session_state.animando = True
with col2:
    if st.button("⏹️ Detener"):
        st.session_state.animando = False

pos = np.empty((0, 2))        # Posiciones de electrones
vel = np.empty((0, 2))        # Velocidades de electrones
fase = np.empty((0,))         # Fase ondulatoria de cada electrón (para que no se muevan todos igual en y)
cooldown = np.empty((0,))     # Tiempo de espera tras colisión para evitar múltiples colisiones seguidas


atoms_x = np.linspace(1, 8, 80) #80 posiciones uniformemente distribuidas entre x = 1 y x = 8 (dentro del tubo).
atoms_y = np.random.uniform(0.5, altura - 0.5, 80) # 80 posiciones aleatorias verticales entre y = 0.5 y altura - 0.5, para no pegarlos a los bordes.
atoms = np.column_stack((atoms_x, atoms_y)) # ombina esas dos listas en coordenadas [x, y], una por átomo → tendrás 80 átomos dispersos.

fig, ax = plt.subplots(figsize=(8, 4))
canvas = st.empty()
dt = 0.05 #paso del tiempo arbitrario, puedes ajustarlo para que la animación vaya más rápida o más lenta.

while st.session_state.animando:
    # Emisión continua
    nuevos = flujo_electrones
    nuevas_pos = np.zeros((nuevos, 2))
    nuevas_pos[:, 0] = x_filamento
    nuevas_pos[:, 1] = np.random.uniform(0.5, altura - 0.5, nuevos)

    nuevas_vel = np.zeros((nuevos, 2))
    nuevas_vel[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6
    nuevas_fase = np.random.uniform(0, 2*np.pi, nuevos)
    nuevas_cd = np.zeros(nuevos)

    pos = np.vstack([pos, nuevas_pos])
    vel = np.vstack([vel, nuevas_vel])
    fase = np.concatenate([fase, nuevas_fase])
    cooldown = np.concatenate([cooldown, nuevas_cd])

    # Movimiento
    pos[:, 0] += vel[:, 0] * dt
    pos[:, 1] += 0.05 * np.sin(4 * pos[:, 0] + fase)

    # ------------------ Colisiones con átomos -----------------
    for i in range(len(pos)): # iterar sobre cada electrón
        if cooldown[i] > 0: # Si está en cooldown, no colisiona
            cooldown[i] -= 1
            continue
        for atom in atoms: # iterar sobre cada átomo
            if np.linalg.norm(pos[i] - atom) < 0.3: # si la distancia al átomo es menor a 0.3
                ec = 0.5 * 9.1e-31 * (vel[i, 0] / 1e-6)**2 # Energía cinética del electrón
                ee = pot_excitacion * 1.6e-19 # Energía de excitación del átomo
                if ec >= ee: # Si la energía cinética es suficiente para excitar el átomo
                    prob = min(1.0, (ec / ee - 1) * 0.2) # Probabilidad de excitación 
                    if np.random.rand() < prob:     # Si se excita el átomo
                        vel[i, 0] = np.sqrt(2 * (ec - ee) / 9.1e-31) * 1e-6 # Actualiza la velocidad del electrón
                        cooldown[i] = 10
                break


    # Definir factores de conversión
    escala = 0.1        # 1 unidad = 0.1 metros
    factor_v = 1e-6     # La velocidad simulada = velocidad física (m/s) * factor_v

    # Parámetros físicos
    q = 1.6e-19  # Carga del electrón (C)
    m = 9.1e-31  # Masa del electrón (kg)

    # Al inicializar la velocidad de los electrones:
    nuevas_vel[:, 0] = np.sqrt(2 * q * voltaje_max / m) * factor_v

    # … (más adelante, en el loop de simulación) …

    # Campo eléctrico uniformente entre ánodo y colector:
    dist_frenado = (x_colector - x_anodo) * escala  # distancia en metros
    if dist_frenado > 0:
        E_frenado = voltaje_frenado / dist_frenado    # V/m
        a_frenado = -q * E_frenado / m                 # aceleración en m/s²

        # Convertir la aceleración a unidades de simulación:
        # dt (segundos) se multiplica por la aceleración física; luego se convierte a unidades de simulación con factor_v
        a_frenado_sim = a_frenado * dt * factor_v

        # Definir la zona de frenado (entre x_anodo y x_colector en unidades de simulación)
        zona_frenado = (pos[:, 0] >= x_anodo) & (pos[:, 0] <= x_colector)

        # Aplicar la desaceleración sobre la velocidad (en unidades de simulación/s)
        vel[zona_frenado, 0] += a_frenado_sim

        # Calcular la energía cinética: convertir la velocidad simulada a m/s
        energia = 0.5 * m * (vel[:, 0] / factor_v)**2  # Ahora vel/factor_v es la velocidad física en m/s
        umbral = voltaje_frenado * q                    # Umbral en Joules

        # Forzar cero velocidad en caso de que, por errores numéricos, se tengan energías negativas
        vel[vel[:, 0] < 0, 0] = 0
        detener = zona_frenado & (energia < umbral)
        vel[detener, 0] = 0

    # Eliminar fuera de rango
    dentro = (pos[:, 0] <= ancho)
    pos = pos[dentro]
    vel = vel[dentro]
    fase = fase[dentro]
    cooldown = cooldown[dentro]

    # -----------------------------DIBUJO-----------------------------
    ax.clear()
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_facecolor('#0e1b28')
    ax.tick_params(colors='white', left=False, bottom=False, labelleft=False, labelbottom=False) # Ocultar ejes
    ax.set_title("Simulación de electrones", color='white') 
    ax.add_patch(patches.Rectangle((x_catodo, 0), 0.05, altura, color='gray')) # Cátodo
    ax.plot([x_filamento]*2, [0.8, altura - 0.8], color='orange', linewidth=3) # Filamento
    ax.add_patch(patches.Rectangle((x_anodo, 0), 0.05, altura, color='green')) # Ánodo
    ax.axvspan(x_anodo + 0.1, ancho, color='red', alpha=0.1) # Zona de frenado
    ax.scatter(atoms[:, 0], atoms[:, 1], c='#ffaa00', s=60, edgecolors='black', linewidths=0.5) 
    ax.scatter(pos[:, 0], pos[:, 1], c='#00ffff', s=10)
    canvas.pyplot(fig)
    time.sleep(0.03)
