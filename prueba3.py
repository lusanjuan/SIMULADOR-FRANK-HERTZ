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
        h1, h2, h3, label, .stMarkdown, .css-qrbaxs { color: white !important; font-family: 'Segoe UI', sans-serif; }
        .stButton > button { background-color: #0a141a; color: white !important; border: 1px solid #444; padding: 0.5em 1em; border-radius: 8px; font-weight: bold; }
        .stButton > button:hover { background-color: #444444; color: white !important; }
        .stSlider { background-color: #0a141a; padding: 1rem; border: 1px solid #9da9b0; border-radius: 10px; margin-bottom: 1rem; }
        .stSlider label { color: white !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ EXPLICACION TEORICA ------------------------------
st.markdown("""
<h1>Experimento de Franck y Hertz</h1>
<h3>🔬 Fundamento físico</h3>

<p>
El experimento de Franck y Hertz demuestra que los electrones al colisionar con átomos (como los de mercurio) pueden perder energía en cantidades discretas, evidenciando que los niveles de energía en los átomos están cuantizados.
<br><br>
Un filamento caliente (cátodo) emite electrones por emisión termoiónica, un proceso explicado por la ley de Richardson-Dushman. A mayor temperatura, más electrones adquieren suficiente energía térmica para escapar del metal.
<br><br>
Los electrones son acelerados hacia una rejilla (ánodo) mediante un voltaje acelerador. Durante su recorrido, pueden colisionar con átomos de mercurio.
<br><br>
Si la energía cinética de un electrón es igual o superior a la energía de excitación del átomo (por ejemplo, 4.9 eV para Hg), el electrón puede excitar al átomo y pierde esa energía.
<br><br>
Después del ánodo, un pequeño voltaje de frenado puede impedir que los electrones lleguen al colector, afectando la corriente detectada.
<br><br>
Al aumentar el voltaje acelerador, se observan picos y valles en la corriente, lo cual refleja los momentos en los que los electrones pierden energía por colisiones inelásticas con los átomos.
</p>

<p>
El filamento caliente (cátodo) emite electrones por <strong>emisión termoiónica</strong>, descrita por la ley de Richardson-Dushman:
</p>
<p style="text-align:center">
J = A T² e^(−ϕ / kT)
</p>
<p>
donde <em>T</em> es la temperatura, <em>ϕ</em> el trabajo de función y <em>A</em> una constante. A mayor T, mayor flujo de electrones.
</p>
""", unsafe_allow_html=True)

# ------------------------------ SLIDERS ------------------------------
sliders, grafico = st.columns([1, 2])
with sliders:
    pot_excitacion  = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
    voltaje_max     = st.slider("Voltaje de aceleración (V)",   0.0, 50.0, 8.0, 0.1)
    voltaje_frenado = st.slider("Voltaje de frenado (V)",       0.0, 10.0, 1.0, 0.1)
    temp_filamento  = st.slider("Temperatura del filamento (K)", 1000, 3000, 2000, 100)

# ------------------------------ FLUJO DE ELECTRONES ------------------------------
phi, kB, A = 4.5, 8.617e-5, 1.0e6
J = A * temp_filamento**2 * np.exp(-phi / (kB * temp_filamento))
flujo_electrones = max(1, int(J*1e-4))
with sliders:
    st.markdown(f"Flujo de electrones: **{flujo_electrones} e⁻/frame**")

# ------------------------------ GRÁFICO I-V ------------------------------
def corriente_simulada(V, e_exc, esc):
    V = np.array(V)
    A, B, α = 1.0, 0.7, 1.2
    I = A*V**α * (1 - B*np.sin(np.pi*V/e_exc)**2)
    I[V<=0] = 0
    return I*esc

V_arr = np.linspace(0, 50, 500)
I_arr = corriente_simulada(V_arr, pot_excitacion, flujo_electrones)
I_arr[V_arr > voltaje_max] = np.nan
with grafico:
    fig_IV, ax_IV = plt.subplots()
    fig_IV.patch.set_facecolor('#0e1b28')
    ax_IV.plot(V_arr, I_arr, color="cyan")
    ax_IV.axvline(voltaje_max, color="red", ls="--")
    ax_IV.tick_params(colors='white')
    for s in ax_IV.spines.values(): s.set_color('white')
    ax_IV.set_xlabel("Voltaje (V)", color='white')
    ax_IV.set_ylabel("Corriente (u.a.)", color='white')
    ax_IV.set_title("Corriente vs Voltaje", color='white')
    st.pyplot(fig_IV)

# ------------------------------ SIMULACIÓN VISUAL ------------------------------
st.markdown("## Simulación visual")
ancho, altura = 10, 5
x_catodo, x_filamento, x_anodo, x_colector = 0.3, 0.2, 8.0, 10
escala = 0.08
FACTOR_V = 1e-6
x_catodo_m, x_filamento_m, x_anodo_m, x_colector_m = \
    np.array([x_catodo, x_filamento, x_anodo, x_colector]) * escala

if "animando" not in st.session_state: st.session_state.animando = False
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Comenzar"): st.session_state.animando = True
with col2:
    if st.button("⏹️ Detener"): st.session_state.animando = False

pos, vel = np.empty((0,2)), np.empty((0,2))
fase, cooldown = np.empty((0,)), np.empty((0,))
atoms_x = np.linspace(1, 8, 80)
atoms_y = np.random.uniform(0.5, altura-0.5, 80)
atoms = np.column_stack((atoms_x, atoms_y))

# ----- NIVELES Hg: 0 eV, 4.9 eV, 6.7 eV -----
ENERG_N = [0.0, 4.9, 6.7]      # eV absolutos
T_RELAX = [  0,   60, 120]     # frames
COL_N   = ['#ffaa00', '#ff4444', '#d48bff']
nivel_atom = np.zeros(len(atoms), int)
relax_t    = np.zeros(len(atoms), int)

fig, ax = plt.subplots(figsize=(8,4))
fig.patch.set_facecolor('#0e1b28')
canvas = st.empty()
dt, q, m = 0.07, 1.6e-19, 9.1e-31
RCOL, P_INEL = 0.22, 0.12

while st.session_state.animando:
    # Emisión
    n = flujo_electrones
    pos = np.vstack([pos,
        np.column_stack([np.full(n, x_filamento),
                         np.random.uniform(0.5, altura-0.5, n)])])
    v0 = np.sqrt(2*q*voltaje_max/m)*FACTOR_V
    vel = np.vstack([vel, np.column_stack([np.full(n, v0), np.zeros(n)])])
    fase = np.concatenate([fase, np.random.uniform(0, 2*np.pi, n)])
    cooldown = np.concatenate([cooldown, np.zeros(n)])

    # Movimiento base
    pos[:,0] += vel[:,0]*dt
    pos[:,1] += 0.05*np.sin(4*pos[:,0]+fase)

    # Colisiones (energía incremental)
    for i in range(len(pos)):
        if cooldown[i]>0: cooldown[i]-=1; continue
        for j, at in enumerate(atoms):
            if np.linalg.norm(pos[i]-at) < RCOL:
                Ec = 0.5*m*(vel[i,0]/FACTOR_V)**2
                lvl = nivel_atom[j]
                if lvl < 2:
                    deltaE = (ENERG_N[lvl+1] - ENERG_N[lvl]) * q
                    if Ec >= deltaE and np.random.rand() < P_INEL:
                        vel[i,0] = np.sqrt(max(0, 2*(Ec-deltaE)/m))*FACTOR_V
                        nivel_atom[j] += 1
                        relax_t[j] = T_RELAX[nivel_atom[j]]
                        cooldown[i] = 10
                break

    # Relajación de átomos
    relax_t = np.maximum(relax_t-1, 0)
    for j in np.where(relax_t==0)[0]:
        if nivel_atom[j]>0:
            nivel_atom[j]-=1; relax_t[j]=T_RELAX[nivel_atom[j]]

    # Zona de frenado (sin retroceso)
    dist_m = (x_colector_m - x_anodo_m)
    if dist_m>0:
        dv = (-q*voltaje_frenado/m) / dist_m * FACTOR_V**2 * dt
        zona = (pos[:,0]>=x_anodo)&(pos[:,0]<=x_colector)
        rev = zona & (vel[:,0]+dv <= 0)
        vel[rev,0]=0; vel[zona & ~rev,0] += dv

    # Limpiar fuera de la pantalla
    keep = pos[:,0]<=ancho
    pos, vel, fase, cooldown = pos[keep], vel[keep], fase[keep], cooldown[keep]

    # Dibujo
    ax.clear()
    ax.set_xlim(0, ancho); ax.set_ylim(0, altura)
    ax.set_facecolor('#0e1b28'); ax.axis('off')
    ax.add_patch(patches.Rectangle((x_catodo,0),0.05,altura,color='gray'))
    ax.plot([x_filamento]*2,[0.8,altura-0.8], color='orange', lw=3)
    ax.add_patch(patches.Rectangle((x_anodo,0),0.05,altura,color='green'))
    ax.axvspan(x_anodo+0.1, ancho, color='red', alpha=0.15)
    ax.scatter(atoms[:,0], atoms[:,1],
               c=[COL_N[k] for k in nivel_atom], s=60,
               edgecolors='black', linewidths=0.5)
    ax.scatter(pos[:,0], pos[:,1],
               c='#ff9cbb', s=10, edgecolors='white', linewidths=0.2)
    canvas.pyplot(fig); time.sleep(0.03)


# ---------------- LEYENDA ----------------
st.markdown("""
**Átomos:** 🟠 fundamental · 🔴 1ª excitación · 🟣 2ª excitación  
**Electrones:** puntos rosa claro
""")