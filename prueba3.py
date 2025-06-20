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
        .stApp { background-color: #0d1c2c; }                    /* azul marino */
        h1, h2, h3, label, .stMarkdown, .css-qrbaxs { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
        .stButton > button { background-color: #102237; color: #e8e8ea !important; border: 1px solid #445; padding: 0.5em 1em; border-radius: 8px; font-weight: bold; }
        .stButton > button:hover { background-color: #334a69; color: white !important; }
        .stSlider { background-color: #102237; padding: 1rem; border: 1px solid #445; border-radius: 10px; margin-bottom: 1rem; }
        .stSlider label { color: #ffffff !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ EXPLICACION TEORICA ------------------------------
st.markdown("""
<h1>Experimento de Franck y Hertz</h1>
<h3>🔬 Fundamento físico</h3>

<p>
El experimento de Franck y Hertz fue una de las primeras evidencias experimentales de que los átomos poseen <strong>niveles de energía cuantizados</strong>. Consiste en un tubo lleno de vapor de mercurio (Hg), dentro del cual se acelera un haz de electrones libres generados por <em>emisión termoiónica</em> desde un filamento caliente (cátodo). Esta emisión sigue la ley de Richardson-Dushman:
</p>

<p style="text-align:center">
J = A·T²·e<sup>−ϕ / kT</sup>
</p>

<p>
donde <em>J</em> es la densidad de corriente de electrones emitidos, <em>T</em> la temperatura del filamento, <em>ϕ</em> el trabajo de extracción, <em>k</em> la constante de Boltzmann y <em>A</em> una constante material.
</p>

<p>
Una vez emitidos, los electrones son acelerados por un voltaje <em>V<sub>acel</sub></em> hacia un ánodo. En su trayecto, pueden colisionar con átomos de mercurio. Si la energía cinética del electrón coincide con la diferencia entre niveles electrónicos permitidos del átomo, puede producirse una <strong>colisión inelástica</strong>, en la que el electrón pierde una cantidad fija de energía para excitar al átomo.
</p>

<p>
Las primeras excitaciones posibles corresponden a transiciones desde el estado fundamental a niveles superiores:
</p>
<ul>
  <li><strong>4,9 eV</strong> → excitación al primer estado excitado (<em>n = 2</em>)</li>
  <li><strong>6,7 eV</strong> → excitación al segundo estado excitado (<em>n = 3</em>)</li>
  <li>Y así sucesivamente, para niveles <em>n ≥ 4</em>, con energía creciente</li>
</ul>

<p>
Cada transición electrónica hacia un nivel superior es temporal: tras un breve intervalo, el átomo tiende a desexcitarse y emitir la energía sobrante en forma de <strong>fotón ultravioleta (UV)</strong>. En la simulación, estos fotones se visualizan como destellos breves.
</p>

<p>
Después del ánodo se encuentra una región de <strong>frenado</strong> (barrera de retención) controlada por un voltaje <em>V<sub>frenado</sub></em>. Esta zona impide que los electrones que no conservaron suficiente energía cinética lleguen al colector. Así, al aumentar <em>V<sub>acel</sub></em>, se observan aumentos y caídas periódicas en la corriente de colector, reflejando la pérdida de energía de los electrones al excitar a los átomos.
</p>

<p>
Este comportamiento ondulatorio en la curva corriente–voltaje confirma que la energía interna de los átomos solo puede variar en valores discretos, demostrando experimentalmente la <strong>cuantización de la energía</strong>.
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
    st.markdown(f"<span style='color:#00e6ff'>Flujo: {flujo_electrones} e⁻/frame</span>", unsafe_allow_html=True)

# ------------------------------ GRÁFICO I-V ------------------------------
def corriente_simulada(V, e_exc, esc, V_f):
    V = np.array(V)
    V_efectivo = np.maximum(0, V - V_f)
    A, B, alpha = 1.0, 0.7, 1.2
    I = A*V_efectivo**alpha * (1 - B*np.sin(np.pi*V_efectivo/e_exc)**2)
    I[V_efectivo<=0] = 0
    return I * esc

V_arr = np.linspace(0, 50, 500)
I_arr = corriente_simulada(V_arr, pot_excitacion, flujo_electrones, voltaje_frenado)

with grafico:
    fig_IV, ax_IV = plt.subplots()
    fig_IV.patch.set_facecolor('#0d1c2c')
    ax_IV.set_facecolor('#0d1c2c')
    ax_IV.plot(V_arr, I_arr, color="#00e6ff", linewidth=2)
    ax_IV.axvline(voltaje_max, color="#ff4d4d", linestyle="--", linewidth=1.2)
    ax_IV.grid(alpha=0.25, color="#445")
    ax_IV.tick_params(colors='white')
    for s in ax_IV.spines.values(): s.set_color('white')
    ax_IV.set_xlabel("Voltaje (V)", color='white')
    ax_IV.set_ylabel("Corriente ", color='white')
    ax_IV.set_title("Curva característica I-V", color='white')
    ax_IV.text(voltaje_max+0.5, I_arr[np.searchsorted(V_arr, voltaje_max)], f"{voltaje_max:.1f} V", color='white', size=6)
    st.pyplot(fig_IV, clear_figure=True)
# ---------- GEOMETRÍA ----------
ancho,altura=10,5
x_catodo,x_filamento,x_anodo,x_colector=0.5,0.2,8.0,10
escala=0.08;FACTOR_V=1e-6
x_catodo_m,x_filamento_m,x_anodo_m,x_colector_m=np.array([x_catodo,x_filamento,x_anodo,x_colector])*escala

# ---------- ÁTOMOS (fijos) ----------
atoms=np.column_stack([np.linspace(1,8,80),
                       np.random.uniform(0.5,altura-0.5,80)])
ENERG_N=[0.0,4.9,6.7];T_RELAX=[0,20,40];COL_N=['#ffaa00','#ff4444','#bb88ff']

# ---------- ESTADO PERSISTENTE ----------
if "pos" not in st.session_state:
    st.session_state.pos       = np.empty((0,2))
    st.session_state.vel       = np.empty((0,2))
    st.session_state.fase      = np.empty((0,))
    st.session_state.cooldown  = np.empty((0,))
    st.session_state.nivel_atom= np.zeros(len(atoms),int)
    st.session_state.relax_t   = np.zeros(len(atoms),int)
    st.session_state.phot_pos  = np.empty((0,2))
    st.session_state.phot_life = np.empty((0,))

pos       = st.session_state.pos
vel       = st.session_state.vel
fase      = st.session_state.fase
cooldown  = st.session_state.cooldown
nivel_atom= st.session_state.nivel_atom
relax_t   = st.session_state.relax_t
phot_pos  = st.session_state.phot_pos
phot_life = st.session_state.phot_life

# ---------- BOTONES ----------
if "animando" not in st.session_state: st.session_state.animando=False
col1,col2=st.columns(2)
if col1.button("▶️ Comenzar"): st.session_state.animando=True
if col2.button("⏹️ Detener"):  st.session_state.animando=False

# ---------- FIGURA ----------
fig,ax=plt.subplots(figsize=(8,4));fig.patch.set_facecolor('#0d1c2c')
canvas=st.empty()
dt,q,m=0.07,1.6e-19,9.1e-31
RCOL,P_INEL=0.15,0.15
# ------------- PARÁMETROS -------------
PH_SPAN  = 15      # vida en frames
PH_SPEED = 0.6     # desplazamiento vertical
PH_SIZE  = 6       # tamaño base

while st.session_state.animando:
    n = flujo_electrones
    pos = np.vstack([pos, np.column_stack([
        np.full(n, x_filamento),
        np.random.uniform(0.5, altura - 0.5, n)])])
    v0 = np.sqrt(2 * q * voltaje_max / m) * FACTOR_V
    vel = np.vstack([vel, np.column_stack([
        np.full(n, v0), np.zeros(n)])])
    fase = np.concatenate([fase, np.random.uniform(0, 2 * np.pi, n)])
    cooldown = np.concatenate([cooldown, np.zeros(n)])

    pos[:, 0] += vel[:, 0] * dt
    pos[:, 1] += 0.04 * np.sin(4 * pos[:, 0] + fase)

    for i in range(len(pos)):
        if cooldown[i] > 0:
            cooldown[i] -= 1
            continue
        for j, at in enumerate(atoms):
            if np.linalg.norm(pos[i] - at) < RCOL:
                Ec = 0.5 * m * (vel[i, 0] / FACTOR_V)**2
                lvl = nivel_atom[j]
                if lvl < 2:
                    dE = (ENERG_N[lvl + 1] - ENERG_N[lvl]) * q
                    p_eff = P_INEL * (1 - dE / Ec)
                    p_eff = max(0, min(p_eff, 1))
                    if Ec >= dE and np.random.rand() < p_eff:
                        vel[i, 0] = np.sqrt(max(0, 2 * (Ec - dE) / m)) * FACTOR_V
                        nivel_atom[j] += 1
                        relax_t[j] = T_RELAX[nivel_atom[j]]
                        cooldown[i] = 10
                break

    relax_t = np.maximum(relax_t - 1, 0)
    for j in np.where(relax_t == 0)[0]:
        if nivel_atom[j] > 0:
            phot_pos = np.vstack([phot_pos, atoms[j]])
            phot_life = np.append(phot_life, PH_SPAN)
            nivel_atom[j] -= 1
            relax_t[j] = T_RELAX[nivel_atom[j]]

    dist_m = (x_colector_m - x_anodo_m)
    if dist_m > 0:
        dv = (-q * voltaje_frenado / m) / dist_m * FACTOR_V**2 * dt * 0.5
        zona = (pos[:, 0] >= x_anodo) & (pos[:, 0] <= x_colector)
        vel[zona, 0] += dv

    if phot_pos.size:
        phot_pos[:, 1] += PH_SPEED
        phot_life -= 1
        keep = phot_life > 0
        phot_pos, phot_life = phot_pos[keep], phot_life[keep]

    keep = (pos[:, 0] >= 0) & (pos[:, 0] <= ancho)
    pos, vel, fase, cooldown = pos[keep], vel[keep], fase[keep], cooldown[keep]

    ax.clear()
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_facecolor('#0d1c2c')
    ax.axis('off')

    ax.add_patch(patches.Rectangle((x_catodo, 0), 0.05, altura, color='#b0b0b0'))
    t = np.linspace(0, 1, 200)
    ysp = 0.8 + (altura - 1.6) * t
    xsp = x_filamento + 0.15 * np.sin(2 * np.pi * 15 * t)
    ax.plot(xsp, ysp, color='#ffa64d', lw=2, solid_capstyle='round')
    ax.plot([x_anodo + 0.025, x_anodo + 0.025], [0, altura],
            color='#2ecc71', linewidth=2, linestyle=':')
    ax.axvspan(x_anodo + 0.1, ancho, color='#ff4757', alpha=0.18)

    ax.scatter(atoms[:, 0], atoms[:, 1],
               c=[COL_N[k] for k in nivel_atom],
               s=60, edgecolors='white', lw=0.3, alpha=0.9)
    ax.scatter(pos[:, 0], pos[:, 1], c='#ff9cbb', s=6, edgecolors='none')
    if phot_pos.size:
        alpha = phot_life / PH_SPAN
        sizes = PH_SIZE * (0.8 + alpha)
        ax.scatter(phot_pos[:, 0], phot_pos[:, 1],
                   c='#ffffff', s=sizes,
                   alpha=alpha, edgecolors='none')

    canvas.pyplot(fig)
    time.sleep(0.05)

    st.session_state.pos = pos
    st.session_state.vel = vel
    st.session_state.fase = fase
    st.session_state.cooldown = cooldown
    st.session_state.nivel_atom = nivel_atom
    st.session_state.relax_t = relax_t
    st.session_state.phot_pos = phot_pos
    st.session_state.phot_life = phot_life


# ---------- LEYENDA ----------
st.markdown("""
<span style='color:#ffaa00'>●</span> fundamental (n=1)&nbsp;&nbsp;
<span style='color:#ff4444'>●</span> n=2&nbsp;&nbsp;
<span style='color:#bb88ff'>●</span> n=3&nbsp;&nbsp;
<span style='color:#ffffff'>●</span> fotón emitido cuando el átomo se desexcita&nbsp;&nbsp;  
<span style='color:#ffa64d'>●</span> filamento&nbsp;&nbsp; 
<span style='color:#b0b0b0'>●</span> cátodo&nbsp;&nbsp; 
<span style='color:#2ecc71'>●</span> ánodo&nbsp;&nbsp;   
""",unsafe_allow_html=True)


