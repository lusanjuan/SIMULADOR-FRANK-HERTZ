import streamlit as st, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches, itertools, io, time

# ───── CONFIGURACIÓN GLOBAL ─────
SUB_STEPS, DRAW_EVERY = 2, 1
MAX_E, PH_MAX         = 4000, 600
FACTOR_V, P_INEL      = 1e-6, 0.15
PH_SPAN, PH_SPEED, PH_SIZE = 15, 0.6, 6
DT                    = 0.07

# ───────── SET-UP STREAMLIT ─────────
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

st.markdown("""
<style>
  .block-container{padding-left:1rem;padding-right:1rem;max-width:80%!important}
  .stApp{background:#0d1c2c}
  h1,h2,h3,label,.stMarkdown,.css-qrbaxs{color:#fff!important;font-family:"Segoe UI",sans-serif}
  .stButton>button{background:#102237;color:#e8e8ea!important;border:1px solid #445;
                   padding:.5em 1em;border-radius:8px;font-weight:bold}
  .stButton>button:hover{background:#334a69;color:#fff!important}
  .stSlider{background:#102237;padding:1rem;border:1px solid #445;border-radius:10px;margin-bottom:1rem}
  .stSlider label{color:#fff!important;font-weight:bold}
</style>""", unsafe_allow_html=True)

# ───────── TEXTO INTRO ─────────
st.markdown("""
<h1>Experimento de Franck y Hertz</h1>
<h3>🔬 Fundamento físico</h3>
<p>El experimento de Franck&nbsp;y&nbsp;Hertz demuestra la <strong>cuantización</strong>
de los niveles de energía atómicos.</p>""", unsafe_allow_html=True)

# ───── CONTROLES ─────
sl, gr = st.columns([1,2])
with sl:
    exc   = st.slider("Potencial de excitación (eV)", 0.1,20.0,4.9,0.1)
    Vacc  = st.slider("Voltaje de aceleración (V)",    0.0,50.0,8.0,0.1)
    Vfilt = st.slider("Voltaje de frenado (V)",        0.0,10.0,1.0,0.1)
    Tfil  = st.slider("Temperatura filamento (K)", 1000,3000,2000,100)

phi,kB,A_R = 4.5,8.617e-5,1e6
J = A_R*Tfil**2*np.exp(-phi/(kB*Tfil))
flujo = max(1,int(J*1e-5))
st.markdown(f"<span style='color:#00e6ff'>Flujo: {flujo} e⁻/frame</span>",
            unsafe_allow_html=True)

# ───── CURVA I-V ─────
def I_sim(V,e_exc,scale):
    A0,B,a=1,0.7,1.2
    I=A0*V**a*(1-B*np.sin(np.pi*V/e_exc)**2); I[V<=0]=0; return I*scale
Varr=np.linspace(0,50,500); Iarr=I_sim(Varr,exc,flujo)
fig_iv,ax_iv=plt.subplots(); fig_iv.patch.set_facecolor('#0d1c2c')
ax_iv.set_facecolor('#0d1c2c'); ax_iv.plot(Varr,Iarr,c="#00e6ff")
ax_iv.axvline(Vacc,c="#ff4d4d",ls="--"); [s.set_color("w") for s in ax_iv.spines.values()]
ax_iv.tick_params(colors="w"); ax_iv.set_xlim(0,50)
gr.pyplot(fig_iv, clear_figure=True)

# ─────  SIMULATION STATE ─────
state = st.session_state
if "run" not in state:              # flags & dynamic data
    state.update(
        run=False, frame=0,
        pos=np.empty((0,2)), vel=np.empty((0,2)), fase=np.empty((0,)),
        cd=np.empty((0,)), nivel=np.zeros(80,int), relax=np.zeros(80,int),
        ppos=np.empty((0,2)), pprev=np.empty((0,2)), plife=np.empty((0,))
    )

c1,c2 = st.columns(2)
if c1.button("▶️ Comenzar"): state.run=True
if c2.button("⏹️ Detener"):  state.run=False

# ───── GEOMETRÍA & FIGURA ─────
ancho,alto=10,5
x_cat,x_fil,x_an,x_col=0.5,0.2,8.0,10
escala=0.08
atoms=np.column_stack([np.linspace(1,8,80),
                       np.random.uniform(0.5,alto-0.5,80)])
ENERG,TREL = [0,4.9,6.7],[0,20,40]
COL        = ['#ffaa00','#ff4444','#bb88ff']

if "fig_main" not in state:
    fig,ax=plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor('#0d1c2c'); ax.set_facecolor('#0d1c2c')
    ax.set_xlim(0,ancho); ax.set_ylim(0,alto); ax.axis("off")
    ax.add_patch(patches.Rectangle((x_cat,0),0.05,alto,color="#b0b0b0"))
    t=np.linspace(0,1,400)
    ax.plot(x_fil+0.15*np.sin(2*np.pi*15*t), 0.8+(alto-1.6)*t,
            color="#ffa64d", lw=2)
    ax.plot([x_an+0.025]*2,[0,alto],color="#2ecc71",lw=2,ls=":")
    ax.axvspan(x_an+0.1,ancho,color="#ff4757",alpha=0.18)

    art=dict(
        scat_atoms=ax.scatter([],[],s=70,ec='w',lw=.4),
        scat_e   =ax.scatter([],[],s=8 ,c='#ff9cbb',ec='none'),
        scat_p   =ax.scatter([],[],s=PH_SIZE,c='#fff',ec='none'),
        lines    =[ax.plot([],[],c='#fff',lw=1.1)[0] for _ in range(PH_MAX)]
    )
    state.fig_main, state.ax_main, state.art = fig, ax, art
placeholder = st.empty()

# ───── FÍSICA ─────
def step():
    pos,vel,fase,cd   = state.pos,state.vel,state.fase,state.cd
    nivel,relax       = state.nivel,state.relax
    ppos,pprev,plife  = state.ppos,state.pprev,state.plife

    # emisión
    n=flujo
    pos=np.vstack([pos,
        np.column_stack([np.full(n,x_fil),
                         np.random.uniform(0.5,alto-0.5,n)])])[-MAX_E:]
    v0=np.sqrt(2*1.6e-19*Vacc/9.1e-31)*FACTOR_V
    vel=np.vstack([vel,np.column_stack([np.full(n,v0),np.zeros(n)])])[-MAX_E:]
    fase=np.concatenate([fase,np.random.uniform(0,2*np.pi,n)])[-MAX_E:]
    cd=np.concatenate([cd,np.zeros(n)])[-MAX_E:]

    # movimiento
    pos[:,0]+=vel[:,0]*DT
    pos[:,1]+=0.04*np.sin(4*pos[:,0]+fase)

    # colisiones
    for i in range(len(pos)):
        if cd[i]>0: cd[i]-=1; continue
        for j,at in enumerate(atoms):
            if np.linalg.norm(pos[i]-at)<0.15:
                Ec=0.5*9.1e-31*(vel[i,0]/FACTOR_V)**2
                lvl=nivel[j]
                if lvl<2:
                    dE=(ENERG[lvl+1]-ENERG[lvl])*1.6e-19
                    if Ec>dE and np.random.rand()<P_INEL*(1-dE/Ec):
                        vel[i,0]=np.sqrt(max(0,2*(Ec-dE)/9.1e-31))*FACTOR_V
                        nivel[j]+=1; relax[j]=TREL[nivel[j]]; cd[i]=10
                break

    # relajación / fotones
    relax=np.maximum(relax-1,0)
    for j in np.where(relax==0)[0]:
        if nivel[j]>0:
            ppos  = np.vstack([ppos , atoms[j]])[-PH_MAX:]
            pprev = np.vstack([pprev, atoms[j]])[-PH_MAX:]
            plife = np.append(plife, PH_SPAN)[-PH_MAX:]
            nivel[j]-=1; relax[j]=TREL[nivel[j]]

    # frenado
    dist=(x_col-x_an)*escala
    if dist>0:
        dv=(-1.6e-19*Vfilt/9.1e-31)/dist*FACTOR_V**2*DT*0.5
        mask=(pos[:,0]>=x_an)&(pos[:,0]<=x_col)
        vel[mask,0]+=dv

    # fotones
    if ppos.size:
        pprev=ppos.copy(); ppos[:,1]+=PH_SPEED; plife-=1
        k=plife>0; ppos,pprev,plife=ppos[k],pprev[k],plife[k]

    # limpiar e- fuera
    k=(pos[:,0]>=0)&(pos[:,0]<=ancho)
    state.update(pos=pos[k], vel=vel[k], fase=fase[k], cd=cd[k],
                 nivel=nivel, relax=relax,
                 ppos=ppos, pprev=pprev, plife=plife)

def draw():
    art=state.art
    art["scat_atoms"].set_offsets(atoms)
    art["scat_atoms"].set_color([COL[k] for k in state.nivel])

    art["scat_e"].set_offsets(state.pos)

    if state.ppos.size:
        a = state.plife/PH_SPAN
        art["scat_p"].set_offsets(state.ppos)
        art["scat_p"].set_sizes(PH_SIZE*(0.8+a))
        art["scat_p"].set_alpha(a)
        for ln,p0,p1,al in itertools.zip_longest(
                art["lines"], state.pprev,
                state.ppos-[[0,PH_SPEED]], a, fillvalue=None):
            if ln is not None:
                ln.set_data([p1[0],p0[0]],[p1[1],p0[1]])
                ln.set_alpha(al*0.7)
    else:
        art["scat_p"].set_offsets(np.empty((0,2)))   # ← fix vacío (0,2)
        for ln in art["lines"]: ln.set_alpha(0)

# ───── BUCLE ─────
if state.run:
    for _ in range(SUB_STEPS): step()
    if state["frame"] % DRAW_EVERY == 0:
        draw()
        # — render a PNG en memoria para evitar parpadeo —
        buf = io.BytesIO()
        state.fig_main.canvas.draw()
        state.fig_main.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        buf.seek(0)
        placeholder.image(buf, clamp=True)      ### ← SIN titileo
    state["frame"] += 1
    time.sleep(0.002)
    st.rerun()

# ───── LEYENDA ─────
st.markdown("""
<span style='color:#ffaa00'>●</span> fundamental&nbsp;
<span style='color:#ff4444'>●</span> n=2&nbsp;
<span style='color:#bb88ff'>●</span> n=3&nbsp;
<span style='color:#ffffff'>●</span> fotón""", unsafe_allow_html=True)