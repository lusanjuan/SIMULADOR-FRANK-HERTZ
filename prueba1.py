import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


#hola delfi!
#holaaa

#PAGINA EN GOOGLE 
st.set_page_config(page_title="Simulador Franck-Hertz", layout="centered")

#FONDO Y ESTILO GENERAL
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

#TITULO 
st.markdown("""
    <h1>Experimento de Franck y Hertz</h1>
""", unsafe_allow_html=True)

#CUADRO DE TEXTO
st.markdown("""
Este simulador te permite explorar cómo los electrones colisionan con átomos a medida que aumenta el voltaje. 
Observa los picos de corriente cuando los electrones pierden energía por excitación atómica. 
""")

#POTENCIAL ACELERADOR
e_exc = st.slider("Potencial de excitación (eV)", 0.1, 20.0, 4.9, 0.1)
#ENERGIA DE EXCITACION DEL ATOMO
voltaje_max = st.slider("Voltaje de aceleración (V)", 0.0, 50.0, 0.0, 0.1)
#NUMERO DE ELECTRONES
num_electrones = st.slider("Número de electrones", 5, 100, 20)

#SIMULACION DE LA CORRIENTE
def corriente_simulada(V, e_exc):
    corriente = []

    for v in V:
        # Aumenta la corriente con el voltaje (lineal o logarítmica)
        base = v ** 1.2

        # Cada múltiplo de e_exc produce una caída abrupta
        # Detectamos si estamos justo antes de una colisión inelástica
        residuo = v % e_exc
        if residuo < 0.3:  # cerca del inicio del pico de caída
            caída = 0.4  # cae bruscamente
        else:
            caída = 1.0  # no hay colisión

        corriente.append(base * caída)

    return np.array(corriente)



# Generar datos
voltajes = np.linspace(0, 50, 500)
corrientes = corriente_simulada(voltajes, e_exc)


# Mostrar valor actual

corriente_actual = corriente_simulada([voltaje_max], e_exc)[0]
st.write(f" Corriente estimada para {voltaje_max:.1f} V: **{corriente_actual:.3f}** (unidades arbitrarias)")

# Cortar la curva después del voltaje actual
corrientes_visibles = corriente_simulada(voltajes, e_exc)
corrientes_visibles[voltajes > voltaje_max] = np.nan

# Gráfico
fig, ax = plt.subplots()
ax.plot(voltajes, corrientes_visibles, color="blue", label="Corriente simulada hasta voltaje actual")
ax.axvline(voltaje_max, color="red", linestyle="--", label=f"Voltaje actual: {voltaje_max:.1f} V")
ax.set_xlabel("Voltaje (V)")
ax.set_ylabel("Corriente (u.a.)")
ax.set_title("Corriente vs Voltaje")
ax.legend()
st.pyplot(fig)





st.title("Simulación visual del experimento de Franck y Hertz")

ancho = 10
altura = 5
dt = 0.1

# Inicialización
np.random.seed(1)
posiciones = np.zeros((num_electrones, 2))
posiciones[:, 1] = np.random.uniform(0.5, altura - 0.5, num_electrones)
velocidades = np.zeros((num_electrones, 2))
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

# Contenedor para animación
grafico = st.empty()

# Inicializar variables de sesión
if "animando" not in st.session_state:
    st.session_state.animando = False

# Botones
col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Comenzar animación"):
        st.session_state.animando = True
with col2:
    if st.button("🛑 Detener animación"):
        st.session_state.animando = False

# Dimensiones del tubo
ancho = 10
altura = 5
dt = 0.1

# Inicializar posiciones y velocidades
np.random.seed(1)
posiciones = np.zeros((num_electrones, 2))
posiciones[:, 1] = np.random.uniform(0.5, altura - 0.5, num_electrones)
velocidades = np.zeros((num_electrones, 2))
velocidades[:, 0] = np.sqrt(2 * 1.6e-19 * voltaje_max / 9.1e-31) * 1e-6

# Contenedor del gráfico
grafico = st.empty()

# Animación en tiempo real controlada por botón
while st.session_state.animando:
    posiciones[:, 0] += velocidades[:, 0] * dt

    # Reubicar los que llegan al final
    reiniciar = posiciones[:, 0] > ancho
    posiciones[reiniciar, 0] = 0
    posiciones[reiniciar, 1] = np.random.uniform(0.5, altura - 0.5, np.sum(reiniciar))

    # Dibujar gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, altura)
    ax.set_title("Movimiento de electrones en el tubo")
    ax.set_xlabel("Distancia (cm)")
    ax.set_ylabel("Altura (cm)")
    ax.scatter(posiciones[:, 0], posiciones[:, 1], color='cyan')
    grafico.pyplot(fig)

    # Pausa para efecto de animación
    time.sleep(0.1)