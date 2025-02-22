import streamlit as st
import matplotlib.pyplot as plt
import time
import show_robot_logic_gpt

# Diese Funktion zeichnet das Spielfeld und markiert:
# - Block-Positionen als rote Quadrate,
# - Reward-Felder als rote Quadrate mit der Reward-Zahl,
# - Den Roboter (current_state) als blauen Kreis,
# - Den Startpunkt als gelbes Quadrat.
def draw_environment(ql):
    env = ql.show_environment
    fig, ax = plt.subplots()
    ax.set_title("Show-Robot Environment")
    ax.set_xlim(0.5, env.dim_x + 0.5)
    ax.set_ylim(0.5, env.dim_y + 0.5)
    ax.set_xticks(range(1, env.dim_x+1))
    ax.set_yticks(range(1, env.dim_y+1))
    ax.grid(True)

    # Zeichne Block-Positionen
    for bx, by in env.block_list:
        ax.scatter(bx, by, marker='s', color='red', s=200)

    # Zeichne Reward-Felder
    for (rx, ry), rew in env.reward_matrix.items():
        ax.scatter(rx, ry, marker='s', color='lightgreen', s=200)
        ax.text(rx, ry, f"{rew}", color='black', ha='center', va='center')

    # Zeichne Startpunkt (gelb)
    sx, sy = env.start_pos
    ax.scatter(sx, sy, marker='s', color='yellow', s=200)

    # Zeichne den Roboter (current_state) als blauen Kreis
    rx, ry = ql.current_state
    ax.scatter(rx, ry, marker='o', color='blue', s=200)

    st.pyplot(fig)
    plt.close(fig)

st.title("Live Show-Robot Simulation")

service = st.selectbox("Wähle Service", [
    "QR-Code",
    "Survey feedback",
    "Customer profile enrichment",
    "Exchange scheduling",
    "gallery enquiry",
    "Concierge service"
])

if st.button("Simulation starten"):
    ql = show_robot_logic_gpt.Q_Learning(service)
    st.write("Simulation gestartet. Die Karte aktualisiert sich einmal pro Sekunde.")

    # Beispiel: 20 Schritte (kann beliebig erweitert werden)
    for i in range(20):
        ql.simulate_step(epsilon=0.1)
        draw_environment(ql)
        time.sleep(1)
