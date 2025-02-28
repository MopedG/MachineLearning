import streamlit as st
import matplotlib.pyplot as plt
import time
import show_robot_logic as backend


def draw_environment(ql, placeholder, selected_service, steps):
    env = ql.show_environment
    fig, ax = plt.subplots()
    ax.set_title(f"Show-Robot Environment (Steps: {steps})")  # Steps im Titel anzeigen
    ax.set_xlim(0.5, env.dim_x + 0.5)
    ax.set_ylim(0.5, env.dim_y + 0.5)
    ax.set_xticks(range(1, env.dim_x + 1))
    ax.set_yticks(range(1, env.dim_y + 1))
    ax.grid(True)

    # Zeichne Block-Positionen
    for bx, by in env.block_list:
        ax.scatter(bx, by, marker='s', color='red', s=200)

    # Zeichne Reward-Felder
    for (rx, ry), rew in env.reward_matrix.items():
        ax.scatter(rx, ry, marker='s', color='lightgreen', s=200)
        ax.text(rx, ry, f"{rew}", color='black', ha='center', va='center')

    # Markiere Service-Koordinaten (falls ausgewählt, dunkelgrün)
    if selected_service in env.services:
        for sx, sy in env.services[selected_service]:
            ax.scatter(sx, sy, marker='s', color='darkgreen', s=200)

    # Zeichne Startpunkt (gelb)
    sx, sy = env.start_pos
    ax.scatter(sx, sy, marker='s', color='yellow', s=200)

    # Zeichne den Roboter (current_state) als blauen Kreis
    rx, ry = ql.current_state
    ax.scatter(rx, ry, marker='o', color='blue', s=200)

    placeholder.pyplot(fig)
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

if "running" not in st.session_state:
    st.session_state.running = False
if "steps" not in st.session_state:
    st.session_state.steps = 0  # Schrittzähler initialisieren
if "goal_reached" not in st.session_state:
    st.session_state.goal_reached = False  # Zustand, ob Ziel erreicht wurde

if st.button("Simulation starten"):
    st.session_state.running = True
    st.session_state.steps = 0  # Schritte zurücksetzen
    st.session_state.goal_reached = False  # Zielstatus zurücksetzen

plot_placeholder = st.empty()  # Platzhalter für die Karte

# Stop-Button immer sichtbar halten
if st.button("Stop Simulation"):
    st.session_state.running = False

if st.session_state.running:
    ql = backend.Q_Learning(service)
    st.write("Simulation gestartet.")

    while st.session_state.running:
        ql.simulate_step(epsilon=0.1)

        # Überprüfe, ob der Roboter das Ziel erreicht hat
        target_reached = False
        if service in ql.show_environment.services:
            for sx, sy in ql.show_environment.services[service]:
                if ql.current_state == (sx, sy):
                    target_reached = True
                    break

        # Wenn das Ziel erreicht wurde, speichern wir das und warten auf den Rückweg zur Startposition
        if target_reached and not st.session_state.goal_reached:
            st.session_state.goal_reached = True  # Ziel erreicht
            st.session_state.steps += 1  # Schritte zählen, wenn das Ziel erreicht wurde

        # Wenn der Roboter an der Startposition ist und das Ziel vorher erreicht wurde, Zähler zurücksetzen
        elif ql.current_state == ql.show_environment.start_pos and st.session_state.goal_reached:
            st.session_state.steps = 0  # Zähler zurücksetzen
            st.session_state.goal_reached = False  # Zielstatus zurücksetzen

        else:
            st.session_state.steps += 1  # Schritte hochzählen, wenn das Ziel noch nicht erreicht wurde

        draw_environment(ql, plot_placeholder, service, st.session_state.steps)
        time.sleep(0.4)
