import streamlit as st
import matplotlib.pyplot as plt
import time
import show_robot_logic_gpt as backend

def draw_environment(ql, placeholder, selected_services, steps):
    env = ql.show_environment
    fig, ax = plt.subplots()
    ax.set_title(f"Show-Robot Environment (Steps: {steps})")
    ax.set_xlim(0.5, env.dim_x + 0.5)
    ax.set_ylim(0.5, env.dim_y + 0.5)
    ax.set_xticks(range(1, env.dim_x + 1))
    ax.set_yticks(range(1, env.dim_y + 1))
    ax.grid(True)

    for bx, by in env.block_list:
        ax.scatter(bx, by, marker='s', color='red', s=200)

    for (rx, ry), rew in env.reward_matrix.items():
        ax.scatter(rx, ry, marker='s', color='lightgreen', s=200)
        ax.text(rx, ry, f"{rew}", color='black', ha='center', va='center')

    for service in selected_services:
        if service in env.services:
            for sx, sy in env.services[service]:
                ax.scatter(sx, sy, marker='s', color='darkgreen', s=200)

    sx, sy = env.start_pos
    ax.scatter(sx, sy, marker='s', color='yellow', s=200)

    rx, ry = ql.current_state
    ax.scatter(rx, ry, marker='o', color='blue', s=200)

    placeholder.pyplot(fig)
    plt.close(fig)

st.title("Live Show-Robot Simulation")

services = st.multiselect("Wähle zwei Services", [
    "QR-Code",
    "Survey feedback",
    "Customer profile enrichment",
    "Exchange scheduling",
    "Gallery enquiry",
    "Concierge service",
    "Sales enquiry",
    "Catering cleanup"
], max_selections=2)

epsilon = st.slider("Wähle Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
gamma = st.slider("Wähle Gamma (discount factor)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
beta = st.slider("Wähle Beta (supervised update learning rate", min_value=0.0, max_value=1.0, value=0.5, step=0.1)  
autosupervision = st.checkbox("Autosupervision", value=True)
update_rate = st.slider("Aktualisierungsrate des Bildes", min_value=0.0, max_value=1.0, value=0.4, step=0.1)

if len(services) == 2:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "steps" not in st.session_state:
        st.session_state.steps = 0
    if "goal_reached" not in st.session_state:
        st.session_state.goal_reached = False

    if st.button("Simulation starten"):
        st.session_state.running = True
        st.session_state.steps = 0
        st.session_state.goal_reached = False

    plot_placeholder = st.empty()

    if st.button("Stop Simulation"):
        st.session_state.running = False


    if st.session_state.running:
        ql = backend.Q_Learning(services = services, gamma = gamma, beta = beta)
        st.write("Simulation gestartet.")

        while st.session_state.running:
            ql.simulate_step(epsilon=epsilon, auto_supervision=autosupervision)

            target_reached = False
            for service in services:
                if service in ql.show_environment.services:
                    for sx, sy in ql.show_environment.services[service]:
                        if ql.current_state == (sx, sy):
                            target_reached = True
                            break
                if target_reached:
                    break

            if target_reached and not st.session_state.goal_reached:
                st.session_state.goal_reached = True
                st.session_state.steps += 1

            elif ql.current_state == ql.show_environment.start_pos and st.session_state.goal_reached:
                st.session_state.steps = 0
                st.session_state.goal_reached = False

            else:
                st.session_state.steps += 1

            draw_environment(ql, plot_placeholder, services, st.session_state.steps)
            time.sleep(update_rate)



