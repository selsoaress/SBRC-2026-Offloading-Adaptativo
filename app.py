import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta

st.set_page_config(layout="wide", page_title="Offloading Simulation")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] { background-color: #262730; border: 1px solid #41444C; border-radius: 8px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #FAFAFA; }
    section[data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #41444C; }
    </style>
""", unsafe_allow_html=True)


st.title("Offloading Adaptativo com Agentes de Multi-Armed Bandits")
st.markdown("### Simulação baseada no algoritmo UCB")


with st.sidebar:
    st.header("Configurações")
    
    st.subheader("Simulação")
    # seed fixa para visualizar os resultados do paper com aqueles parametros
    use_seed = st.checkbox("Fixar Semente (Seed 42)", value=True)
    n_episodes = st.number_input("Episódios", value=40)
    n_turns = st.number_input("Turnos", value=1500)
    w_smooth = st.slider("Suavização", 10, 200, 60)

    st.subheader("Parâmetros de Custo")
    
    cost_fixed = st.number_input("Custo Fixo", 0.0, 1.0, 0.05)
    cost_penalty = st.number_input("Penalidade Max", 0.0, 5.0, 1.0)
    
    st.subheader("Ambiente (QoS)")
    col1, col2 = st.columns(2)
    ci_a = col1.number_input("Alpha Local", value=8.0)
    ci_b = col2.number_input("Beta Local", value=4.0)
    cl_a = col1.number_input("Alpha Nuvem", value=12.0)
    cl_b = col2.number_input("Beta Nuvem", value=3.0)

    st.subheader("Agentes")
    st.markdown("Agente 1")
    bat_d1 = st.number_input("Bateria D1", value=500.0)
    prob_d1 = st.slider("Prob. Rede D1", 0.0, 1.0, 0.7)
    
    st.markdown("Agente 2")
    bat_d2 = st.number_input("Bateria D2", value=150.0)
    prob_d2 = st.slider("Prob. Rede D2", 0.0, 1.0, 0.3)


N_ACTIONS = 100
THRESHOLDS = np.linspace(0.0, 1.0, N_ACTIONS)

class UCBAgent:
    def __init__(self, energy):
        self.counts = np.zeros(N_ACTIONS)
        self.values = np.zeros(N_ACTIONS)
        self.total_n = 0
        self.e0 = energy
        self.e = energy

    def select_action(self):
        self.total_n += 1
        if 0 in self.counts: return np.argmin(self.counts)
        confidence = np.sqrt((2 * np.log(self.total_n)) / (self.counts + 1e-7))
        return np.argmax(self.values + confidence)

    def update(self, idx, reward, consumed):
        if consumed and self.e > 0: self.e -= 1.0
        self.counts[idx] += 1
        self.values[idx] += (reward - self.values[idx]) / self.counts[idx]

# Correção aqui: Adicionados os parâmetros cost_fixed e cost_penalty na definição
def calc_reward(agente, intent, success, ci, cl, cost_fixed, cost_penalty):

    # se não tem intenção de offloading, processa local (ci)
    if not intent: return 0.0

    # incorporando escassez de recursos à função de custo de offloading
    consumption_bonus = (agente.e0 - agente.e) / agente.e0
    
    return (cl + consumption_bonus) if success else consumption_bonus


def smooth(x, w):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, 'valid')


# Removida a "Sensibilidade" da lista de tabs
tab1, tab2 = st.tabs(["Ambiente", "Simulação"])

with tab1:
    rv_ci = beta(ci_a, ci_b)
    rv_cl = beta(cl_a, cl_b)
    x = np.linspace(0, 1, 1000)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=rv_ci.pdf(x), fill='tozeroy', name='Local (CI)', line=dict(color='#7f7f7f')))
    fig.add_trace(go.Scatter(x=x, y=rv_cl.pdf(x), fill='tozeroy', name='Nuvem (CL)', line=dict(color='#00CC96')))
    fig.update_layout(title="Distribuições de Probabilidade (QoS)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Média Local", f"{rv_ci.mean():.3f}")
    c2.metric("Média Nuvem", f"{rv_cl.mean():.3f}")
    c3.metric("Gap", f"{rv_cl.mean() - rv_ci.mean():.3f}")

with tab2:
    if st.button("Rodar Simulação", type="primary"):
        if use_seed:
            np.random.seed(42) # Fixando a aleatoriedade
            
        h_alpha1 = np.zeros((n_episodes, n_turns))
        h_alpha2 = np.zeros((n_episodes, n_turns))
        h_rew1 = np.zeros((n_episodes, n_turns))
        h_rew2 = np.zeros((n_episodes, n_turns))

        bar = st.progress(0)
        
        for ep in range(n_episodes):
            d1 = UCBAgent(bat_d1)
            d2 = UCBAgent(bat_d2)

            for t in range(n_turns):
                ci1, cl1 = np.random.beta(ci_a, ci_b), np.random.beta(cl_a, cl_b)
                ci2, cl2 = np.random.beta(ci_a, ci_b), np.random.beta(cl_a, cl_b)

                i1, i2 = d1.select_action(), d2.select_action()
                a1, a2 = THRESHOLDS[i1], THRESHOLDS[i2]

                int1 = (ci1 < a1) and (d1.e > 0)
                int2 = (ci2 < a2) and (d2.e > 0)

                s1, s2 = False, False
                if int1 and int2:
                    if np.random.rand() < prob_d1: s1 = True
                    else: s2 = True
                elif int1: s1 = True
                elif int2: s2 = True

                # A chamada agora corresponde à definição da função
                r1 = calc_reward(d1, int1, s1, ci1, cl1, cost_fixed, cost_penalty)
                r2 = calc_reward(d2, int2, s2, ci2, cl2, cost_fixed, cost_penalty)

                d1.update(i1, r1, int1)
                d2.update(i2, r2, int2)

                h_alpha1[ep, t] = a1
                h_alpha2[ep, t] = a2
                h_rew1[ep, t] = r1
                h_rew2[ep, t] = r2
            
            bar.progress((ep + 1) / n_episodes)

        s_a1 = smooth(np.mean(h_alpha1, axis=0), w_smooth)
        s_a2 = smooth(np.mean(h_alpha2, axis=0), w_smooth)
        s_r1 = smooth(np.mean(h_rew1, axis=0), w_smooth)
        s_r2 = smooth(np.mean(h_rew2, axis=0), w_smooth)
        x_axis = np.arange(len(s_a1))

        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=x_axis, y=s_a1, name='Agente 1', line=dict(color='#636EFA')))
        fig_a.add_trace(go.Scatter(x=x_axis, y=s_a2, name='Agente 2', line=dict(color='#EF553B')))
        fig_a.update_layout(title="Convergência do Limiar (Alpha)", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_a, use_container_width=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=x_axis, y=s_r1, name='Agente 1', line=dict(color='#636EFA')))
        fig_r.add_trace(go.Scatter(x=x_axis, y=s_r2, name='Agente 2', line=dict(color='#EF553B')))
        fig_r.update_layout(title="Recompensa Média", template="plotly_dark")
        st.plotly_chart(fig_r, use_container_width=True)
