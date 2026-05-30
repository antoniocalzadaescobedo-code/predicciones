# ui_components.py
import streamlit as st
import numpy as np

# Regla de decisión: partido impredecible si diff < 5%
def get_dominant_verdict(probs, team_names):
    sorted_probs = sorted(probs, reverse=True)
    p1, p2 = sorted_probs[0], sorted_probs[1]
    idx_max = np.argmax(probs)
    diff = p1 - p2
    leader_prob = p1

    if diff < 0.05:
        main_text = "🔴 Partido equilibrado"
        confidence = leader_prob
        show_advantage = False
    else:
        main_text = team_names[idx_max]
        confidence = leader_prob
        if diff >= 0.12 and leader_prob >= 0.45:
            show_advantage = True
        else:
            show_advantage = False

    others = []
    for i, (p, name) in enumerate(zip(probs, team_names)):
        if i != idx_max:
            others.append((name, p))
    others.sort(key=lambda x: x[1], reverse=True)
    other_details = {others[0][0]: others[0][1], others[1][0]: others[1][1]}

    return main_text, confidence, show_advantage, other_details

# Tarjeta única de resultado (estilo ESPN)
def result_card_unique(probs, team_names):
    main_text, confidence, show_advantage, others = get_dominant_verdict(probs, team_names)

    st.markdown("""
    <style>
    .unique-card {
        background: linear-gradient(145deg, #0f172a 0%, #020617 100%);
        border-radius: 32px;
        padding: 1.8rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.5);
        border: 1px solid #334155;
    }
    .unique-title {
        font-size: 0.85rem;
        letter-spacing: 2px;
        color: #facc15;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    .unique-main {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .unique-confidence {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        border-left: 3px solid #facc15;
        padding-left: 12px;
    }
    .unique-advantage {
        background: rgba(250, 204, 21, 0.15);
        border-radius: 30px;
        padding: 0.2rem 0.8rem;
        display: inline-block;
        font-size: 0.75rem;
        color: #facc15;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .unique-details {
        background: #1e293b;
        border-radius: 20px;
        padding: 0.8rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }
    .detail-item {
        display: flex;
        justify-content: space-between;
        margin: 6px 0;
        color: #cbd5e1;
    }
    .detail-percent {
        font-weight: 600;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    html = f"""
    <div class="unique-card">
        <div class="unique-title">🏆 RESULTADO PROBABLE</div>
        <div class="unique-main">{main_text}</div>
        <div class="unique-confidence">Confianza: {confidence:.1%}</div>
    """
    if show_advantage:
        html += '<div class="unique-advantage">🔥 Ventaja significativa</div>'
    else:
        html += '<div class="unique-advantage" style="background:#2d3748; color:#a0aec0;">⚖️ Sin ganador claro</div>'

    other_items = "".join([f'<div class="detail-item"><span>{name}</span><span class="detail-percent">{p:.1%}</span></div>'
                           for name, p in others.items()])
    html += f"""
        <div class="unique-details">
            {other_items}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
