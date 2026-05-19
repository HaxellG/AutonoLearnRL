import json
import os
import matplotlib.pyplot as plt

def generate_graph(json_path, output_path, title, color):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    curve = data.get('checkpoints', [])
    if not curve:
        print(f"No checkpoints found in {json_path}")
        return
        
    episodes = [entry['episode'] for entry in curve]
    avg_scores = [entry['mean'] for entry in curve]
    max_scores = [entry.get('max', 0) for entry in curve]
    
    plt.figure(figsize=(10, 6))
    
    # Set professional light aesthetic style
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    plt.gcf().set_facecolor('#ffffff')
    
    plt.plot(episodes, avg_scores, label='Average Score', color=color, linewidth=2.5)
    plt.fill_between(episodes, 0, avg_scores, color=color, alpha=0.15)
    plt.plot(episodes, max_scores, label='Max Score', color='#6c757d', linewidth=1.5, linestyle='--', alpha=0.8)
    
    plt.title(title, fontsize=16, color='#212529', pad=20, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=12, color='#495057', fontweight='bold')
    plt.ylabel('Score', fontsize=12, color='#495057', fontweight='bold')
    
    plt.grid(True, color='#dee2e6', linestyle='-', linewidth=0.5, alpha=0.8)
    plt.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#ced4da', labelcolor='#212529')
    
    ax.spines['bottom'].set_color('#adb5bd')
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#adb5bd')
    
    ax.tick_params(axis='x', colors='#495057')
    ax.tick_params(axis='y', colors='#495057')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print(f"Saved {output_path}")

os.makedirs('results/graphs', exist_ok=True)

configs = [
    ('results/medium/qlearning_results.json', 'results/graphs/medium_qlearning.png', 'Tabular Q-Learning (Normal Difficulty, Gap=125)', '#005b96'),
    ('results/medium/dqn_results.json', 'results/graphs/medium_dqn.png', 'Vanilla DQN (Normal Difficulty, Gap=125)', '#d9534f'),
    ('results/medium/ddqn_results.json', 'results/graphs/medium_ddqn.png', 'Double DQN (Normal Difficulty, Gap=125)', '#5cb85c'),
    ('results/hard/qlearning_results.json', 'results/graphs/hard_qlearning.png', 'Tabular Q-Learning (Hard Difficulty, Gap=100)', '#005b96'),
    ('results/hard/dqn_results.json', 'results/graphs/hard_dqn.png', 'Vanilla DQN (Hard Difficulty, Gap=100)', '#d9534f'),
    ('results/hard/ddqn_results.json', 'results/graphs/hard_ddqn.png', 'Double DQN (Hard Difficulty, Gap=100)', '#5cb85c'),
]

for c in configs:
    generate_graph(*c)
