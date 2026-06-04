"""Train the Webots evader with recurrent PPO."""
import os
import sys
import time
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback

import controllers.evader_env

def main() -> None:
    # Configurações iniciais
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    robot_name = os.environ.get("WEBOTS_ROBOT_NAME", "evader")

    # Orçamento de treino (10 milhões para não parar cedo demais)
    timesteps = 10_000_000

    # ==========================================
    # MENU INTERATIVO NO TERMINAL
    # ==========================================
    print("\n" + "="*50)
    print("🚗 MENU DE TREINO - PROJETO ROBÓTICA 🚗")
    print("="*50)
    print("1. Começar um treino NOVO (do ZERO)")
    print("2. CONTINUAR um treino salvo (Branching)")

    escolha = input("\n👉 Escolha a opção (1 ou 2): ").strip()

    load_path = None
    base_name = "evader_recurrent_ppo"

    if escolha == "2":
        all_models = []

        # 1. Procurar na pasta de checkpoints
        checkpoints_dir = os.path.join(log_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            for f in os.listdir(checkpoints_dir):
                if f.endswith('.zip'):
                    # Expressão regular para separar o NOME BASE dos STEPS
                    match = re.search(r'^(.*?)_(\d+)_steps\.zip$', f)
                    if match:
                        base = match.group(1)
                        steps = int(match.group(2))
                    else:
                        base = f.replace('.zip', '')
                        steps = 0

                    all_models.append({
                        "name": f,
                        "base_name": base,
                        "path": os.path.join(checkpoints_dir, f),
                        "type": "CHECKPOINT",
                        "steps": steps
                    })

        # 2. Procurar na pasta raiz (logs) pelos modelos FINAIS
        if os.path.exists(log_dir):
            for f in os.listdir(log_dir):
                if f.endswith('.zip'):
                    base = f.replace('.zip', '')
                    all_models.append({
                        "name": f,
                        "base_name": base,
                        "path": os.path.join(log_dir, f),
                        "type": "FINAL",
                        "steps": float('inf') # Coloca os finais sempre no fim do seu grupo
                    })

        if all_models:
            # MAGIA DA ORGANIZAÇÃO:
            # 1º Ordem Alfabética do Nome Base
            # 2º Checkpoints primeiro, Finais depois
            # 3º Ordem Numérica de Steps
            all_models.sort(key=lambda x: (x["base_name"].lower(), x["type"] == "FINAL", x["steps"]))

            print("\n📂 Modelos salvos encontrados (Agrupados por Nome e Steps):")
            current_group = ""
            for i, m in enumerate(all_models):
                # Imprime um separador quando muda o nome do modelo para ficar mais bonito
                if m["base_name"] != current_group:
                    current_group = m["base_name"]
                    print(f"\n   --- {current_group.upper()} ---")

                prefix = "🏁 FINAL     " if m["type"] == "FINAL" else "⏳ CHECKPOINT"
                print(f"  [{i:2d}] {prefix} -> {m['name']}")

            idx = input(f"\n👉 Digite o NÚMERO do modelo que quer carregar (0 a {len(all_models)-1}): ").strip()
            try:
                selected = all_models[int(idx)]
                load_path = selected["path"]
                file_name = selected["name"]
                print(f"\n[OK] Modelo selecionado: {file_name}")

                # Limpa o nome para sugerir no próximo passo
                base_name = selected["base_name"]
            except (ValueError, IndexError):
                print("\n[AVISO] Número inválido! O treino vai começar do ZERO.")
        else:
            print("\n[AVISO] Nenhum ficheiro .zip encontrado. Vai começar do ZERO.")

    print("\n" + "-"*50)
    if load_path:
        print(f"O nome base atual é: '{base_name}'")
        save_name = input("💾 [ENTER] para manter ou digite NOVO NOME para ramificar:\n👉 ").strip()
    else:
        save_name = input("💾 Qual o NOME para salvar este treino novo?\n👉 ").strip()

    if not save_name:
        save_name = base_name

    # ==========================================
    # INÍCIO DO AMBIENTE E ALGORITMO
    # ==========================================
    env: gym.Env = gym.make("Evader-v0", robot_name=robot_name)

    if load_path is not None:
        print(f"🧠 A carregar inteligência anterior...")
        model: BaseAlgorithm = RecurrentPPO.load(
            load_path,
            env=env,
            tensorboard_log=os.path.join(log_dir, "tensorboard_logs"),
        )
        reset_timesteps = False
        tb_name = save_name + "_continuacao"
    else:
        print("👶 A criar um cérebro novo (em branco)...")
        model: BaseAlgorithm = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=128,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log=os.path.join(log_dir, "tensorboard_logs"),
        )
        reset_timesteps = True
        tb_name = save_name

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=save_name,
    )

    print(f"\n🚀 Treino iniciado: {save_name}")
    print("💡 Minimiza o Webots para acelerar o processo!")

    try:
        model.learn(
            total_timesteps=timesteps,
            log_interval=10,
            tb_log_name=tb_name,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_timesteps,
        )
    except KeyboardInterrupt:
        print("\n[STOP] Treino interrompido pelo utilizador. A guardar modelo final...")

    model.save(os.path.join(log_dir, save_name))
    print(f"✅ Modelo final guardado como: {save_name}.zip na pasta logs/")
    env.close()

if __name__ == "__main__":
    main()
