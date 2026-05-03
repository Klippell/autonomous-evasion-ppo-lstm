import sys
import os

# =================================================================
# 1. CONFIGURAÇÃO DE AMBIENTE (RESOLUÇÃO DE DLLs E CAMINHOS)
# =================================================================
# Define o local de instalação do Webots
WEBOTS_HOME = r"C:\Program Files\Webots"
os.environ['WEBOTS_HOME'] = WEBOTS_HOME

# Pastas críticas que contêm as DLLs de física e do motorista (Driver)
dll_paths = [
    os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin"),
    os.path.join(WEBOTS_HOME, "msys64", "mingw64", "bin", "cpp"),
    os.path.join(WEBOTS_HOME, "lib", "controller"),
    os.path.join(WEBOTS_HOME, "projects", "vehicles", "lib")
]

# Adiciona as pastas ao sistema para que o Python 3.10 permita o carregamento
for path in dll_paths:
    if os.path.exists(path):
        os.add_dll_directory(path)
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')

# Adiciona a biblioteca Python oficial do Webots ao PATH do sistema
sys.path.append(os.path.join(WEBOTS_HOME, "lib", "controller", "python"))

# Agora importamos o Driver (específico para veículos Ackermann como carros)
from vehicle import Driver


# =================================================================
# 2. FUNÇÃO PRINCIPAL DO CONTROLADOR DO EVADIDO
# =================================================================
def run_evader():
    # Inicializa a interface de controle do veículo
    driver = Driver()

    # O timestep define a frequência da simulação (essencial para sincronismo)
    timestep = int(driver.getBasicTimeStep())

    print("=====================================================")
    print("SISTEMA DO EVADIDO: Conectado e pronto para o teste.")
    print("=====================================================")

    # --- CONFIGURAÇÕES DE MOVIMENTO (Ação Inicial) ---
    # Para o carro se mover no Webots, precisamos de 4 comandos:
    driver.setGear(1)  # 1. Engata a primeira marcha
    driver.setCruisingSpeed(40.0)  # 2. Define velocidade alvo (km/h)
    driver.setThrottle(0.8)  # 3. Aplica 80% de aceleração
    driver.setBrakeIntensity(0.0)  # 4. Garante que o freio está solto
    driver.setSteeringAngle(0.0)  # Rodas retas

    # --- LOOP PRINCIPAL (Sense-Think-Act) ---
    # Este loop roda a cada passo da simulação [cite: 13]
    while driver.step() != -1:
        # SENSE (Sentir): Espaço para Camera e LiDAR (Próxima Fase) [cite: 16, 33]

        # MONITORAMENTO: Verifica se o carro ganhou velocidade
        current_speed = driver.getCurrentSpeed()
        if current_speed > 0.1:
            print(f"MOVIMENTO DETECTADO: {current_speed:.2f} km/h")

        # ACT (Agir): Mantém os comandos de aceleração ativos
        driver.setThrottle(0.8)
        driver.setSteeringAngle(0.0)


# =================================================================
# 3. PONTO DE ENTRADA
# =================================================================
if __name__ == "__main__":
    run_evader()