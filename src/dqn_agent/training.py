import numpy as np
import time
import os
from datetime import datetime
# Importaciones relativas dentro del paquete dqn_agent
from .agent import EnhancedDQNAgentPyTorch as EnhancedDQNAgent
from .environment import EVChargingEnv
from typing import Dict

# Importaciones de common
from src.common.logger import SimpleEpisodeLogger
from src.common.config import load_system_config

def create_episode_data(episode_num, total_reward, energy_metrics, assign_count, skip_count,
                       total_vehicles, episode_time, agent_epsilon):
    """
    Crea un diccionario estructurado con los datos del episodio para el logger.
    Compatibilidad con el main anterior que funciona.
    """
    # CÓDIGO CORREGIDO
    satisfaction_pct = energy_metrics.get("energy_satisfaction_pct", 0.0)
    assign_ratio = (assign_count / total_vehicles * 100) if total_vehicles > 0 else 0

    return {
        "episode": episode_num,
        "reward": total_reward,
        "satisfaction_pct": satisfaction_pct,
        "vehicles_assigned": assign_count,
        "vehicles_skipped": skip_count,
        "total_vehicles": total_vehicles,
        "assign_ratio": assign_ratio,
        "epsilon": agent_epsilon,
        "episode_time": episode_time
    }

class SystemProgressLogger:
    """
    Logger del sistema compatible con el main anterior.
    """
    def __init__(self):
        self.systems_data = {}
    
    def save_system_progress(self, system_id, episodes_data, system_info):
        """Guarda el progreso de un sistema."""
        self.systems_data[system_id] = {
            "episodes_data": episodes_data,
            "system_info": system_info
        }
        print(f"Progreso del sistema {system_id} guardado.")

def train_agent(systems, config):
    """
    Entrena un agente RL con múltiples sistemas.
    REPLICACIÓN EXACTA de la función del main anterior que funciona.
    """
    import time
    from datetime import datetime
    
    # Configuración del agente optimizada
    state_size = config.get("state_size", 32)
    action_size = config.get("action_size", 50)
    batch_size = config.get("batch_size", 32)
    episodes_per_system = config.get("episodes_per_system", 15)
    checkpoint_frequency = config.get("checkpoint_frequency", 5)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    model_path = config.get("model_path", "./ev_scheduler_model_pytorch.pt")
    patience = config.get("patience", 3)
    resume_from_checkpoint = config.get("resume_from_checkpoint", True)
    
    # Crear directorio de checkpoints si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    progress_logger = SystemProgressLogger()
    
    print(f"\nINICIANDO ENTRENAMIENTO DQN OPTIMIZADO")
    print(f"Configuración:")
    print(f"   - State size: {state_size}")
    print(f"   - Action size: {action_size}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Episodios por sistema: {episodes_per_system}")
    print(f"   - Total de sistemas: {len(systems)}")
    print(f"   - Total episodios estimados: {len(systems) * episodes_per_system}")
    
    # Crear agente avanzado con configuración optimizada
    print(f"\nCreando agente DQN...")
    agent = EnhancedDQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=config.get("learning_rate", 0.0005),
        gamma=config.get("gamma", 0.95),
        epsilon=config.get("epsilon", 0.9),
        epsilon_min=config.get("epsilon_min", 0.05),
        epsilon_decay=config.get("epsilon_decay", 0.995),
        memory_size=config.get("memory_size", 10000),
        batch_size=batch_size,
        dueling_network=config.get("dueling_network", True),
        target_update_freq=config.get("target_update_freq", 50),
        use_per=config.get("use_per", True),
        use_noisy=config.get("use_noisy", False),
        n_step=config.get("n_step", 3)
    )
    print(f"Agente creado exitosamente")
    
    # Orden de entrenamiento: sistemas más simples primero
    system_order = sorted(systems.keys())
    
    # Resumir entrenamiento desde checkpoint si existe
    current_system_idx = 0
    current_episode = 0
    
    if resume_from_checkpoint:
        print(f"\nBuscando checkpoints...")
        checkpoint_files = []
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith('.pt') and 'checkpoint' in file:  # Cambio .h5 -> .pt
                    checkpoint_files.append(os.path.join(root, file))
        
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
            latest_checkpoint = checkpoint_files[-1]
            
            try:
                filename = os.path.basename(latest_checkpoint)
                parts = filename.split('_')
                current_system_idx = int(parts[2])
                current_episode = int(parts[4].split('.')[0])
                
                print(f"Encontrado checkpoint: {latest_checkpoint}")
                print(f"Reanudando desde sistema {current_system_idx}, episodio {current_episode}")
                
                agent.load(latest_checkpoint)
                current_episode += 1
                
                if current_episode >= episodes_per_system:
                    current_system_idx += 1
                    current_episode = 0
            except Exception as e:
                print(f"Error al analizar checkpoint, iniciando desde cero: {e}")
                current_system_idx = 0
                current_episode = 0
        else:
            print(f"No se encontraron checkpoints")
    
    # Si no se encontró checkpoint, intentar cargar modelo pre-entrenado
    if current_system_idx == 0 and current_episode == 0:
        try:
            print(f"Intentando cargar modelo pre-entrenado desde {model_path}...")
            model_loaded = agent.load(model_path)
            if model_loaded:
                print("Modelo pre-entrenado cargado exitosamente.")
                agent.epsilon = 0.5  # Epsilon intermedio para modelo pre-entrenado
                print(f"Epsilon ajustado a {agent.epsilon}")
        except Exception as e:
            print(f"No se pudo cargar modelo pre-entrenado: {e}")
            print("Entrenando nuevo modelo desde cero...")
    
    # Archivo de registro para el progreso
    log_path = os.path.join(checkpoint_dir, "training_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\n--- Nueva sesión de entrenamiento optimizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        # Variables para estadísticas globales
        global_start_time = time.time()
        total_episodes_completed = 0
        
        # Entrenar en cada sistema, comenzando desde el índice actual
        for system_idx in range(current_system_idx, len(system_order)):
            system_num = system_order[system_idx]
            system_config = systems[system_num]
            
            log_message = f"\n============================================================\nSISTEMA {system_num} ({system_idx+1}/{len(systems)})\n============================================================"
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            system_info = f"Info del sistema: {len(system_config['arrivals'])} vehículos, {system_config['n_spots']} plazas, {len(system_config['chargers'])} cargadores"
            print(system_info)
            log_file.write(f"{system_info}\n")
            
            # Crear entorno avanzado (usar EVChargingEnv en lugar de EnhancedEVChargingEnv)
            print(f"Creando entorno para sistema {system_num}...")
            env = EVChargingEnv(system_config)
            print(f"Entorno creado")
            
            agent.epsilon = config.get("epsilon", 0.9)
            print(f"Epsilon reiniciado a {agent.epsilon}")
            
            # Comenzar desde el episodio actual o desde 0
            start_episode = current_episode if system_idx == current_system_idx else 0
            
            log_message = f"Ejecutando episodios {start_episode+1}-{episodes_per_system}..."
            print(log_message)
            log_file.write(f"{log_message}\n")
            
            # Variables para early stopping
            best_reward = -float('inf')
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, f"best_model_sys_{system_idx}.pt")  # Cambio .h5 -> .pt
            
            # Variables para estadísticas del sistema
            system_start_time = time.time()
            episode_times = []
            rewards_history = []
            satisfaction_history = []
            
            # Lista para el logger
            episodes_data = []
            
            for e in range(start_episode, episodes_per_system):
                episode_start_time = time.time()
                
                print(f"\nEpisodio {e+1}/{episodes_per_system} (Sistema {system_num})")
                print(f"   Epsilon actual: {agent.epsilon:.4f}")
                print(f"   Memoria: {len(agent.memory)}/{agent.memory.maxlen}")
                
                state = env.reset()
                total_reward = 0
                done = False
                step_count = 0
                
                # Contadores para diagnóstico
                skip_count = 0
                assign_count = 0
                decisions_made = 0
                total_vehicles = len(env.arrivals)
                
                print(f"   Procesando {total_vehicles} vehículos...", end="", flush=True)
                
                while not done:
                    # Mostrar progreso cada 20 vehículos
                    if decisions_made > 0 and decisions_made % 20 == 0:
                        progress = (decisions_made / total_vehicles) * 100
                        print(f" {progress:.0f}%", end="", flush=True)
                    
                    possible_actions = env._get_possible_actions(state)
                    
                    if len(possible_actions) == 0:
                        print(f"\n   No hay acciones posibles para el vehículo actual")
                        break
                        
                    # Solo mostrar detalles de acción cada 50 vehículos para no saturar
                    verbose_action = (decisions_made % 50 == 0)
                    action = agent.act(state, possible_actions, verbose=verbose_action)
                    
                    if action == -1:
                        print(f"\n   Agente no pudo seleccionar acción válida")
                        break
                        
                    selected_action = possible_actions[action]

                    # Comprueba el VALOR de la clave 'action' para ver si es una acción de no-asignación.
                    # Usamos .get() para más seguridad y comprobamos todas las acciones que no son una asignación directa.
                    non_assignment_actions = ["skip", "reject", "strategic_reject", "continue_waiting"]

                    if selected_action.get("action") in non_assignment_actions:
                        skip_count += 1
                    else:
                        assign_count += 1
                        
                    next_state, reward, done = env.step(action)
                    info = {'action_type': selected_action.get('action', 'unknown')}

                    agent.remember(state, action, reward, next_state, done, info=info)
                    
                    state = next_state
                    total_reward += reward
                    step_count += 1
                    decisions_made += 1
                    
                    # Entrenar con experiencia con mayor frecuencia
                    if len(agent.memory) > batch_size and decisions_made % 10 == 0:
                        agent.replay()
                
                print(f" Completado")
                
                # Métricas de satisfacción (usar el método correcto)
                energy_metrics = env.get_performance_metrics()

                satisfaction_pct = energy_metrics["total_satisfaction_pct"]
                
                episode_time = time.time() - episode_start_time
                episode_times.append(episode_time)
                rewards_history.append(total_reward)
                satisfaction_history.append(satisfaction_pct)
                
                # Guardar datos del episodio para el logger
                episode_data = create_episode_data(
                    e + 1, total_reward, energy_metrics, assign_count, skip_count,
                    total_vehicles, episode_time, agent.epsilon
                )
                episodes_data.append(episode_data)
                
                # Calcular tendencias
                avg_reward_last_3 = np.mean(rewards_history[-3:]) if len(rewards_history) >= 3 else total_reward
                avg_satisfaction_last_3 = np.mean(satisfaction_history[-3:]) if len(satisfaction_history) >= 3 else satisfaction_pct
                
                # Estadísticas del episodio
                log_message = (f"Episodio {e+1} completado:\n"
                             f"   Recompensa: {total_reward:.2f} (promedio últimos 3: {avg_reward_last_3:.2f})\n"
                             f"   Satisfacción energética: {satisfaction_pct:.1f}% (promedio últimos 3: {avg_satisfaction_last_3:.1f}%)\n"
                             f"   Vehículos asignados: {assign_count}/{total_vehicles} ({assign_count/total_vehicles*100:.1f}%)\n"
                             f"   Vehículos saltados: {skip_count}/{total_vehicles} ({skip_count/total_vehicles*100:.1f}%)\n"
                             f"   Tiempo: {episode_time:.1f}s\n"
                             f"   Pasos: {step_count}")
                
                print(log_message)
                log_file.write(f"{log_message}\n")
                
                # Entrenar intensivamente al final del episodio
                if len(agent.memory) > batch_size:
                    print(f"   Entrenando red neuronal...")
                    for _ in range(5):  # Múltiples entrenamientos al final
                        agent.replay()
                
                # Guardar checkpoint periódicamente
                if (e + 1) % checkpoint_frequency == 0 or e == episodes_per_system - 1:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_sys_{system_idx}_ep_{e+1}.pt")  # Cambio .h5 -> .pt
                    print(f"Guardando checkpoint: {checkpoint_path}")
                    agent.save(checkpoint_path)
                    
                    # También actualizar modelo principal
                    agent.save(model_path)
                    log_file.write(f"Checkpoint guardado: {checkpoint_path}\n")
                    log_file.flush()
                
                # Early stopping basado en recompensa Y satisfacción
                avg_reward_last_5 = np.mean(rewards_history[-5:]) if len(rewards_history) >= 5 else total_reward
                avg_satisfaction_last_5 = np.mean(satisfaction_history[-5:]) if len(satisfaction_history) >= 5 else satisfaction_pct
                combined_metric = avg_reward_last_5 + avg_satisfaction_last_5

                improvement_threshold = 10.0
                NEGATIVE_REWARD_THRESHOLD = -5000
                
                if combined_metric > best_reward + improvement_threshold:
                    best_reward = combined_metric
                    no_improvement_count = 0
                    agent.save(best_model_path)
                    improvement_msg = f"Nuevo mejor modelo! Métrica combinada: {combined_metric:.2f}"
                    print(improvement_msg)
                    log_file.write(f"{improvement_msg}\n")
                else:
                    no_improvement_count += 1
                    patience_msg = f"Sin mejora. Paciencia: {no_improvement_count}/{patience}"
                    print(patience_msg)
                    log_file.write(f"{patience_msg}\n")

                    if len(rewards_history) >= 5 and all(r < NEGATIVE_REWARD_THRESHOLD for r in rewards_history[-5:]):
                        early_stop_msg = "Detención anticipada: todas las recompensas recientes son muy negativas."
                        print(early_stop_msg)
                        log_file.write(f"{early_stop_msg}\n")
                        break

                    if no_improvement_count >= patience:
                        early_stop_msg = f"Early stopping tras {patience} episodios sin mejora"
                        print(early_stop_msg)
                        log_file.write(f"{early_stop_msg}\n")
                        agent.load(best_model_path)
                        print(f"Cargado mejor modelo desde {best_model_path}")
                        break
                
                total_episodes_completed += 1
                
                # Estimación de tiempo restante
                if len(episode_times) > 2:
                    avg_episode_time = np.mean(episode_times[-3:])
                    remaining_episodes = (len(systems) - system_idx - 1) * episodes_per_system + (episodes_per_system - e - 1)
                    estimated_remaining = remaining_episodes * avg_episode_time
                    
                    eta_msg = f"ETA: {estimated_remaining/60:.1f} minutos restantes"
                    print(eta_msg)
            
            # Guardar progreso del sistema
            system_info_for_logger = {
                "total_vehicles": len(system_config["arrivals"]),
                "n_spots": system_config["n_spots"],
                "n_chargers": len(system_config["chargers"])  # Acceso directo, no parking_config
            }
            progress_logger.save_system_progress(system_num, episodes_data, system_info_for_logger)
            
            # Estadísticas del sistema completado
            system_time = time.time() - system_start_time
            avg_reward = np.mean(rewards_history) if rewards_history else 0
            avg_satisfaction = np.mean(satisfaction_history) if satisfaction_history else 0
            best_satisfaction = max(satisfaction_history) if satisfaction_history else 0
            
            system_summary = f"""
RESUMEN SISTEMA {system_num}:
   Tiempo total: {system_time/60:.1f} minutos
   Recompensa promedio: {avg_reward:.2f}
   Mejor métrica combinada: {best_reward:.2f}
   Satisfacción promedio: {avg_satisfaction:.1f}%
   Mejor satisfacción: {best_satisfaction:.1f}%
   Episodios completados: {len(rewards_history)}
   Epsilon final: {agent.epsilon:.4f}
"""
            print(system_summary)
            log_file.write(f"{system_summary}\n")
            
            # Reset current_episode para el siguiente sistema
            current_episode = 0
    
    # Estadísticas finales
    total_time = time.time() - global_start_time
    final_summary = f"""
ENTRENAMIENTO COMPLETADO!
Tiempo total: {total_time/60:.1f} minutos
Episodios totales: {total_episodes_completed}
Memoria final: {len(agent.memory)} experiencias
Epsilon final: {agent.epsilon:.4f}
Modelo guardado en: {model_path}
"""
    print(final_summary)
    
    # Guardar modelo final
    print(f"\nGuardando modelo final en {model_path}...")
    agent.save(model_path)
    print(f"Modelo guardado exitosamente")
    
    return agent

def train_dqn_agent(agent: EnhancedDQNAgent, env: EVChargingEnv, num_episodes: int,
                    model_save_path: str = None, log_frequency: int = 100,
                    logger: SimpleEpisodeLogger = None, verbose: bool = False):
    """
    Función simplificada para entrenamiento de un solo sistema (compatibilidad).
    Usa la lógica del main anterior pero para un solo sistema.
    
    Args:
        verbose (bool): Si True, muestra información detallada. Si False, solo progreso básico.
    """
    if logger is None:
        logger = SimpleEpisodeLogger(log_frequency=log_frequency if verbose else num_episodes + 1) 

    # Solo mostrar info inicial si es verbose
    if verbose:
        print(f"\n--- Iniciando entrenamiento del DQN para {num_episodes} episodios ---")
        print(f"Hiperparámetros del agente: LR={agent.learning_rate}, Gamma={agent.gamma}, "
              f"Epsilon_decay={agent.epsilon_decay}, Batch_size={agent.batch_size}")

    best_reward = -float('inf')
    
    # Variables para progreso
    milestone_interval = max(1, num_episodes // 4)  # Mostrar cada 25%
    
    for episode in range(1, num_episodes + 1):
        start_time = time.time()
        
        state = env.reset()
        total_reward = 0
        done = False
        
        # Contadores para diagnóstico
        skip_count = 0
        assign_count = 0
        decisions_made = 0
        total_vehicles = len(env.arrivals)
        
        while not done:
            possible_actions = env._get_possible_actions(state)
            
            if len(possible_actions) == 0:
                break
                
            action = agent.act(state, possible_actions)
            
            if action == -1:
                break
                
            selected_action = possible_actions[action]

            # Comprueba el VALOR de la clave 'action' para ver si es una acción de no-asignación.
            # Usamos .get() para más seguridad y comprobamos todas las acciones que no son una asignación directa.
            non_assignment_actions = ["skip", "reject", "strategic_reject", "continue_waiting"]

            if selected_action.get("action") in non_assignment_actions:
                skip_count += 1
            else:
                assign_count += 1
                
            next_state, reward, done = env.step(action)
            info = {'action_type': selected_action.get('action', 'unknown')}

            agent.remember(state, action, reward, next_state, done, info=info)
    
            # Entrenar periódicamente
            if len(agent.memory) > agent.batch_size and decisions_made % 10 == 0:
                agent.replay()
            
            state = next_state
            total_reward += reward
            decisions_made += 1

        episode_time = time.time() - start_time

        # Obtener métricas finales del episodio
        metrics = env.get_performance_metrics()

        
        # Registrar el episodio usando el logger
        logger.log_episode(
            episode, total_reward, metrics,
            assign_count, skip_count,
            total_vehicles, episode_time, agent.epsilon
        )

        # Mostrar progreso solo en hitos (sin verbose) o según log_frequency (con verbose)
        if not verbose:
            if episode % milestone_interval == 0 or episode == num_episodes:
                progress = episode / num_episodes * 100
                print(f"        Entrenando: {progress:.0f}% ({episode}/{num_episodes}) - Reward: {total_reward:.0f}")

        # Guardar el modelo si es el mejor y se especificó una ruta
        if total_reward > best_reward:
            best_reward = total_reward
            if model_save_path:
                agent.save(model_save_path)
    
    # Solo mostrar resumen final si es verbose
    if verbose:
        print(f"--- Entrenamiento finalizado. Recompensa máxima: {best_reward:.2f} ---")
    
    return logger.get_episode_data()

def run_single_dqn_training(config_path: str, dqn_hyperparams: dict, num_episodes: int,
                            output_dir: str = "dqn_training_results"):
    """
    Función de utilidad para ejecutar el entrenamiento de un solo DQN.
    """
    print(f"Cargando configuración del sistema desde: {config_path}")
    system_config = load_system_config(config_path)
    
    # Preparar el entorno
    env = EVChargingEnv(system_config)
    
    # Preparar el agente
    state_size = env.state_size  # Usar la propiedad del entorno
    action_size = env.action_size  # Usar la propiedad del entorno
    agent = EnhancedDQNAgent(state_size, action_size, **dqn_hyperparams)
    
    # Directorio para guardar el modelo
    agent_model_dir = os.path.join(output_dir, "models")
    os.makedirs(agent_model_dir, exist_ok=True)
    model_filepath = os.path.join(agent_model_dir, f"dqn_model_ep{num_episodes}.pt")
    
    # Instanciar el logger para el entrenamiento
    logger = SimpleEpisodeLogger(log_frequency=100)

    # Entrenar
    print(f"Iniciando entrenamiento para {num_episodes} episodios...")
    episode_data_log = train_dqn_agent(
        agent, env, num_episodes, model_save_path=model_filepath, logger=logger
    )
    
    print("\n--- Resultados del Entrenamiento ---")
    if episode_data_log:
        last_episode_data = episode_data_log[-1]
        print(f"Recompensa Total del Último Episodio: {last_episode_data['reward']:.2f}")
        print(f"Satisfacción Energética Final: {last_episode_data['satisfaction_pct']:.2f}%")
        print(f"Vehículos Asignados: {last_episode_data['vehicles_assigned']} / {last_episode_data['total_vehicles']}")
    else:
        print("No se registraron datos de episodios.")

    # Guardar el log completo de episodios
    log_filepath = os.path.join(output_dir, "episode_log.json")
    logger.save_episode_data(log_filepath)

def train_dqn_agent_for_optimization(dqn_params: Dict, reward_weights: Dict, 
                                    system_config: Dict, episodes: int = 100) -> float:
    """Versión específica para Scatter Search que retorna fitness directo"""
    
    # Crear environment y agente
    env = EVChargingEnv(system_config)
    env.update_reward_weights(reward_weights)
    
    agent = EnhancedDQNAgent(**dqn_params)
    
    # Entrenar
    results = train_dqn_agent(agent, env, episodes)
    
    # Retornar fitness (promedio últimos 10 episodios)
    if len(results) >= 10:
        return np.mean([ep['reward'] for ep in results[-10:]])
    else:
        return np.mean([ep['reward'] for ep in results])
    
def evaluate_hyperparameter_config(dqn_params, reward_weights, system_config, episodes=50):
    """Evalúa UNA configuración de hiperparámetros entrenando un DQN completo"""
    
    # Crear environment con pesos específicos
    env = EVChargingEnv(system_config)
    env.update_reward_weights(reward_weights)
    
    # Crear agente con hiperparámetros específicos
    agent = EnhancedDQNAgent(**dqn_params)
    
    # Entrenar completamente
    results = train_dqn_agent(agent, env, episodes)
    
    # Retornar fitness (promedio últimos episodios)
    return np.mean([ep['reward'] for ep in results[-10:]])