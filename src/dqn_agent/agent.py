import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import pickle
import os

class DQNNetwork(nn.Module):
    """
    Red neuronal DQN mejorada con arquitectura más profunda y features adicionales.
    Optimizada para el nuevo environment con gestión completa del parqueadero.
    """
    def __init__(self, state_size, action_size, dueling=True, use_noisy=False):
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        self.use_noisy = use_noisy
        
        # Arquitectura más profunda para manejar la complejidad adicional
        # Primera capa con más neuronas para capturar las nuevas features
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Capas intermedias
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.15)
        
        # Skip connection para mejor flujo de gradientes
        self.skip_connection = nn.Linear(state_size, 64)
        
        if self.dueling:
            # Dueling DQN: separar value y advantage streams
            # Value stream
            self.value_stream = nn.Linear(64, 32)
            self.value_output = nn.Linear(32, 1)
            
            # Advantage stream (más grande para manejar más acciones)
            self.advantage_stream = nn.Linear(64, 32)
            self.advantage_output = nn.Linear(32, action_size)
        else:
            # DQN simple
            self.output = nn.Linear(64, action_size)
        
        # Noisy layers para mejor exploración (opcional)
        if self.use_noisy:
            self.register_buffer('noise_scale', torch.tensor(0.1))
    
    def forward(self, x):
        # Asegurar que x tenga la forma correcta
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Forward pass con skip connection
        # Capa 1
        h1 = F.relu(self.bn1(self.fc1(x)))
        
        # Capa 2
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout1(h2)
        
        # Capa 3
        h3 = F.relu(self.bn3(self.fc3(h2)))
        h3 = self.dropout2(h3)
        
        # Skip connection
        skip = self.skip_connection(x)
        h3 = h3 + skip  # Residual connection
        
        if self.dueling:
            # Dueling DQN
            value = F.relu(self.value_stream(h3))
            value = self.value_output(value)
            
            advantage = F.relu(self.advantage_stream(h3))
            advantage = self.advantage_output(advantage)
            
            # Combinar value y advantage
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            
            # Agregar ruido si está habilitado
            if self.use_noisy and self.training:
                noise = torch.randn_like(q_values) * self.noise_scale
                q_values = q_values + noise
            
            return q_values
        else:
            # DQN simple
            return self.output(h3)

class EnhancedDQNAgentPyTorch:
    """
    Agente DQN mejorado para el nuevo environment con gestión completa del parqueadero.
    Incluye mejoras como Prioritized Experience Replay y n-step returns.
    """
    def __init__(self, state_size, action_size, learning_rate=0.0005, 
                 gamma=0.95, epsilon=0.9, epsilon_min=0.05, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, target_update_freq=50,
                 dueling_network=True, use_per=True, use_noisy=False,
                 n_step=3, alpha=0.6, beta=0.4):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dueling_network = dueling_network
        self.use_per = use_per
        self.use_noisy = use_noisy
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        
        # Contadores
        self.target_update_counter = 0
        self.steps = 0
        
        # Detectar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent usando: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Crear redes neuronales mejoradas
        self.q_network = DQNNetwork(state_size, action_size, dueling_network, use_noisy).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, dueling_network, use_noisy).to(self.device)
        
        # Optimizer con configuración optimizada
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, eps=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Inicializar target network
        self.update_target_network()
        
        # Memory - usar Prioritized Experience Replay si está habilitado
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(memory_size, alpha=alpha, beta=beta)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # Buffer para n-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Memoria específica por tipo de acción para balance
        self.action_type_memories = defaultdict(lambda: deque(maxlen=2000))
        
        # Estadísticas para monitoreo
        self.action_counts = defaultdict(int)
        self.reward_history = deque(maxlen=100)
        
        print(f"Red neuronal creada con {sum(p.numel() for p in self.q_network.parameters())} parámetros")
        print(f"Configuración: Dueling={dueling_network}, PER={use_per}, Noisy={use_noisy}, N-step={n_step}")
    
    def update_target_network(self):
        """Actualiza el target network con soft update."""
        # Soft update con tau=1.0 (hard update)
        tau = 1.0
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """
        Almacena experiencia con soporte para n-step returns y priorización.
        info puede contener metadatos como tipo de acción.
        """
        # Registrar estadísticas
        self.reward_history.append(reward)
        if info and 'action_type' in info:
            self.action_counts[info['action_type']] += 1
        
        # Agregar a n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Si el buffer está lleno o episodio terminó, calcular n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Calcular n-step return
            n_step_return = 0
            for i in range(len(self.n_step_buffer)):
                n_step_return += (self.gamma ** i) * self.n_step_buffer[i][2]
            
            # Usar el primer estado y último next_state
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            # Crear experiencia n-step
            experience = (first_state, first_action, n_step_return, last_next_state, last_done)
            
            # Almacenar en memoria principal
            if self.use_per:
                # Calcular prioridad inicial (TD error se calculará después)
                self.memory.add(experience, priority=abs(n_step_return) + 1e-6)
            else:
                self.memory.append(experience)
            
            # Almacenar también por tipo de acción si hay info
            if info and 'action_type' in info:
                self.action_type_memories[info['action_type']].append(experience)
            
            # Si terminó el episodio, limpiar buffer
            if done:
                self.n_step_buffer.clear()
                
    def _get_action_type(self, action):
        if isinstance(action, dict):
            if 'action' in action:
                return action['action']
            elif len(action) == 1 and 'action' in list(action.values())[0]:
                return list(action.values())[0]['action']
            else:
                return 'combination'
        return 'unknown'
    
    def act(self, state, possible_actions, verbose=False):
        """
        Selecciona acción con epsilon-greedy mejorado y consideración del tipo de acción.
        """
        if len(possible_actions) == 0:
            if verbose:
                print("      No hay acciones posibles")
            return -1
        
        if np.random.rand() <= self.epsilon:
            action_types = []
            for action in possible_actions:
                if isinstance(action, dict):
                    if 'action' in action:
                        action_types.append(action['action'])
                    else:
                        action_types.append('combination')
                else:
                    action_types.append('unknown')
            
            action_counts = [self.action_counts[atype] + 1 for atype in action_types]
            probs = 1.0 / np.array(action_counts)
            probs = probs / probs.sum()
            
            action = np.random.choice(len(possible_actions), p=probs)
            
            if verbose:
                print(f"      Acción exploratoria: {action} (tipo: {action_types[action]}, "
                    f"epsilon: {self.epsilon:.3f})")
            return action
        
        try:
            state_vector = self._process_state(state)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            q_values_np = q_values.cpu().numpy().flatten()
            
            action_values = []
            for i, action in enumerate(possible_actions[:len(q_values_np)]):
                if i < len(q_values_np):
                    q_val = q_values_np[i]
                    action_type = self._get_action_type(action)
                    exploration_bonus = 1.0 / (self.action_counts[action_type] + 10)
                    action_values.append((i, q_val + exploration_bonus * 0.1))
            
            if not action_values:
                action = np.random.choice(len(possible_actions))
            else:
                action = max(action_values, key=lambda x: x[1])[0]
            
            if verbose:
                action_type = self._get_action_type(possible_actions[action])
                q_value = action_values[action][1] if action < len(action_values) else 0
                print(f"      Acción DQN: {action} (tipo: {action_type}, Q-value: {q_value:.3f})")
            
            return action
            
        except Exception as e:
            if verbose:
                print(f"      Error en act(): {e}")
            return np.random.choice(len(possible_actions))
    
    def _get_action_type(self, action):
        """Extrae el tipo de acción del diccionario de acción."""
        if isinstance(action, dict) and 'action' in action:
            return action['action']
        return 'unknown'
    
    def _process_state(self, state):
        """
        Procesa el estado del nuevo environment con todas las features adicionales.
        """
        if state is None:
            return np.zeros(self.state_size, dtype=np.float32)
        
        features = []
        
        try:
            # 1. Características del parqueadero (fijas)
            parking_features = state.get("parking_features", {})
            if isinstance(parking_features, dict):
                features.extend([
                    parking_features.get("total_occupancy_ratio", 0.0),
                    parking_features.get("charger_occupancy_ratio", 0.0),
                    parking_features.get("charger_availability_ratio", 0.0),
                    parking_features.get("waiting_spots_availability_ratio", 0.0),
                    parking_features.get("total_available_spots", 0) / 100.0,
                    parking_features.get("evs_charging", 0) / 50.0,
                    parking_features.get("evs_waiting_inside", 0) / 50.0,
                    parking_features.get("evs_outside", 0) / 50.0,
                    parking_features.get("transformer_usage_ratio", 0.0)
                ])
            else:
                features.extend([0.0] * 9)
            
            # 2. Características de cola (fijas)
            queue_features = state.get("queue_features", {})
            if isinstance(queue_features, dict):
                features.extend([
                    queue_features.get("queue_length", 0.0),
                    queue_features.get("avg_wait_time", 0.0),
                    queue_features.get("fairness_score", 1.0)
                ])
            else:
                features.extend([0.0, 0.0, 1.0])
            
            # 3. CAMBIO CRÍTICO: Características de vehículos (dinámicas)
            ev_features_batch = state.get("ev_features_batch", [])
            num_evs_actual = len(ev_features_batch)
            
            # NUEVO: Estadísticas agregadas de todos los vehículos
            if num_evs_actual > 0:
                # Agregar información agregada de TODOS los vehículos
                all_urgencies = []
                all_energy_needs = []
                all_wait_times = []
                
                for ev_features in ev_features_batch:
                    if len(ev_features) >= 7:
                        # Calcular urgencia aproximada: energía_requerida / tiempo_restante
                        time_remaining = max(0.01, ev_features[5])  # time_remaining_from_now
                        energy_delivered_ratio = ev_features[3]    # energy_delivered_ratio
                        energy_remaining = 1.0 - energy_delivered_ratio
                        urgency = energy_remaining / time_remaining
                        all_urgencies.append(urgency)
                        all_energy_needs.append(energy_remaining)
                        
                        if len(ev_features) >= 7:
                            all_wait_times.append(ev_features[6])  # wait_time
                
                # Estadísticas agregadas
                features.extend([
                    num_evs_actual / 50.0,  # Número de vehículos normalizado
                    np.mean(all_urgencies) if all_urgencies else 0.0,
                    np.std(all_urgencies) if len(all_urgencies) > 1 else 0.0,
                    np.max(all_urgencies) if all_urgencies else 0.0,
                    np.mean(all_energy_needs) if all_energy_needs else 0.0,
                    np.mean(all_wait_times) if all_wait_times else 0.0,
                ])
            else:
                features.extend([0.0] * 6)
            
            # 4. NUEVO: Características de los vehículos más críticos (Top-K)
            # En lugar de procesar TODOS individualmente, tomar los más importantes
            max_individual_evs = min(8, num_evs_actual)  # Máximo 8 vehículos individuales
            ev_features_per_vehicle = 12
            
            if num_evs_actual > 0:
                # Ordenar por urgencia (aproximada) para tomar los más críticos
                ev_with_urgency = []
                for i, ev_features in enumerate(ev_features_batch):
                    if len(ev_features) >= 6:
                        time_remaining = max(0.01, ev_features[5])
                        energy_delivered_ratio = ev_features[3]
                        urgency = (1.0 - energy_delivered_ratio) / time_remaining
                        ev_with_urgency.append((urgency, ev_features))
                
                # Tomar los más urgentes
                ev_with_urgency.sort(key=lambda x: x[0], reverse=True)
                top_evs = ev_with_urgency[:max_individual_evs]
                
                # Procesar vehículos individuales más importantes
                for urgency, ev_features in top_evs:
                    padded_features = ev_features[:ev_features_per_vehicle]
                    while len(padded_features) < ev_features_per_vehicle:
                        padded_features.append(0.0)
                    features.extend(padded_features)
                
                # Pad si hay menos vehículos que el máximo
                for i in range(len(top_evs), max_individual_evs):
                    features.extend([0.0] * ev_features_per_vehicle)
            else:
                # No hay vehículos, llenar con ceros
                for i in range(max_individual_evs):
                    features.extend([0.0] * ev_features_per_vehicle)
            
            # 5. Características del sistema (fijas)
            features.extend([
                state.get("total_occupancy_ratio", 0.0),
                state.get("charger_availability_ratio", 0.5),
                state.get("waiting_spots_availability_ratio", 0.5),
                state.get("queue_length", 0.0),
                state.get("avg_wait_time_current", 0.0)
            ])
            
            features.extend([
                state.get("system_type", 0) / 20.0,
                state.get("n_spots_total", 100) / 200.0,
                state.get("n_chargers_total", 10) / 50.0,
                state.get("transformer_limit", 50) / 200.0
            ])
            
            features.extend([
                state.get("current_time_idx", 0) / 100.0,
                state.get("current_time_normalized", 0.0)
            ])
            
            # 6. NUEVO: Información de carga de decisión
            num_evs_needing_decision = state.get("num_evs_needing_decision", 0)
            features.extend([
                num_evs_needing_decision / 50.0,  # Número absoluto normalizado
                1.0 if num_evs_needing_decision > 10 else 0.0,  # Indicador de alta carga
                1.0 if num_evs_needing_decision > 20 else 0.0,  # Indicador de carga extrema
                min(1.0, num_evs_needing_decision / 30.0)  # Factor de saturación
            ])
            
            # Convertir a array y ajustar tamaño
            features_array = np.array(features, dtype=np.float32)
            
            if len(features_array) < self.state_size:
                padding = np.zeros(self.state_size - len(features_array), dtype=np.float32)
                features_array = np.concatenate([features_array, padding])
            elif len(features_array) > self.state_size:
                features_array = features_array[:self.state_size]
            
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)
            
            return features_array
            
        except Exception as e:
            print(f"Error en _process_state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)
        
    
    
    def replay(self, beta=None):
        """
        Entrena la red con experiencia pasada usando las mejoras implementadas.
        """
        if self.use_per:
            if len(self.memory) < self.batch_size:
                return
            
            # Obtener batch con prioridades
            batch, indices, weights = self.memory.sample(self.batch_size, beta or 0.4)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return
            
            # Sampling balanceado: 70% uniforme, 30% de acciones específicas
            uniform_size = int(self.batch_size * 0.7)
            balanced_size = self.batch_size - uniform_size
            
            # Sample uniforme
            batch = random.sample(self.memory, uniform_size)
            
            # Sample balanceado por tipo de acción
            action_types = list(self.action_type_memories.keys())
            if action_types and balanced_size > 0:
                samples_per_type = balanced_size // len(action_types)
                for action_type in action_types:
                    if len(self.action_type_memories[action_type]) > 0:
                        type_samples = random.sample(
                            self.action_type_memories[action_type],
                            min(samples_per_type, len(self.action_type_memories[action_type]))
                        )
                        batch.extend(type_samples)
            
            # Pesos uniformes si no usamos PER
            weights = torch.ones(len(batch)).to(self.device)
            indices = None
        
        # Procesar batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            state_vector = self._process_state(state)
            next_state_vector = self._process_state(next_state)
            
            states.append(state_vector)
            actions.append(action)
            rewards.append(float(reward))
            next_states.append(next_state_vector)
            dones.append(done)
        
        # Convertir a tensores
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Calcular Q-values actuales
        self.q_network.train()
        current_q_values = self.q_network(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1))
        
        # Calcular Q-values target (Double DQN)
        with torch.no_grad():
            # Seleccionar acciones con main network
            next_q_values_main = self.q_network(next_states_tensor)
            next_actions = next_q_values_main.argmax(1)
            
            # Evaluar con target network
            next_q_values_target = self.target_network(next_states_tensor)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1))
            
            # Calcular targets con n-step return
            n_step_discount = self.gamma ** self.n_step
            target_q_values = rewards_tensor + (n_step_discount * next_q_values.squeeze() * ~dones_tensor)
        
        # Calcular TD errors para PER
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach()
        
        # Calcular loss con importance sampling weights
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        loss = (loss * weights).mean()
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping más agresivo para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Actualizar prioridades en PER
        if self.use_per and indices is not None:
            priorities = td_errors.cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Decrementar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Actualizar target network
        self.steps += 1
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.update_target_network()
            self.target_update_counter = 0
    
    def get_statistics(self):
        """Retorna estadísticas del agente para monitoreo."""
        stats = {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "memory_size": len(self.memory) if not self.use_per else len(self.memory),
            "action_distribution": dict(self.action_counts),
            "avg_reward_last_100": np.mean(self.reward_history) if self.reward_history else 0,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
        return stats
    
    def save(self, filepath):
        """Guarda el modelo y estadísticas."""
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
                'action_counts': dict(self.action_counts),
                'config': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'dueling': self.dueling_network,
                    'use_per': self.use_per,
                    'use_noisy': self.use_noisy,
                    'n_step': self.n_step
                }
            }, filepath)
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False
    
    def load(self, filepath):
        """Carga el modelo y estadísticas."""
        try:
            if not os.path.exists(filepath):
                return False
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            
            if 'action_counts' in checkpoint:
                self.action_counts = defaultdict(int, checkpoint['action_counts'])
            
            return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False


class PrioritizedReplayBuffer:
    """
    Buffer de replay con priorización para importance sampling.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, experience, priority=None):
        """Agrega experiencia con prioridad."""
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size, beta=None):
        """Samplea batch con importance sampling weights."""
        if beta is None:
            beta = self.beta
        
        size = len(self.buffer)
        
        # Calcular probabilidades
        priorities = self.priorities[:size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Samplear índices
        indices = np.random.choice(size, batch_size, p=probs)
        
        # Calcular importance sampling weights
        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Obtener experiencias
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Actualiza prioridades después del entrenamiento."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)