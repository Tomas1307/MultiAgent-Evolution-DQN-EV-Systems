import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

class EVChargingEnv:
    """
    Entorno mejorado de simulación para la carga de vehículos eléctricos con RL.
    
    Principales mejoras:
    - Considera TODOS los spots del parqueadero (con y sin cargador)
    - Permite admitir EVs a spots de espera
    - Sistema de recompensas balanceado
    - Gestión completa del parqueadero (no solo cargadores)
    """
    
    def __init__(self, config):
        """Inicializa el entorno con la configuración del sistema."""
        # Constantes de recompensa y penalización
        self.REWARD_ADMIT_AND_CHARGE = 50.0
        self.REWARD_ADMIT_TO_WAIT = 20.0
        self.REWARD_COMPLETE_CHARGE = 30.0
        self.PENALTY_REJECT_CAPACITY = -5.0
        self.PENALTY_REJECT_STRATEGIC = -15.0
        self.ENERGY_COST_WEIGHT = 0.5
        self.EFFICIENCY_BONUS_WEIGHT = 10.0
        self.FAIRNESS_BONUS_WEIGHT = 5.0
        self.REWARD_FREE_UP_CHARGER = 25.0
        self.PENALTY_BLOCKING_CHARGER = -15.0
        
        # Configuración e inicialización del entorno
        self.config = config
        self.process_config()
        self.reset()
        
        # --- INICIO DEL CÁLCULO ROBUSTO DE STATE_SIZE ---
        
        # 1. Importar la clase del agente localmente para evitar dependencias circulares a nivel de módulo.
        from .agent import EnhancedDQNAgentPyTorch as EnhancedDQNAgent

        # 2. Crear un "estado ficticio" (dummy_state) que contenga todas las claves
        #    posibles que la función _process_state del agente espera encontrar.
        dummy_state = {
            "ev_features": [0.0] * 15,
            "parking_features": {
                "total_occupancy_ratio": 0.5, "charger_occupancy_ratio": 0.5,
                "charger_availability_ratio": 0.5, "waiting_spots_availability_ratio": 0.5,
                "total_available_spots": self.n_spots, "evs_charging": 0,
                "evs_waiting_inside": 0, "evs_outside": 0, "transformer_usage_ratio": 0.5
            },
            "queue_features": {
                "queue_length": 0.0, "position_in_queue": -1.0,
                "avg_wait_time": 0.0, "fairness_score": 1.0
            },
            "system_type": 0, "n_spots_total": self.n_spots, "n_chargers_total": self.n_charger_spots,
            "transformer_limit": self.station_limit, "current_time_idx": 0,
            "current_time_normalized": 0.0, "representative_ev": "dummy_ev",
            "ev_current_status": "outside", "ev_current_location": "outside"
        }

        # 3. Crear un agente temporal con un state_size GRANDE y permisivo (ej. 256)
        #    para asegurar que la función _process_state no trunque el vector de salida.
        temp_agent_for_sizing = EnhancedDQNAgent(state_size=256, action_size=100)
        print(f"##############################: {temp_agent_for_sizing}")
        # 4. Calcular el tamaño real y definitivo del vector de estado usando el agente
        #    temporal y el estado ficticio.
        self.state_size = len(temp_agent_for_sizing._process_state(dummy_state))
        print(f"##############################: {self.state_size}")
        # 5. El tamaño de la acción puede ser un valor fijo y grande, ya que el agente
        #    solo elige de la lista de acciones posibles que genera el entorno.
        self.action_size = 100 
        
        # --- FIN DEL CÁLCULO ROBUSTO ---

        print(f"Environment initialized with State Size: {self.state_size} and Action Size: {self.action_size}")
        
    
    def process_config(self):
        """Procesa la configuración para extraer parámetros."""
        # Mantener toda la lógica actual de process_config
        self.times = self.config["times"]
        self.prices = self.config["prices"]
        self.arrivals = self.config["arrivals"]
        self.chargers = self.config["parking_config"]["chargers"]
        self.station_limit = self.config["parking_config"]["transformer_limit"]
        self.dt = self.config.get("dt", 0.25)
        self.n_spots = self.config["parking_config"]["n_spots"]
        self.test_number = self.config.get("test_number", 0)
        
        # NUEVO: Identificar spots con y sin cargador
        self.n_charger_spots = len(self.chargers)
        self.n_waiting_spots = self.n_spots - self.n_charger_spots
        self.charger_to_spot = {c["charger_id"]: i for i, c in enumerate(self.chargers)}
        self.spot_to_charger = {i: c for c, i in self.charger_to_spot.items()}
        # Procesar información de vehículos (mantener lógica actual)
        self.has_brand_info = any('brand' in ev for ev in self.arrivals)
        self.has_battery_info = any('battery_capacity' in ev for ev in self.arrivals)
        self.has_priority_info = any('priority' in ev for ev in self.arrivals)
        self.has_willingness_info = any('willingness_to_pay' in ev for ev in self.arrivals)
        self.has_efficiency_info = any('efficiency' in ev for ev in self.arrivals)
        self.has_charge_rate_info = any('min_charge_rate' in ev for ev in self.arrivals)
        
        # Normalización de precios
        self.min_price = min(self.prices)
        self.max_price = max(self.prices)
        self.normalized_prices = [(p - self.min_price) / (self.max_price - self.min_price + 1e-6) 
                                 for p in self.prices]
        
        # Estadísticas del sistema
        self.max_charger_power = max(c["power"] for c in self.chargers)
        self.total_charging_capacity = sum(c["power"] for c in self.chargers)
        self.avg_required_energy = np.mean([arr["required_energy"] for arr in self.arrivals])
        self.max_required_energy = max([arr["required_energy"] for arr in self.arrivals])
        self.avg_stay_duration = np.mean([(arr["departure_time"] - arr["arrival_time"]) 
                                         for arr in self.arrivals])
        
        # Mapeos básicos
        self.ev_ids = [arr["id"] for arr in self.arrivals]
        self.arrival_time = {arr["id"]: arr["arrival_time"] for arr in self.arrivals}
        self.departure_time = {arr["id"]: arr["departure_time"] for arr in self.arrivals}
        self.required_energy = {arr["id"]: arr["required_energy"] for arr in self.arrivals}
        
        # Procesar información adicional de EVs
        self._process_ev_advanced_features()
        
        # Información de cargadores
        self._process_charger_info()
        
    def _process_ev_advanced_features(self):
        """Procesa características avanzadas de los EVs."""
        if self.has_battery_info:
            self.battery_capacity = {arr["id"]: arr.get("battery_capacity", 40) 
                                   for arr in self.arrivals}
        
        if self.has_brand_info:
            self.brands = {arr["id"]: arr.get("brand", "Unknown") for arr in self.arrivals}
            unique_brands = list(set(self.brands.values()))
            self.brand_to_id = {brand: i/max(1, len(unique_brands)) 
                               for i, brand in enumerate(unique_brands)}
        
        if self.has_priority_info:
            self.priority = {arr["id"]: arr.get("priority", 1) for arr in self.arrivals}
            self.max_priority = max(self.priority.values())
        
        if self.has_willingness_info:
            self.willingness_to_pay = {arr["id"]: arr.get("willingness_to_pay", 1.0) 
                                     for arr in self.arrivals}
            
        if self.has_efficiency_info:
            self.efficiency = {arr["id"]: arr.get("efficiency", 0.9) for arr in self.arrivals}
            
        if self.has_charge_rate_info:
            self.min_charge_rate = {arr["id"]: arr.get("min_charge_rate", 3.5) 
                                   for arr in self.arrivals}
            self.max_charge_rate = {arr["id"]: arr.get("max_charge_rate", 50) 
                                   for arr in self.arrivals}
            self.ac_charge_rate = {arr["id"]: arr.get("ac_charge_rate", 7) 
                                  for arr in self.arrivals}
            self.dc_charge_rate = {arr["id"]: arr.get("dc_charge_rate", 50) 
                                  for arr in self.arrivals}
    
    def _process_charger_info(self):
        """Procesa información de cargadores y compatibilidad."""
        self.charger_ids = [c["charger_id"] for c in self.chargers]
        self.max_charger_power_dict = {c["charger_id"]: c["power"] for c in self.chargers}
        
        self.charger_type = {}
        self.compatible_vehicles = {}
        
        for c in self.chargers:
            charger_id = c["charger_id"]
            self.charger_type[charger_id] = c.get("type", "AC")
            
            if "compatible_vehicles" in c:
                self.compatible_vehicles[charger_id] = c["compatible_vehicles"]
            else:
                self.compatible_vehicles[charger_id] = list(set(self.brands.values())) if self.has_brand_info else ["All"]
        
        # Mapeo de compatibilidad EV-Cargador
        self.ev_charger_compatible = {}
        for ev_id in self.ev_ids:
            self.ev_charger_compatible[ev_id] = []
            
            if self.has_brand_info:
                ev_brand = self.brands.get(ev_id, "")
                
                for charger_id in self.charger_ids:
                    compatible = True
                    if ev_brand and charger_id in self.compatible_vehicles:
                        base_brand = ev_brand.split()[0] if " " in ev_brand else ev_brand
                        compatible = False
                        for comp_brand in self.compatible_vehicles[charger_id]:
                            if comp_brand in ev_brand or base_brand in comp_brand:
                                compatible = True
                                break
                    
                    if compatible:
                        charger_power = self.max_charger_power_dict[charger_id]
                        charger_type = self.charger_type.get(charger_id, "AC")
                        
                        if self.has_charge_rate_info:
                            if charger_type == "AC" and charger_power <= self.ac_charge_rate.get(ev_id, 7):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                            elif charger_type == "DC" and charger_power <= self.dc_charge_rate.get(ev_id, 50):
                                self.ev_charger_compatible[ev_id].append(charger_id)
                        else:
                            self.ev_charger_compatible[ev_id].append(charger_id)
            else:
                self.ev_charger_compatible[ev_id] = self.charger_ids
    
    def reset(self):
        """Reinicia el entorno al estado inicial."""
        # Estado temporal del entorno
        self.current_time_idx = 0
        self.evs_processed = set()
        
        # NUEVO: Estados y ubicaciones de EVs
        self.ev_status = {}  # 'outside', 'waiting_inside', 'charging', 'charged_waiting', 'rejected'
        self.ev_location = {}  # spot_id o 'outside'
        self.ev_admission_time = {}  # Cuándo entró al parqueadero
        self.ev_charge_start_time = {}  # Cuándo empezó a cargar
        self.waiting_queue = deque()  # EVs esperando cargador dentro del parqueadero
        
        # Estados del entorno (actualizado para todos los spots)
        self.all_spots_occupied = {t: set() for t in range(len(self.times))}  # NUEVO
        self.charger_spots_occupied = {t: set() for t in range(len(self.times))}  # Renombrado
        self.occupied_chargers = {t: set() for t in range(len(self.times))}
        self.power_used = {t: 0 for t in range(len(self.times))}
        self.energy_delivered = {ev_id: 0 for ev_id in self.ev_ids}
        
        # Schedule de carga
        self.charging_schedule = []
        
        # Métricas para tracking
        self.rejection_reasons = defaultdict(int)
        self.total_wait_times = {}
        self.fairness_scores = []
        
        # Inicializar todos los EVs como 'outside'
        for ev_id in self.ev_ids:
            self.ev_status[ev_id] = 'outside'
            self.ev_location[ev_id] = 'outside'
        
        return self._get_state()
    
    def _get_state(self):
        """
        Obtiene el estado actual del entorno mejorado.
        Incluye información sobre ocupación total del parqueadero y colas internas.
        """
        # Verificar inicialización
        if not hasattr(self, 'current_time_idx'):
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        # Auto-skip de períodos sin vehículos (mantener lógica actual)
        max_skips = len(self.times)
        skips_made = 0
        
        while self.current_time_idx < len(self.times) and skips_made < max_skips:
            current_time = self.times[self.current_time_idx]
            
            # Obtener vehículos que necesitan decisión (outside o waiting_inside)
            evs_needing_decision = []
            for ev_id in self.ev_ids:
                if (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id] and
                    self.ev_status[ev_id] in ['outside', 'waiting_inside'] and
                    ev_id not in self.evs_processed):
                    
                    energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
                    if energy_needed > 0.01:
                        evs_needing_decision.append(ev_id)
            
            if evs_needing_decision:
                break
            
            self.current_time_idx += 1
            skips_made += 1
        
        if self.current_time_idx >= len(self.times):
            return None
        
        # Seleccionar EV representativo (priorizar los que están afuera esperando)
        evs_outside = [ev for ev in evs_needing_decision if self.ev_status[ev] == 'outside']
        evs_waiting = [ev for ev in evs_needing_decision if self.ev_status[ev] == 'waiting_inside']
        
        representative_ev = self._select_representative_ev(evs_outside if evs_outside else evs_waiting)
        if representative_ev is None:
            self.current_time_idx += 1
            return self._get_state()
        
        ev_id = representative_ev
        current_time = self.times[self.current_time_idx]
        
        # Calcular features del EV (mantener lógica actual)
        ev_features = self._calculate_ev_features(ev_id, current_time)
        
        # NUEVO: Calcular features del parqueadero completo
        parking_features = self._calculate_parking_features(current_time)
        
        # NUEVO: Calcular features de cola y fairness
        queue_features = self._calculate_queue_features(ev_id, current_time)
        
        # Construir estado completo
        state = {
            "ev_features": ev_features,
            "parking_features": parking_features,
            "queue_features": queue_features,
            
            # Features agregadas del sistema
            "total_occupancy_ratio": parking_features["total_occupancy_ratio"],
            "charger_availability_ratio": parking_features["charger_availability_ratio"],
            "waiting_spots_availability_ratio": parking_features["waiting_spots_availability_ratio"],
            "queue_length": queue_features["queue_length"],
            "ev_position_in_queue": queue_features["position_in_queue"],
            "avg_wait_time_current": queue_features["avg_wait_time"],
            
            # Información del sistema
            "system_type": self.test_number,
            "n_spots_total": self.n_spots,
            "n_chargers_total": len(self.charger_ids),
            "transformer_limit": self.station_limit,
            
            # Información temporal
            "current_time_idx": self.current_time_idx,
            "current_time_normalized": self.current_time_idx / len(self.times),
            
            # EV representativo y su estado
            "representative_ev": ev_id,
            "ev_current_status": self.ev_status[ev_id],
            "ev_current_location": self.ev_location[ev_id]
        }
        
        return state
    
    def _calculate_ev_features(self, ev_id, current_time):
        """Calcula las características del EV (mantiene lógica actual)."""
        stay_duration = self.departure_time[ev_id] - self.arrival_time[ev_id]
        energy_requirement_normalized = self.required_energy[ev_id] / self.max_required_energy
        stay_duration_normalized = stay_duration / max(self.times)
        energy_delivered_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id] if self.required_energy[ev_id] > 0 else 0
        time_remaining_from_now = (self.departure_time[ev_id] - current_time) / max(self.times)
        
        # NUEVO: Tiempo esperando (si está dentro)
        wait_time = 0
        if self.ev_status[ev_id] == 'waiting_inside' and ev_id in self.ev_admission_time:
            wait_time = (current_time - self.ev_admission_time[ev_id]) / max(self.times)
        
        ev_features = [
            self.arrival_time[ev_id] / max(self.times),
            self.departure_time[ev_id] / max(self.times),
            energy_requirement_normalized,
            energy_delivered_ratio,
            self.current_time_idx / len(self.times),
            time_remaining_from_now,
            wait_time  # NUEVO
        ]
        
        # Agregar features adicionales si están disponibles
        if self.has_battery_info:
            ev_features.append(self.battery_capacity[ev_id] / 100.0)
            
        if self.has_priority_info:
            ev_features.append(self.priority[ev_id] / self.max_priority)
            
        if self.has_willingness_info:
            ev_features.append(self.willingness_to_pay[ev_id] / 1.5)
        
        return ev_features
    
    def _calculate_parking_features(self, current_time):
        """NUEVO: Calcula características del estado del parqueadero completo."""
        current_idx = self.current_time_idx
        
        # Ocupación total
        total_occupied = len(self.all_spots_occupied[current_idx])
        total_occupancy_ratio = total_occupied / self.n_spots
        
        # Ocupación de spots con cargador
        charger_spots_occupied = len(self.charger_spots_occupied[current_idx])
        charger_occupancy_ratio = charger_spots_occupied / self.n_charger_spots if self.n_charger_spots > 0 else 1.0
        
        # Disponibilidad
        total_available = self.n_spots - total_occupied
        charger_spots_available = self.n_charger_spots - charger_spots_occupied
        waiting_spots_available = self.n_waiting_spots - (total_occupied - charger_spots_occupied)
        
        # EVs por estado
        evs_charging = len([ev for ev in self.ev_status if self.ev_status[ev] == 'charging'])
        evs_waiting_inside = len([ev for ev in self.ev_status if self.ev_status[ev] == 'waiting_inside'])
        evs_outside = len([ev for ev in self.ev_status if self.ev_status[ev] == 'outside'])
        
        return {
            "total_occupancy_ratio": total_occupancy_ratio,
            "charger_occupancy_ratio": charger_occupancy_ratio,
            "charger_availability_ratio": charger_spots_available / self.n_charger_spots if self.n_charger_spots > 0 else 0,
            "waiting_spots_availability_ratio": waiting_spots_available / self.n_waiting_spots if self.n_waiting_spots > 0 else 0,
            "total_available_spots": total_available,
            "evs_charging": evs_charging,
            "evs_waiting_inside": evs_waiting_inside,
            "evs_outside": evs_outside,
            "transformer_usage_ratio": self.power_used[current_idx] / self.station_limit
        }
    
    def _calculate_queue_features(self, ev_id, current_time):
        """NUEVO: Calcula características relacionadas con colas y fairness."""
        # Longitud de la cola
        queue_length = len(self.waiting_queue)
        
        # Posición en la cola (si está esperando)
        position_in_queue = -1
        if ev_id in self.waiting_queue:
            position_in_queue = list(self.waiting_queue).index(ev_id) / max(1, queue_length)
        
        # Tiempo promedio de espera de los que están esperando
        waiting_times = []
        for waiting_ev in self.waiting_queue:
            if waiting_ev in self.ev_admission_time:
                wait_time = current_time - self.ev_admission_time[waiting_ev]
                waiting_times.append(wait_time)
        
        avg_wait_time = np.mean(waiting_times) / max(self.times) if waiting_times else 0
        
        # Fairness score (basado en prioridades y tiempos de espera)
        fairness_score = self._calculate_current_fairness()
        
        return {
            "queue_length": queue_length / max(1, self.n_spots),
            "position_in_queue": position_in_queue,
            "avg_wait_time": avg_wait_time,
            "fairness_score": fairness_score
        }
    
    def _calculate_current_fairness(self):
        """Calcula un score de fairness basado en la distribución actual de recursos."""
        if not self.ev_status:
            return 1.0
        
        # Calcular fairness basado en prioridad vs servicio recibido
        priority_service_ratios = []
        
        for ev_id in self.ev_ids:
            if self.ev_status[ev_id] in ['charging', 'charged_waiting']:
                priority = self.priority.get(ev_id, 1) if self.has_priority_info else 1
                service_ratio = self.energy_delivered[ev_id] / self.required_energy[ev_id]
                priority_service_ratios.append(service_ratio / priority)
        
        if not priority_service_ratios:
            return 1.0
        
        # Calcular coeficiente de variación (menor = más fair)
        mean_ratio = np.mean(priority_service_ratios)
        std_ratio = np.std(priority_service_ratios)
        cv = std_ratio / mean_ratio if mean_ratio > 0 else 0
        
        # Convertir a score 0-1 (1 = perfectamente fair)
        fairness_score = 1.0 / (1.0 + cv)
        
        return fairness_score
    
    def _get_possible_actions(self, state):
        """
        Determina las acciones posibles con el nuevo sistema de admisión.
        Ahora considera admitir a spots de espera y gestión completa del parqueadero.
        """
        if state is None:
            return []
        
        ev_id = state["representative_ev"]
        ev_status = state["ev_current_status"]
        current_time_idx = state["current_time_idx"]
        current_time = self.times[current_time_idx]
        
        # Verificar que el EV está presente
        if not (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id]):
            return [{"action": "skip"}]
        
        # Verificar energía necesaria
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        if energy_needed <= 0.01:
            return [{"action": "skip"}]
        
        actions = []
        
        # Obtener ocupación actual
        total_occupied = len(self.all_spots_occupied[current_time_idx])
        
        # Si el parqueadero está lleno, solo podemos rechazar
        if total_occupied >= self.n_spots and ev_status == 'outside':
            return [{
                "action": "reject",
                "reason": "parking_full",
                "ev_id": ev_id
            }]
        
        # Si el EV está afuera, puede ser admitido o rechazado estratégicamente
        if ev_status == 'outside':
            # Opción 1: Admitir y cargar (si hay cargador disponible)
            available_charger_spots = self._get_available_charger_spots(current_time_idx)
            
            for spot_id in available_charger_spots[:2]:  # Limitar opciones
                charger_id = self.spot_to_charger.get(spot_id)
                if charger_id and charger_id in self.ev_charger_compatible.get(ev_id, []):
                    # Verificar capacidad del transformador
                    charger_power = self.max_charger_power_dict[charger_id]
                    if self.power_used[current_time_idx] + charger_power <= self.station_limit:
                        actions.append({
                            "action": "admit_and_charge",
                            "ev_id": ev_id,
                            "spot": spot_id,
                            "charger": charger_id,
                            "power": min(charger_power, self.max_charge_rate.get(ev_id, 50))
                        })
            
            # Opción 2: Admitir a spot de espera (si no hay cargadores)
            if len(actions) == 0 and total_occupied < self.n_spots:
                available_waiting_spots = self._get_available_waiting_spots(current_time_idx)
                
                for spot_id in available_waiting_spots[:2]:  # Limitar opciones
                    estimated_wait = self._estimate_wait_time_for_charger(ev_id, current_time)
                    actions.append({
                        "action": "admit_and_wait",
                        "ev_id": ev_id,
                        "spot": spot_id,
                        "estimated_wait": estimated_wait
                    })
            
            # Opción 3: Rechazo estratégico (siempre disponible)
            actions.append({
                "action": "strategic_reject",
                "ev_id": ev_id,
                "reason": "optimization"
            })
        
        # Si el EV está esperando dentro, puede ser movido a cargador
        elif ev_status == 'waiting_inside':
            available_charger_spots = self._get_available_charger_spots(current_time_idx)
            
            for spot_id in available_charger_spots[:1]:  # Solo la mejor opción
                charger_id = self.spot_to_charger.get(spot_id)
                if charger_id and charger_id in self.ev_charger_compatible.get(ev_id, []):
                    charger_power = self.max_charger_power_dict[charger_id]
                    if self.power_used[current_time_idx] + charger_power <= self.station_limit:
                        actions.append({
                            "action": "move_to_charger",
                            "ev_id": ev_id,
                            "from_spot": self.ev_location[ev_id],
                            "to_spot": spot_id,
                            "charger": charger_id,
                            "power": min(charger_power, self.max_charge_rate.get(ev_id, 50))
                        })
            
            # Opción de seguir esperando
            actions.append({
                "action": "continue_waiting",
                "ev_id": ev_id
            })
            
        elif ev_status == 'charged_waiting':
            # ¡El EV representativo es un bloqueador!
            # La única acción útil es moverlo si hay espacio de espera.

            available_waiting_spots = self._get_available_waiting_spots(current_time_idx)
            if available_waiting_spots:
                # Proponer moverlo al primer spot de espera disponible para simplificar
                to_spot = available_waiting_spots[0]
                actions.append({
                    "action": "move_to_wait",
                    "ev_id": ev_id,
                    "from_spot": self.ev_location[ev_id], # El cargador que ocupa
                    "to_spot": to_spot
                })

            # Si no hay a dónde moverse, la única opción es que siga esperando (skip)
            actions.append({"action": "skip"})
        
        # Si no hay acciones, skip
        if len(actions) == 0:
            actions.append({"action": "skip"})
        
        return actions
    
    def _get_available_charger_spots(self, time_idx):
        """Obtiene spots con cargador disponibles."""
        occupied = self.charger_spots_occupied[time_idx]
        return [i for i in range(self.n_charger_spots) if i not in occupied]
    
    def _get_available_waiting_spots(self, time_idx):
        """Obtiene spots de espera disponibles."""
        total_occupied = self.all_spots_occupied[time_idx]
        # Los spots de espera empiezan después de los spots con cargador
        waiting_spots = []
        for i in range(self.n_charger_spots, self.n_spots):
            if i not in total_occupied:
                waiting_spots.append(i)
        return waiting_spots
    
    def _estimate_wait_time_for_charger(self, ev_id, current_time):
        """Estima tiempo de espera para obtener un cargador."""
        # Simplificado: basado en número de EVs esperando y tasa de liberación
        evs_waiting = len(self.waiting_queue)
        evs_charging = len([ev for ev in self.ev_status if self.ev_status[ev] == 'charging'])
        
        if evs_charging == 0:
            return 0.5  # Tiempo mínimo
        
        # Estimar basado en energía restante promedio de los que están cargando
        avg_time_to_complete = []
        for charging_ev in self.ev_ids:
            if self.ev_status[charging_ev] == 'charging':
                energy_remaining = self.required_energy[charging_ev] - self.energy_delivered[charging_ev]
                # Asumir potencia promedio
                estimated_time = energy_remaining / (self.total_charging_capacity / evs_charging)
                avg_time_to_complete.append(estimated_time)
        
        if avg_time_to_complete:
            avg_completion = np.mean(avg_time_to_complete)
            # Posición en cola afecta tiempo estimado
            queue_position = evs_waiting + 1
            estimated_wait = (queue_position / self.n_charger_spots) * avg_completion
            return min(estimated_wait, self.departure_time[ev_id] - current_time)
        
        return 1.0
    
    def _select_representative_ev(self, evs_present):
        """Selecciona el vehículo más adecuado para representar el estado actual."""
        if not evs_present:
            return None
        
        current_time = self.times[self.current_time_idx]
        
        # Calcular scoring para cada EV
        ev_scores = []
        for ev_id in evs_present:
            # Factor de urgencia
            energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
            time_remaining = max(1e-6, self.departure_time[ev_id] - current_time)
            urgency_score = energy_needed / time_remaining
            
            # Factor de prioridad
            priority_score = self.priority.get(ev_id, 1) if self.has_priority_info else 1
            
            # Factor de willingness to pay
            willingness_score = self.willingness_to_pay.get(ev_id, 1) if self.has_willingness_info else 1
            
            # Factor de estado (priorizar los que están afuera)
            status_score = 1.5 if self.ev_status[ev_id] == 'outside' else 1.0
            
            # Score combinado
            combined_score = (urgency_score * 0.4 + priority_score * 0.3 + 
                            willingness_score * 0.2 + status_score * 0.1)
            
            ev_scores.append((ev_id, combined_score))
        
        # Seleccionar el EV con mayor score
        ev_scores.sort(key=lambda x: x[1], reverse=True)
        return ev_scores[0][0]
    
    def step(self, action_idx):
        """
        Ejecuta una acción y avanza el entorno al siguiente estado.
        Maneja las nuevas acciones de admisión y gestión del parqueadero.
        """
        state = self._get_state()
        if state is None:
            return None, 0, True
        
        actions = self._get_possible_actions(state)
        
        # Protección contra índices inválidos
        if action_idx < 0 or action_idx >= len(actions):
            action_idx = len(actions) - 1
        
        action = actions[action_idx]
        ev_id = state["representative_ev"]
        current_time_idx = state["current_time_idx"]
        current_time = self.times[current_time_idx]
        
        # Procesar según tipo de acción
        if action["action"] == "skip":
            reward = 0
            if state["ev_current_status"] == 'charged_waiting':
                # ¡El agente decidió no mover un bloqueador!
                reward = -self.PENALTY_BLOCKING_CHARGER
                queue_length = len(self.waiting_queue)
                reward -= queue_length * 3

            self.current_time_idx += 1
            
        elif action["action"] == "reject":
            reward = self._execute_reject_action(action, ev_id, state)
            
        elif action["action"] == "strategic_reject":
            reward = self._execute_strategic_reject_action(action, ev_id, state)
            
        elif action["action"] == "admit_and_charge":
            reward = self._execute_admit_and_charge_action(action, ev_id, state)
            
        elif action["action"] == "admit_and_wait":
            reward = self._execute_admit_and_wait_action(action, ev_id, state)
            
        elif action["action"] == "move_to_charger":
            reward = self._execute_move_to_charger_action(action, ev_id, state)
            
        elif action["action"] == "continue_waiting":
            reward = self._execute_continue_waiting_action(action, ev_id, state)
            
        elif action["action"] == "move_to_wait":
            reward = self._execute_move_to_wait_action(action, ev_id, state)    
            
        
        else:
            reward = 0
            self.current_time_idx += 1
        
        # Actualizar métricas y limpiar EVs que ya salieron
        self._update_system_state()
        
        return self._get_state(), reward, self.current_time_idx >= len(self.times)
    
    def _execute_reject_action(self, action, ev_id, state):
        """Ejecuta rechazo por capacidad llena."""
        self.ev_status[ev_id] = 'rejected'
        self.rejection_reasons[action["reason"]] += 1
        self.evs_processed.add(ev_id)
        
        # Penalización mínima - no había opción
        reward = -self.PENALTY_REJECT_CAPACITY
        
        # Factor de prioridad (penalizar más rechazar EVs de alta prioridad)
        if self.has_priority_info:
            priority_factor = self.priority[ev_id] / self.max_priority
            reward *= (1 + priority_factor * 0.5)
        
        self.current_time_idx += 1
        return reward
    
    def _execute_move_to_wait_action(self, action, ev_id, state):
        """Ejecuta el movimiento de un VE ya cargado a un spot de espera."""
        from_spot = action["from_spot"]
        to_spot = action["to_spot"]

        # 1. Liberar el spot con cargador
        self.all_spots_occupied[self.current_time_idx].remove(from_spot)
        self.charger_spots_occupied[self.current_time_idx].remove(from_spot)
        charger_id = self.spot_to_charger.get(from_spot)
        if charger_id in self.occupied_chargers[self.current_time_idx]:
            self.occupied_chargers[self.current_time_idx].remove(charger_id)

        # 2. Ocupar el nuevo spot de espera
        self.all_spots_occupied[self.current_time_idx].add(to_spot)

        # 3. Actualizar la ubicación y estado del EV
        self.ev_location[ev_id] = to_spot
        self.ev_status[ev_id] = 'finished_waiting' # Un nuevo estado para no confundirlo con los que esperan carga

        # 4. Asignar recompensa positiva por la buena gestión
        reward = self.REWARD_FREE_UP_CHARGER

        # Bonus: La recompensa es mayor si hay una cola larga de espera
        queue_length = len(self.waiting_queue)
        reward += queue_length * 5 # +5 de recompensa por cada coche en cola

        self.current_time_idx += 1
        return reward
    
    def _execute_strategic_reject_action(self, action, ev_id, state):
        """Ejecuta rechazo estratégico."""
        self.ev_status[ev_id] = 'rejected'
        self.rejection_reasons['strategic'] += 1
        self.evs_processed.add(ev_id)
        
        # Penalización moderada - fue una decisión
        reward = -self.PENALTY_REJECT_STRATEGIC
        
        # Factores de ajuste
        energy_deficit = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        urgency_factor = min(2.0, energy_deficit / self.avg_required_energy)
        
        if self.has_priority_info:
            priority_factor = self.priority[ev_id] / self.max_priority
            reward *= (1 + priority_factor * 0.5) * urgency_factor
        
        self.current_time_idx += 1
        return reward
    
    def _execute_admit_and_charge_action(self, action, ev_id, state):
        """Ejecuta admisión directa a cargador."""
        spot = action["spot"]
        charger = action["charger"]
        power = action["power"]
        current_time = self.times[self.current_time_idx]
        
        # Actualizar estado del EV
        self.ev_status[ev_id] = 'charging'
        self.ev_location[ev_id] = spot
        self.ev_admission_time[ev_id] = current_time
        self.ev_charge_start_time[ev_id] = current_time
        
        # Registrar ocupación
        self.all_spots_occupied[self.current_time_idx].add(spot)
        self.charger_spots_occupied[self.current_time_idx].add(spot)
        self.occupied_chargers[self.current_time_idx].add(charger)
        self.power_used[self.current_time_idx] += power
        
        # Registrar en schedule
        self.charging_schedule.append((ev_id, self.current_time_idx, charger, spot, power))
        
        # Calcular energía a entregar
        efficiency = self.efficiency.get(ev_id, 0.9)
        charger_eff = 0.95
        energy_to_deliver = power * self.dt * efficiency * charger_eff
        remaining_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        actual_energy = min(energy_to_deliver, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy
        
        
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id]:
            self.ev_status[ev_id] = 'charged_waiting' # ¡Este VE es ahora un bloqueador potencial!
            # También podríamos marcarlo como procesado para que no vuelva a ser seleccionado para cargar
            self.evs_processed.add(ev_id)
        
        # Calcular recompensa base
        reward = self.REWARD_ADMIT_AND_CHARGE
        
        # Bonus por eficiencia
        charger_max_power = self.max_charger_power_dict[charger]
        efficiency_ratio = power / charger_max_power
        if efficiency_ratio > 0.8:  # Uso eficiente del cargador
            reward += self.EFFICIENCY_BONUS_WEIGHT
        
        # Ajuste por costo
        current_price = self.prices[self.current_time_idx]
        normalized_price = (current_price - self.min_price) / (self.max_price - self.min_price + 1e-6)
        cost_penalty = -self.ENERGY_COST_WEIGHT * actual_energy * normalized_price
        reward += cost_penalty
        
        # Factor de prioridad
        if self.has_priority_info:
            priority_multiplier = 0.8 + 0.4 * (self.priority[ev_id] / self.max_priority)
            reward *= priority_multiplier
        
        # Verificar si se completó la carga
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
            reward += self.REWARD_COMPLETE_CHARGE
            self.evs_processed.add(ev_id)
        
        self.current_time_idx += 1
        return reward
    
    def _execute_admit_and_wait_action(self, action, ev_id, state):
        """Ejecuta admisión a spot de espera."""
        spot = action["spot"]
        estimated_wait = action["estimated_wait"]
        current_time = self.times[self.current_time_idx]
        
        # Actualizar estado del EV
        self.ev_status[ev_id] = 'waiting_inside'
        self.ev_location[ev_id] = spot
        self.ev_admission_time[ev_id] = current_time
        
        # Agregar a cola de espera
        if ev_id not in self.waiting_queue:
            self.waiting_queue.append(ev_id)
        
        # Registrar ocupación
        self.all_spots_occupied[self.current_time_idx].add(spot)
        
        # Calcular recompensa base
        reward = self.REWARD_ADMIT_TO_WAIT
        
        # Penalización por tiempo de espera estimado
        wait_penalty = -estimated_wait * 2.0
        reward += wait_penalty
        
        # Bonus por mantener fairness
        fairness_bonus = self._calculate_fairness_bonus()
        reward += fairness_bonus
        
        # Factor de prioridad (menor impacto que en carga directa)
        if self.has_priority_info:
            priority_multiplier = 0.9 + 0.2 * (self.priority[ev_id] / self.max_priority)
            reward *= priority_multiplier
        
        self.current_time_idx += 1
        return reward
    
    def _execute_move_to_charger_action(self, action, ev_id, state):
        """Ejecuta movimiento de spot de espera a cargador."""
        from_spot = action["from_spot"]
        to_spot = action["to_spot"]
        charger = action["charger"]
        power = action["power"]
        current_time = self.times[self.current_time_idx]
        
        # Liberar spot anterior
        if from_spot in self.all_spots_occupied[self.current_time_idx]:
            self.all_spots_occupied[self.current_time_idx].remove(from_spot)
        
        # Actualizar estado del EV
        self.ev_status[ev_id] = 'charging'
        self.ev_location[ev_id] = to_spot
        self.ev_charge_start_time[ev_id] = current_time
        
        # Remover de cola de espera
        if ev_id in self.waiting_queue:
            self.waiting_queue.remove(ev_id)
        
        # Registrar nueva ocupación
        self.all_spots_occupied[self.current_time_idx].add(to_spot)
        self.charger_spots_occupied[self.current_time_idx].add(to_spot)
        self.occupied_chargers[self.current_time_idx].add(charger)
        self.power_used[self.current_time_idx] += power
        
        # Registrar en schedule
        self.charging_schedule.append((ev_id, self.current_time_idx, charger, to_spot, power))
        
        # Calcular tiempo de espera
        wait_time = current_time - self.ev_admission_time[ev_id]
        self.total_wait_times[ev_id] = wait_time
        
        # Calcular energía
        efficiency = self.efficiency.get(ev_id, 0.9)
        charger_eff = 0.95
        energy_to_deliver = power * self.dt * efficiency * charger_eff
        remaining_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        actual_energy = min(energy_to_deliver, remaining_needed)
        self.energy_delivered[ev_id] += actual_energy
        
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id]:
            self.ev_status[ev_id] = 'charged_waiting' # ¡Este VE es ahora un bloqueador potencial!
            # También podríamos marcarlo como procesado para que no vuelva a ser seleccionado para cargar
            self.evs_processed.add(ev_id)
        
        # Recompensa base por iniciar carga después de espera
        reward = self.REWARD_ADMIT_AND_CHARGE * 0.8  # Ligeramente menor que carga directa
        
        # Bonus por tiempo de espera razonable
        if wait_time < self.avg_stay_duration * 0.2:  # Esperó menos del 20% del tiempo promedio
            reward += self.EFFICIENCY_BONUS_WEIGHT * 0.5
        
        # Ajuste por costo
        current_price = self.prices[self.current_time_idx]
        normalized_price = (current_price - self.min_price) / (self.max_price - self.min_price + 1e-6)
        cost_penalty = -self.ENERGY_COST_WEIGHT * actual_energy * normalized_price
        reward += cost_penalty
        
        # Verificar si se completó la carga
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
            reward += self.REWARD_COMPLETE_CHARGE
            self.evs_processed.add(ev_id)
        
        self.current_time_idx += 1
        return reward
    
    def _execute_continue_waiting_action(self, action, ev_id, state):
        """Ejecuta continuar esperando."""
        # No hay cambios de estado significativos
        
        # Pequeña penalización por no aprovechar oportunidad
        reward = -1.0
        
        # Ajustar por urgencia
        current_time = self.times[self.current_time_idx]
        time_remaining = self.departure_time[ev_id] - current_time
        energy_remaining = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        
        if time_remaining < energy_remaining / self.min_charge_rate.get(ev_id, 3.5):
            # Se está quedando sin tiempo
            reward -= 5.0
        
        self.current_time_idx += 1
        return reward
    
    def _calculate_fairness_bonus(self):
        """Calcula bonus por mantener fairness en el sistema."""
        fairness_score = self._calculate_current_fairness()
        return self.FAIRNESS_BONUS_WEIGHT * fairness_score
    
    def _update_system_state(self):
        """Actualiza el estado del sistema después de cada step."""
        current_time = self.times[self.current_time_idx] if self.current_time_idx < len(self.times) else self.times[-1]
        
        # Liberar EVs que ya deberían haber salido
        for ev_id in list(self.ev_status.keys()):
            if self.departure_time[ev_id] <= current_time and self.ev_status[ev_id] not in ['outside', 'rejected']:
                # Liberar recursos
                if ev_id in self.ev_location and self.ev_location[ev_id] != 'outside':
                    spot = self.ev_location[ev_id]
                    # Liberar el spot en todos los tiempos futuros
                    for t in range(self.current_time_idx, len(self.times)):
                        if spot in self.all_spots_occupied[t]:
                            self.all_spots_occupied[t].remove(spot)
                        if spot in self.charger_spots_occupied[t]:
                            self.charger_spots_occupied[t].remove(spot)
                
                # Actualizar estado
                self.ev_status[ev_id] = 'departed'
                self.ev_location[ev_id] = 'outside'
                self.evs_processed.add(ev_id)
                
                # Remover de cola si estaba esperando
                if ev_id in self.waiting_queue:
                    self.waiting_queue.remove(ev_id)
        
        # Propagar ocupación a tiempos futuros para EVs que siguen
        if self.current_time_idx < len(self.times) - 1:
            next_idx = self.current_time_idx + 1
            
            # Copiar ocupación base del tiempo actual al siguiente
            self.all_spots_occupied[next_idx] = self.all_spots_occupied[self.current_time_idx].copy()
            self.charger_spots_occupied[next_idx] = self.charger_spots_occupied[self.current_time_idx].copy()
            self.occupied_chargers[next_idx] = self.occupied_chargers[self.current_time_idx].copy()
            
            # Actualizar power_used para el siguiente tiempo
            self.power_used[next_idx] = 0
            for (ev_id, t, charger, spot, power) in self.charging_schedule:
                if t == self.current_time_idx and self.ev_status.get(ev_id) == 'charging':
                    # Continuar cargando en el siguiente período
                    self.power_used[next_idx] += power
    
    def get_performance_metrics(self):
        """Retorna métricas detalladas del desempeño del sistema."""
        # Métricas de energía (mantener las actuales)
        total_required = sum(self.required_energy.values())
        total_delivered = sum(self.energy_delivered.values())
        energy_satisfaction_pct = (total_delivered / total_required) * 100 if total_required > 0 else 100
        
        # Métricas de admisión y rechazo
        total_arrived = len(self.ev_ids)
        evs_admitted = len([ev for ev in self.ev_status if self.ev_status[ev] not in ['outside', 'rejected']])
        evs_rejected = len([ev for ev in self.ev_status if self.ev_status[ev] == 'rejected'])
        evs_rejected_capacity = self.rejection_reasons.get('parking_full', 0)
        evs_rejected_strategic = self.rejection_reasons.get('strategic', 0)
        
        # Métricas de espera
        avg_wait_time = np.mean(list(self.total_wait_times.values())) if self.total_wait_times else 0
        max_wait_time = max(self.total_wait_times.values()) if self.total_wait_times else 0
        evs_waited = len(self.total_wait_times)
        
        # Métricas de utilización
        avg_parking_util = np.mean([len(occupied) / self.n_spots 
                                   for occupied in self.all_spots_occupied.values()])
        avg_charger_util = np.mean([len(occupied) / self.n_charger_spots 
                                   for occupied in self.charger_spots_occupied.values()])
        
        # Métricas por prioridad
        priority_metrics = {}
        if self.has_priority_info:
            for priority in range(1, self.max_priority + 1):
                priority_evs = [ev for ev in self.ev_ids if self.priority[ev] == priority]
                if priority_evs:
                    delivered = sum(self.energy_delivered[ev] for ev in priority_evs)
                    required = sum(self.required_energy[ev] for ev in priority_evs)
                    admitted = len([ev for ev in priority_evs if self.ev_status[ev] not in ['outside', 'rejected']])
                    
                    priority_metrics[priority] = {
                        "count": len(priority_evs),
                        "admitted": admitted,
                        "admission_rate": admitted / len(priority_evs),
                        "energy_satisfaction": delivered / required if required > 0 else 1.0
                    }
        
        return {
            # Métricas de energía
            "total_required_energy": total_required,
            "total_delivered_energy": total_delivered,
            "energy_satisfaction_pct": energy_satisfaction_pct,
            
            # Métricas de admisión
            "total_evs_arrived": total_arrived,
            "evs_admitted": evs_admitted,
            "evs_rejected": evs_rejected,
            "evs_rejected_capacity": evs_rejected_capacity,
            "evs_rejected_strategic": evs_rejected_strategic,
            "admission_rate": evs_admitted / total_arrived if total_arrived > 0 else 0,
            
            # Métricas de espera
            "evs_waited": evs_waited,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": max_wait_time,
            "wait_rate": evs_waited / evs_admitted if evs_admitted > 0 else 0,
            
            # Métricas de utilización
            "avg_parking_utilization": avg_parking_util,
            "avg_charger_utilization": avg_charger_util,
            
            # Métricas por prioridad
            "priority_metrics": priority_metrics,
            
            # Fairness
            "final_fairness_score": self._calculate_current_fairness()
        }
    
    def get_schedule(self):
        """Retorna el schedule de carga completo generado."""
        return self.charging_schedule
    
    def update_reward_weights(self, weights_dict: Dict[str, float]):
        """Actualiza pesos de recompensa dinámicamente."""
        self.REWARD_ADMIT_AND_CHARGE = weights_dict.get("reward_admit_charge", self.REWARD_ADMIT_AND_CHARGE)
        self.REWARD_ADMIT_TO_WAIT = weights_dict.get("reward_admit_wait", self.REWARD_ADMIT_TO_WAIT)
        self.REWARD_COMPLETE_CHARGE = weights_dict.get("reward_complete", self.REWARD_COMPLETE_CHARGE)
        self.PENALTY_REJECT_CAPACITY = weights_dict.get("penalty_reject_capacity", self.PENALTY_REJECT_CAPACITY)
        self.PENALTY_REJECT_STRATEGIC = weights_dict.get("penalty_reject_strategic", self.PENALTY_REJECT_STRATEGIC)
        self.ENERGY_COST_WEIGHT = weights_dict.get("energy_cost_weight", self.ENERGY_COST_WEIGHT)
        self.EFFICIENCY_BONUS_WEIGHT = weights_dict.get("efficiency_bonus", self.EFFICIENCY_BONUS_WEIGHT)
        self.FAIRNESS_BONUS_WEIGHT = weights_dict.get("fairness_bonus", self.FAIRNESS_BONUS_WEIGHT)