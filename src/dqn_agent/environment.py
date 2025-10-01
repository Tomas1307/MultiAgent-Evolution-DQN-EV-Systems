
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import itertools
import random

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
        # En la función __init__
        self.PENALTY_DEPART_UNSATISFIED = 100.0 # Penalización muy fuerte por fallo
        self.PENALTY_REJECT_CAPACITY = 5.0
        self.PENALTY_REJECT_STRATEGIC = 15.0
        self.ENERGY_COST_WEIGHT = 0.5
        self.EFFICIENCY_BONUS_WEIGHT = 10.0
        self.FAIRNESS_BONUS_WEIGHT = 5.0
        self.REWARD_FREE_UP_CHARGER = 25.0
        self.PENALTY_BLOCKING_CHARGER = 15.0

        # NUEVO (Top-K y acción explícita para avanzar tiempo)
        self.ADVANCE_TIME_ACTION_NAME = "advance_time"
        self.K_PER_TYPE = 8
        
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
        # 4. Calcular el tamaño real y definitivo del vector de estado usando el agente
        #    temporal y el estado ficticio.
        self.state_size = len(temp_agent_for_sizing._process_state(dummy_state))
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
            
        # CÓDIGO CORREGIDO en environment.py

        if self.has_efficiency_info:
            self.efficiency = {arr["id"]: arr.get("efficiency", 0.9) for arr in self.arrivals}
        else:
            ## IMPORTANTE OTRA VEZ PARA QUE FUNCIONE
            # Si la info no existe, crea el atributo con un valor por defecto para cada EV.
            self.efficiency = {arr["id"]: 0.9 for arr in self.arrivals}
        # CÓDIGO CORREGIDO

        if self.has_charge_rate_info:
            self.min_charge_rate = {arr["id"]: arr.get("min_charge_rate", 3.5) for arr in self.arrivals}
            self.max_charge_rate = {arr["id"]: arr.get("max_charge_rate", 50) for arr in self.arrivals}
            self.ac_charge_rate = {arr["id"]: arr.get("ac_charge_rate", 7) for arr in self.arrivals}
            self.dc_charge_rate = {arr["id"]: arr.get("dc_charge_rate", 50) for arr in self.arrivals}
        else:
            ####
            #IMPORTANTE ESTO ES PARA QUE AQUELLOS QUE NO TIENEN ESTO, SE TENGA IGUAL PARA QUE FUNCIONE BIEN
            #
            # Si la info no existe en el JSON, crea los atributos con valores por defecto para todos los EVs.
            self.min_charge_rate = {arr["id"]: 3.5 for arr in self.arrivals}
            self.max_charge_rate = {arr["id"]: 50 for arr in self.arrivals}
            self.ac_charge_rate = {arr["id"]: 7 for arr in self.arrivals}
            self.dc_charge_rate = {arr["id"]: 50 for arr in self.arrivals}
    
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
        if not hasattr(self, 'current_time_idx'):
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        max_skips = len(self.times)
        skips_made = 0
        
        while self.current_time_idx < len(self.times) and skips_made < max_skips:
            current_time = self.times[self.current_time_idx]
            
            evs_needing_decision = []
            for ev_id in self.ev_ids:
                if (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id] and
                    self.ev_status[ev_id] in ['outside', 'waiting_inside', 'charged_waiting'] and
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
        
        current_time = self.times[self.current_time_idx]
        
        parking_features = self._calculate_parking_features(current_time)
        queue_features = self._calculate_queue_features_global(current_time)
        
        # CAMBIO CRÍTICO: Eliminar límite artificial
        # max_evs_per_state = 10  ← ELIMINAR ESTA LÍNEA
        
        ev_features_batch = []
        # NUEVO: Procesar TODOS los vehículos que necesitan decisión
        for ev_id in evs_needing_decision:  # ← SIN LÍMITE [:max_evs_per_state]
            ev_features = self._calculate_ev_features(ev_id, current_time)
            ev_features_batch.append(ev_features)
        
        state = {
            "parking_features": parking_features,
            "queue_features": queue_features,
            "all_evs_needing_decision": evs_needing_decision,
            "ev_features_batch": ev_features_batch,
            "total_occupancy_ratio": parking_features["total_occupancy_ratio"],
            "charger_availability_ratio": parking_features["charger_availability_ratio"],
            "waiting_spots_availability_ratio": parking_features["waiting_spots_availability_ratio"],
            "queue_length": queue_features["queue_length"],
            "avg_wait_time_current": queue_features["avg_wait_time"],
            "system_type": self.test_number,
            "n_spots_total": self.n_spots,
            "n_chargers_total": len(self.charger_ids),
            "transformer_limit": self.station_limit,
            "current_time_idx": self.current_time_idx,
            "current_time_normalized": self.current_time_idx / len(self.times),
            "num_evs_needing_decision": len(evs_needing_decision)
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
    
    def _calculate_queue_features_global(self, current_time):
        queue_length = len(self.waiting_queue)
        
        waiting_times = []
        for waiting_ev in self.waiting_queue:
            if waiting_ev in self.ev_admission_time:
                wait_time = current_time - self.ev_admission_time[waiting_ev]
                waiting_times.append(wait_time)
        
        avg_wait_time = np.mean(waiting_times) / max(self.times) if waiting_times else 0
        fairness_score = self._calculate_current_fairness()
        
        return {
            "queue_length": queue_length / max(1, self.n_spots),
            "avg_wait_time": avg_wait_time,
            "fairness_score": fairness_score
        }
        
    
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
    
    def _stable_sort_key(self, act: Dict):
        """
        Clave de ordenamiento determinista por tipo de acción para estabilizar el set.
        Evita que la red vea ordenes aleatorios en cada step.
        """
        t = act.get("action", "")
        if t == "admit_and_charge":
            return (act.get("spot", 10**9), act.get("charger", 10**9), -act.get("power", 0.0), act.get("ev_id", ""))
        if t == "admit_and_wait":
            return (act.get("spot", 10**9), act.get("estimated_wait", 10**9), act.get("ev_id", ""))
        if t == "move_to_charger":
            return (act.get("to_spot", 10**9), act.get("charger", 10**9), -act.get("power", 0.0), act.get("ev_id", ""))
        if t == "move_to_wait":
            return (act.get("to_spot", 10**9), act.get("ev_id", ""))
        if t in ("reject", "strategic_reject"):
            return (act.get("reason", ""), act.get("ev_id", ""))
        if t == "continue_waiting":
            return (act.get("ev_id", ""),)
        # 'skip' y otros
        return (0,)
    
    def _stabilize_and_topk_actions(self, actions: List[Dict]):
        """
        Ordena de forma determinista y aplica Top-K por tipo de acción.
        Además, asegura que siempre exista una acción explícita para avanzar el tiempo.
        """
        if not actions:
            actions = [{"action": "skip"}]
        
        # Asegurar que exista 'advance_time' explícita al final
        has_advance = any(a.get("action") == self.ADVANCE_TIME_ACTION_NAME for a in actions)
        if not has_advance:
            actions.append({"action": self.ADVANCE_TIME_ACTION_NAME})
        
        # Agrupar por tipo
        groups = defaultdict(list)
        for a in actions:
            groups[a.get("action", "skip")].append(a)
        
        # Orden recomendado (prioriza acciones "más resolutivas" primero)
        type_order = [
            "admit_and_charge",
            "move_to_charger",
            "admit_and_wait",
            "move_to_wait",
            "continue_waiting",
            "strategic_reject",
            "reject",
            "skip",
            self.ADVANCE_TIME_ACTION_NAME
        ]
        
        pruned = []
        for t in type_order:
            if t not in groups:
                continue
            bucket = groups[t]
            # orden estable y determinista
            bucket.sort(key=self._stable_sort_key)
            # top-k por tipo (excepto advance_time y skip, que dejamos solo 1)
            if t in ("skip", self.ADVANCE_TIME_ACTION_NAME):
                pruned.extend(bucket[:1])
            else:
                pruned.extend(bucket[: self.K_PER_TYPE])
        
        return pruned
    
    
    def _apply_continuous_charging(self):
        """Aplica la energía de este *slot* a todos los EVs que están cargando.

        - Usa la potencia asignada en el momento de mover/admitir al cargador (si existe).
        - Si no hay registro explícito de potencia, se calcula por compatibilidad (máx. del cargador y del EV).
        - Actualiza energy_delivered y, si llega al 95% del requerimiento, pasa a 'charged_waiting'.
        - Recalcula power_used del *slot actual* como la suma de potencias de los EVs cargando.
        """
        current_idx = self.current_time_idx
        if current_idx >= len(self.times):
            return

        # Sumar potencias y aplicar energía
        total_power = 0.0
        charger_eff = 0.95

        # Diccionarios que crearemos si no existen
        if not hasattr(self, 'ev_current_power'):
            self.ev_current_power = {}
        if not hasattr(self, 'ev_assigned_charger'):
            self.ev_assigned_charger = {}

        for ev_id, status in list(self.ev_status.items()):
            if status != 'charging':
                continue

            # Potencia base (priorizar la que guardamos en la acción)
            power = self.ev_current_power.get(ev_id, None)
            charger_id = self.ev_assigned_charger.get(ev_id, None)

            # Si no hay potencia registrada, la inferimos
            if power is None:
                # Si sabemos el cargador asignado, usamos su potencia con límite del EV
                if charger_id is not None:
                    ch_power = self.max_charger_power_dict.get(charger_id, self.max_charger_power)
                else:
                    ch_power = self.max_charger_power
                max_ev_rate = self.max_charge_rate.get(ev_id, 50) if hasattr(self, 'max_charge_rate') else 50
                power = min(ch_power, max_ev_rate)
                self.ev_current_power[ev_id] = power  # persistir

            total_power += power

            # Aplicar energía de este dt
            eff = self.efficiency.get(ev_id, 0.9) if hasattr(self, 'efficiency') else 0.9
            energy_to_deliver = power * self.dt * eff * charger_eff
            remaining = self.required_energy[ev_id] - self.energy_delivered.get(ev_id, 0.0)
            actual = min(energy_to_deliver, remaining)
            self.energy_delivered[ev_id] = self.energy_delivered.get(ev_id, 0.0) + actual

            # Verificar si ya quedó "casi completo"
            if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
                self.ev_status[ev_id] = 'charged_waiting'
                # NOTA: lo dejamos ocupando el cargador hasta que el agente lo mueva (move_to_wait)

        # Respetar el límite del transformador de forma informativa (no abortamos, pero medimos uso)
        self.power_used[current_idx] = min(total_power, self.station_limit)
    
    def _is_valid_combination(self, action_combination, current_time_idx):
        spots_needed = 0
        power_needed = 0
        chargers_needed = set()
        spots_used = set()
        
        # ✓ AÑADIR DEBUGGING
        from_spots_used = set()  # Tracking de spots que se liberan
        
        for ev_id, action in action_combination.items():
            action_type = action.get("action")
            
            if action_type == "admit_and_charge":
                spot_id = action.get("spot")
                charger_id = action.get("charger")
                
                if spot_id in spots_used or charger_id in chargers_needed:
                    return False
                    
                spots_needed += 1
                power_needed += action.get("power", 0)
                chargers_needed.add(charger_id)
                spots_used.add(spot_id)
                
            elif action_type == "admit_and_wait":
                spot_id = action.get("spot")
                
                if spot_id in spots_used:
                    return False
                    
                spots_needed += 1
                spots_used.add(spot_id)
                
            elif action_type == "move_to_charger":
                charger_id = action.get("charger")
                to_spot = action.get("to_spot")
                from_spot = action.get("from_spot")
                
                # ✓ AÑADIR VALIDACIÓN CRÍTICA
                if from_spot in from_spots_used:
                    print(f" INVALID COMBINATION: Two EVs trying to move from same spot {from_spot}")
                    return False
                
                from_spots_used.add(from_spot)
                
                if charger_id in chargers_needed or to_spot in spots_used:
                    return False
                    
                power_needed += action.get("power", 0)
                chargers_needed.add(charger_id)
                spots_used.add(to_spot)
            
            elif action_type == "move_to_wait":
                to_spot = action.get("to_spot")
                from_spot = action.get("from_spot")
                
                # ✓ AÑADIR VALIDACIÓN CRÍTICA
                if from_spot in from_spots_used:
                    print(f" INVALID COMBINATION: Two EVs trying to move from same spot {from_spot}")
                    return False
                
                from_spots_used.add(from_spot)
                
                if to_spot in spots_used:
                    return False
                    
                spots_used.add(to_spot)
        
        total_occupied = len(self.all_spots_occupied[current_time_idx])
        
        if total_occupied + spots_needed > self.n_spots:
            return False
        
        if self.power_used[current_time_idx] + power_needed > self.station_limit:
            return False
        
        current_chargers = self.occupied_chargers[current_time_idx]
        if len(chargers_needed & current_chargers) > 0:
            return False
        
        return True

    def _generate_valid_action_combinations(self, individual_actions, current_time_idx):
        evs = list(individual_actions.keys())
        
        if not evs:
            return [{"action": "advance_time"}]
        
        # ✅ LÍMITE MÁS AGRESIVO para prevenir explosión combinatoria
        max_combinations = min(100, 10 ** min(4, len(evs)))  # Reducido de 500 a 100
        valid_combinations = []
        
        action_lists = [individual_actions[ev] for ev in evs]
        
        #  TIMEOUT: Si hay demasiadas combinaciones posibles, usar solo muestreo
        total_possible = 1
        for action_list in action_lists:
            total_possible *= len(action_list)
            if total_possible > 10000:  # Si excede 10k combinaciones
                print(f"  Too many combinations ({total_possible}), using sampling only")
                break
        
        if len(evs) > 10 or total_possible > 10000:  #  Forzar muestreo si es complejo
            # Muestreo inteligente
            attempts = 0
            max_attempts = max_combinations * 5  #  Límite de intentos
            
            while len(valid_combinations) < max_combinations // 2 and attempts < max_attempts:
                attempts += 1
                combination = {}
                for ev_id in evs:
                    combination[ev_id] = random.choice(individual_actions[ev_id])
                
                if self._is_valid_combination(combination, current_time_idx):
                    valid_combinations.append(combination)
            
            if attempts >= max_attempts:
                print(f"  Reached max attempts ({max_attempts}), returning {len(valid_combinations)} combinations")
        else:
            # Producto cartesiano con límite estricto

            
            checked = 0
            max_checks = 5000  #  Máximo de combinaciones a revisar
            
            for combination in itertools.product(*action_lists):
                checked += 1
                if checked > max_checks:
                    print(f"  Checked {max_checks} combinations, stopping")
                    break
                
                action_dict = dict(zip(evs, combination))
                
                # Pre-filtro de from_spot
                from_spots_in_use = set()
                has_conflict = False
                
                for ev_id, action in action_dict.items():
                    from_spot = action.get("from_spot")
                    if from_spot is not None:
                        if from_spot in from_spots_in_use:
                            has_conflict = True
                            break
                        from_spots_in_use.add(from_spot)
                
                if has_conflict:
                    continue
                
                if self._is_valid_combination(action_dict, current_time_idx):
                    valid_combinations.append(action_dict)
                    
                    if len(valid_combinations) >= max_combinations:
                        break
        
        if not valid_combinations:
            skip_combination = {ev_id: {"action": "skip", "ev_id": ev_id} 
                            for ev_id in evs}
            valid_combinations.append(skip_combination)
        
        valid_combinations.append({"action": "advance_time"})
        
        return valid_combinations

    def _get_possible_actions(self, state):
        evs_needing_decision = state["all_evs_needing_decision"]
        current_time_idx = state["current_time_idx"]
        
        if not evs_needing_decision:
            return [{"action": "advance_time"}]
        
        individual_actions = {}
        for ev_id in evs_needing_decision:
            individual_actions[ev_id] = self._get_actions_for_single_ev(ev_id, state)
        
        combined_actions = self._generate_valid_action_combinations(individual_actions, current_time_idx)
        
        return combined_actions
    
    def _get_actions_for_single_ev(self, ev_id, state):
        current_time_idx = state["current_time_idx"]
        current_time = self.times[current_time_idx]
        ev_status = self.ev_status[ev_id]
        
        if not (self.arrival_time[ev_id] <= current_time < self.departure_time[ev_id]):
            return [{"action": "skip", "ev_id": ev_id}]
        
        energy_needed = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        if energy_needed <= 0.01:
            return [{"action": "skip", "ev_id": ev_id}]
        
        actions = []
        total_occupied = len(self.all_spots_occupied[current_time_idx])
        
        if total_occupied >= self.n_spots and ev_status == 'outside':
            actions = [{"action": "reject", "reason": "parking_full", "ev_id": ev_id}]
            return actions
        
        if ev_status == 'outside':
            available_charger_spots = self._get_available_charger_spots(current_time_idx)
            
            for spot_id in available_charger_spots:
                charger_id = self.spot_to_charger.get(spot_id)
                if charger_id and charger_id in self.ev_charger_compatible.get(ev_id, []):
                    charger_power = self.max_charger_power_dict[charger_id]
                    actions.append({
                        "action": "admit_and_charge",
                        "ev_id": ev_id,
                        "spot": spot_id,
                        "charger": charger_id,
                        "power": min(charger_power, self.max_charge_rate.get(ev_id, 50))
                    })
            
            if total_occupied < self.n_spots:
                available_waiting_spots = self._get_available_waiting_spots(current_time_idx)
                for spot_id in available_waiting_spots:
                    estimated_wait = self._estimate_wait_time_for_charger(ev_id, current_time)
                    actions.append({
                        "action": "admit_and_wait",
                        "ev_id": ev_id,
                        "spot": spot_id,
                        "estimated_wait": estimated_wait
                    })
            
            actions.append({
                "action": "strategic_reject",
                "ev_id": ev_id,
                "reason": "optimization"
            })
        
        elif ev_status == 'waiting_inside':
            available_charger_spots = self._get_available_charger_spots(current_time_idx)
            
            for spot_id in available_charger_spots:
                charger_id = self.spot_to_charger.get(spot_id)
                if charger_id and charger_id in self.ev_charger_compatible.get(ev_id, []):
                    charger_power = self.max_charger_power_dict[charger_id]
                    actions.append({
                        "action": "move_to_charger",
                        "ev_id": ev_id,
                        "from_spot": self.ev_location[ev_id],
                        "to_spot": spot_id,
                        "charger": charger_id,
                        "power": min(charger_power, self.max_charge_rate.get(ev_id, 50))
                    })
            
            actions.append({
                "action": "continue_waiting",
                "ev_id": ev_id
            })
            
        elif ev_status == 'charged_waiting':
            available_waiting_spots = self._get_available_waiting_spots(current_time_idx)
            for to_spot in available_waiting_spots:
                actions.append({
                    "action": "move_to_wait",
                    "ev_id": ev_id,
                    "from_spot": self.ev_location[ev_id],
                    "to_spot": to_spot
                })
            
            actions.append({"action": "skip", "ev_id": ev_id})
        
        if len(actions) == 0:
            actions.append({"action": "skip", "ev_id": ev_id})
        
        # CAMBIO: Aumentar límite pero mantener control
        # return actions[:self.K_PER_TYPE]  ← ANTES: K_PER_TYPE = 8
        max_actions = min(20, len(actions))  # ← NUEVO: Máximo 20 acciones por vehículo
        return actions[:max_actions]
    
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
        
        if action_idx < 0 or action_idx >= len(actions):
            action_idx = len(actions) - 1
        
        action_combination = actions[action_idx]
        
        if "action" in action_combination and action_combination["action"] == "advance_time":
            self._apply_continuous_charging()
            self.current_time_idx += 1
            return self._get_state(), 0, self.current_time_idx >= len(self.times)
        
        total_reward = 0
        
        # Ejecutar todas las acciones SIN incrementar tiempo individualmente
        for ev_id, action in action_combination.items():
            reward = self._execute_single_action(action, ev_id, state)
            total_reward += reward
        
        # Actualizar sistema UNA VEZ después de todas las acciones
        final_penalty = self._update_system_state()
        total_step_reward = total_reward + final_penalty
        
        # Incrementar tiempo UNA VEZ al final
        self.current_time_idx += 1
        
        return self._get_state(), total_step_reward, self.current_time_idx >= len(self.times)

    def _execute_admit_and_charge_action(self, action, ev_id, state):
        spot = action["spot"]
        charger = action["charger"]
        power = action["power"]
        current_time = self.times[self.current_time_idx]
        
        self.ev_status[ev_id] = 'charging'
        self.ev_location[ev_id] = spot
        self.ev_charge_start_time[ev_id] = current_time
        
        self.all_spots_occupied[self.current_time_idx].add(spot)
        self.charger_spots_occupied[self.current_time_idx].add(spot)
        self.occupied_chargers[self.current_time_idx].add(charger)
        
        self.ev_current_power = getattr(self, 'ev_current_power', {})
        self.ev_assigned_charger = getattr(self, 'ev_assigned_charger', {})
        self.ev_current_power[ev_id] = power
        self.ev_assigned_charger[ev_id] = charger
        
        actual_energy = power * self.dt * self.efficiency.get(ev_id, 0.9) * 0.95
        self.energy_delivered[ev_id] += actual_energy
        
        reward = self.REWARD_ADMIT_AND_CHARGE
        energy_requirement_bonus = self.required_energy[ev_id] * 0.5
        reward += energy_requirement_bonus
        
        charger_max_power = self.max_charger_power_dict[charger]
        efficiency_ratio = power / charger_max_power
        if efficiency_ratio > 0.8:
            reward += self.EFFICIENCY_BONUS_WEIGHT
        
        current_price = self.prices[self.current_time_idx]
        normalized_price = (current_price - self.min_price) / (self.max_price - self.min_price + 1e-6)
        cost_penalty = -self.ENERGY_COST_WEIGHT * actual_energy * normalized_price
        reward += cost_penalty
        
        if self.has_priority_info:
            priority_multiplier = 0.8 + 0.4 * (self.priority.get(ev_id, 1) / self.max_priority)
            reward *= priority_multiplier
        
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
            reward += self.REWARD_COMPLETE_CHARGE
            self.ev_status[ev_id] = 'charged_waiting'
            self.evs_processed.add(ev_id)
        
        return reward
    
    def _execute_move_to_charger_action(self, action, ev_id, state):
        from_spot = action["from_spot"]
        to_spot = action["to_spot"]
        charger = action["charger"]
        power = action["power"]
        current_time = self.times[self.current_time_idx]
        
        if ev_id in self.waiting_queue:
            self.waiting_queue.remove(ev_id)
        
        actual_location = self.ev_location.get(ev_id, 'outside')
        
        if actual_location != 'outside' and actual_location != from_spot:
            # Si hay discrepancia, usar la ubicación real
            from_spot = actual_location
        
        if isinstance(from_spot, int) and from_spot in self.all_spots_occupied[self.current_time_idx]:
            self.all_spots_occupied[self.current_time_idx].remove(from_spot)
        
        self.all_spots_occupied[self.current_time_idx].add(to_spot)
        self.charger_spots_occupied[self.current_time_idx].add(to_spot)
        self.occupied_chargers[self.current_time_idx].add(charger)
        
        self.ev_status[ev_id] = 'charging'
        self.ev_location[ev_id] = to_spot
        self.ev_charge_start_time[ev_id] = current_time
        
        wait_time = current_time - self.ev_admission_time.get(ev_id, current_time)
        
        self.ev_current_power = getattr(self, 'ev_current_power', {})
        self.ev_assigned_charger = getattr(self, 'ev_assigned_charger', {})
        self.ev_current_power[ev_id] = power
        self.ev_assigned_charger[ev_id] = charger
        
        actual_energy = power * self.dt * self.efficiency.get(ev_id, 0.9) * 0.95
        self.energy_delivered[ev_id] += actual_energy
        
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
            self.ev_status[ev_id] = 'charged_waiting'
            self.evs_processed.add(ev_id)
        
        reward = self.REWARD_ADMIT_AND_CHARGE * 0.8
        
        if wait_time < self.avg_stay_duration * 0.2:
            reward += self.EFFICIENCY_BONUS_WEIGHT * 0.5
        
        current_price = self.prices[self.current_time_idx]
        normalized_price = (current_price - self.min_price) / (self.max_price - self.min_price + 1e-6)
        cost_penalty = -self.ENERGY_COST_WEIGHT * actual_energy * normalized_price
        reward += cost_penalty
        
        if self.energy_delivered[ev_id] >= self.required_energy[ev_id] * 0.95:
            reward += self.REWARD_COMPLETE_CHARGE
        
        return reward
    
    def _execute_move_to_wait_action(self, action, ev_id, state):
        from_spot = action["from_spot"]
        to_spot = action["to_spot"]
        
        if hasattr(self, 'ev_current_power') and ev_id in self.ev_current_power:
            self.ev_current_power.pop(ev_id, None)
        if hasattr(self, 'ev_assigned_charger') and ev_id in self.ev_assigned_charger:
            self.ev_assigned_charger.pop(ev_id, None)
        
        actual_location = self.ev_location.get(ev_id, 'outside')
        
        if actual_location != 'outside' and actual_location != from_spot:
            from_spot = actual_location
        
        if isinstance(from_spot, int):
            if from_spot in self.charger_spots_occupied[self.current_time_idx]:
                self.charger_spots_occupied[self.current_time_idx].remove(from_spot)
            if from_spot in self.all_spots_occupied[self.current_time_idx]:
                self.all_spots_occupied[self.current_time_idx].remove(from_spot)
        
        # Agregar al nuevo spot
        self.all_spots_occupied[self.current_time_idx].add(to_spot)
        
        charger_id = self.spot_to_charger.get(from_spot)
        if charger_id:
            self.occupied_chargers[self.current_time_idx].discard(charger_id)
        
        self.ev_status[ev_id] = 'waiting_inside'
        self.ev_location[ev_id] = to_spot
        
        if ev_id not in self.waiting_queue:
            self.waiting_queue.append(ev_id)
        
        reward = self.REWARD_FREE_UP_CHARGER
        queue_length = len(self.waiting_queue)
        reward += queue_length * 5
        
        return reward

    def _execute_single_action(self, action, ev_id, state):
        action_type = action["action"]
        current_time_idx = state["current_time_idx"]
        
        if action_type == "skip":
            reward = 0
            if state.get("ev_current_status") == 'charged_waiting':
                reward = -self.PENALTY_BLOCKING_CHARGER
                queue_length = len(self.waiting_queue)
                reward -= queue_length * 3
            self._apply_continuous_charging()
            
        elif action_type == "reject":
            reward = self._execute_reject_action(action, ev_id, state)
            
        elif action_type == "strategic_reject":
            reward = self._execute_strategic_reject_action(action, ev_id, state)
            
        elif action_type == "admit_and_charge":
            reward = self._execute_admit_and_charge_action(action, ev_id, state)
            
        elif action_type == "admit_and_wait":
            reward = self._execute_admit_and_wait_action(action, ev_id, state)
            
        elif action_type == "move_to_charger":
            reward = self._execute_move_to_charger_action(action, ev_id, state)
            
        elif action_type == "continue_waiting":
            reward = self._execute_continue_waiting_action(action, ev_id, state)
            
        elif action_type == "move_to_wait":
            reward = self._execute_move_to_wait_action(action, ev_id, state)
            
        else:
            reward = 0
            self._apply_continuous_charging()
        
        return reward

    def _execute_reject_action(self, action, ev_id, state):
        """Ejecuta rechazo por capacidad llena."""
        self.ev_status[ev_id] = 'rejected'
        self.rejection_reasons[action["reason"]] += 1
        self.evs_processed.add(ev_id)
        
        reward = -self.PENALTY_REJECT_CAPACITY
        
        if self.has_priority_info:
            priority_factor = self.priority[ev_id] / self.max_priority
            reward *= (1 + priority_factor * 0.5)
        
        # NO incrementar current_time_idx aquí
        return reward
    
    def _execute_strategic_reject_action(self, action, ev_id, state):
        """Ejecuta rechazo estratégico."""
        self.ev_status[ev_id] = 'rejected'
        self.rejection_reasons['strategic'] += 1
        self.evs_processed.add(ev_id)
        
        reward = -self.PENALTY_REJECT_STRATEGIC
        
        energy_deficit = self.required_energy[ev_id] - self.energy_delivered[ev_id]
        urgency_factor = min(2.0, energy_deficit / self.avg_required_energy)
        
        if self.has_priority_info:
            priority_factor = self.priority[ev_id] / self.max_priority
            reward *= (1 + priority_factor * 0.5) * urgency_factor
        
        # NO incrementar current_time_idx aquí
        return reward
    
    def _execute_admit_and_wait_action(self, action, ev_id, state):
        """Ejecuta admisión a spot de espera."""
        spot = action["spot"]
        estimated_wait = action["estimated_wait"]
        current_time = self.times[self.current_time_idx]
        
        self.ev_status[ev_id] = 'waiting_inside'
        self.ev_location[ev_id] = spot
        self.ev_admission_time[ev_id] = current_time
        
        if ev_id not in self.waiting_queue:
            self.waiting_queue.append(ev_id)
        
        self.all_spots_occupied[self.current_time_idx].add(spot)
        
        reward = self.REWARD_ADMIT_TO_WAIT
        wait_penalty = -estimated_wait * 2.0
        reward += wait_penalty
        
        fairness_bonus = self._calculate_fairness_bonus()
        reward += fairness_bonus
        
        if self.has_priority_info:
            priority_multiplier = 0.9 + 0.2 * (self.priority[ev_id] / self.max_priority)
            reward *= priority_multiplier
        
        # NO incrementar current_time_idx aquí
        return reward
    
    # CÓDIGO CORREGIDO Y COMPLETO
    def _execute_continue_waiting_action(self, action, ev_id, state):
        """Ejecuta continuar esperando con penalización inteligente."""
        current_time_idx = state["current_time_idx"]
        available_chargers = self._get_available_charger_spots(current_time_idx)

        # Si hay cargadores libres y el agente AÚN ASÍ decide esperar,
        # la penalización es FUERTE.
        if available_chargers:
            reward = -25.0  # Penalización fuerte por ignorar una oportunidad de carga.
        # Si todos los cargadores están ocupados, esperar es una acción válida,
        # por lo que la penalización es LEVE.
        else:
            reward = -1.0

        # La lógica de urgencia que ya tenías se mantiene y se suma a la penalización
        current_time = self.times[current_time_idx]
        time_remaining = self.departure_time[ev_id] - current_time
        energy_remaining = self.required_energy[ev_id] - self.energy_delivered.get(ev_id, 0)
        
        min_charge_rate = self.min_charge_rate.get(ev_id, 3.5)
        if min_charge_rate > 0 and time_remaining < energy_remaining / min_charge_rate:
            reward -= 15.0
        
        # NO llamar _apply_continuous_charging aquí (se hace en step())
        # NO incrementar current_time_idx aquí
        return reward
    
    def _calculate_fairness_bonus(self):
        """Calcula bonus por mantener fairness en el sistema."""
        fairness_score = self._calculate_current_fairness()
        return self.FAIRNESS_BONUS_WEIGHT * fairness_score
    
    def _update_system_state(self):
        """
        Actualiza el estado del sistema, libera recursos de EVs que parten
        y SINCRONIZA el estado de ocupación con ev_location.
        """
        current_time = self.times[self.current_time_idx] if self.current_time_idx < len(self.times) else self.times[-1]
        final_penalty = 0.0

        # Liberar EVs que ya deberían haber salido
        for ev_id in list(self.ev_status.keys()):
            if self.departure_time[ev_id] <= current_time and self.ev_status[ev_id] not in ['outside', 'rejected', 'departed']:
                
                satisfaction_ratio = self.energy_delivered.get(ev_id, 0) / self.required_energy[ev_id]
                if satisfaction_ratio < 0.9:
                    final_penalty += (-self.PENALTY_DEPART_UNSATISFIED) * (1 - satisfaction_ratio)

                # Actualizar estado
                self.ev_status[ev_id] = 'departed'
                self.ev_location[ev_id] = 'outside'
                self.evs_processed.add(ev_id)
                
                if ev_id in self.waiting_queue:
                    self.waiting_queue.remove(ev_id)
        
        self.all_spots_occupied[self.current_time_idx] = set()
        self.charger_spots_occupied[self.current_time_idx] = set()
        
        for ev_id, location in self.ev_location.items():
            status = self.ev_status.get(ev_id)
            
            if (location != 'outside' and 
                status not in ['outside', 'rejected', 'departed']):
                
                spot = location
                self.all_spots_occupied[self.current_time_idx].add(spot)
                
                # Si está en un spot con cargador (0 a n_charger_spots-1)
                if isinstance(spot, int) and spot < self.n_charger_spots:
                    self.charger_spots_occupied[self.current_time_idx].add(spot)
        
        # Propagación al siguiente timestep
        if self.current_time_idx < len(self.times) - 1:
            next_idx = self.current_time_idx + 1
            self.all_spots_occupied[next_idx] = self.all_spots_occupied[self.current_time_idx].copy()
            self.charger_spots_occupied[next_idx] = self.charger_spots_occupied[self.current_time_idx].copy()
            self.occupied_chargers[next_idx] = self.occupied_chargers[self.current_time_idx].copy()
            
            self.power_used[next_idx] = 0
            for (ev_id, t, charger, spot, power) in self.charging_schedule:
                if t == self.current_time_idx and self.ev_status.get(ev_id) == 'charging':
                    self.power_used[next_idx] += power
                    
        return final_penalty
    
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
