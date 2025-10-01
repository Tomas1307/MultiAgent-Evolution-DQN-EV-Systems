import argparse
import json
import os
import numpy as np
import time
import sys
import re
from datetime import datetime
from collections import defaultdict
import glob
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
import yaml
from src.common.config import load_data, load_all_test_systems, load_system_config, load_hyperparameters
from src.dqn_agent.agent import EnhancedDQNAgentPyTorch as EnhancedDQNAgent
from src.dqn_agent.environment import EVChargingEnv
from src.dqn_agent.training import train_agent, train_dqn_agent
from src.milp_optimizer.optimizer import EVChargingMILP
from src.common.logger import SimpleEpisodeLogger
from src.common.visualize import plot_dqn_learning_progress, visualize_solution, visualize_milp_solution, visualize_rl_solution
from src.common.utils import ensure_directory_exists, get_timestamp_filepath
from src.tests.evaluator import EfficientSlotBySlotSimulator
import gc
import torch

def map_hyperparameters(hyperparams):
    """
    Maps hyperparameter names from the YAML file to match the
    EnhancedDQNAgentPyTorch constructor parameters.

    Args:
        hyperparams (dict): A dictionary of hyperparameters loaded from a YAML file.

    Returns:
        dict: A dictionary of mapped hyperparameters suitable for the agent's constructor.
    """
    try:
        mapped = {}

        param_mappings = {
            'learning_rate': 'learning_rate',
            'gamma': 'gamma',
            'epsilon_start': 'epsilon',
            'epsilon_min': 'epsilon_min',
            'epsilon_decay': 'epsilon_decay',
            'memory_size': 'memory_size',
            'batch_size': 'batch_size',
            'target_update_freq': 'target_update_freq',
            'target_update_frequency': 'target_update_freq',
            'dueling_network': 'dueling_network',
            'dueling': 'dueling_network',
            'epsilon_end': 'epsilon_min'
        }

        for yaml_key, agent_key in param_mappings.items():
            if yaml_key in hyperparams:
                mapped[agent_key] = hyperparams[yaml_key]

        defaults = {
            'learning_rate': 0.0005,
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 5000,
            'batch_size': 32,
            'target_update_freq': 50,
            'dueling_network': True
        }

        for key, default_value in defaults.items():
            if key not in mapped:
                mapped[key] = default_value

        return mapped
    except Exception as e:
        print(f"Error mapping hyperparameters: {e}")


def get_all_system_ids(data_dir: str) -> list[int]:
    """
    Scans the data directory to find all available system_ids
    based on the 'test_system_X.json' pattern.

    Args:
        data_dir (str): The path to the directory containing the test system JSON files.

    Returns:
        list[int]: A sorted list of found system IDs.
    """
    system_ids = []
    try:
        for filename in os.listdir(data_dir):
            match = re.match(r"test_system_(\d+)\.json", filename)
            if match:
                system_ids.append(int(match.group(1)))
        system_ids.sort()
    except FileNotFoundError:
        print(f"Warning: Directory {data_dir} not found.")
    return system_ids


def generate_solution(config, agent=None, model_path="./ev_scheduler_model_pytorch.pt", state_size: int = 256, action_size: int = 60):
    """
    Generates a solution using the RL agent.

    Args:
        config (dict): The configuration dictionary for the environment.
        agent (EnhancedDQNAgent, optional): An initialized RL agent. If None, an agent will be loaded.
        model_path (str, optional): The path to the pre-trained model. Defaults to "./ev_scheduler_model_pytorch.pt".
        state_size (int, optional): The size of the state space. Defaults to 256.
        action_size (int, optional): The size of the action space. Defaults to 60.

    Returns:
        tuple: A tuple containing the generated schedule (list) and performance metrics (dict).
    """
    try:
        print("Generating RL solution...")
        print("DEBUG: Starting generate_solution function")

        if agent is None:
            print(f"Attempting to load model from {model_path}...")
            print("DEBUG: About to create temp_env")
            
            temp_env = EVChargingEnv(config)
            print("DEBUG: temp_env created successfully")

            print("DEBUG: About to create agent")
            agent = EnhancedDQNAgent(
                state_size=state_size,
                action_size=action_size,
                dueling_network=True
            )
            print("DEBUG: Agent created successfully")

            try:
                print("DEBUG: About to load model")
                model_loaded = agent.load(model_path)
                print(f"DEBUG: Model loaded result: {model_loaded}")
                if not model_loaded:
                    raise ValueError("The model could not be loaded correctly.")
            except Exception as e:
                raise RuntimeError(f"Critical error loading the model from {model_path}: {e}")

        print("DEBUG: About to create main env")
        env = EVChargingEnv(config)
        print("DEBUG: Main env created, about to reset")
        state = env.reset()
        print(f"DEBUG: Environment reset, state shape: {np.array(state).shape if hasattr(state, 'shape') else len(state) if state else 'None'}")
        
        done = False
        step_count = 0
        max_steps = 1000  # Agregar límite de pasos

        original_epsilon = agent.epsilon
        agent.epsilon = 0.05
        print(f"DEBUG: Set epsilon to {agent.epsilon}")

        while not done and step_count < max_steps:
            print(f"DEBUG: Step {step_count}")
            possible_actions = env._get_possible_actions(state)
            print(f"DEBUG: Possible actions: {len(possible_actions) if possible_actions else 0}")
            
            if not possible_actions:
                print("DEBUG: No possible actions, breaking")
                break
                
            action = agent.act(state, possible_actions)
            print(f"DEBUG: Selected action: {action}")

            if action == -1 or action >= len(possible_actions):
                print(f"DEBUG: Invalid action {action}, breaking")
                break

            state, _, done = env.step(action)
            print(f"DEBUG: Step completed, done: {done}")
            step_count += 1

        print(f"DEBUG: Loop finished after {step_count} steps")

        agent.epsilon = original_epsilon

        schedule_rl = env.get_schedule()
        energy_metrics = env.get_energy_satisfaction_metrics()

        total_cost = sum(
            config["prices"][time_idx] * config["dt"] * power
            for (_, time_idx, _, _, power) in schedule_rl
        )

        metrics = {
            "energy_metrics": energy_metrics,
            "total_cost": total_cost,
            "cost_per_kwh": total_cost / energy_metrics["total_delivered_energy"] if energy_metrics["total_delivered_energy"] > 0 else 0,
            "schedule_size": len(schedule_rl),
            "evs_served": len(set(entry[0] for entry in schedule_rl)),
            "total_evs": len(config["arrivals"])
        }

        print(f"RL solution generated: {len(schedule_rl)} assignments for {metrics['evs_served']}/{metrics['total_evs']} EVs ({metrics['evs_served']/metrics['total_evs']*100:.2f}%)")
        print(f"Energy satisfaction: {energy_metrics['total_satisfaction_pct']:.2f}%")
        print(f"Total cost: ${total_cost:.2f}")

        return schedule_rl, metrics
    except Exception as e:
        print(f"Error on generate_solution: {e}")


def optimize_solution(config, rl_schedule, alpha_cost=0.6, alpha_satisfaction=0.4, time_limit=900):
    """
    Optimizes the RL solution using multi-objective MILP.

    Args:
        config (dict): The configuration dictionary for the environment.
        rl_schedule (list): The schedule generated by the RL agent.
        alpha_cost (float, optional): Weight for the cost objective. Defaults to 0.6.
        alpha_satisfaction (float, optional): Weight for the satisfaction objective. Defaults to 0.4.
        time_limit (int, optional): Time limit for the MILP solver in seconds. Defaults to 900.

    Returns:
        tuple: A tuple containing the optimized MILP schedule (list) and performance metrics (dict).
               Returns (None, {"error": "Optimization failed"}) if optimization fails.
    """
    try:
        print("\nOptimizing solution with multi-objective MILP...")
        print(f"Weights: Cost ({alpha_cost:.2f}), Satisfaction ({alpha_satisfaction:.2f})")

        optimizer = EVChargingMILP(config)

        start_time = time.time()
        model, schedule, rejected_details, obj_values = optimizer.solve(
            penalty_unmet=1000.0,
            rl_schedule=rl_schedule,
            time_limit=time_limit,
            epsilon_satisfaction=0.6,
            return_infeasible=True
        )

        solve_time = time.time() - start_time

        if model is None:
            print("Could not optimize. Returning original RL solution.")
            return None, {"error": "Optimization failed"}

        schedule_milp = []
        for ev_id, entries in schedule.items():
            for (start_time, end_time, charger_id, slot, power) in entries:
                time_idx = None
                for idx, t in enumerate(config["times"]):
                    if abs(t - start_time) < 1e-5:
                        time_idx = idx
                        break

                if time_idx is not None:
                    schedule_milp.append((ev_id, time_idx, charger_id, slot, power))

        metrics = {
            "obj_values": obj_values,
            "rejected_details": rejected_details,
            "solve_time": solve_time,
            "schedule_size": len(schedule_milp),
            "evs_served": len(set(entry[0] for entry in schedule_milp)),
            "total_evs": len(config["arrivals"]),
            "milp_status": model.status if model else None,
            "milp_objective": model.objective.value() if model else None
        }

        print(f"MILP solution generated in {solve_time:.2f} seconds")
        print(f"Energy satisfied: {obj_values['energy_satisfaction_pct']:.2f}%")
        print(f"Total cost: ${obj_values['energy_cost']:.2f}")
        print(f"Weighted satisfaction: {obj_values['weighted_satisfaction']:.4f}")

        if rejected_details:
            print(f"Vehicles with unmet energy: {len(rejected_details)}")

        return schedule_milp, metrics
    except Exception as e:
        print(f"Error on optimize_solution: {e}")


def save_solution(schedule, metrics, filename):
    """
    Saves the solution and metrics to a JSON file.

    Args:
        schedule (list): The generated or optimized schedule.
        metrics (dict): The performance metrics.
        filename (str): The name of the file to save the data.
    """
    try:
        serializable_schedule = []
        for entry in schedule:
            serializable_schedule.append(list(entry))

        data = {
            "schedule": serializable_schedule,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Solution saved to {filename}")
    except Exception as e:
        print(f"Error on save_solution: {e}")


def compare_solutions(rl_metrics, milp_metrics):
    """
    Compares the RL and MILP solutions.

    Args:
        rl_metrics (dict): Metrics from the RL solution.
        milp_metrics (dict): Metrics from the MILP solution.
    """
    try:
        print("\n=== SOLUTION COMPARISON ===")

        rl_satisfaction = rl_metrics["energy_metrics"]["total_satisfaction_pct"]
        milp_satisfaction = milp_metrics["obj_values"]["energy_satisfaction_pct"]
        satisfaction_diff = milp_satisfaction - rl_satisfaction

        print(f"Energy satisfaction:")
        print(f"  RL:   {rl_satisfaction:.2f}%")
        print(f"  MILP: {milp_satisfaction:.2f}% ({'+' if satisfaction_diff >= 0 else ''}{satisfaction_diff:.2f}%)")

        rl_cost = rl_metrics["total_cost"]
        milp_cost = milp_metrics["obj_values"]["energy_cost"]
        cost_diff_pct = ((milp_cost - rl_cost) / rl_cost) * 100 if rl_cost > 0 else 0

        print(f"Total cost:")
        print(f"  RL:   ${rl_cost:.2f}")
        print(f"  MILP: ${milp_cost:.2f} ({'+' if cost_diff_pct >= 0 else ''}{cost_diff_pct:.2f}%)")

        rl_evs = rl_metrics["evs_served"]
        milp_evs = milp_metrics["evs_served"]
        total_evs = rl_metrics["total_evs"]

        print(f"Vehicles served:")
        print(f"  RL:   {rl_evs}/{total_evs} ({rl_evs/total_evs*100:.1f}%)")
        print(f"  MILP: {milp_evs}/{total_evs} ({milp_evs/total_evs*100:.1f}%)")
    except Exception as e:
        print(f"Error on compare_solutions: {e}")


def run_scatter_search_optimization(args, ask_resume: bool = True):
    """
    Executes hyperparameter optimization using Scatter Search.

    Args:
        args: An object containing command-line arguments (e.g., output_base_dir, data_dir, scatter_config).
        ask_resume (bool, optional): If True, prompts the user to resume from a checkpoint if found. Defaults to True.
    """
    try:
        print("=== SCATTER SEARCH OPTIMIZATION MODE ===")

        # IMPORTS AT THE BEGINNING
        try:
            from src.scatter_search.scatter_algorithm import ScatterSearchOptimizer
            from src.scatter_search.results_analyzer import ResultsAnalyzer
        except ImportError as e:
            print(f"Error importing Scatter Search modules: {e}")
            print("Ensure all scatter_search files are in src/scatter_search/")
            return

        # CONFIGURE DIRECTORIES AND DATA
        scatter_output_dir = args.output_base_dir
        ensure_directory_exists(scatter_output_dir)

        print("Loading data systems...")
        systems_data = load_all_test_systems(args.data_dir)

        if not systems_data:
            print("No data systems found for optimization.")
            return

        print(f"Systems loaded: {list(systems_data.keys())}")

        # CONFIGURATION
        scatter_config_path = args.scatter_config or "src/configs/scatter_search_config.yaml"

        # RESUME HANDLING
        checkpoint_result = find_latest_checkpoint()
        resume_from = None

        if checkpoint_result is not None:
            latest_checkpoint, iteration_num = checkpoint_result
            if ask_resume:
                response = input(f"Resume from checkpoint (iteration {iteration_num})? (y/n): ")
                if response.lower() == 'y':
                    resume_from = latest_checkpoint
        else:
            print("No previous checkpoints found. Starting new optimization.")

        # INITIALIZE OPTIMIZER
        print(f"Initializing Scatter Search with configuration: {scatter_config_path}")
        optimizer = ScatterSearchOptimizer(scatter_config_path, systems_data, args.output_base_dir)

        print("Starting hyperparameter optimization...")
        start_time = time.time()

        try:
            # EXECUTE OPTIMIZATION (WITH RESUME IF APPLICABLE)
            results = optimizer.run_optimization(resume_from=resume_from)
            execution_time = time.time() - start_time

            print(f"\nOptimization completed in {execution_time/3600:.2f} hours")
            print(f"Best solutions found: {len(results['best_solutions'])}")

            # SHOW TOP 3 (WITH VALIDATION)
            if results['best_solutions']:
                print("\nTOP 3 SOLUTIONS FOUND:")
                for i, solution in enumerate(results['best_solutions'][:3]):
                    archetype = solution.get('archetype', 'unknown')
                    fitness = solution.get('fitness', 'N/A')
                    print(f"{i+1}. {archetype.upper()} - Fitness: {fitness}")

            # ANALYSIS AND EXPORT
            print("\nAnalyzing and exporting results...")
            analyzer = ResultsAnalyzer(scatter_output_dir)

            analyzer.save_complete_results(results)
            analyzer.generate_summary_report(results)

            if results['best_solutions']:
                analyzer.create_configuration_files(results['best_solutions'])

            analyzer.generate_visualizations(results)
            analyzer.export_to_excel(results)
            analyzer.create_deployment_package(results)

            print(f"\nAll results saved in: {scatter_output_dir}")
            print("Check 'optimization_report.txt' for a complete summary")
            print("Optimized configurations are in 'configurations/'")
            print("Visualizations are in 'visualizations/'")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            print("Partial results may be available in checkpoints")
        except Exception as e:
            print(f"\nError during optimization: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error on run_scatter_search_optimization: {e}")

def load_scatter_search_solution(system_id, model_name, output_base_dir):
    """
    Load a pre-calculated RL solution from scatter search results
    
    Args:
        system_id (int): System ID
        model_name (str): Model name (e.g., 'efficiency_focused_rank_1')
        output_base_dir (str): Base output directory
    
    Returns:
        tuple: (schedule_rl, rl_metrics) or (None, None) if error
    """
    try:
        solution_file = os.path.join(
            output_base_dir, 
            "scatter_search", 
            "model_solution", 
            model_name, 
            f"config_{system_id}.json"
        )
        
        print(f"Loading scatter search solution from: {solution_file}")
        
        if not os.path.exists(solution_file):
            print(f"Error: Scatter search solution not found: {solution_file}")
            print(f"Hint: Run 'solution_scatter_search' mode first to generate solutions")
            return None, None
        
        with open(solution_file, 'r') as f:
            solution_data = json.load(f)
        
        schedule_rl = solution_data.get('schedule_detail', [])
        if not schedule_rl:
            print(f"Error: No 'schedule_detail' found in {solution_file}")
            return None, None
        
        rl_metrics = {
            "energy_metrics": {
                "total_satisfaction_pct": solution_data.get('energy_performance', {}).get('overall_satisfaction_pct', 0),
                "total_delivered_energy": solution_data.get('energy_performance', {}).get('total_energy_delivered_kwh', 0)
            },
            "total_cost": solution_data.get('economic_performance', {}).get('total_energy_cost', 0),
            "cost_per_kwh": solution_data.get('economic_performance', {}).get('cost_efficiency', 0),
            "schedule_size": len(schedule_rl),
            "evs_served": solution_data.get('vehicle_performance', {}).get('vehicles_assigned', 0),
            "total_evs": solution_data.get('vehicle_performance', {}).get('vehicles_total', 0),
            "execution_time": solution_data.get('execution_performance', {}).get('execution_time_seconds', 0),
            "model_info": solution_data.get('model_info', {}),
            "source": "scatter_search_pre_calculated"
        }
        
        print(f"Loaded scatter search solution:")
        print(f"   Schedule entries: {len(schedule_rl)}")
        print(f"   Vehicles served: {rl_metrics['evs_served']}/{rl_metrics['total_evs']}")
        print(f"   Energy satisfaction: {rl_metrics['energy_metrics']['total_satisfaction_pct']:.1f}%")
        print(f"   Model: {rl_metrics['model_info'].get('model_name', 'Unknown')}")
        
        return schedule_rl, rl_metrics
        
    except Exception as e:
        print(f"Error loading scatter search solution: {e}")
        return None, None
    
def create_default_scatter_config(config_path):
    """
    Creates a default configuration file for Scatter Search.

    Args:
        config_path (str): The path where the default configuration file will be created.
    """
    default_config = {
        'algorithm': {
            'population_size': 30,
            'ref_set_size': 10,
            'elite_count': 6,
            'diverse_count': 4,
            'max_iterations': 15,
            'max_time_hours': 8.0,
            'combination_probability': 0.7,
            'improvement_probability': 0.3
        },
        'evaluation': {
            'fast': {
                'systems': [1, 2, 3],
                'episodes': 20
            },
            'medium': {
                'systems': [1, 2, 3, 4, 5],
                'episodes': 30
            },
            'full': {
                'systems': [1, 2, 3, 4, 5, 6, 7],
                'episodes': 50
            }
        },
        'output': {
            'save_frequency': 2,
            'checkpoint_dir': './checkpoints/scatter_search'
        }
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Default configuration created in: {config_path}")


def find_latest_checkpoint(checkpoint_dir="./results/scatter_search/checkpoints"):
    """
    Finds the most recent Scatter Search checkpoint.

    Args:
        checkpoint_dir (str, optional): The directory where checkpoints are saved.
                                        Defaults to "./results/scatter_search/checkpoints".

    Returns:
        tuple or None: A tuple containing the path to the latest checkpoint and its iteration number,
                       or None if no checkpoints are found.
    """
    pattern = os.path.join(checkpoint_dir, "scatter_checkpoint_iter_*.json")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    def extract_iteration(path):
        filename = os.path.basename(path)
        return int(filename.split('_')[-1].split('.')[0])

    latest = max(checkpoints, key=extract_iteration)
    iteration = extract_iteration(latest)

    print(f"Most recent checkpoint found: iteration {iteration}")
    return latest, iteration


def run_solution_scatter_search(args):
    """
    Ejecuta evaluación de soluciones usando modelos entrenados de scatter search
    Procesa sistemas según --all_systems, --system_id, o sistema por defecto
    """
    try:
        print("=== SOLUTION SCATTER SEARCH MODE ===")
        
        # Validar argumentos requeridos
        if not args.model_to_evaluate:
            print("Error: --model_to_evaluate is required for solution_scatter_search mode")
            return
        
        if not args.model_name:
            print("Error: --model_name is required for solution_scatter_search mode")
            return
        
        if not os.path.exists(args.model_to_evaluate):
            print(f"Error: Model file not found: {args.model_to_evaluate}")
            return
        
        solution_output_dir = os.path.join(args.output_base_dir, "scatter_search", "model_solution", args.model_name)
        ensure_directory_exists(solution_output_dir)
        
        print(f"Model: {args.model_name}")
        print(f"Model file: {args.model_to_evaluate}")
        print(f"Output directory: {solution_output_dir}")
        
        systems_to_process = []
        
        if args.all_systems:
            print("Processing all systems...")
            all_system_ids = get_all_system_ids(args.data_dir)
            if not all_system_ids:
                print(f"No test_system_*.json files found in {args.data_dir}")
                return
            systems_to_process = all_system_ids
            print(f"Found systems: {systems_to_process}")
        else:
            systems_to_process = [args.system_id]
            print(f"Processing system: {args.system_id}")
        

        temp_evaluator = EfficientSlotBySlotSimulator(
            models_dir="", 
            systems_dir=args.data_dir,
            output_dir=solution_output_dir
        )
        
        print(f"\nLoading model: {args.model_to_evaluate}")
        agent = temp_evaluator.load_agent_once(args.model_to_evaluate)
        
        successful_evaluations = 0
        failed_evaluations = 0
        
        for system_id in systems_to_process:
            print(f"\n{'='*50}")
            print(f"PROCESSING SYSTEM {system_id}")
            print(f"{'='*50}")
            
            try:
                system_config_path = os.path.join(args.data_dir, f"test_system_{system_id}.json")
                
                if not os.path.exists(system_config_path):
                    print(f"Warning: System file not found: {system_config_path}")
                    failed_evaluations += 1
                    continue
                
                system_config = load_system_config(system_config_path)
                print(f"System {system_id} loaded: {len(system_config['arrivals'])} vehicles, {system_config['n_spots']} spots, {len(system_config['chargers'])} chargers")
                
                start_time = time.time()
                schedule_entries = temp_evaluator.simulate_system_efficient(agent, system_config)
                execution_time = time.time() - start_time
                
                metrics = temp_evaluator._calculate_metrics_from_schedule(schedule_entries, system_config)
                
                result = {
                    'system_id': system_id,
                    'model_info': {
                        'model_name': args.model_name,
                        'model_path': args.model_to_evaluate,
                        'filename': os.path.basename(args.model_to_evaluate)
                    },
                    'system_metrics': {
                        'num_vehicles': len(system_config['arrivals']),
                        'num_slots': system_config['n_spots'],
                        'num_chargers': len(system_config['chargers']),
                        'system_context': metrics['system_context']
                    },
                    'execution_performance': {
                        'execution_time_seconds': round(execution_time, 3),
                        'schedule_entries': len(schedule_entries),
                        'timesteps_used': metrics['timesteps_used'],
                        'time_coverage_pct': metrics['time_coverage_pct']
                    },
                    'vehicle_performance': {
                        'vehicles_assigned': metrics['vehicles_assigned'],
                        'vehicles_total': metrics['total_vehicles'],
                        'assignment_ratio_pct': round(metrics['assignment_ratio'] * 100, 1),
                        'vehicles_fully_satisfied': metrics['vehicles_fully_satisfied'],
                        'vehicles_partially_satisfied': metrics['vehicles_partially_satisfied'],
                        'vehicles_not_satisfied': metrics['vehicles_not_satisfied'],
                        'charging_assignments': metrics['charging_assignments'],
                        'parking_assignments': metrics['parking_assignments']
                    },
                    'energy_performance': {
                        'total_energy_required_kwh': round(metrics['total_energy_required'], 2),
                        'total_energy_delivered_kwh': round(metrics['total_energy_delivered'], 2),
                        'overall_satisfaction_pct': round(metrics['overall_satisfaction_pct'], 1),
                        'energy_deficit_kwh': round(metrics['energy_deficit'], 2),
                        'satisfaction_distribution': metrics['satisfaction_distribution']
                    },
                    'economic_performance': {
                        'total_energy_cost': round(metrics['total_energy_cost'], 2),
                        'avg_energy_price_per_kwh': round(metrics['avg_energy_price'], 3),
                        'cost_efficiency': round(metrics['cost_per_kwh_delivered'], 3)
                    },
                    'resource_utilization': {
                        'capacity_utilization_pct': round(metrics['capacity_utilization_pct'], 1),
                        'spots_utilization_pct': round(metrics['spots_utilization_pct'], 1),
                        'chargers_utilization_pct': round(metrics['chargers_utilization_pct'], 1)
                    },
                    'detailed_ev_metrics': metrics['detailed_ev_metrics'],
                    'schedule_detail': schedule_entries,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Guardar resultado como config_{system_id}.json
                output_filename = f"config_{system_id}.json"
                output_filepath = os.path.join(solution_output_dir, output_filename)
                
                with open(output_filepath, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f" SYSTEM {system_id} COMPLETED:")
                print(f"   Satisfaction: {metrics['overall_satisfaction_pct']:.1f}%")
                print(f"   Vehicles satisfied: {metrics['vehicles_fully_satisfied']}/{metrics['total_vehicles']}")
                print(f"   Energy delivered: {metrics['total_energy_delivered']:.1f}/{metrics['total_energy_required']:.1f} kWh")
                print(f"   Execution time: {execution_time:.1f}s")
                print(f"   Saved: {output_filename}")
                
                successful_evaluations += 1
                
            except Exception as e:
                print(f" ERROR processing system {system_id}: {e}")
                import traceback
                traceback.print_exc()
                failed_evaluations += 1
            
            finally:
                # Limpieza de memoria entre sistemas
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Resumen final
        print(f"\n{'='*50}")
        print(f"SOLUTION SCATTER SEARCH COMPLETED")
        print(f"{'='*50}")
        print(f"Model: {args.model_name}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Failed evaluations: {failed_evaluations}")
        print(f"Results saved in: {solution_output_dir}")
        
        if successful_evaluations > 0:
            print(f"\nFiles created:")
            for system_id in systems_to_process:
                config_file = os.path.join(solution_output_dir, f"config_{system_id}.json")
                if os.path.exists(config_file):
                    print(f"   config_{system_id}.json")
                else:
                    print(f"   config_{system_id}.json")
        
    except Exception as e:
        print(f"Error in run_solution_scatter_search: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to parse arguments and run the EV Charging Management System
    in different modes: train, solve (RL), optimize (RL+MILP), run_milp,
    visualize_solution, or scatter_search for hyperparameter optimization.
    """
    try:
        parser = argparse.ArgumentParser(description="EV Charging Management System")
        parser.add_argument("--mode", type=str, required=True,
                    choices=["train", "train_dqn", "solve", "optimize", "run_milp", "visualize_solution", "scatter_search", "solution_scatter_search"],
                    help="Operating mode")
        parser.add_argument("--data_dir", type=str, default="src/configs/system_data",
                            help="Directory containing system configuration files (JSON).")
        parser.add_argument("--output_base_dir", type=str, default="results",
                            help="Base directory for saving all outputs.")
        parser.add_argument("--system_id", type=int, default=1,
                            help="ID of the system configuration to use")
        parser.add_argument("--system", type=str, default=None,
                            help="Specific system to process (for solve/optimize modes)")
        parser.add_argument("--all_systems", action="store_true",
                            help="Process all systems")
        parser.add_argument("--all", action="store_true", help="Alias for --all_systems")
        parser.add_argument("--model_path", type=str, default="./model/ev_scheduler_model_pytorch.pt",
                            help="Path to save/load the DQN model.")
        parser.add_argument("--hyperparameters_path", type=str, default="./src/configs/hyperparameters.yaml",
                            help="Path to the YAML file containing DQN hyperparameters.")
        parser.add_argument("--num_episodes", type=int, default=1000,
                            help="Number of training episodes for DQN")
        parser.add_argument("--episodes", type=int, default=30,
                            help="Episodes per system for training (train mode)")
        parser.add_argument("--log_frequency", type=int, default=50,
                            help="Frequency of logging episode data to console")
        parser.add_argument("--alpha_cost", type=float, default=0.6,
                            help="Weight for the cost objective (0-1)")
        parser.add_argument("--alpha_satisfaction", type=float, default=0.4,
                            help="Weight for the satisfaction objective (0-1)")
        parser.add_argument("--time_limit", type=int, default=900,
                            help="Time limit for MILP solver (seconds)")
        parser.add_argument("--solution_to_visualize", type=str, default=None,
                            help="Path to a saved solution JSON file to visualize")
        parser.add_argument("--scatter_config", type=str, default=None,
                            help="Path to Scatter Search configuration YAML file")
        parser.add_argument("--state_size", type=int, default=256,
                            help="State size for DQN Agent")
        parser.add_argument("--action_size", type=int, default=500,
                            help="Action size for DQN Agent")
        
        parser.add_argument("--model_to_evaluate", type=str, default='./results/scatter_search/trained_models/efficiency_focused_rank_1.pt',
                            help="Path to the specific model (.pt file) to evaluate")

        parser.add_argument("--model_name", type=str, default='efficiency_focused_rank_1',  # Sin .pt
                            help="Name for the model output directory (e.g., 'efficiency_focused_rank_1')")
        args = parser.parse_args()

        if args.all:
            args.all_systems = True

        os.makedirs(args.output_base_dir, exist_ok=True)

        hyperparameters = {}
        if args.mode in ["train_dqn", "train"]:
            try:
                raw_hyperparameters = load_hyperparameters(args.hyperparameters_path)
                hyperparameters = map_hyperparameters(raw_hyperparameters)
                print(f"Hyperparameters loaded from: {args.hyperparameters_path}")
                print(f"Mapped parameters: {list(hyperparameters.keys())}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading hyperparameters: {e}")
                sys.exit(1)

        if args.mode == "scatter_search":
            run_scatter_search_optimization(args)
            
        elif args.mode == "solution_scatter_search":
            run_solution_scatter_search(args)

        elif args.mode == "train":
            print("=== TRAINING MODE ===")

            systems = load_all_test_systems(args.data_dir)

            if not systems:
                print("No systems found for training.")
                return

            print(f"Loaded {len(systems)} test systems.")

            train_config = {
                "state_size": args.state_size,
                "action_size": args.action_size,
                "batch_size": 64,
                "episodes_per_system": args.episodes,
                "checkpoint_frequency": 5,
                "checkpoint_dir": "./checkpoints",
                "model_path": args.model_path,
                "patience": 8,
                "resume_from_checkpoint": True,
                "learning_rate": 0.0005,
                "gamma": 0.99,
                "epsilon": 0.9,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.995,
                "memory_size": 12000,
                "target_update_freq": 10,
                "dueling_network": True
            }

            start_time = time.time()
            agent = train_agent(systems, train_config)
            train_time = time.time() - start_time

            print(f"\nTraining completed in {train_time:.2f} seconds.")
            print(f"Model saved to: {args.model_path}")

        elif args.mode == "train_dqn":
            print("\n--- Starting DQN Training Mode ---")

            systems_to_train = []
            if args.all_systems:
                systems_to_train = get_all_system_ids(args.data_dir)
                if not systems_to_train:
                    print(f"No test_system_*.json files found in {args.data_dir}. Exiting.")
                    sys.exit(1)
                print(f"Found systems to train: {systems_to_train}")
            else:
                systems_to_train.append(args.system_id)

            for current_system_id in systems_to_train:
                print(f"\n--- Training DQN for System ID: {current_system_id} ---")

                dqn_output_root = os.path.join(args.output_base_dir, f"dqn_agent_system_{current_system_id}")
                dqn_models_dir = os.path.join(dqn_output_root, "models")
                dqn_logs_dir = os.path.join(dqn_output_root, "logs")
                dqn_plots_dir = os.path.join(dqn_output_root, "plots")

                ensure_directory_exists(dqn_models_dir)
                ensure_directory_exists(dqn_logs_dir)
                ensure_directory_exists(dqn_plots_dir)

                system_config_filepath = os.path.join(args.data_dir, f"test_system_{current_system_id}.json")
                try:
                    system_config = load_system_config(system_config_filepath)
                    print(f"System configuration loaded from: {system_config_filepath}")
                except FileNotFoundError as e:
                    print(f"Error: {e}. Skipping system {current_system_id}.")
                    continue
                except KeyError as e:
                    print(f"Error in system config file structure for system {current_system_id}: {e}. Skipping.")
                    continue

                env = EVChargingEnv(system_config)

                state_size = args.state_size
                action_size = args.action_size

                print(f"Creating agent with hyperparameters: {hyperparameters}")
                agent = EnhancedDQNAgent(state_size, action_size, **hyperparameters)

                logger = SimpleEpisodeLogger(log_frequency=args.log_frequency)

                if args.model_path:
                    base_model_name = os.path.basename(args.model_path).replace(".pt", "")
                    model_save_path = os.path.join(dqn_models_dir, f"{base_model_name}_system_{current_system_id}.pt")
                else:
                    model_save_path = get_timestamp_filepath(dqn_models_dir, prefix=f"dqn_model_system_{current_system_id}_", suffix=".pt")

                print(f"Training DQN agent for {args.num_episodes} episodes for system {current_system_id}...")
                episode_data_log = train_dqn_agent(
                    agent, env, args.num_episodes, model_save_path=model_save_path, logger=logger
                )

                episode_log_filename = f"dqn_episode_log_system_{current_system_id}.json"
                episode_log_filepath = os.path.join(dqn_logs_dir, episode_log_filename)
                logger.save_episode_data(episode_log_filepath)

                plot_dqn_learning_progress(episode_data_log, save_dir=dqn_plots_dir, title=f"DQN Learning Progress - System {current_system_id}")
                plot_output_filename = "dqn_learning_progress.png"
                print(f"DQN learning progress plot saved to: {os.path.join(dqn_plots_dir, plot_output_filename)}")

                # Save final schedule
                final_schedule = env.get_schedule()
                if final_schedule:
                    schedule_filename = f"dqn_final_schedule_system_{current_system_id}.json"
                    schedule_filepath = os.path.join(dqn_logs_dir, schedule_filename)
                    with open(schedule_filepath, 'w') as f:
                        json.dump(final_schedule, f, indent=4)
                    print(f"Final DQN schedule saved to: {schedule_filepath}")

                    # Visualize final DQN schedule
                    dqn_schedule_plot_filename = f"dqn_final_schedule_system_{current_system_id}.png"
                    visualize_rl_solution(final_schedule, system_config, output_dir=dqn_plots_dir, filename=dqn_schedule_plot_filename)
                    print(f"Final DQN schedule visualization saved to: {os.path.join(dqn_plots_dir, dqn_schedule_plot_filename)}")
                else:
                    print("No final DQN schedule to save or visualize for this system.")

        elif args.mode == "solve":
            print("=== RL SOLUTION MODE ===")

            if args.all_systems:
                print("Processing all systems with RL...")
                systems = load_all_test_systems(args.data_dir)
                for system_id, config in systems.items():
                    print(f"\nSystem {system_id}")
                    schedule_rl, rl_metrics = generate_solution(config, model_path=args.model_path, state_size=args.state_size, action_size=args.action_size)
                    rl_metrics["execution_time"] = time.time()

                    # Save solution
                    save_solution(schedule_rl, rl_metrics, os.path.join(args.output_base_dir, f"rl_solution_{system_id}.json"))

                    # Visualize
                    visualize_solution(schedule_rl, config, f"RL System {system_id}: ")
                return

            # Load specific system
            config = None
            if args.system:
                try_paths = [
                    args.system,
                    os.path.join(args.data_dir, args.system),
                    os.path.join(args.data_dir, f"test_system_{args.system}.json") if args.system.isdigit() else None
                ]
            else:
                try_paths = [os.path.join(args.data_dir, f"test_system_{args.system_id}.json")]

            for path in try_paths:
                if path is None:
                    continue
                try:
                    config = load_data(path)
                    print(f"System loaded: {path}")
                    break
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if config is None:
                print("Could not load the system.")
                return

            # Generate RL solution
            start_time = time.time()
            schedule_rl, metrics = generate_solution(config, model_path=args.model_path)
            solve_time = time.time() - start_time
            metrics["execution_time"] = solve_time

            # Save solution
            test_number = config.get("test_number", "unknown")
            output_file = os.path.join(args.output_base_dir, f"rl_solution_{test_number}.json")
            save_solution(schedule_rl, metrics, output_file)

            # Visualize
            visualize_solution(schedule_rl, config, "RL: ")

        elif args.mode == "optimize":
            print("=== RL+MILP OPTIMIZATION MODE (Using Scatter Search Solutions) ===")

            if args.all_systems:
                print("Processing all systems with Scatter Search + MILP...")
                
                available_systems = get_all_system_ids(args.data_dir)
                if not available_systems:
                    print(f"No test_system_*.json files found in {args.data_dir}")
                    return
                
                for system_id in available_systems:
                    print(f"\nOPTIMIZING SYSTEM {system_id}")
                    
                    try:
                        config = load_data(os.path.join(args.data_dir, f"test_system_{system_id}.json"))
                    except Exception as e:
                        print(f"Error loading system {system_id}: {e}")
                        continue
                    
                    schedule_rl, rl_metrics = load_scatter_search_solution(
                        system_id, args.model_name, args.output_base_dir
                    )
                    
                    if schedule_rl is None:
                        print(f"Skipping system {system_id} - no scatter search solution available")
                        continue
                    
                    print(f"OPTIMIZING WITH MILP...")
                    schedule_milp, milp_metrics = optimize_solution(
                        config,
                        schedule_rl,
                        alpha_cost=args.alpha_cost,
                        alpha_satisfaction=args.alpha_satisfaction,
                        time_limit=args.time_limit
                    )
                    
                    if schedule_milp is not None:
                        compare_solutions(rl_metrics, milp_metrics)
                        
                        # Create organized directory structure
                        system_dir = os.path.join(args.output_base_dir, "MILP_RL", f"milp_optimizer_system_{system_id}")
                        ensure_directory_exists(system_dir)
                        
                        rl_output = os.path.join(system_dir, f"scatter_rl_solution_{system_id}.json")
                        milp_output = os.path.join(system_dir, f"scatter_milp_solution_{system_id}.json")
                        
                        save_solution(schedule_rl, rl_metrics, rl_output)
                        save_solution(schedule_milp, milp_metrics, milp_output)
                        
                        print(f"Generating visualizations...")
                        visualize_rl_solution(schedule_rl, config, output_dir=system_dir, 
                                            filename=f"scatter_rl_solution_{system_id}.png")
                        visualize_milp_solution(schedule_milp, config, output_dir=system_dir, 
                                            filename=f"scatter_milp_solution_{system_id}.png")
                    else:
                        print(f"MILP optimization failed for system {system_id}")
                
                return

            system_id = args.system_id
            if args.system:
                if args.system.isdigit():
                    system_id = int(args.system)
                else:
                    print(f"Error: --system should be a number, got: {args.system}")
                    return

            print(f"Processing system {system_id} with Scatter Search + MILP...")

            config = None
            system_file = os.path.join(args.data_dir, f"test_system_{system_id}.json")
            
            try:
                config = load_data(system_file)
                print(f"System {system_id} loaded: {system_file}")
            except Exception as e:
                print(f"Error loading system {system_id}: {e}")
                return

            print(f"1. LOADING SCATTER SEARCH SOLUTION")
            schedule_rl, rl_metrics = load_scatter_search_solution(
                system_id, args.model_name, args.output_base_dir
            )
            
            if schedule_rl is None:
                print(f"Cannot proceed without scatter search solution")
                print(f"Run this first: python main.py --mode solution_scatter_search --model_name {args.model_name} --system_id {system_id}")
                return

            print(f"2. OPTIMIZING WITH MULTI-OBJECTIVE MILP")
            schedule_milp, milp_metrics = optimize_solution(
                config,
                schedule_rl,
                alpha_cost=args.alpha_cost,
                alpha_satisfaction=args.alpha_satisfaction,
                time_limit=args.time_limit
            )

            if schedule_milp is not None:
                print(f"3. COMPARING SOLUTIONS")
                compare_solutions(rl_metrics, milp_metrics)

                # Create organized directory structure
                test_number = config.get("test_number", system_id)
                system_dir = os.path.join(args.output_base_dir, "MILP_RL", f"milp_optimizer_system_{test_number}")
                ensure_directory_exists(system_dir)

                rl_output = os.path.join(system_dir, f"scatter_rl_solution_{test_number}.json")
                milp_output = os.path.join(system_dir, f"scatter_milp_solution_{test_number}.json")

                save_solution(schedule_rl, rl_metrics, rl_output)
                save_solution(schedule_milp, milp_metrics, milp_output)

                print(f"4. GENERATING VISUALIZATIONS")
                visualize_rl_solution(schedule_rl, config, output_dir=system_dir, 
                                    filename=f"scatter_rl_solution_{test_number}.png")
                visualize_milp_solution(schedule_milp, config, output_dir=system_dir, 
                                    filename=f"scatter_milp_solution_{test_number}.png")
                
                print(f"OPTIMIZATION COMPLETED")
                print(f"Results saved in: {system_dir}")
                print(f"Results based on {rl_metrics['model_info'].get('model_name', 'scatter search')} solution")
            else:
                print(f"MILP optimization did not produce a valid solution.")
                print(f"The scatter search solution will be saved for analysis:")
                
                test_number = config.get("test_number", system_id)
                system_dir = os.path.join(args.output_base_dir, "MILP_RL", f"milp_optimizer_system_{test_number}")
                ensure_directory_exists(system_dir)
                
                rl_output = os.path.join(system_dir, f"scatter_rl_solution_{test_number}.json")
                save_solution(schedule_rl, rl_metrics, rl_output)
                visualize_rl_solution(schedule_rl, config, output_dir=system_dir, 
                                    filename=f"scatter_rl_solution_{test_number}.png")
                print(f"RL solution saved in: {system_dir}")

        elif args.mode == "run_milp":
            print("\n--- Starting MILP Optimization Mode ---")

            # Output subdirectories for the current system
            milp_output_root = os.path.join(args.output_base_dir, f"milp_optimizer_system_{args.system_id}")
            milp_solutions_dir = os.path.join(milp_output_root, "solutions")
            milp_plots_dir = os.path.join(milp_output_root, "plots")

            ensure_directory_exists(milp_solutions_dir)
            ensure_directory_exists(milp_plots_dir)

            # Load System Configuration
            system_config_filepath = os.path.join(args.data_dir, f"test_system_{args.system_id}.json")
            try:
                system_config = load_system_config(system_config_filepath)
                print(f"System configuration loaded from: {system_config_filepath}")
            except FileNotFoundError as e:
                print(f"Error: {e}. Exiting MILP mode.")
                sys.exit(1)
            except KeyError as e:
                print(f"Error in system config file structure for system {args.system_id}: {e}. Exiting.")
                sys.exit(1)

            # Initialize MILP Optimizer
            milp_optimizer = EVChargingMILP(system_config)

            print("Solving MILP problem...")
            try:
                model, milp_schedule, rejected_details, obj_values = milp_optimizer.solve(
                    penalty_unmet=1000.0,
                    time_limit=300,
                    epsilon_satisfaction=0.8
                )

                if milp_schedule:
                    print("MILP solution found.")
                    print(f"Objective values: {obj_values}")

                    # Save MILP solution
                    milp_solution_filename = f"milp_solution_system_{args.system_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    milp_solution_filepath = os.path.join(milp_solutions_dir, milp_solution_filename)
                    solution_data = {
                        "schedule": milp_schedule,
                        "objective_values": obj_values,
                        "rejected_details": rejected_details,
                        "system_id": args.system_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(milp_solution_filepath, 'w') as f:
                        json.dump(solution_data, f, indent=4)
                    print(f"MILP solution saved to: {milp_solution_filepath}")

                    # Visualize MILP solution
                    milp_plot_filename = f"milp_solution_schedule_system_{args.system_id}.png"
                    visualize_milp_solution(milp_schedule, system_config, output_dir=milp_plots_dir, filename=milp_plot_filename)
                    print(f"MILP solution visualization saved to: {os.path.join(milp_plots_dir, milp_plot_filename)}")
                else:
                    print("No MILP solution found.")

            except Exception as e:
                print(f"Error solving MILP: {e}")
                sys.exit(1)

        elif args.mode == "visualize_solution":
            print("\n--- Starting Visualization Mode ---")

            # Output root for visualizations
            visualization_output_root = os.path.join(args.output_base_dir, "visualizations")
            visualize_plots_dir = os.path.join(visualization_output_root, "plots")
            ensure_directory_exists(visualize_plots_dir)

            if not args.solution_to_visualize:
                print("Error: Please provide --solution_to_visualize path for visualization mode.")
                sys.exit(1)

            solution_filepath = args.solution_to_visualize
            if not os.path.exists(solution_filepath):
                print(f"Error: Solution file not found at {solution_filepath}")
                sys.exit(1)

            try:
                with open(solution_filepath, 'r') as f:
                    solution_data = json.load(f)

                schedule = solution_data.get("schedule")
                # If the loaded JSON is just the schedule list
                if not schedule and isinstance(solution_data, list):
                    schedule = solution_data

                if not schedule:
                    print("Error: 'schedule' key not found in the provided solution file or file format is incorrect.")
                    sys.exit(1)

                # Try to infer system_id from the solution filename or content
                system_id_from_filename = None
                match = re.search(r"system_(\d+)", os.path.basename(solution_filepath))
                if match:
                    system_id_from_filename = int(match.group(1))

                # Use provided system_id or try to infer from solution
                current_system_id = args.system_id
                if system_id_from_filename:
                    current_system_id = system_id_from_filename

                # Load System Configuration using the inferred/provided system_id
                system_config_filepath = os.path.join(args.data_dir, f"test_system_{current_system_id}.json")
                try:
                    system_config = load_system_config(system_config_filepath)
                    print(f"System configuration loaded from: {system_config_filepath}")
                except FileNotFoundError as e:
                    print(f"Error: {e}. Cannot load system config for visualization.")
                    sys.exit(1)
                except KeyError as e:
                    print(f"Error in system config file structure for system {current_system_id}: {e}. Cannot load config for visualization.")
                    sys.exit(1)

                output_filename_base = os.path.basename(solution_filepath).replace(".json", "")

                if isinstance(schedule, list) and all(isinstance(item, (list, tuple)) and len(item) >= 4 for item in schedule):
                    print("Visualizing DQN-style schedule.")
                    output_filename = f"{output_filename_base}_viz_dqn.png"
                    visualize_rl_solution(schedule, system_config, output_dir=visualize_plots_dir, filename=output_filename)
                elif isinstance(schedule, dict) and all(isinstance(v, list) for v in schedule.values()):
                    print("Visualizing MILP-style schedule.")
                    output_filename = f"{output_filename_base}_viz_milp.png"
                    visualize_milp_solution(schedule, system_config, output_dir=visualize_plots_dir, filename=output_filename)
                else:
                    print("Unknown schedule format. Cannot visualize.")
                    sys.exit(1)

                print(f"Solution visualization saved to {os.path.join(visualize_plots_dir, output_filename)}")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {solution_filepath}: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred during visualization: {e}")
                sys.exit(1)

        else:
            print(f"Unrecognized mode: {args.mode}. Use 'train', 'train_dqn', 'solve', 'optimize', 'run_milp', 'visualize_solution', or 'scatter_search'.")
    except Exception as e:
        print(f"Error on main: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()