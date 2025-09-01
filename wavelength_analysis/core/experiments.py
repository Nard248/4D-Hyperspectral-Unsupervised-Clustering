"""
Experimental framework for wavelength analysis

This module provides tools for running systematic experiments
across different configurations and parameters.
"""

from typing import List, Dict, Any
from .analyzer import WavelengthAnalyzer  
from .config import AnalysisConfig, EXPERIMENTAL_CONFIGS
import json
from pathlib import Path


class ExperimentFramework:
    """
    Framework for running systematic wavelength analysis experiments
    """
    
    def __init__(self, base_output_dir: str = "./experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_parameter_sweep(self, base_config: AnalysisConfig, 
                           parameter_ranges: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Run experiments across different parameter values
        
        Args:
            base_config: Base configuration to modify
            parameter_ranges: Dictionary of parameter names to value lists
            
        Returns:
            Dictionary containing all experiment results
        """
        results = {}
        
        for param_name, param_values in parameter_ranges.items():
            for value in param_values:
                # Create modified config
                config = AnalysisConfig(**base_config.to_dict())
                setattr(config, param_name, value)
                
                # Update output directory
                config.output_dir = str(self.base_output_dir / f"{param_name}_{value}")
                
                # Run analysis
                try:
                    analyzer = WavelengthAnalyzer(config)
                    result = analyzer.run_complete_analysis()
                    results[f"{param_name}_{value}"] = result
                except Exception as e:
                    results[f"{param_name}_{value}"] = {"status": "error", "message": str(e)}
        
        return results
    
    def compare_configurations(self, configs: Dict[str, AnalysisConfig]) -> Dict[str, Any]:
        """
        Compare different analysis configurations
        
        Args:
            configs: Dictionary of configuration name to AnalysisConfig
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for config_name, config in configs.items():
            try:
                config.output_dir = str(self.base_output_dir / config_name)
                analyzer = WavelengthAnalyzer(config)
                result = analyzer.run_complete_analysis()
                results[config_name] = result
            except Exception as e:
                results[config_name] = {"status": "error", "message": str(e)}
        
        return results