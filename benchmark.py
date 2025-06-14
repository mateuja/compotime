#!/usr/bin/env python3
"""Simple benchmark script for compotime models."""

import time
import numpy as np
import pandas as pd
from compotime import LocalLevelForecaster, LocalTrendForecaster, preprocess


def create_small_synthetic_data(n_timesteps=20, n_series=3, seed=42):
    """Create a small synthetic compositional time series dataset.
    
    Parameters
    ----------
    n_timesteps : int
        Number of time steps (default: 20 - very small for fast testing)
    n_series : int  
        Number of compositional series (default: 3)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Compositional time series that sum to 1 at each timestep
    """
    np.random.seed(seed)
    
    # Generate random walk data
    raw_data = np.random.randn(n_timesteps, n_series).cumsum(axis=0)
    
    # Make it compositional (sum to 1 at each timestep)
    raw_data = np.exp(raw_data)  # Ensure positive
    compositional_data = raw_data / raw_data.sum(axis=1, keepdims=True)
    
    # Create DataFrame with simple range index (more reliable for small datasets)
    df = pd.DataFrame(
        compositional_data, 
        index=pd.RangeIndex(start=0, stop=n_timesteps, step=1),
        columns=[f'Series_{i}' for i in range(n_series)]
    )
    
    return df


def benchmark_model(model_class, data, model_name, n_runs=3):
    """Benchmark a single model.
    
    Parameters
    ----------
    model_class : class
        Model class to benchmark
    data : pd.DataFrame
        Time series data
    model_name : str
        Name of the model for reporting
    n_runs : int
        Number of runs to average over
        
    Returns
    -------
    dict
        Benchmark results
    """
    print(f"\nBenchmarking {model_name}...")
    print(f"Data shape: {data.shape}")
    
    fit_times = []
    predict_times = []
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}", end="")
        
        # Benchmark fitting
        model = model_class()
        start_time = time.time()
        model.fit(data)
        fit_time = time.time() - start_time
        fit_times.append(fit_time)
        print(f" - Fit: {fit_time:.3f}s", end="")
        
        # Benchmark prediction
        start_time = time.time()
        predictions = model.predict(horizon=5)
        predict_time = time.time() - start_time
        predict_times.append(predict_time)
        print(f" - Predict: {predict_time:.3f}s")
    
    results = {
        'model': model_name,
        'data_shape': data.shape,
        'fit_time_mean': np.mean(fit_times),
        'fit_time_std': np.std(fit_times),
        'predict_time_mean': np.mean(predict_times),
        'predict_time_std': np.std(predict_times),
        'total_time_mean': np.mean(fit_times) + np.mean(predict_times)
    }
    
    return results


def print_results(results_list):
    """Print benchmark results in a nice format."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for results in results_list:
        print(f"\nModel: {results['model']}")
        print(f"Data shape: {results['data_shape']}")
        print(f"Fit time:     {results['fit_time_mean']:.3f} ± {results['fit_time_std']:.3f} seconds")
        print(f"Predict time: {results['predict_time_mean']:.3f} ± {results['predict_time_std']:.3f} seconds") 
        print(f"Total time:   {results['total_time_mean']:.3f} seconds")
    
    print("\n" + "="*60)


def main():
    """Run the benchmark."""
    print("Compotime Performance Benchmark")
    print("Using small synthetic dataset for fast testing")
    
    # Create small test data
    print("\nGenerating synthetic data...")
    data = create_small_synthetic_data(n_timesteps=50, n_series=5)
    
    # Apply preprocessing (treat small values)
    data = preprocess.treat_small(data, 1e-6)
    
    print(f"Data preview:")
    print(data.head())
    print(f"Data sums (should be ~1.0): {data.sum(axis=1).head()}")
    
    # Benchmark both models
    results = []
    
    try:
        # Benchmark LocalLevelForecaster
        level_results = benchmark_model(
            LocalLevelForecaster, 
            data, 
            "LocalLevelForecaster"
        )
        results.append(level_results)
        
        # Benchmark LocalTrendForecaster  
        trend_results = benchmark_model(
            LocalTrendForecaster,
            data,
            "LocalTrendForecaster"
        )
        results.append(trend_results)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        return
    
    # Print results
    print_results(results)
    
    # Save results to file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_file = f"benchmark_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()