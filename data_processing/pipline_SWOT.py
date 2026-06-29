"""
Complete Working Example: SWOT + IS2 Lead & Polynya Detection Pipeline

This script demonstrates:
1. Loading synthetic SWOT+IS2 data
2. Running lead and polynya detection
3. Visualizing results
4. Saving output
5. Batch processing multiple tracks

Run with: python example_complete_pipeline.py
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

# Import detector and visualization modules
from swot_polynya_detector import UniversalLeadDetector, DetectionParams
from polynya_visualization import DetectionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_beam(
    n_segments: int = 500,
    scenario: str = 'mixed'
) -> Dict:
    """
    Create synthetic SWOT+IS2 beam data for testing
    
    Parameters
    ----------
    n_segments : int
        Number of along-track segments
    scenario : str
        'pure_pack' - only pack ice
        'polynyas' - high polynya density
        'mixed' - mix of pack ice and polynyas
        
    Returns
    -------
    dict
        Beam structure with IS2 and SWOT data
    """
    
    logger.info(f"Creating synthetic beam ({scenario} scenario, {n_segments} segments)")
    
    # Coordinates
    lat = np.linspace(-80, -85, n_segments)
    lon = np.linspace(0, 15, n_segments)
    
    # Base elevation profile
    h_ice2 = np.ones(n_segments) * 0.8
    
    if scenario == 'pure_pack':
        # Pure pack ice: smooth elevation, high backscatter
        h_ice2 += 0.1 * np.sin(np.linspace(0, 4*np.pi, n_segments))
        sigma0_db = np.random.normal(-10, 1.5, n_segments)
        
    elif scenario == 'polynyas':
        # High polynya density
        # Add polynyas: low elevation, low backscatter
        poly_centers = [100, 250, 400]
        for center in poly_centers:
            i_min = max(0, center - 30)
            i_max = min(n_segments, center + 30)
            h_ice2[i_min:i_max] -= 0.3
        
        sigma0_db = np.random.normal(-12, 2, n_segments)
        sigma0_db[100:130] -= 5  # Low backscatter polynyas
        sigma0_db[250:280] -= 5
        sigma0_db[400:430] -= 5
        
    else:  # mixed
        # Mix of pack and polynyas
        # Add some polynyas
        h_ice2[100:140] -= 0.4
        h_ice2[350:380] -= 0.3
        
        # Sigma0 profile
        sigma0_db = np.random.normal(-10, 2, n_segments)
        sigma0_db[100:140] -= 4  # Polynya backscatter
        sigma0_db[350:380] -= 3
        
        # Add some leads
        h_ice2[200:210] -= 0.5
        sigma0_db[200:210] += 3  # Bright leads
        
        h_ice2[420:430] -= 0.4
        sigma0_db[420:430] += 2
    
    # Add noise
    h_ice2 += np.random.normal(0, 0.05, n_segments)
    sigma0_db += np.random.normal(0, 0.3, n_segments)
    
    # SWOT collocation
    n_swot_per_segment = np.random.randint(5, 20, n_segments)
    
    dist_min = []
    sigma_swot = []
    ssh_swot = []
    
    for i, n_swot in enumerate(n_swot_per_segment):
        # Random distances to SWOT samples
        dist_min.append(np.random.uniform(0, 2000, n_swot))
        
        # SWOT backscatter (correlated with IS2)
        sigma_swot.append(sigma0_db[i] + np.random.normal(0, 0.5, n_swot))
        
        # SWOT SSH
        ssh_swot.append(np.random.normal(0, 0.2, n_swot))
    
    beam = {
        'lat': lat,
        'lon': lon,
        'height_ice2': h_ice2,
        'DIST_MIN': [dist_min],  # Note: list of arrays in cell format
        'sigma_swot': [sigma_swot],
        'ssh_swot': [ssh_swot],
    }
    
    logger.info(f"Synthetic beam created: {len(lat)} segments")
    return beam


def create_synthetic_swot_2d(
    height: int = 256,
    width: int = 256,
    scenario: str = 'mixed'
) -> np.ndarray:
    """
    Create synthetic 2D SWOT sigma0 image for neural network
    
    Parameters
    ----------
    height, width : int
        Image dimensions
    scenario : str
        Same as beam scenarios
        
    Returns
    -------
    ndarray [height, width]
        Synthetic sigma0 image in dB
    """
    
    logger.info(f"Creating synthetic 2D SWOT image ({scenario})")
    
    # Base background
    swot_2d = np.random.normal(-12, 2, (height, width))
    
    if scenario == 'pure_pack':
        # Uniform pack ice
        pass
    
    elif scenario == 'polynyas':
        # Multiple polynya regions
        for _ in range(5):
            cy, cx = np.random.randint(30, height-30), \
                    np.random.randint(30, width-30)
            ry, rx = np.random.randint(20, 50), np.random.randint(20, 50)
            
            y, x = np.ogrid[:height, :width]
            mask = (y - cy)**2 / ry**2 + (x - cx)**2 / rx**2 <= 1
            
            swot_2d[mask] -= np.random.uniform(4, 8)  # Lower backscatter
    
    else:  # mixed
        # Few polynya patches and leads
        # Polynyas
        for _ in range(2):
            cy, cx = np.random.randint(40, height-40), \
                    np.random.randint(40, width-40)
            ry, rx = np.random.randint(20, 40), np.random.randint(20, 40)
            
            y, x = np.ogrid[:height, :width]
            mask = (y - cy)**2 / ry**2 + (x - cx)**2 / rx**2 <= 1
            swot_2d[mask] -= np.random.uniform(3, 6)
        
        # Leads (bright backscatter)
        for _ in range(3):
            y1, y2 = np.random.randint(0, height-50), \
                    np.random.randint(0, height-50)
            x1, x2 = np.random.randint(0, width-50), \
                    np.random.randint(0, width-50)
            
            swot_2d[y1:y1+50, x1:x1+10] += np.random.uniform(2, 4)
    
    return swot_2d


def run_detection(
    beam: Dict,
    swot_2d: np.ndarray = None,
    params: DetectionParams = None
) -> Dict:
    """
    Run lead and polynya detection on a single track
    
    Parameters
    ----------
    beam : dict
        Beam structure
    swot_2d : ndarray, optional
        2D SWOT sigma0 for NN-based polynya detection
    params : DetectionParams, optional
        Detection parameters
        
    Returns
    -------
    dict
        Detection results
    """
    
    if params is None:
        params = DetectionParams(use_nn_polynya=swot_2d is not None)
    
    logger.info("Initializing detector")
    detector = UniversalLeadDetector(params)
    
    logger.info("Running detection")
    results = detector.detect(beam, swot_2d=swot_2d)
    
    # Print summary
    summary = results['summary']
    logger.info(f"Detection complete:")
    logger.info(f"  Pack leads: {summary['pack_leads']}")
    logger.info(f"  Polynyas: {summary['polynya']}")
    logger.info(f"  Open water: {summary['open_water']}")
    logger.info(f"  Thin ice: {summary['thin_ice']}")
    
    return results


def save_results(
    results: Dict,
    output_dir: Path,
    track_name: str = 'track'
) -> Dict:
    """
    Save detection results to disk
    
    Parameters
    ----------
    results : dict
        Detection results
    output_dir : Path
        Output directory
    track_name : str
        Track identifier for filenames
        
    Returns
    -------
    dict
        Saved file paths
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as NPZ
    npz_path = output_dir / f'{track_name}_results.npz'
    np.savez_compressed(
        npz_path,
        lat=results['lat'],
        lon=results['lon'],
        h_ice2=results['h_ice2'],
        sigma0_mean=results['sigma0_mean'],
        is_lead=results['is_lead_final'],
        is_polynya=results['is_polynya'],
        is_open_water=results['is_open_water'],
        is_thin_ice=results['is_thin_ice'],
    )
    logger.info(f"Results saved to {npz_path}")
    
    # Save summary as CSV
    csv_path = output_dir / f'{track_name}_summary.csv'
    import pandas as pd
    summary_df = pd.DataFrame({
        'lat': results['lat'],
        'lon': results['lon'],
        'elevation_m': results['h_ice2'],
        'sigma0_dB': results['sigma0_mean'],
        'is_lead': results['is_lead_final'].astype(int),
        'is_polynya': results['is_polynya'].astype(int),
        'is_open_water': results['is_open_water'].astype(int),
        'is_thin_ice': results['is_thin_ice'].astype(int),
    })
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Summary saved to {csv_path}")
    
    return {
        'npz': npz_path,
        'csv': csv_path,
    }


def visualize_results(
    results: Dict,
    swot_2d: np.ndarray = None,
    output_dir: Path = None,
    track_name: str = 'track'
) -> Dict:
    """
    Create visualization plots
    
    Parameters
    ----------
    results : dict
        Detection results
    swot_2d : ndarray, optional
        2D SWOT background
    output_dir : Path, optional
        Directory to save plots
    track_name : str
        Track identifier for filenames
        
    Returns
    -------
    dict
        Saved plot paths
    """
    
    viz = DetectionVisualizer(figsize=(16, 10))
    plots = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection map
        map_path = output_dir / f'{track_name}_detection_map.png'
        viz.plot_detection_map(results, swot_2d=swot_2d, figname=str(map_path))
        plots['map'] = map_path
        
        # Along-track signatures
        sig_path = output_dir / f'{track_name}_signatures.png'
        viz.plot_along_track_signatures(results, figname=str(sig_path))
        plots['signatures'] = sig_path
        
        # Histograms
        hist_path = output_dir / f'{track_name}_histograms.png'
        viz.plot_sigma0_distributions(results, figname=str(hist_path))
        plots['histograms'] = hist_path
        
        # Statistics
        stat_path = output_dir / f'{track_name}_statistics.png'
        viz.plot_detection_statistics(results, figname=str(stat_path))
        plots['statistics'] = stat_path
        
        logger.info(f"Plots saved to {output_dir}")
    
    return plots


def batch_process(
    scenarios: List[str] = ['pure_pack', 'mixed', 'polynyas'],
    n_segments: int = 500,
    output_dir: Path = None
) -> Dict:
    """
    Batch process multiple scenarios
    
    Parameters
    ----------
    scenarios : list
        Scenarios to process
    n_segments : int
        Segments per track
    output_dir : Path, optional
        Output directory
        
    Returns
    -------
    dict
        Summary statistics
    """
    
    if output_dir is None:
        output_dir = Path('swot_detection_output')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Batch processing {len(scenarios)} scenarios")
    batch_results = {}
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scenario: {scenario}")
        logger.info(f"{'='*60}")
        
        # Create synthetic data
        beam = create_synthetic_beam(n_segments, scenario)
        swot_2d = create_synthetic_swot_2d(256, 256, scenario)
        
        # Detection parameters
        params = DetectionParams(
            use_nn_polynya=True,  # Use NN if available
            make_plots=False,
        )
        
        # Run detection
        results = run_detection(beam, swot_2d, params)
        
        # Save results
        saved = save_results(results, output_dir / scenario, scenario)
        
        # Visualize
        plots = visualize_results(
            results, swot_2d,
            output_dir / scenario / 'plots',
            scenario
        )
        
        # Store batch results
        batch_results[scenario] = {
            'results': results,
            'saved_files': saved,
            'plots': plots,
        }
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    
    for scenario, data in batch_results.items():
        summary = data['results']['summary']
        print(f"{scenario:15s}: Leads={summary['pack_leads']:3d} | " +
              f"Polynya={summary['polynya']:3d} | " +
              f"OpenWater={summary['open_water']:3d} | " +
              f"ThinIce={summary['thin_ice']:3d}")
    
    return batch_results


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("SWOT + IS2 Lead & Polynya Detection - Complete Example")
    print("="*70 + "\n")
    
    # Configuration
    output_dir = Path('swot_detection_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example 1: Single track processing
    print("\n[1/3] SINGLE TRACK PROCESSING")
    print("-" * 70)
    
    logger.info("Creating synthetic data for 'mixed' scenario")
    beam = create_synthetic_beam(500, scenario='mixed')
    swot_2d = create_synthetic_swot_2d(256, 256, scenario='mixed')
    
    logger.info("Setting up detection parameters")
    params = DetectionParams(
        use_nn_polynya=True,
        nn_confidence_thr=0.5,
        ensemble_weight_nn=0.6,
        score_min=2,
    )
    
    logger.info("Running detection")
    results = run_detection(beam, swot_2d, params)
    
    logger.info("Saving results")
    saved = save_results(results, output_dir / 'single_track', 'example')
    
    logger.info("Creating visualizations")
    plots = visualize_results(
        results, swot_2d,
        output_dir / 'single_track' / 'plots',
        'example'
    )
    
    # Example 2: Batch processing
    print("\n[2/3] BATCH PROCESSING")
    print("-" * 70)
    
    batch_results = batch_process(
        scenarios=['pure_pack', 'mixed', 'polynyas'],
        n_segments=300,
        output_dir=output_dir / 'batch'
    )
    
    # Example 3: Parameter sensitivity
    print("\n[3/3] PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 70)
    
    beam = create_synthetic_beam(500, scenario='mixed')
    score_mins = [1, 2, 3, 4]
    
    logger.info("Testing different score_min thresholds")
    sensitivity = {}
    
    for score_min in score_mins:
        params = DetectionParams(score_min=score_min)
        results = run_detection(beam, params=params)
        sensitivity[f'score_min={score_min}'] = results['summary']
        print(f"score_min={score_min}: Leads={results['summary']['pack_leads']}")
    
    # Final summary
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - Single track results: {output_dir / 'single_track'}")
    print(f"  - Batch results: {output_dir / 'batch'}")
    print(f"  - Visualizations: {output_dir / '*' / 'plots'}")
    
    logger.info("All examples completed successfully!")


if __name__ == '__main__':
    main()
